import argparse
import contextlib
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_LOGS, PROCESSED_LOGS, INSTRUCTION, EVAL_DATASET_PATH, PREPROCESS_MAX_SUSPICIOUS, CATEGORIES_FILE
from dataset_recipe import UNCATEGORIZED, RecipeError, line_key, load_recipe, read_categories, recipe_path, select_lines
from eval_metrics import load_eval_dataset
import manifest
from overlap import EvalOverlap
from valsplit import is_validation
from prompts import STYLES, build_prompt, style_for_template
from text_rules import HANGEUL_PATTERN, JP_PATTERN

# --- Filters (from the TranslateGemma pipeline's preprocess) ---

# A row is skipped for exactly one of these, checked in this order.
REASONS = (
    "empty field",
    "hangeul in original",
    "JP residual in translation",
    "JP in original (ko_ja)",
    "Hangeul residual in translation (ko_ja)",
    "hallucination",
    "recruitment spam",
    "eval overlap",
    "eval overlap (near)",
    "duplicate",
    "json error",
)


MAX_SUSPICIOUS = PREPROCESS_MAX_SUSPICIOUS

# Rows dropped because the line is bad data: too many of them means the wrong file or a broken export (see max_drop).
SUSPICIOUS = ("hangeul in original", "JP residual in translation", "hallucination")
# Drops that are normal for chat logs and say nothing about the file's health: not counted in the guard's share.
EXPECTED = ("empty field", "recruitment spam", "eval overlap", "eval overlap (near)", "duplicate", "not selected (recipe)")
# Only counted (and reported) when a dataset recipe chose the training lines: passed lines the recipe left out.
RECIPE_REASON = "not selected (recipe)"


# Output row layouts, by pipeline: unsloth trains on instruction/input/output rows,
# LLaMA-Factory reads original/translated columns (see update_dataset_info.py).
FORMATS = {
    "instruction": lambda original, translated: {"instruction": INSTRUCTION, "input": original, "output": translated},
    "pair": lambda original, translated: {"original": original, "translated": translated},
}
DEFAULT_FORMAT = "instruction"
# The share of the lines written to the validation file for the pair format, unless --val-fraction says otherwise.
PAIR_VAL_FRACTION = 0.05


def forward_row(original, translated, style=None):
    """The ja->ko `pair` row with the instruction of `style` (None = the raw line)."""
    return {"original": build_prompt(style, "ja-ko", original), "translated": translated}


def reverse_row(original, translated, style=None):
    """The ko->ja row of the same pair: the Korean is the input, the Japanese the answer."""
    return {"original": build_prompt(style, "ko-ja", translated), "translated": original}


def pair_rows(original, translated, style=None, reverse=False):
    """The forward row, and with `reverse` the ko->ja row after it (no filtering here)."""
    rows = [forward_row(original, translated, style)]
    if reverse:
        rows.append(reverse_row(original, translated, style))
    return rows


def clean_reason(original, translated, direction="ja-ko", keep=()):
    """Why a row must not be trained on, or None when it is clean.

    Duplicates are not decided here (they need the rows seen so far). A dataset recipe can `keep` a filter by name
    (only "recruitment spam" so far): that filter lets the row through.
    """
    original = (original or "").strip()
    translated = (translated or "").strip()
    # resonance-stream writes `translated: null` for lines it never translated.
    if not original or not translated:
        return "empty field"
    if direction == "ko-ja":
        # Reverse rows (Korean in, Japanese out): kana/kanji in the Korean input, or Hangeul left in the Japanese answer.
        if JP_PATTERN.search(original):
            return "JP in original (ko_ja)"
        if HANGEUL_PATTERN.search(translated):
            return "Hangeul residual in translation (ko_ja)"
    else:
        # The source is Japanese chat; Hangeul means a wrong row.
        if HANGEUL_PATTERN.search(original):
            return "hangeul in original"
        # Kana/kanji left in the Korean output: the line was not translated.
        if JP_PATTERN.search(translated):
            return "JP residual in translation"
    # Looping output: the translation is more than 10x longer than the input.
    if len(translated) > len(original) * 10:
        return "hallucination"
    # Guild recruitment walls: long lines with several IDs.
    if len(original) > 150 and original.count('ID:') > 1 and "recruitment spam" not in keep:
        return "recruitment spam"
    return None


def row_outcomes(lines, eval_overlap, seen, keep=()):
    """(original, translated, reason) of each raw JSONL line, in order: reason is None when the ja->ko row passes every
    filter, else why it is dropped (`clean_reason`, then "eval overlap" / "eval overlap (near)", "duplicate", or
    "json error" with original and translated None).

    `seen` is the set of inputs already taken (the duplicate check): a passed original is added to it before it is
    yielded, and the caller may add to it too (the reverse rows' Korean inputs share it).
    """
    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            yield None, None, "json error"
            continue
        original = (data.get("original") or "").strip()
        translated = (data.get("translated") or "").strip()

        reason = clean_reason(original, translated, keep=keep)
        if reason is None:
            found = eval_overlap.check(original)
            if found:
                reason = "eval overlap" if found == "exact" else "eval overlap (near)"
        if reason is None and original in seen:
            reason = "duplicate"  # the first occurrence wins
        if reason is None:
            seen.add(original)
        yield original, translated, reason


def _report(counts):
    skipped = counts["total"] - counts["passed"]
    print("\n--- Preprocessing Report ---")
    width = max(len(reason) for reason in REASONS + (RECIPE_REASON,))
    print(f"{'Total input':<{width + 2}}: {counts['total']}")
    print(f"{'Passed':<{width + 2}}: {counts['passed']}")
    print(f"{'Skipped':<{width + 2}}: {skipped}")
    for reason in REASONS + ((RECIPE_REASON,) if RECIPE_REASON in counts else ()):
        print(f"  {reason:<{width}}: {counts[reason]}")
    if counts["validation"]:
        print(f"{'Validation rows':<{width + 2}}: {counts['validation']} of the passed rows (the rest are training rows)")


def _report_recipe(name, recipe, selection):
    """What the recipe asked for and got, per key: its weight, the lines available and the lines selected."""
    allocation = selection.allocation
    print(f"\n--- Recipe {name} (seed {recipe.seed}) ---")
    width = max(len(key) for key in recipe.weights)
    print(f"{'category':<{width}}  {'share':>6}  {'available':>9}  {'selected':>8}")
    for key, share in recipe.shares.items():
        print(f"{key:<{width}}  {float(share):>6.1%}  {selection.available[key]:>9}  {allocation.targets[key]:>8}")
    ended = f", limited by '{allocation.limited_by}' (it has no more lines)" if allocation.limited_by else ""
    asked = f" of the {allocation.requested} asked for" if allocation.requested and allocation.requested != allocation.total else ""
    print(f"{allocation.total} lines selected{asked}{ended}. A line is its normalised text: variants count once.")


def transform_for_lora(input_file, output_file, fmt=DEFAULT_FORMAT, style=None, reverse=False, eval_originals=(),
                       val_file=None, val_fraction=0.0, max_drop=None, recipe=None, categories=None, recipe_report=None,
                       recipe_name="recipe"):
    """Clean raw {original, translated} rows into training rows of layout `fmt`.

    Rows whose original is one of `eval_originals` (exactly or nearly: see overlap.py) are dropped, so the
    model is never trained on what it is evaluated on; their reverse rows go with them.
    With `val_file` and `val_fraction`, about that share of the lines is written to `val_file` instead of `output_file`,
    chosen per line by valsplit.py (both directions of a pair go to the same file).
    With `max_drop`, exits 1 when more than that share of the usable rows (not untranslated, duplicate, spam or eval
    lines) were dropped as suspicious (Hangeul in the source, Japanese left in the output, runaway-long output).
    With a dataset `recipe` (dataset_recipe.py) and the `categories` of the lines ({line key: category}), the training
    file holds only the lines the recipe selects by category weight; the validation lines are decided before the recipe
    and do not depend on it, so every recipe has the same validation file. The lines it leaves out are counted as
    "not selected (recipe)". `recipe_report`, a dict, is filled with the plan (available / targets / total / limited_by /
    requested). The recipe's `keep` lets recruitment walls through the filters.
    Returns the counts per outcome (`total`, `passed` -- training and validation rows together --, `validation`
    and one per reason).
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose one of: {', '.join(sorted(FORMATS))}")
    if (style or reverse) and fmt != "pair":
        raise ValueError("a prompt style or --reverse needs the pair format (--format pair)")
    if val_file and fmt != "pair":
        raise ValueError("a validation split needs the pair format (--format pair); the instruction format is split by unsloth/split_dataset.py")
    if not os.path.exists(input_file):
        print(f"[ERROR] Raw data not found at: {input_file}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    counts = {"total": 0, "passed": 0, **{reason: 0 for reason in REASONS}, "validation": 0}
    keep = recipe.keep if recipe else ()
    if recipe:
        counts[RECIPE_REASON] = 0
    seen_inputs = set()
    eval_overlap = EvalOverlap(eval_originals)
    usable = suspicious = 0  # forward direction only: the guard's denominator and numerator
    pending = []  # with a recipe: the training-side lines (original, translated), held until the recipe has chosen
    selection = None

    def emit(sink, original, translated):
        """Write a passed line to `sink`, and with `reverse` its ko->ja row (tried on its own, counted on its own)."""
        counts["validation"] += sink is f_val
        if fmt == "pair":
            sink.write(json.dumps(forward_row(original, translated, style), ensure_ascii=False) + '\n')
        else:
            sink.write(json.dumps(FORMATS[fmt](original, translated), ensure_ascii=False) + '\n')

        # The reverse direction is tried only for rows whose forward direction passed, with its own filters and
        # counted on its own (as experiment/translategemma's bidirectional preprocess did).
        if reverse:
            counts["total"] += 1
            reason = clean_reason(translated, original, direction="ko-ja", keep=keep)
            if reason is None and translated in seen_inputs:
                reason = "duplicate"
            if reason:
                counts[reason] += 1
                return
            seen_inputs.add(translated)
            counts["passed"] += 1
            counts["validation"] += sink is f_val
            sink.write(json.dumps(reverse_row(original, translated, style), ensure_ascii=False) + '\n')

    with contextlib.ExitStack() as files:
        f_in = files.enter_context(open(input_file, 'r', encoding='utf-8'))
        # LF on every OS: the files' sha256 goes into the manifest and MLflow.
        f_out = files.enter_context(open(output_file, 'w', encoding='utf-8', newline='\n'))
        f_val = files.enter_context(open(val_file, 'w', encoding='utf-8', newline='\n')) if val_file else None
        for original, translated, reason in row_outcomes(f_in, eval_overlap, seen_inputs, keep):
            counts["total"] += 1
            if reason == "json error":
                counts[reason] += 1
                continue
            usable += reason not in EXPECTED
            suspicious += reason in SUSPICIOUS
            if reason:
                counts[reason] += 1
                continue

            counts["passed"] += 1
            # The pair's side is decided by its Japanese line, so the reverse row follows the forward row.
            sink = f_val if f_val and is_validation(original, val_fraction) else f_out
            if recipe is not None and sink is f_out:
                pending.append((original, translated))
                continue
            emit(sink, original, translated)

        if recipe is not None:
            keyed = [(line_key(original), original, translated) for original, translated in pending]
            known = categories or {}
            selection = select_lines(recipe, [(key, known.get(key, UNCATEGORIZED)) for key, _, _ in keyed])
            for key, original, translated in keyed:
                if key in selection.keys:
                    emit(f_out, original, translated)
                else:
                    counts["passed"] -= 1  # it passed the filters, but the recipe did not take it
                    counts[RECIPE_REASON] += 1

    _report(counts)
    if selection is not None:
        _report_recipe(recipe_name, recipe, selection)
        if recipe_report is not None:
            allocation = selection.allocation
            recipe_report.update(available=selection.available, targets=allocation.targets, total=allocation.total,
                                 limited_by=allocation.limited_by, requested=allocation.requested)
        if not selection.allocation.total:
            print("[ERROR] The recipe selected no training lines: a category of the recipe has no lines in this log "
                  f"('{selection.allocation.limited_by}'). Check the categories file and the recipe's category names.")
            sys.exit(1)
    share = suspicious / usable if usable else 0.0
    print(f"{'Suspicious drops':<18}: {suspicious}/{usable} usable rows ({share:.1%})"
          + (f", limit {max_drop:.0%}" if max_drop is not None else ""))
    if max_drop is not None and share > max_drop:
        print(f"[ERROR] {share:.1%} of the usable rows were dropped as suspicious (Hangeul in the source, Japanese left in the output, "
              f"runaway-long output); the limit is {max_drop:.0%}. Is this the right raw log? Look at the counts above; "
              "--max-drop raises the limit once you know why.")
        sys.exit(1)
    if counts["passed"] == 0:
        print("[ERROR] Preprocessing produced 0 lines. Check your raw input file.")
        sys.exit(1)
    print(f"Successfully preprocessed {counts['passed']} lines.")
    return counts


def _profile_template(model):
    """The LLaMA-Factory template of the model profile (needs pyyaml; only --prompt auto gets here)."""
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "llamafactory"))
    from lf_tools import load_profile, model_name
    return load_profile(model_name(model)).template


def main(argv=None):
    parser = argparse.ArgumentParser(description="Clean the raw chat log into training rows.")
    parser.add_argument("--format", choices=sorted(FORMATS), default=DEFAULT_FORMAT, dest="fmt",
                        help="row layout (default: %(default)s)")
    parser.add_argument("--prompt", default="none", choices=["none", "auto", *sorted(STYLES)],
                        help="pair format only: instruction put before each line; auto = the style of --model's template "
                             "(default: %(default)s = the raw line)")
    parser.add_argument("--model", help="llamafactory profile for --prompt auto (default: $RESONANCE_LF_PROFILE, else the default)")
    parser.add_argument("--eval-set", default=EVAL_DATASET_PATH,
                        help="eval dataset whose lines are kept out of the training data (default: %(default)s)")
    parser.add_argument("--keep-eval", action="store_true",
                        help="do not drop the eval set's lines from the training data (experiments only: it inflates every score)")
    parser.add_argument("--val-fraction", type=float, default=None,
                        help="share of the lines written to the validation file lora_train_data.val.jsonl "
                             "(default: 0.05 for --format pair, which LLaMA-Factory validates on; 0 for instruction)")
    parser.add_argument("--max-drop", type=float, default=MAX_SUSPICIOUS,
                        help="stop when more than this share of the usable rows is dropped as suspicious (default: %(default)s; 1 = never)")
    parser.add_argument("--reverse", action="store_true",
                        help="pair format only: also write every clean row as ko->ja, Korean as the input")
    parser.add_argument("--recipe", default=os.environ.get("RESONANCE_RECIPE") or None, metavar="NAME_OR_FILE",
                        help="choose the training lines by category weight: a recipe name (configs/datasets/<name>.json) or a "
                             "JSON file (default: $RESONANCE_RECIPE; none = every clean line is trained on)")
    parser.add_argument("--categories", default=CATEGORIES_FILE,
                        help="with --recipe: the file saying which category each line is in (default: %(default)s)")
    args = parser.parse_args(argv)
    style = {"none": None, "auto": None}.get(args.prompt, args.prompt)
    if args.prompt == "auto":
        style = style_for_template(_profile_template(args.model))
    val_fraction = (PAIR_VAL_FRACTION if args.fmt == "pair" else 0.0) if args.val_fraction is None else args.val_fraction
    if not 0 <= val_fraction < 1:
        parser.error("--val-fraction must be at least 0 and below 1")
    eval_set, eval_originals = _eval_originals(args)
    manifest.remove_manifest(PROCESSED_LOGS)  # a failed run must not leave the old file's manifest or validation rows behind
    val_file = manifest.val_path(PROCESSED_LOGS) if val_fraction > 0 else None
    recipe, recipe_inputs = _load_recipe(args)
    recipe_report = {}
    counts = transform_for_lora(RAW_LOGS, PROCESSED_LOGS, args.fmt, style, args.reverse, eval_originals, val_file, val_fraction, args.max_drop,
                                recipe=recipe, categories=recipe_inputs and recipe_inputs["categories"], recipe_report=recipe_report,
                                recipe_name=recipe_inputs["name"] if recipe_inputs else "recipe")
    if val_file and not counts["validation"]:
        print("[WARNING] No line fell into the validation split (too few lines?): training will refuse this data.")
    block = _recipe_block(recipe, recipe_inputs, recipe_report) if recipe else None
    path = manifest.write_manifest(PROCESSED_LOGS, args.fmt, style, args.reverse, RAW_LOGS, counts, eval_set, len(eval_originals), val_fraction,
                                   recipe=block)
    print(f"Manifest written -> {path}")


def _load_recipe(args):
    """(Recipe, inputs) of `--recipe`, or (None, None) without one; exits 1 with a message when it cannot be used.

    A recipe given by name is configs/datasets/<name>.json; anything with a path separator or ending in .json is a file.
    """
    if not args.recipe:
        return None, None
    try:
        by_path = "/" in args.recipe or os.sep in args.recipe or args.recipe.endswith(".json")
        path = args.recipe if by_path else recipe_path(args.recipe)
        recipe, digest = load_recipe(path)
        categories = read_categories(args.categories)
    except RecipeError as error:
        print(f"[ERROR] {error}")
        sys.exit(1)
    inputs = {"name": os.path.splitext(os.path.basename(path))[0], "sha256": digest, "categories": categories,
              "categories_file": args.categories}
    return recipe, inputs


def _recipe_block(recipe, inputs, report):
    """What the manifest says about the recipe: which one (name and content hash), and what each category offered and gave."""
    return {
        "name": inputs["name"], "sha256": inputs["sha256"], "seed": recipe.seed, "keep": list(recipe.keep),
        "weights": {key: str(weight) for key, weight in recipe.weights.items()},
        "categories_file": os.path.basename(inputs["categories_file"]),
        "categories_sha256": manifest.file_sha256(inputs["categories_file"]),
        "total": report["total"], "requested": report["requested"], "limited_by": report["limited_by"],
        "available": report["available"], "selected": report["targets"],
    }


def _eval_originals(args):
    """(eval file, eval lines) to keep out of training, per the command line: (None, []) when excluding is off or there is no set."""
    if args.keep_eval:
        print("--keep-eval: the eval set's lines stay in the training data.")
        return None, []
    if not os.path.exists(args.eval_set):
        print(f"There is no eval set at {args.eval_set}: nothing is excluded from the training data.")
        return None, []
    try:
        samples = load_eval_dataset(args.eval_set)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    print(f"Excluding eval lines ({len(samples)}) from {args.eval_set}.")
    return args.eval_set, [sample["original"] for sample in samples]


if __name__ == "__main__":
    main()
