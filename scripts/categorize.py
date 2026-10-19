"""Gate 1: categorise every raw chat message into a root category -- by rules, or by a judge: a local llama-server, or the Kev model loaded in this process.

    python scripts/categorize.py                          raw log -> data/processed/categories.jsonl (rule baseline)
    python scripts/categorize.py --judge-url URL          the same with a decision model as the judge (llama-server, /v1/systemone)
    python scripts/categorize.py --judge-local [RUN]      the same, the Kev model loaded in this process (no server; .venv-kev, judge_local.py)
    python scripts/categorize.py --probe [SAMPLE]         score the baseline on a hand-labelled sample; writes nothing
    python scripts/categorize.py --probe [SAMPLE] --judge-url URL    ... and the judge, with a cutoff sweep

The categories file (what `preprocess.py --recipe` reads) has one row per distinct line (variants such as `乙です` / `乙です！` are
one key; the first occurrence decides): {"key", "category", "by"}, and "p" / "margin" for a judge. A line that is not placed
(no rule fired, or the judge's margin is below the cutoff) is left out of the file: it stays "uncategorized" for a recipe, never
forced. Untranslated rows are categorised too: they are the new messages whose Korean is still to be written.

The judge pass is resumable: every answer goes to a journal (config.JUDGE_PASS_FILE) as it arrives, a rerun asks only about the
lines not in it, and a new --cutoff needs no new request. With --use-channel a raw row's `channel` (the app's chat channel, kept by
fetch_data.py) is given to the judge with the line. If the judge server cannot be reached the rules are used instead, with a
warning (never over a file a judge made). The data never leaves this machine: the server is a local llama-server.
"""

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from categorizer import BASELINE, categorize
from config import CATEGORIES_FILE, GATE1_COMPARE_DIR, GATE1_SAMPLE, JUDGE_CUTOFF, JUDGE_LOCAL_RUN, JUDGE_PASS_FILE, JUDGE_URL, RAW_LOGS
from dataset_recipe import line_key
from gate_compare import check_label, collect_answers, probe_path, probe_record, save_probe
from gate_eval import SampleError, evaluate_categorizer, format_gate_report, read_sample
from gate_judge import (SWEEP_CUTOFFS, GateJudge, JournalError, categories_from, cutoff_sweep, format_sweep,
                        judge_id, journal_row, read_journal, trim_torn_tail, write_categories)
from judge_client import JudgeError, SystemOneClient
from judge_prompts import DEFAULT as DEFAULT_PROMPT
from judge_prompts import PromptError, judge_question, list_variants, load_variant
from judge_local import DTYPES, LocalKevClient
from taxonomy import load_taxonomy

MAX_CONSECUTIVE_FAILURES = 20  # requests that fail one after the other: the server is gone, stop instead of skipping the log
PROGRESS_EVERY = 1000


def distinct_lines(raw_path, stats):
    """(key, original, channel) of every distinct line of the raw log, the first occurrence deciding; `stats["damaged"]` counts
    the rows that are not JSON objects."""
    seen = set()
    stats["damaged"] = 0
    with open(raw_path, encoding="utf-8") as raw:
        for line in raw:
            try:
                row = json.loads(line)
            except ValueError:
                stats["damaged"] += 1
                continue
            original = (row.get("original") or "").strip() if isinstance(row, dict) else ""
            if not original:
                if not isinstance(row, dict):
                    stats["damaged"] += 1
                continue
            key = line_key(original)
            if key in seen:
                continue
            seen.add(key)
            channel = row.get("channel")
            yield key, original, channel if isinstance(channel, str) and channel else None


def categorise_raw(raw_path, out_path):
    """Write the categories file for the raw log; returns (lines per root, uncategorized lines, damaged rows)."""
    per_root, uncategorized, stats = {}, 0, {}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:  # LF on every OS: a recipe records its sha256
        for key, original, _ in distinct_lines(raw_path, stats):
            category = categorize(original)
            if category is None:
                uncategorized += 1
                continue
            per_root[category] = per_root.get(category, 0) + 1
            out.write(json.dumps({"key": key, "category": category, "by": BASELINE}, ensure_ascii=False) + "\n")
    return per_root, uncategorized, stats["damaged"]


def judge_raw(raw_path, journal_path, judge, by, use_channel, done):
    """Ask the judge about every distinct line that is not in `done` yet, journaling each answer as it comes.
    Returns (keys of this raw log, lines the server failed on, damaged rows, whether the server stopped answering)."""
    keys, unjudged, consecutive, asked, stats = [], 0, 0, 0, {}
    os.makedirs(os.path.dirname(os.path.abspath(journal_path)), exist_ok=True)
    with open(journal_path, "a", encoding="utf-8", newline="\n") as journal:
        for key, original, channel in distinct_lines(raw_path, stats):
            keys.append(key)
            if key in done:
                continue
            answer = judge.answer(original, channel if use_channel else None)
            if answer is None:
                unjudged += 1
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    return keys, unjudged, stats["damaged"], True
                continue
            consecutive = 0
            done[key] = journal_row(key, answer, by)
            journal.write(json.dumps(done[key], ensure_ascii=False) + "\n")
            journal.flush()
            asked += 1
            if asked % PROGRESS_EVERY == 0:
                print(f"  ... {asked} lines judged")
    return keys, unjudged, stats["damaged"], False


def report_distribution(per_root, uncategorized, damaged, out_path, by=BASELINE, why="no rule fired: left out of the file"):
    total = sum(per_root.values()) + uncategorized
    print(f"\n--- Gate 1 ({by}): {total} distinct lines -> {out_path} ---")
    width = max([len("uncategorized"), *map(len, per_root)])
    for root, count in sorted(per_root.items(), key=lambda item: -item[1]):
        print(f"  {root:<{width}}  {count:>8}  {count / total:>6.1%}")
    print(f"  {'uncategorized':<{width}}  {uncategorized:>8}  {uncategorized / total if total else 0:>6.1%}   ({why})")
    if damaged:
        print(f"{damaged} damaged row(s) skipped")


class PassError(Exception):
    """The judge pass cannot go on (a journal of another judge, a judge that stopped answering); the message says what to do."""


def judge_pass(raw_path, journal_path, out_path, judge, by, cutoff, use_channel, force=False, where="the judge", who="the judge"):
    """The whole judge pass over the raw log: resume from the journal, ask about the lines not in it, derive the categories file
    for `cutoff`, print the split. What `main` does after it has a judge, a function so a notebook can run it on the judge it has
    already loaded (a second process would load the model twice). Raises PassError."""
    if force and os.path.exists(journal_path):
        os.remove(journal_path)
    trim_torn_tail(journal_path)
    try:
        done = read_journal(journal_path, by)
    except JournalError as error:
        raise PassError(str(error)) from None
    print(f"Judge {by} at {where}: {len(done)} lines already in {journal_path}")
    keys, unjudged, damaged, gone = judge_raw(raw_path, journal_path, judge, by, use_channel, done)
    if gone:
        raise PassError(f"the judge failed {MAX_CONSECUTIVE_FAILURES} requests in a row: is {who} still working? "
                        f"The answers so far are in {journal_path}: run again to resume.")
    categories, uncertain = categories_from({key: done[key] for key in keys if key in done}, cutoff)
    write_categories(out_path, categories)
    per_root = {}
    for row in categories:
        per_root[row["category"]] = per_root.get(row["category"], 0) + 1
    report_distribution(per_root, uncertain + unjudged, damaged, out_path, by=by,
                        why=f"margin below the cutoff {cutoff}, or {who} failed: left out of the file")
    if unjudged:
        print(f"{unjudged} line(s) unjudged ({who} failed on them): run again to ask about them.")


def existing_judges(path):
    """The `by` of every judge that wrote rows into the categories file (the rule baseline does not count)."""
    found = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    by = json.loads(line).get("by")
                except (ValueError, AttributeError):
                    continue
                if by and by != BASELINE:
                    found.add(by)
    except OSError:
        pass
    return found


def probe(args, rows, judge, by):
    print(format_gate_report(evaluate_categorizer(rows, categorize), BASELINE))
    if judge is None:
        return
    channels = {row["original"]: row.get("channel") for row in rows}
    started = time.time()
    sweep = cutoff_sweep(rows, judge, SWEEP_CUTOFFS)
    seconds_per_line = (time.time() - started) / len(rows)
    print("\n" + format_gate_report(evaluate_categorizer(rows, lambda text: judge.predict(text, channels.get(text))), f"{by}, cutoff {args.cutoff}"))
    if judge.failures:
        print(f"\n{judge.failures} request(s) failed: counted as no answer.")
    print("\n" + format_sweep(sweep, by))
    print(f"\n{seconds_per_line:.2f} s per line on this pass")
    if args.save_probe:
        path = save_probe(args.probe_dir, probe_record(args.save_probe, by, collect_answers(rows, judge, args.use_channel),
                                                       seconds_per_line, args.use_channel,
                                                       notes=f"prompt {args.judge_prompt}"))
        print(f"probe saved: {path} (compare judges with scripts/compare_judges.py)")


def main(argv=None, post=None, loader=None):
    parser = argparse.ArgumentParser(description="Categorise the raw chat log into root categories (rules, or a judge on llama-server).")
    parser.add_argument("--raw", default=RAW_LOGS, help="raw log to categorise (default: %(default)s)")
    parser.add_argument("--out", default=CATEGORIES_FILE, help="categories file to write (default: %(default)s)")
    parser.add_argument("--probe", nargs="?", const=GATE1_SAMPLE, metavar="SAMPLE",
                        help="score the baseline (and the judge) on a hand-labelled sample instead (default sample: %s)" % GATE1_SAMPLE)
    parser.add_argument("--judge-url", metavar="URL", help=f"llama-server serving a decision model, e.g. {JUDGE_URL}: the judge decides")
    parser.add_argument("--judge-local", nargs="?", const=JUDGE_LOCAL_RUN, metavar="RUN",
                        help=f"the judge is the Kev model loaded in this process instead (a Hub id[@revision] or a directory; default {JUDGE_LOCAL_RUN})")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16", help="with --judge-local: the precision (default: %(default)s)")
    parser.add_argument("--device", help="with --judge-local: cuda or cpu (default: cuda when there is one)")
    parser.add_argument("--judge-model", metavar="NAME", help="the model to ask, when the server runs several (router mode)")
    parser.add_argument("--cutoff", type=float, default=JUDGE_CUTOFF, help="smallest margin (top probability minus the second) that counts "
                        "as an answer (default: %(default)s, a guess: choose it with --probe)")
    parser.add_argument("--save-probe", metavar="LABEL", help="with --probe and a judge: save what the judge answered as data/eval/gate1-compare/LABEL.json, "
                        "to compare judges with scripts/compare_judges.py (a plain name like 9b-q8; never replaces a saved probe)")
    parser.add_argument("--probe-dir", default=GATE1_COMPARE_DIR, help="where saved probes go (default: %(default)s)")
    parser.add_argument("--judge-prompt", default=DEFAULT_PROMPT, metavar="NAME",
                        help=f"what the judge is asked: a file of configs/judge_prompts/ (there are: {', '.join(list_variants())}); "
                        "default = the taxonomy's root descriptions as they are. Another wording is another judge: its journal and probe never mix")
    parser.add_argument("--journal", default=JUDGE_PASS_FILE, help="the judge's answers, resumable (default: %(default)s)")
    parser.add_argument("--use-channel", action="store_true", help="give the judge the chat channel of a line (a `channel` field of the row)")
    parser.add_argument("--force", action="store_true", help="start the judge journal again instead of stopping on rows of another judge")
    args = parser.parse_args(argv)
    if args.judge_local and (args.judge_url or args.judge_model):
        parser.error("--judge-local runs the model in this process: it excludes --judge-url and --judge-model")
    if args.judge_prompt != DEFAULT_PROMPT and not (args.judge_url or args.judge_local):
        parser.error("--judge-prompt changes what a judge is asked: it needs --judge-url or --judge-local")
    try:  # before any model loads
        instructions, options = judge_question(load_taxonomy(), load_variant(args.judge_prompt))
    except PromptError as error:
        parser.error(str(error))
    if args.save_probe:
        if args.probe is None or not (args.judge_url or args.judge_local):
            parser.error("--save-probe saves what a judge answered on the sample: it needs --probe and --judge-url or --judge-local")
        try:
            check_label(args.save_probe)
        except ValueError as error:
            parser.error(str(error))
        if os.path.exists(probe_path(args.probe_dir, args.save_probe)):  # before the model loads: not after a long run
            print(f"[ERROR] {probe_path(args.probe_dir, args.save_probe)} exists: use another label, or delete that file to probe again")
            sys.exit(1)

    judge = by = None
    if args.judge_local:
        client, where, who, kind = LocalKevClient(args.judge_local, device=args.device, dtype=args.dtype, loader=loader), args.judge_local, "the model", "kev-local"
    elif args.judge_url:
        client, where, who, kind = SystemOneClient(args.judge_url, model=args.judge_model, post=post), args.judge_url, "the server", "systemone"
    if args.judge_local or args.judge_url:
        try:
            by = judge_id(client.check_server(), instructions, options, kind=kind)
            judge = GateJudge(client, options, args.cutoff, instructions)
        except JudgeError as error:
            if args.probe is not None:
                print(f"[ERROR] the judge cannot be used: {error}")
                sys.exit(1)
            made_by = existing_judges(args.out)
            if made_by:
                print(f"[ERROR] the judge cannot be used ({error}) and {args.out} was made by {', '.join(sorted(made_by))}: "
                      "not replacing it with the rules. Fix that, or use another --out.", file=sys.stderr)
                sys.exit(1)
            print(f"[WARNING] the judge cannot be used ({error}): categorising with the rule baseline ({BASELINE}) instead.", file=sys.stderr)

    if args.probe is not None:
        try:
            rows = read_sample(args.probe)
        except SampleError as error:
            print(f"[ERROR] {error}")
            sys.exit(1)
        if not args.use_channel:
            rows = [{"original": row["original"], "category": row["category"]} for row in rows]
        probe(args, rows, judge, by)
        return

    if not os.path.exists(args.raw):
        print(f"[ERROR] Raw data not found at: {args.raw}")
        sys.exit(1)
    if judge is None:
        per_root, uncategorized, damaged = categorise_raw(args.raw, args.out)
        report_distribution(per_root, uncategorized, damaged, args.out)
        return

    try:
        judge_pass(args.raw, args.journal, args.out, judge, by, args.cutoff, args.use_channel, args.force, where, who)
    except PassError as error:
        print(f"[ERROR] {error}")
        sys.exit(1)

if __name__ == "__main__":
    main()
