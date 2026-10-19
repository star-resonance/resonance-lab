import json
from fractions import Fraction

import pytest

import gate_pipeline
from dataset_recipe import Recipe, RecipeError, line_key
from gate_eval import SampleError, read_sample


def lines_of(**by_text):
    """distinct lines as categorize.distinct_lines yields them: (key, original, channel)."""
    return [(line_key(text), text, channel) for text, channel in by_text.items()]


def numbered(prefix, n, channel=None):
    return [(line_key(f"{prefix}{i}"), f"{prefix}{i}", channel) for i in range(n)]


def guess_by_prefix(original, channel):
    return {"s": "social", "r": "recruitment", "q": "question"}.get(original[0])  # anything else: no guess


# --- the draft of the labelled sample ----------------------------------------------------------------------------------


def test_the_draft_takes_the_same_number_from_each_guessed_category():
    lines = numbered("s", 30) + numbered("r", 30) + numbered("q", 30)

    rows = gate_pipeline.draft_sample(lines, 12, seed=1, guess=guess_by_prefix)

    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert counts == {"social": 4, "recruitment": 4, "question": 4}


def test_a_category_that_runs_out_gives_the_rest_to_the_others():
    lines = numbered("s", 2) + numbered("r", 30) + numbered("q", 30)

    rows = gate_pipeline.draft_sample(lines, 12, seed=1, guess=guess_by_prefix)

    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert counts["social"] == 2 and len(rows) == 12 and counts["recruitment"] == counts["question"] == 5


def test_the_draft_is_no_longer_than_the_lines_there_are():
    assert len(gate_pipeline.draft_sample(numbered("s", 3), 10, seed=1, guess=guess_by_prefix)) == 3


def test_the_same_seed_gives_the_same_draft_and_another_seed_another():
    lines = numbered("s", 50) + numbered("r", 50)

    first = gate_pipeline.draft_sample(lines, 20, seed=1, guess=guess_by_prefix)

    assert first == gate_pipeline.draft_sample(list(reversed(lines)), 20, seed=1, guess=guess_by_prefix)
    assert first != gate_pipeline.draft_sample(lines, 20, seed=2, guess=guess_by_prefix)


def test_lines_the_judge_has_no_guess_for_are_a_stratum_of_their_own_marked_so():
    rows = gate_pipeline.draft_sample(numbered("s", 5) + numbered("x", 5), 6, seed=1, guess=guess_by_prefix)

    assert {row["category"] for row in rows} == {"social", gate_pipeline.UNGUESSED}


def test_an_unreviewed_guess_cannot_pass_as_a_label(tmp_path):
    """The marker is not a category: read_sample refuses the file until the human has put a real label on the line."""
    rows = gate_pipeline.draft_sample(numbered("x", 2), 2, seed=1, guess=guess_by_prefix)
    path = tmp_path / "gate1-sample.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    with pytest.raises(SampleError, match="not a category"):
        read_sample(str(path))


def test_the_channel_of_a_line_is_kept_in_the_row():
    rows = gate_pipeline.draft_sample(lines_of(**{"s1": "PARTY", "s2": None}), 2, seed=1, guess=guess_by_prefix)

    assert {row["original"]: row.get("channel") for row in rows} == {"s1": "PARTY", "s2": None}
    assert all("channel" not in row for row in rows if row["original"] == "s2")


def test_lines_already_in_the_sample_are_not_drawn_again():
    lines = numbered("s", 10)
    done = {key for key, _, _ in lines[:6]}

    rows = gate_pipeline.draft_sample(lines, 10, seed=1, guess=guess_by_prefix, exclude=done)

    assert len(rows) == 4 and not {line_key(row["original"]) for row in rows} & done


def test_the_guess_gets_the_channel_too():
    seen = []
    gate_pipeline.draft_sample(lines_of(**{"s1": "PARTY"}), 1, seed=1, guess=lambda text, channel: seen.append((text, channel)))

    assert seen == [("s1", "PARTY")]


def test_the_draft_file_is_written_utf8_and_never_over_an_edited_one(tmp_path):
    path = tmp_path / "gate1-sample.draft.jsonl"
    rows = [{"original": "おはよう", "category": "social"}]

    gate_pipeline.write_draft(str(path), rows)
    assert json.loads(path.read_text(encoding="utf-8")) == rows[0] and "おはよう" in path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        gate_pipeline.write_draft(str(path), [])
    gate_pipeline.write_draft(str(path), [], overwrite=True)
    assert path.read_text(encoding="utf-8") == ""


def test_the_draft_path_sits_next_to_the_sample():
    assert gate_pipeline.draft_path("data/eval/gate1-sample.jsonl") == "data/eval/gate1-sample.draft.jsonl"


# --- the channel mix ---------------------------------------------------------------------------------------------------


def test_the_channel_report_counts_lines_per_channel():
    report = gate_pipeline.channel_report(lines_of(a="PARTY", b="PARTY", c="WORLD", d=None))

    assert report == {"total": 4, "with_channel": 3, "channels": {"PARTY": 2, "WORLD": 1}}
    text = gate_pipeline.format_channel_report(report)
    assert "PARTY" in text and "2" in text and "3 of 4" in text


def test_without_any_channel_the_report_says_what_that_means_for_the_judge():
    text = gate_pipeline.format_channel_report(gate_pipeline.channel_report(lines_of(a=None, b=None)))

    assert "no line has a channel" in text and "--use-channel" in text


# --- the recipe against the categories ---------------------------------------------------------------------------------


def recipe(**weights):
    return Recipe(seed=1, total=None, keep=(), weights={name: Fraction(weight) for name, weight in weights.items()})


def test_the_preview_says_what_each_category_would_contribute():
    keys = [line_key(f"s{i}") for i in range(20)] + [line_key(f"r{i}") for i in range(4)] + [line_key("unknown")]
    categories = {**{line_key(f"s{i}"): "social" for i in range(20)}, **{line_key(f"r{i}"): "recruitment" for i in range(4)}}

    preview = gate_pipeline.recipe_preview(recipe(social=1, recruitment=1), categories, keys)

    rows = {row["key"]: row for row in preview["rows"]}
    assert rows["social"]["available"] == 20 and rows["recruitment"]["available"] == 4
    assert rows["recruitment"]["target"] == 4 and rows["social"]["target"] == 4  # equal weights: the short category ends it
    assert preview["limited_by"] == "recruitment" and preview["total"] == 8
    assert preview["uncategorized"] == 1 and preview["distinct"] == 25
    assert rows["social"]["natural"] == pytest.approx(20 / 24) and rows["social"]["share"] == pytest.approx(0.5)


def test_children_count_for_their_root_and_a_category_the_recipe_does_not_name_is_outside():
    keys = [line_key("a"), line_key("b"), line_key("c")]
    categories = {line_key("a"): "game/combat", line_key("b"): "game", line_key("c"): "spam"}

    preview = gate_pipeline.recipe_preview(recipe(game=1), categories, keys)

    assert preview["rows"][0]["available"] == 2 and preview["outside"] == 1


def test_a_recipe_of_the_repo_is_previewed_by_its_name():
    """The notebook passes a name; the file's loading (load_recipe returns the recipe AND its hash) is not its business."""
    keys = [line_key(f"s{i}") for i in range(10)]

    preview = gate_pipeline.recipe_preview_by_name("balanced-v1", {key: "social" for key in keys}, keys)

    assert {row["key"] for row in preview["rows"]} >= {"social", "chat", "game"}
    social = next(row for row in preview["rows"] if row["key"] == "social")
    assert social["available"] == 10


def test_an_unknown_recipe_name_is_a_recipe_error_naming_the_file():
    with pytest.raises(RecipeError, match="not-a-recipe"):
        gate_pipeline.recipe_preview_by_name("not-a-recipe", {}, [])


def test_the_preview_is_text_with_a_marker_for_the_limiting_category():
    keys = [line_key("a")]
    preview = gate_pipeline.recipe_preview(recipe(social=1, chat=1), {line_key("a"): "social"}, keys)

    text = gate_pipeline.format_recipe_preview(preview, "balanced-v1")

    assert "balanced-v1" in text and "social" in text and "limits" in text


# --- the training side: what preprocess.py would actually train on ------------------------------------------------------


def raw_rows(**translated_by_original):
    return [{"original": original, "translated": translated} for original, translated in translated_by_original.items()]


def test_the_training_side_holds_only_the_lines_that_pass_preprocess_filters(write_jsonl):
    raw = write_jsonl("raw.jsonl", [
        {"original": "おはよう", "translated": "좋은 아침"},
        {"original": "おはよう", "translated": "안녕"},  # duplicate: the first occurrence wins
        {"original": "ログイン", "translated": None},  # no Korean yet
        {"original": "한국어", "translated": "한국어"},  # Hangeul in the source
        {"original": "スカイ", "translated": "스카이"},  # an eval line
        "{broken",
    ])

    side = gate_pipeline.train_side(raw, {}, eval_originals=["スカイ"], val_fraction=0.0)

    assert side.usable == frozenset({line_key("おはよう")})
    assert side.pending == frozenset({line_key("ログイン")})
    assert side.lines[line_key("おはよう")] == "uncategorized"  # no entry in the categories file: preprocess's default


def test_validation_lines_are_not_on_the_training_side(write_jsonl):
    from valsplit import is_validation

    originals = [f"テスト{i}" for i in range(40)]
    raw = write_jsonl("raw.jsonl", [{"original": text, "translated": f"테스트{i}"} for i, text in enumerate(originals)])

    side = gate_pipeline.train_side(raw, {}, val_fraction=0.3)

    validation = {line_key(text) for text in originals if is_validation(text, 0.3)}
    assert validation and side.usable == frozenset(line_key(text) for text in originals) - validation


def test_a_line_that_is_translated_once_is_not_pending(write_jsonl):
    raw = write_jsonl("raw.jsonl", [{"original": "ありがとう", "translated": None}, {"original": "ありがとう", "translated": "고마워"}])

    side = gate_pipeline.train_side(raw, {}, val_fraction=0.0)

    assert side.pending == frozenset() and side.usable == frozenset({line_key("ありがとう")})
    on_the_validation_side = gate_pipeline.train_side(raw, {}, val_fraction=0.999)
    assert on_the_validation_side.usable == frozenset() and on_the_validation_side.pending == frozenset()


def test_the_overview_counts_each_category_with_its_natural_share_and_pending_lines(write_jsonl):
    raw = write_jsonl("raw.jsonl", [
        *[{"original": f"挨拶{i}", "translated": f"인사{i}"} for i in range(3)],
        {"original": "挨拶まだ", "translated": None},
        {"original": "雑談", "translated": "잡담"},
    ])
    categories = {**{line_key(f"挨拶{i}"): "social/greeting" for i in range(3)}, line_key("挨拶まだ"): "social/greeting"}

    rows = {row["category"]: row for row in gate_pipeline.category_overview(gate_pipeline.train_side(raw, categories, val_fraction=0.0))}

    assert rows["social/greeting"] == {"category": "social/greeting", "lines": 4, "usable": 3, "natural": pytest.approx(0.75), "pending": 1}
    assert rows["uncategorized"]["usable"] == 1 and rows["uncategorized"]["natural"] == pytest.approx(0.25)


def test_the_training_side_preview_selects_exactly_what_preprocess_selects(write_jsonl, tmp_path):
    import preprocess

    rows = [{"original": f"挨拶{i}", "translated": f"인사{i}"} for i in range(30)] + \
           [{"original": f"募集{i}", "translated": f"모집{i}"} for i in range(9)] + \
           [{"original": f"雑談{i}", "translated": f"잡담{i}"} for i in range(12)] + \
           [{"original": "ID:1 " + "あ" * 150 + " ID:2", "translated": "모집 벽"}]  # a recruitment wall: kept by the recipe
    raw = write_jsonl("raw.jsonl", rows)
    categories = {**{line_key(f"挨拶{i}"): "social" for i in range(30)}, **{line_key(f"募集{i}"): "recruitment/party" for i in range(9)},
                  line_key(rows[-1]["original"]): "spam/wall"}
    chosen = Recipe(seed=3, total=None, keep=("recruitment spam",),
                    weights={"social": Fraction(2), "recruitment": Fraction(1), "uncategorized": Fraction(1), "spam": Fraction(1, 10)})
    report = {}
    preprocess.transform_for_lora(raw, str(tmp_path / "out.jsonl"), fmt="pair", val_file=str(tmp_path / "val.jsonl"), val_fraction=0.2,
                                  recipe=chosen, categories=categories, recipe_report=report)

    preview = gate_pipeline.train_side_preview(chosen, gate_pipeline.train_side(raw, categories, val_fraction=0.2, keep=chosen.keep))

    assert {row["key"]: row["target"] for row in preview["rows"]} == report["targets"]
    assert {row["key"]: row["available"] for row in preview["rows"]} == report["available"]
    assert (preview["total"], preview["limited_by"]) == (report["total"], report["limited_by"])


def test_the_preview_refuses_lines_read_with_another_keep_than_the_recipes(write_jsonl):
    raw = write_jsonl("raw.jsonl", [{"original": "おはよう", "translated": "좋은 아침"}])
    side = gate_pipeline.train_side(raw, {}, val_fraction=0.0)

    with pytest.raises(RecipeError, match="recruitment spam"):
        gate_pipeline.train_side_preview(Recipe(seed=1, total=None, keep=("recruitment spam",), weights={"social": Fraction(1)}), side)


def test_the_training_side_preview_is_text_with_the_limit_and_what_falls_outside(write_jsonl):
    raw = write_jsonl("raw.jsonl", raw_rows(**{"挨拶": "인사", "雑談": "잡담", "募集": "모집"}))
    side = gate_pipeline.train_side(raw, {line_key("挨拶"): "social", line_key("雑談"): "chat"}, val_fraction=0.0)

    preview = gate_pipeline.train_side_preview(recipe(social=1, chat=1), side)
    text = gate_pipeline.format_train_side_preview(preview, "my-mix")

    assert preview["outside"] == 1 and preview["usable"] == 3
    assert "my-mix" in text and "would select 2" in text and "1 usable line is in a category the recipe does not name" in text
    overview = gate_pipeline.format_category_overview(gate_pipeline.category_overview(side))
    assert "social" in overview and "pending" in overview


# --- choosing and saving a recipe ----------------------------------------------------------------------------------


def test_a_chosen_mix_is_a_checked_recipe():
    mapping, chosen = gate_pipeline.choose_recipe({"social": 2, "chat/casual": 1.5}, seed=7, total=100, keep=["recruitment spam"],
                                                  description="mix A")

    assert mapping == {"description": "mix A", "seed": 7, "total": 100, "keep": ["recruitment spam"],
                       "categories": {"social": {"weight": 2}, "chat/casual": {"weight": 1.5}}}
    assert chosen.weights == {"social": Fraction(2), "chat/casual": Fraction(3, 2)} and chosen.keep == ("recruitment spam",)


def test_a_mix_with_a_bad_weight_or_category_is_refused_with_the_reason():
    with pytest.raises(RecipeError, match="weight must be a number above 0"):
        gate_pipeline.choose_recipe({"social": 0})
    with pytest.raises(RecipeError, match="lower-case path"):
        gate_pipeline.choose_recipe({"Social": 1})


def test_a_saved_recipe_reads_back_as_the_same_recipe_with_lf_line_endings(tmp_path):
    mapping, chosen = gate_pipeline.choose_recipe({"social": 1, "game": 3}, description="게임 위주")

    path = gate_pipeline.save_recipe("game-heavy", mapping, directory=str(tmp_path))

    from dataset_recipe import load_recipe
    assert path == str(tmp_path / "game-heavy.json")
    assert load_recipe(path)[0] == chosen
    data = (tmp_path / "game-heavy.json").read_bytes()
    assert b"\r" not in data and data.endswith(b"\n") and "게임 위주".encode() in data


def test_saving_never_replaces_a_recipe_unless_asked(tmp_path):
    mapping, _ = gate_pipeline.choose_recipe({"social": 1})
    gate_pipeline.save_recipe("mine", mapping, directory=str(tmp_path))
    other, _ = gate_pipeline.choose_recipe({"chat": 1})

    with pytest.raises(RecipeError, match="already exists"):
        gate_pipeline.save_recipe("mine", other, directory=str(tmp_path))
    gate_pipeline.save_recipe("mine", other, directory=str(tmp_path), overwrite=True)
    assert json.loads((tmp_path / "mine.json").read_text(encoding="utf-8"))["categories"] == {"chat": {"weight": 1}}


def test_a_recipe_name_that_is_not_a_file_name_is_refused(tmp_path):
    mapping, _ = gate_pipeline.choose_recipe({"social": 1})

    with pytest.raises(RecipeError, match="not a recipe name"):
        gate_pipeline.save_recipe("../escape", mapping, directory=str(tmp_path))


# --- the decision text -----------------------------------------------------------------------------------


def report(accuracy, coverage=1.0, precision=None, n=200):
    covered = round(n * coverage)
    return {"n": n, "covered": covered, "abstained": n - covered, "correct": round(n * accuracy), "wrong": 0, "accuracy": accuracy,
            "coverage": coverage, "precision": precision if precision is not None else accuracy, "per_root": {}, "confusion": {}}


def test_the_decision_text_names_the_judge_the_cutoff_and_the_scores_against_the_rules():
    text = gate_pipeline.decision_text(by="kev-local:kev-9b@v1.0+bf16:ab12cd34", cutoff=0.3, judge=report(0.80, 0.9, 0.88),
                                       rules=report(0.62, 0.7, 0.85), seconds_per_line=0.2, lines=50000, categorized=41000)

    assert "kev-local:kev-9b@v1.0+bf16:ab12cd34" in text and "0.3" in text
    assert "80%" in text and "62%" in text and "200 lines" in text
    assert "beats the rules" in text
    assert "41,000" in text and "50,000" in text and "0.2 s" in text


def test_the_decision_text_says_when_the_judge_does_not_beat_the_rules():
    text = gate_pipeline.decision_text(by="x", cutoff=0.3, judge=report(0.55), rules=report(0.62))

    assert "does NOT beat the rules" in text


def test_the_decision_text_leaves_out_what_it_was_not_given():
    text = gate_pipeline.decision_text(by="x", cutoff=0.3, judge=report(0.9), rules=report(0.6))

    assert "full pass" not in text and "None" not in text


# --- will the model fit the card ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("run, dtype, gb", [
    ("jaredpalmer/kev-0.8b@v1.0", "bf16", 2.6),
    ("jaredpalmer/kev-4b@v1.0", "bf16", 9.0),
    ("jaredpalmer/kev-9b@v1.0", "bf16", 19.0),
    ("jaredpalmer/kev-9b", "fp32", 37.0),
    ("jaredpalmer/kev-27b@v1.0", "bf16", 55.0),
])
def test_the_memory_estimate_follows_the_parameter_count_and_the_precision(run, dtype, gb):
    assert gate_pipeline.estimated_gb(run, dtype) == pytest.approx(gb)


def test_a_run_whose_size_is_not_in_its_name_has_no_estimate():
    assert gate_pipeline.estimated_gb("D:/models/my-kev", "bf16") is None


def test_a_model_that_does_not_fit_gets_a_warning_naming_a_smaller_one():
    text = gate_pipeline.memory_warning("jaredpalmer/kev-9b@v1.0", "bf16", free_gb=15.2)

    assert "19" in text and "15.2" in text and "kev-4b" in text


def test_a_model_that_fits_or_cannot_be_judged_gets_no_warning():
    assert gate_pipeline.memory_warning("jaredpalmer/kev-4b@v1.0", "bf16", free_gb=15.2) is None
    assert gate_pipeline.memory_warning("jaredpalmer/kev-9b@v1.0", "bf16", free_gb=None) is None  # CPU: no VRAM to compare
    assert gate_pipeline.memory_warning("D:/models/my-kev", "bf16", free_gb=1.0) is None
