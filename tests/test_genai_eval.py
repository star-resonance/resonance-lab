import json

import pytest

import eval_metrics
import genai_eval

SAMPLES = [
    {"original": "こんにちは", "translated": "안녕하세요", "category": "chat"},
    {"original": "ヒグマを倒した", "translated": "산적 두목을 쓰러뜨렸다", "category": "boss"},
]


def row(**fields):
    return {"original": "こんにちは", "reference": "안녕하세요", "prediction": "안녕하세요", "raw_output": "안녕하세요",
            "category": "chat", **fields}


# --- the predictions file --------------------------------------------------------------------------


def test_rows_pair_each_sample_with_its_prediction_and_raw_output():
    rows = genai_eval.prediction_rows(SAMPLES, ["안녕", "두목 처치"], ["<think></think>안녕", "두목 처치"])
    assert rows[0] == {"original": "こんにちは", "reference": "안녕하세요", "prediction": "안녕",
                       "raw_output": "<think></think>안녕", "category": "chat"}
    assert rows[1]["category"] == "boss"


def test_a_sample_without_a_category_is_unknown():
    rows = genai_eval.prediction_rows([{"original": "a", "translated": "b"}], ["c"], ["c"])
    assert rows[0]["category"] == "unknown"


def test_rows_refuse_a_different_number_of_predictions():
    with pytest.raises(ValueError, match="2 samples but 1 predictions"):
        genai_eval.prediction_rows(SAMPLES, ["x"], ["x"])


def test_the_predictions_path_names_the_profile_and_the_prompt(tmp_path):
    path = genai_eval.predictions_path(str(tmp_path), "hy-mt2-1.8b", "training")
    assert path == str(tmp_path / "hy-mt2-1.8b-training.jsonl")


def test_predictions_round_trip_as_readable_utf8(tmp_path):
    path = tmp_path / "out" / "p.jsonl"
    rows = genai_eval.prediction_rows(SAMPLES, ["안녕", "두목"], ["안녕", "두목"])
    genai_eval.write_predictions(str(path), rows)
    assert "안녕하세요" in path.read_text(encoding="utf-8")  # ensure_ascii=False
    assert genai_eval.read_predictions(str(path)) == rows


def test_a_missing_predictions_file_says_which_stage_writes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="eval.py"):
        genai_eval.read_predictions(str(tmp_path / "none.jsonl"))


def test_a_row_without_a_prediction_is_refused_with_its_line_number(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps(row()) + "\n\n" + json.dumps({"original": "a", "reference": "b"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3.*prediction"):
        genai_eval.read_predictions(str(path))


def test_an_unreadable_line_is_refused_with_its_line_number(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        genai_eval.read_predictions(str(path))


def test_a_row_without_a_raw_output_falls_back_to_the_prediction(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"original": "a", "reference": "b", "prediction": "c"}) + "\n", encoding="utf-8")
    [loaded] = genai_eval.read_predictions(str(path))
    assert loaded["raw_output"] == "c" and loaded["category"] == "unknown"


# --- the evaluation data ---------------------------------------------------------------------------


def test_evaluation_data_has_the_shape_mlflow_genai_evaluate_reads():
    [item] = genai_eval.evaluation_data([row()])
    assert item == {
        "inputs": {"text": "こんにちは", "category": "chat"},
        "outputs": {"translation": "안녕하세요", "raw_output": "안녕하세요"},
        "expectations": {"expected_response": "안녕하세요"},
    }


# --- the per-sample scores -------------------------------------------------------------------------


def test_chrf_is_100_for_the_reference_and_lower_for_anything_else():
    assert genai_eval.chrf_score("안녕하세요", "안녕하세요") == pytest.approx(100)
    assert genai_eval.chrf_score("고블린", "안녕하세요") < 20


def test_jp_leak_is_true_when_japanese_is_left_in_the_translation():
    assert genai_eval.jp_leak("안녕 ありがとう") is True
    assert genai_eval.jp_leak("안녕") is False


def test_think_leak_is_true_only_for_real_reasoning_in_the_raw_output():
    assert genai_eval.think_leak("<think>he says hi</think>안녕") is True
    assert genai_eval.think_leak("<think></think>안녕") is False
    assert genai_eval.think_leak("안녕") is False


def test_term_ok_has_no_value_when_the_line_holds_no_term():
    assert genai_eval.term_ok("こんにちは", "안녕") is None


def test_term_ok_is_true_only_when_every_term_in_the_line_is_used():
    assert genai_eval.term_ok("ヒグマを倒した", "산적 두목을 쓰러뜨렸다") is True
    assert genai_eval.term_ok("ヒグマを倒した", "곰을 쓰러뜨렸다") is False


def test_discord_must_stay_latin():
    assert genai_eval.discord_violation("discordで話そう", "디스코드에서 얘기하자") is True
    assert genai_eval.discord_violation("discordで話そう", "Discord에서 얘기하자") is False


def test_exact_match_compares_with_the_reference():
    assert genai_eval.exact_match("안녕하세요", "안녕하세요") is True
    assert genai_eval.exact_match("안녕", "안녕하세요") is False


def test_every_scorer_has_a_unique_name_and_takes_the_same_four_values():
    names = [name for name, _ in genai_eval.SCORERS]
    assert len(names) == len(set(names))
    for _, function in genai_eval.SCORERS:
        function("こんにちは", "안녕", "안녕하세요", "안녕")  # original, prediction, reference, raw output


def test_per_sample_scores_agree_with_the_run_level_report():
    """The traces must not disagree with the numbers the runs already carry (eval_metrics.evaluate)."""
    predictions = ["산적 두목을 쓰러뜨렸다 ありがとう", "곰"]
    raws = ["<think>x</think>산적 두목을 쓰러뜨렸다 ありがとう", "곰"]
    samples = [{"original": "ヒグマを倒した", "translated": "산적 두목을 쓰러뜨렸다", "category": "boss"},
               {"original": "ヒグマ discord", "translated": "두목 Discord", "category": "boss"}]
    report = eval_metrics.evaluate(samples, predictions, raws)
    scorers = dict(genai_eval.SCORERS)

    def count(name):
        return sum(bool(scorers[name](s["original"], p, s["translated"], r)) for s, p, r in zip(samples, predictions, raws))

    assert count("jp_leak") == report["jp_leakage"]
    assert count("think_leak") == report["think_leakage"]
    assert count("discord_violation") == report["discord_violations"]
    assert count("exact_match") == report["exact_match"]
    values = [scorers["term_ok"](s["original"], p, s["translated"], r) for s, p, r in zip(samples, predictions, raws)]
    assert values == [True, False]  # the second line's term is missed: the report counts one term miss
    assert report["term_total"] - report["term_hits"] == 1


# --- the MLflow scorers ----------------------------------------------------------------------------


def test_build_scorers_wraps_each_score_for_mlflow():
    registered = {}

    def fake_scorer(name):
        def decorate(function):
            registered[name] = function
            return function
        return decorate

    scorers = genai_eval.build_scorers(fake_scorer)
    assert len(scorers) == len(genai_eval.SCORERS) and set(registered) == {name for name, _ in genai_eval.SCORERS}

    item = genai_eval.evaluation_data([row(prediction="안녕 ありがとう", raw_output="안녕 ありがとう")])[0]
    call = {"inputs": item["inputs"], "outputs": item["outputs"], "expectations": item["expectations"]}
    assert registered["jp_leak"](**call) is True
    assert registered["exact_match"](**call) is False
    assert registered["chrf"](**call) < 100


def test_predictions_end_lines_with_lf_on_every_os(tmp_path):
    path = tmp_path / "p.jsonl"
    genai_eval.write_predictions(str(path), genai_eval.prediction_rows(SAMPLES, ["안녕", "두목"], ["안녕", "두목"]))
    data = path.read_bytes()
    assert data.count(b"\n") == 2 and b"\r" not in data
