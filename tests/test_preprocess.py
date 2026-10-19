import json
import os
import re

import pytest

import config
import preprocess
from config import INSTRUCTION
from conftest import read_jsonl


def test_maps_raw_logs_to_instruction_rows(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"original": "おやすみ！", "translated": "잘 자!"}])
    out = tmp_path / "processed" / "train.jsonl"

    preprocess.transform_for_lora(src, str(out))

    assert read_jsonl(out) == [{"instruction": INSTRUCTION, "input": "おやすみ！", "output": "잘 자!"}]


def test_exits_when_raw_file_is_missing(tmp_path):
    with pytest.raises(SystemExit) as exc:
        preprocess.transform_for_lora(str(tmp_path / "missing.jsonl"), str(tmp_path / "out.jsonl"))
    assert exc.value.code == 1


def test_exits_when_raw_file_is_empty(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [])

    with pytest.raises(SystemExit) as exc:
        preprocess.transform_for_lora(src, str(tmp_path / "out.jsonl"))
    assert exc.value.code == 1


def test_skips_rows_the_app_never_translated(write_jsonl, tmp_path, capsys):
    # resonance-stream writes `translated: null` for lines it never translated.
    src = write_jsonl(
        "raw.jsonl",
        [
            {"pid": 1, "original": "おやすみ！", "translated": None, "timestamp": 1},
            {"pid": 2, "original": "遺跡1F", "translated": "유적 1F", "timestamp": 2},
            {"pid": 3, "original": "スカイ", "translated": "  ", "timestamp": 3},
            {"pid": 4, "original": None, "translated": "번역", "timestamp": 4},
        ],
    )
    out = tmp_path / "out.jsonl"

    preprocess.transform_for_lora(src, str(out))

    assert read_jsonl(out) == [{"instruction": INSTRUCTION, "input": "遺跡1F", "output": "유적 1F"}]
    out = capsys.readouterr().out
    assert re.search(r"Skipped\s*: 3", out)
    assert re.search(r"empty field\s*: 3", out)


def test_exits_when_no_row_was_translated(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"original": "おやすみ！", "translated": None}])

    with pytest.raises(SystemExit) as exc:
        preprocess.transform_for_lora(src, str(tmp_path / "out.jsonl"))
    assert exc.value.code == 1


@pytest.mark.parametrize(
    "original, translated, reason",
    [
        ("おやすみ！", "잘 자!", None),
        ("", "잘 자!", "empty field"),
        ("おやすみ！", "   ", "empty field"),
        ("ムクボ3돌 완료", "무크보 3돌 완료", "hangeul in original"),
        ("おやすみ！", "おやすみ!", "JP residual in translation"),
        ("遺跡1F", "遺跡 1F", "JP residual in translation"),
        ("あ", "가" * 11, "hallucination"),
        ("あ", "가" * 10, None),  # exactly 10x is allowed
        ("ID:1 " + "あ" * 150 + " ID:2", "모집", "recruitment spam"),
        ("ID:1 " + "あ" * 150, "모집", None),  # one ID is a normal recruitment line
        ("あ" * 150 + " ID:1 ID:2", "모집", "recruitment spam"),
        ("ID:1 ID:2", "모집", None),  # short lines are not spam
    ],
)
def test_clean_reason(original, translated, reason):
    assert preprocess.clean_reason(original, translated) == reason


def test_filters_keep_clean_rows_in_order_and_report_each_reason(write_jsonl, tmp_path, capsys):
    src = write_jsonl(
        "raw.jsonl",
        [
            {"original": "遺跡1F", "translated": "유적 1F"},
            {"original": "ムクボ3돌", "translated": "무크보 3돌"},  # hangeul in original
            {"original": "おやすみ", "translated": "おやすみ"},  # JP residual
            {"original": "遺跡1F", "translated": "다른 번역"},  # duplicate of line 1
            "{broken json",
            {"original": "スカイ", "translated": "스카이"},
            {"original": "あ", "translated": "가" * 50},  # hallucination
        ],
    )
    out = tmp_path / "out.jsonl"

    counts = preprocess.transform_for_lora(src, str(out))

    assert [r["input"] for r in read_jsonl(out)] == ["遺跡1F", "スカイ"]
    assert read_jsonl(out)[0]["output"] == "유적 1F"  # the first occurrence wins
    assert counts == {
        "total": 7,
        "passed": 2,
        "empty field": 0,
        "hangeul in original": 1,
        "JP residual in translation": 1,
        "JP in original (ko_ja)": 0,
        "Hangeul residual in translation (ko_ja)": 0,
        "hallucination": 1,
        "recruitment spam": 0,
        "eval overlap": 0,
        "eval overlap (near)": 0,
        "duplicate": 1,
        "json error": 1,
        "validation": 0,
    }
    report = capsys.readouterr().out
    for label, n in (("Total input", 7), ("Passed", 2), ("Skipped", 5), ("duplicate", 1), ("json error", 1)):
        assert re.search(rf"{label}\s*: {n}\b", report), label


def test_cleaned_text_is_stripped(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"original": "  おやすみ！\n", "translated": " 잘 자! "}])
    out = tmp_path / "out.jsonl"

    preprocess.transform_for_lora(src, str(out))

    assert read_jsonl(out)[0]["input"] == "おやすみ！"
    assert read_jsonl(out)[0]["output"] == "잘 자!"


def test_pair_format_is_what_llamafactory_reads(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"pid": 1, "original": "遺跡1F", "translated": "유적 1F", "timestamp": 5}])
    out = tmp_path / "out.jsonl"

    preprocess.transform_for_lora(src, str(out), fmt="pair")

    assert read_jsonl(out) == [{"original": "遺跡1F", "translated": "유적 1F"}]


def test_unknown_format_is_refused_before_anything_is_written(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"original": "遺跡1F", "translated": "유적 1F"}])
    out = tmp_path / "out.jsonl"

    with pytest.raises(ValueError, match="instruction"):
        preprocess.transform_for_lora(src, str(out), fmt="nope")
    assert not out.exists()


def test_main_takes_the_format_from_the_command_line(write_jsonl, tmp_path, monkeypatch):
    src = write_jsonl("raw.jsonl", [{"original": "遺跡1F", "translated": "유적 1F"}])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))

    preprocess.main(["--format", "pair"])
    assert list(read_jsonl(out)[0]) == ["original", "translated"]

    preprocess.main([])  # the default stays the unsloth format
    assert list(read_jsonl(out)[0]) == ["instruction", "input", "output"]


# --- prompt styles and the reverse direction (--format pair) -----------------------------------


def test_pair_rows_wrap_the_line_in_the_models_instruction():
    rows = preprocess.pair_rows("杖@2募集", "법사@2 모집", style="hy")

    assert rows == [{
        "original": "Translate the following text into Korean. Note that you should only output the translated "
                    "result without any additional explanation:\n\n杖@2募集",
        "translated": "법사@2 모집",
    }]


def test_pair_rows_without_a_style_are_the_raw_pair():
    assert preprocess.pair_rows("杖@2募集", "법사@2 모집") == [{"original": "杖@2募集", "translated": "법사@2 모집"}]


def test_reverse_adds_the_ko_to_ja_row_with_its_own_instruction():
    forward, backward = preprocess.pair_rows("杖@2募集", "법사@2 모집", style="translategemma", reverse=True)

    assert forward["original"].endswith("Please translate the following Japanese text into Korean:\n杖@2募集")
    assert forward["translated"] == "법사@2 모집"
    assert backward["original"].endswith("Please translate the following Korean text into Japanese:\n법사@2 모집")
    assert backward["translated"] == "杖@2募集"


def test_transform_writes_forward_and_reverse_rows_for_clean_rows_only(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [
        {"original": "杖@2募集", "translated": "법사@2 모집"},
        {"original": "이미 한글", "translated": "x"},  # Hangeul in the source: dropped, and not reversed either
    ])
    out = tmp_path / "out.jsonl"

    counts = preprocess.transform_for_lora(src, str(out), fmt="pair", style="hy", reverse=True)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and counts["passed"] == 2  # one forward and one reverse row
    assert rows[1]["translated"] == "杖@2募集"


def test_style_and_reverse_need_the_pair_format(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"original": "杖", "translated": "법사"}])

    with pytest.raises(ValueError, match="pair"):
        preprocess.transform_for_lora(src, str(tmp_path / "o.jsonl"), fmt="instruction", style="hy")


def test_main_resolves_the_auto_style_from_the_model_parameter(write_jsonl, tmp_path, monkeypatch):
    src = write_jsonl("raw.jsonl", [{"original": "杖@2募集", "translated": "법사@2 모집"}])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))

    preprocess.main(["--format", "pair", "--prompt", "auto", "--model", "translategemma-4b"])

    assert json.loads(out.read_text(encoding="utf-8"))["original"].startswith("You are a professional Japanese (ja)")


def test_main_without_a_prompt_keeps_the_raw_line(write_jsonl, tmp_path, monkeypatch):
    src = write_jsonl("raw.jsonl", [{"original": "杖@2募集", "translated": "법사@2 모집"}])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))

    preprocess.main(["--format", "pair"])

    assert json.loads(out.read_text(encoding="utf-8"))["original"] == "杖@2募集"


# --- the reverse direction's own filters (experiment/translategemma aab6b66, "process bidirectual training") ---


@pytest.mark.parametrize("japanese,korean,reason", [
    ("杖@2募集", "법사@2 모집", None),
    ("杖@2募集", "법사 杖@2", "JP in original (ko_ja)"),  # kana/kanji left in the Korean that would be the input
    ("杖@2募集", "", "empty field"),
])
def test_reverse_clean_reason_filters_the_korean_side_as_the_source(japanese, korean, reason):
    assert preprocess.clean_reason(korean, japanese, direction="ko-ja") == reason


def test_reverse_clean_reason_flags_hangeul_left_in_the_japanese_answer():
    assert preprocess.clean_reason("법사@2 모집", "杖@2 모집 모집", direction="ko-ja") == "Hangeul residual in translation (ko_ja)"


def test_reverse_clean_reason_keeps_the_ten_times_and_spam_rules():
    assert preprocess.clean_reason("가", "あ" * 11, direction="ko-ja") == "hallucination"
    spam = "ID: 1 ID: 2 " + "가" * 150
    assert preprocess.clean_reason(spam, "あ", direction="ko-ja") == "recruitment spam"


def test_forward_filters_are_unchanged_by_the_direction_argument():
    assert preprocess.clean_reason("이미 한글", "x") == "hangeul in original"
    assert preprocess.clean_reason("杖", "법사") is None


def test_reverse_is_tried_only_for_forward_rows_and_counted_on_its_own(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [
        {"original": "杖@2募集", "translated": "법사@2 모집"},          # forward ok, reverse ok
        {"original": "杖募集" * 4, "translated": "법"},                # forward ok; reverse: Japanese 12x the Korean
        {"original": "이미 한글", "translated": "x"},                  # forward dropped: no reverse attempted
    ])
    out = tmp_path / "out.jsonl"

    counts = preprocess.transform_for_lora(src, str(out), fmt="pair", reverse=True)

    assert counts["total"] == 3 + 2  # three forward attempts, two reverse attempts
    assert counts["passed"] == 3     # 2 forward + 1 reverse
    assert counts["hallucination"] == 1  # the reverse one
    assert counts["hangeul in original"] == 1
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3


def test_reverse_input_that_is_already_a_seen_source_is_a_duplicate(write_jsonl, tmp_path):
    # The second line's Korean answer equals the first line's input: as a reverse source it would repeat a seen input.
    src = write_jsonl("raw.jsonl", [
        {"original": "AAA", "translated": "BBB"},
        {"original": "CCC", "translated": "AAA"},
    ])
    out = tmp_path / "out.jsonl"

    counts = preprocess.transform_for_lora(src, str(out), fmt="pair", reverse=True)

    assert counts["duplicate"] >= 1


def test_the_translategemma_profile_trains_with_room_for_the_instruction():
    import yaml

    for name in ("translategemma-4b", "translategemma-4b-fast", "hy-mt2-1.8b", "hy-mt2-7b"):
        path = f"{config.BASE_DIR}/configs/llamafactory/{name}/train.yaml"
        with open(path, encoding="utf-8") as f:
            assert yaml.safe_load(f)["cutoff_len"] >= 256, name


# --- the eval set stays out of the training data --------------------------------------------------


EVAL_LINE = "今日のレイドは21時から始めます、参加できる人は教えてください"


def test_rows_whose_original_is_in_the_eval_set_are_dropped_and_counted(write_jsonl, tmp_path, capsys):
    src = write_jsonl("raw.jsonl", [
        {"original": "ウルト溜まった", "translated": "궁 찼다!"},  # the eval line
        {"original": "ウルト溜まった！", "translated": "궁 찼다"},  # same line after normalising
        {"original": EVAL_LINE + "ね", "translated": "오늘 레이드는 21시부터예요"},  # near-duplicate
        {"original": "スカイ", "translated": "스카이"},
    ])
    out = tmp_path / "out.jsonl"

    counts = preprocess.transform_for_lora(src, str(out), eval_originals=["ウルト溜まった", EVAL_LINE])

    assert [r["input"] for r in read_jsonl(out)] == ["スカイ"]
    assert counts["eval overlap"] == 2 and counts["eval overlap (near)"] == 1
    assert counts["passed"] == 1
    assert re.search(r"eval overlap\s*: 2", capsys.readouterr().out)


def test_the_reverse_row_of_an_eval_line_is_not_written_either(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [
        {"original": "ウルト溜まった", "translated": "궁 찼다!"},
        {"original": "スカイ", "translated": "스카이"},
    ])
    out = tmp_path / "out.jsonl"

    preprocess.transform_for_lora(src, str(out), fmt="pair", reverse=True, eval_originals=["ウルト溜まった"])

    assert {r["original"] for r in read_jsonl(out)} == {"スカイ", "스카이"}  # both directions of スカイ only, none of the eval line


def test_without_an_eval_set_nothing_is_excluded(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"original": "ウルト溜まった", "translated": "궁 찼다!"}])
    out = tmp_path / "out.jsonl"

    counts = preprocess.transform_for_lora(src, str(out))

    assert counts["passed"] == 1 and counts["eval overlap"] == 0


def _eval_file(tmp_path, rows):
    path = tmp_path / "eval.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def test_main_excludes_the_eval_set_by_default_when_the_file_exists(write_jsonl, tmp_path, monkeypatch, capsys):
    src = write_jsonl("raw.jsonl", [
        {"original": "ウルト溜まった", "translated": "궁 찼다!"},
        {"original": "スカイ", "translated": "스카이"},
    ])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", _eval_file(tmp_path, [{"original": "ウルト溜まった", "translated": "궁 찼다!"}]))

    preprocess.main(["--format", "pair"])

    assert [r["original"] for r in read_jsonl(out)] == ["スカイ"]
    assert "Excluding eval lines" in capsys.readouterr().out


def test_main_keep_eval_turns_the_exclusion_off(write_jsonl, tmp_path, monkeypatch):
    src = write_jsonl("raw.jsonl", [{"original": "ウルト溜まった", "translated": "궁 찼다!"}])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", _eval_file(tmp_path, [{"original": "ウルト溜まった", "translated": "x"}]))

    preprocess.main(["--format", "pair", "--keep-eval"])

    assert len(read_jsonl(out)) == 1


def test_main_says_so_when_there_is_no_eval_set(write_jsonl, tmp_path, monkeypatch, capsys):
    src = write_jsonl("raw.jsonl", [{"original": "スカイ", "translated": "스카이"}])
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(tmp_path / "out.jsonl"))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", str(tmp_path / "missing.jsonl"))

    preprocess.main(["--format", "pair"])

    assert "no eval set" in capsys.readouterr().out


def test_main_stops_on_an_unreadable_eval_set(write_jsonl, tmp_path, monkeypatch):
    src = write_jsonl("raw.jsonl", [{"original": "スカイ", "translated": "스카이"}])
    bad = tmp_path / "eval.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(tmp_path / "out.jsonl"))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", str(bad))

    with pytest.raises(SystemExit) as exc:
        preprocess.main(["--format", "pair"])
    assert exc.value.code == 1


# --- the sidecar manifest says how the training file was made ---------------------------------------


def test_main_writes_the_manifest_next_to_the_output(write_jsonl, tmp_path, monkeypatch):
    import manifest

    src = write_jsonl("raw.jsonl", [
        {"original": "ウルト溜まった", "translated": "궁 찼다!"},
        {"original": "スカイ", "translated": "스카이"},
    ])
    out = tmp_path / "out.jsonl"
    eval_path = _eval_file(tmp_path, [{"original": "ウルト溜まった", "translated": "x"}])
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", eval_path)

    preprocess.main(["--format", "pair", "--prompt", "hy", "--reverse"])

    meta = manifest.read_manifest(manifest.manifest_path(str(out)))
    assert (meta["format"], meta["style"], meta["reverse"]) == ("pair", "hy", True)
    assert meta["data_sha256"] == manifest.file_sha256(str(out))
    assert meta["raw_sha256"] == manifest.file_sha256(src)
    assert meta["counts"]["eval overlap"] == 1 and meta["counts"]["passed"] == 2  # スカイ forward + reverse
    assert meta["eval_lines_excluded_from"] == 1


def test_main_without_a_prompt_records_no_style(write_jsonl, tmp_path, monkeypatch):
    import manifest

    src = write_jsonl("raw.jsonl", [{"original": "スカイ", "translated": "스카이"}])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", str(tmp_path / "missing.jsonl"))

    preprocess.main([])

    meta = manifest.read_manifest(manifest.manifest_path(str(out)))
    assert (meta["format"], meta["style"], meta["reverse"], meta["eval_set"]) == ("instruction", None, False, None)


def test_a_failed_preprocess_leaves_no_stale_manifest(write_jsonl, tmp_path, monkeypatch):
    import manifest

    out = tmp_path / "out.jsonl"
    stale = manifest.manifest_path(str(out))
    with open(stale, "w", encoding="utf-8") as f:
        f.write("{}")
    monkeypatch.setattr(preprocess, "RAW_LOGS", write_jsonl("raw.jsonl", [{"original": "x", "translated": None}]))
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", str(tmp_path / "missing.jsonl"))

    with pytest.raises(SystemExit):
        preprocess.main([])

    assert not os.path.exists(stale)


# --- the validation split: by line, written to its own file ---------------------------------------------


def _many(n=600):
    return [{"original": f"レイド募集 {i} 杖@2", "translated": f"레이드 모집 {i} 법사@2"} for i in range(n)]


def test_validation_rows_go_to_their_own_file_and_never_to_the_training_file(write_jsonl, tmp_path):
    from valsplit import is_validation

    src = write_jsonl("raw.jsonl", _many())
    out, val = tmp_path / "out.jsonl", tmp_path / "val.jsonl"

    counts = preprocess.transform_for_lora(src, str(out), fmt="pair", val_file=str(val), val_fraction=0.1)

    train_rows, val_rows = read_jsonl(out), read_jsonl(val)
    assert counts["validation"] == len(val_rows) and counts["passed"] == len(train_rows) + len(val_rows) == 600
    assert 30 < len(val_rows) < 90
    assert all(is_validation(r["original"], 0.1) for r in val_rows)
    assert not any(is_validation(r["original"], 0.1) for r in train_rows)


def test_both_directions_of_a_pair_stay_on_the_same_side(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", _many(300))
    out, val = tmp_path / "out.jsonl", tmp_path / "val.jsonl"

    preprocess.transform_for_lora(src, str(out), fmt="pair", reverse=True, val_file=str(val), val_fraction=0.2)

    def sides(rows):
        return {r["translated"] if "레이드" in r["original"] else r["original"] for r in rows}  # the Japanese line of each row

    train_rows, val_rows = read_jsonl(out), read_jsonl(val)
    assert len(val_rows) % 2 == 0 and len(train_rows) % 2 == 0  # forward + reverse together
    assert not sides(train_rows) & sides(val_rows)


def test_variants_of_a_line_do_not_straddle_the_split(write_jsonl, tmp_path):
    rows = []
    for i in range(300):
        rows += [{"original": f"レイド {i} 集合", "translated": "a"}, {"original": f"レイド {i} 集合！", "translated": "b"}]
    src = write_jsonl("raw.jsonl", rows)
    out, val = tmp_path / "out.jsonl", tmp_path / "val.jsonl"

    preprocess.transform_for_lora(src, str(out), fmt="pair", val_file=str(val), val_fraction=0.2)

    import re
    key = lambda r: re.sub(r"[！ ]", "", r["original"])  # noqa: E731
    assert not {key(r) for r in read_jsonl(out)} & {key(r) for r in read_jsonl(val)}


def test_without_a_validation_file_everything_is_training_data(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", _many(50))
    out = tmp_path / "out.jsonl"

    counts = preprocess.transform_for_lora(src, str(out), fmt="pair")

    assert len(read_jsonl(out)) == 50 and counts["validation"] == 0


def test_a_validation_split_needs_the_pair_format(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", _many(5))

    with pytest.raises(ValueError, match="pair"):
        preprocess.transform_for_lora(src, str(tmp_path / "o.jsonl"), fmt="instruction", val_file=str(tmp_path / "v.jsonl"), val_fraction=0.1)


def test_main_splits_the_pair_format_by_default_and_the_instruction_format_never(write_jsonl, tmp_path, monkeypatch):
    import manifest

    src = write_jsonl("raw.jsonl", _many(600))
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", str(tmp_path / "missing.jsonl"))

    preprocess.main(["--format", "pair"])
    val = manifest.val_path(str(out))
    meta = manifest.read_manifest(manifest.manifest_path(str(out)))
    assert os.path.exists(val) and meta["val_rows"] == len(read_jsonl(val)) > 0 and meta["val_fraction"] == 0.05
    assert len(read_jsonl(out)) + len(read_jsonl(val)) == 600

    preprocess.main([])  # unsloth's instruction format does its own split: no validation file, and the old one is removed
    assert not os.path.exists(val)
    assert manifest.read_manifest(manifest.manifest_path(str(out)))["val_rows"] == 0
    assert len(read_jsonl(out)) == 600


def test_main_takes_the_validation_fraction_from_the_command_line(write_jsonl, tmp_path, monkeypatch):
    import manifest

    src = write_jsonl("raw.jsonl", _many(600))
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", str(tmp_path / "missing.jsonl"))

    preprocess.main(["--format", "pair", "--val-fraction", "0.3"])

    val_rows = read_jsonl(manifest.val_path(str(out)))
    assert 120 < len(val_rows) < 240


def test_main_refuses_a_fraction_outside_zero_to_one(write_jsonl, tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "RAW_LOGS", write_jsonl("raw.jsonl", _many(5)))
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(tmp_path / "out.jsonl"))

    with pytest.raises(SystemExit):
        preprocess.main(["--format", "pair", "--val-fraction", "1.5"])


# --- the drop-rate guard: a file that is mostly suspicious is the wrong file, not a dataset --------------


def _guard_rows(clean, bad_source=0, bad_output=0, hallucinated=0):
    rows = [{"original": f"レイド {i} 集合", "translated": f"레이드 {i} 집합"} for i in range(clean)]
    rows += [{"original": f"한글 {i}", "translated": "x"} for i in range(bad_source)]  # hangeul in original
    rows += [{"original": f"集合 {i}", "translated": f"集合 {i}"} for i in range(bad_output)]  # JP residual
    rows += [{"original": "あ" + "い" * (i % 2), "translated": "가" * 40} for i in range(hallucinated)]
    return rows


def test_a_mostly_suspicious_file_stops_preprocessing(write_jsonl, tmp_path, capsys):
    src = write_jsonl("raw.jsonl", _guard_rows(clean=6, bad_source=2, bad_output=2))  # 4 of 10 usable rows

    with pytest.raises(SystemExit) as exc:
        preprocess.transform_for_lora(src, str(tmp_path / "out.jsonl"), max_drop=0.3)

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "40.0%" in out and "--max-drop" in out


def test_a_file_under_the_limit_passes(write_jsonl, tmp_path, capsys):
    src = write_jsonl("raw.jsonl", _guard_rows(clean=8, bad_source=1, bad_output=1))  # 20%

    counts = preprocess.transform_for_lora(src, str(tmp_path / "out.jsonl"), max_drop=0.3)

    assert counts["passed"] == 8
    assert "20.0%" in capsys.readouterr().out  # the share is always shown, so the limit can be calibrated


def test_hallucinated_outputs_count_as_suspicious_too(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", _guard_rows(clean=4, hallucinated=6))

    with pytest.raises(SystemExit):
        preprocess.transform_for_lora(src, str(tmp_path / "out.jsonl"), max_drop=0.3)


def test_untranslated_duplicate_and_spam_rows_are_expected_and_not_counted(write_jsonl, tmp_path):
    rows = _guard_rows(clean=5)
    rows += [{"original": f"未訳 {i}", "translated": None} for i in range(50)]  # the app never translated these
    rows += [rows[0]] * 20  # duplicates
    rows += [{"original": "ID:1 " + "あ" * 150 + " ID:2", "translated": "모집"}] * 20  # recruitment spam
    src = write_jsonl("raw.jsonl", rows)

    counts = preprocess.transform_for_lora(src, str(tmp_path / "out.jsonl"), max_drop=0.3)

    assert counts["passed"] == 5


def test_without_a_limit_nothing_is_checked(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", _guard_rows(clean=1, bad_source=9))

    assert preprocess.transform_for_lora(src, str(tmp_path / "out.jsonl"))["passed"] == 1


def test_main_applies_the_configured_limit_and_max_drop_overrides_it(write_jsonl, tmp_path, monkeypatch):
    import manifest

    src = write_jsonl("raw.jsonl", _guard_rows(clean=6, bad_source=4))
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", str(tmp_path / "missing.jsonl"))

    with pytest.raises(SystemExit):
        preprocess.main(["--format", "pair"])  # 40% against the configured limit
    assert not os.path.exists(manifest.manifest_path(str(out)))  # the failed run leaves no manifest: training refuses it

    preprocess.main(["--format", "pair", "--max-drop", "0.5"])
    assert os.path.exists(manifest.manifest_path(str(out)))


def test_the_default_limit_comes_from_config():
    import config

    assert preprocess.MAX_SUSPICIOUS == config.PREPROCESS_MAX_SUSPICIOUS


# --- the output without a recipe is what it was before recipes existed ---------------------------------------------


def test_without_a_recipe_the_output_is_exactly_what_it_was_before_recipes_existed(write_jsonl, tmp_path):
    # Golden values taken from the code before the recipe change (pair format, hy style, reverse, eval line, validation split).
    import manifest

    src = write_jsonl("raw.jsonl", [
        {"original": "遺跡1F", "translated": "유적 1F"},
        {"original": "スカイ", "translated": "스카이"},
        {"original": "おやすみ", "translated": "잘 자"},
        {"original": "ID:1 " + "あ" * 150 + " ID:2", "translated": "모집"},
        {"original": "ムクボ3돌", "translated": "무크보 3돌"},
        {"original": "ウルト溜まった", "translated": "궁 찼다"},
        {"original": "遺跡1F", "translated": "중복"},
        {"original": "杖@2募集", "translated": "법사@2 모집"},
        "{broken",
        {"original": "レイド募集 1", "translated": "레이드 모집 1"},
        {"original": "レイド募集 2", "translated": "레이드 모집 2"},
        {"original": "あ", "translated": "가" * 50},
        {"original": "ログイン", "translated": None},
        {"original": "ありがとう", "translated": "고마워"},
        {"original": "こんにちは", "translated": "안녕하세요"},
    ])
    out, val = tmp_path / "out.jsonl", tmp_path / "val.jsonl"

    counts = preprocess.transform_for_lora(src, str(out), fmt="pair", style="hy", reverse=True, eval_originals=["スカイ"],
                                           val_file=str(val), val_fraction=0.3)

    assert counts == {
        "total": 23, "passed": 16, "empty field": 1, "hangeul in original": 1, "JP residual in translation": 0,
        "JP in original (ko_ja)": 0, "Hangeul residual in translation (ko_ja)": 0, "hallucination": 1,
        "recruitment spam": 1, "eval overlap": 1, "eval overlap (near)": 0, "duplicate": 1, "json error": 1,
        "validation": 10,
    }
    assert manifest.file_sha256(str(out)) == "250aa89d7d22bf84fe57e1cd3b07e11dd0027374c6f858057879bf532bcc51c1"
    assert manifest.file_sha256(str(val)) == "fde8995a9bbd13bfe49cad1a9996c6bbafe8f2e27df5d25a0839596171933eb8"


def test_the_training_and_validation_files_end_lines_with_lf_on_every_os(write_jsonl, tmp_path):
    # Their sha256 goes into the manifest and MLflow: a CRLF file on Windows would hash differently from Linux.
    src = write_jsonl("raw.jsonl", [{"original": f"行{i}", "translated": f"줄{i}"} for i in range(10)])
    out, val = tmp_path / "out.jsonl", tmp_path / "val.jsonl"

    preprocess.transform_for_lora(src, str(out), fmt="pair", val_file=str(val), val_fraction=0.3)

    for path in (out, val):
        data = path.read_bytes()
        assert data.count(b"\n") > 0 and b"\r" not in data, path.name


# --- keep: a recipe can let recruitment walls through ---------------------------------------------------------------

WALL = "ID:1 " + "あ" * 150 + " ID:2"


def test_a_recipe_can_keep_recruitment_walls_but_no_other_filter():
    assert preprocess.clean_reason(WALL, "모집") == "recruitment spam"
    assert preprocess.clean_reason(WALL, "모집", keep=("recruitment spam",)) is None
    assert preprocess.clean_reason(WALL, "가" * 5000, keep=("recruitment spam",)) == "hallucination"
    assert preprocess.clean_reason("모집 " + "あ" * 150 + " ID:1 ID:2", "x", keep=("recruitment spam",)) == "hangeul in original"
    korean_wall = "모집 " + "a" * 150 + " ID:1 ID:2"
    assert preprocess.clean_reason(korean_wall, "募集", direction="ko-ja") == "recruitment spam"
    assert preprocess.clean_reason(korean_wall, "募集", direction="ko-ja", keep=("recruitment spam",)) is None


# --- recipes: the training lines are chosen by category weight ----------------------------------------------------------


def _categorized(write_jsonl, groups):
    """(raw log, categories file) for {category: [(japanese, korean), ...]}; the lines are the made-up ones given."""
    from dataset_recipe import line_key

    raw = [{"original": ja, "translated": ko} for pairs in groups.values() for ja, ko in pairs]
    categories = [{"key": line_key(ja), "category": category} for category, pairs in groups.items() for ja, _ in pairs]
    return write_jsonl("raw.jsonl", raw), write_jsonl("categories.jsonl", categories)


_KOREAN = {"挨拶": "인사", "雑談": "잡담", "売ります": "판매", "募集": "모집", "通知": "알림", "未分類": "미분류", "ギルド": "길드"}


def _lines(prefix, n):
    """Made-up (Japanese, Korean) lines; the Korean differs per prefix so no reverse row is a duplicate of another."""
    return [(f"{prefix} {i}", f"{_KOREAN[prefix]} {i}") for i in range(n)]


def _recipe(**changes):
    from dataset_recipe import parse_recipe

    return parse_recipe({"seed": 3, "categories": {"greetings": {"weight": 1}, "chat": {"weight": 3}}, **changes})


def _run(src, categories_path, out, recipe, **options):
    from dataset_recipe import read_categories

    report = {}
    counts = preprocess.transform_for_lora(src, str(out), fmt="pair", recipe=recipe, categories=read_categories(categories_path),
                                           recipe_report=report, **options)
    return counts, report, read_jsonl(out)


def test_a_recipe_selects_training_lines_by_weight_and_counts_the_rest_as_not_selected(write_jsonl, tmp_path):
    src, cats = _categorized(write_jsonl, {"greetings": _lines("挨拶", 40), "chat": _lines("雑談", 60), "trade": _lines("売ります", 20)})

    counts, report, rows = _run(src, cats, tmp_path / "out.jsonl", _recipe(total=40))

    assert len(rows) == 40
    assert sum(r["original"].startswith("挨拶") for r in rows) == 10
    assert sum(r["original"].startswith("雑談") for r in rows) == 30
    assert not any(r["original"].startswith("売ります") for r in rows)
    assert counts["passed"] == 40 and counts["total"] == 120
    assert counts["not selected (recipe)"] == 80
    assert report["targets"] == {"greetings": 10, "chat": 30}
    assert report["available"] == {"greetings": 40, "chat": 60}
    assert report["limited_by"] is None and report["total"] == 40


def test_without_a_total_the_dataset_ends_where_the_first_category_runs_out(write_jsonl, tmp_path):
    src, cats = _categorized(write_jsonl, {"greetings": _lines("挨拶", 40), "chat": _lines("雑談", 60)})

    counts, report, rows = _run(src, cats, tmp_path / "out.jsonl", _recipe())

    assert (len(rows), report["limited_by"], report["total"]) == (80, "chat", 80)
    assert report["targets"] == {"greetings": 20, "chat": 60}


def test_a_report_without_a_recipe_has_no_recipe_reason(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"original": "おやすみ", "translated": "잘 자"}])

    counts = preprocess.transform_for_lora(src, str(tmp_path / "out.jsonl"))

    assert "not selected (recipe)" not in counts


def test_the_recipe_leaves_validation_rows_alone_so_every_recipe_has_the_same_validation_file(write_jsonl, tmp_path):
    src, cats = _categorized(write_jsonl, {"greetings": _lines("挨拶", 400), "chat": _lines("雑談", 600)})
    one, other = tmp_path / "one", tmp_path / "other"
    one.mkdir(), other.mkdir()

    _, _, train_one = _run(src, cats, one / "out.jsonl", _recipe(total=100), val_file=str(one / "val.jsonl"), val_fraction=0.1)
    _, _, train_two = _run(src, cats, other / "out.jsonl", _recipe(total=200, seed=9), val_file=str(other / "val.jsonl"), val_fraction=0.1)

    val_one, val_two = read_jsonl(one / "val.jsonl"), read_jsonl(other / "val.jsonl")
    assert val_one == val_two and 50 < len(val_one) < 150  # the whole 1000 lines, not just the selected ones
    assert len(train_one) == 100 and len(train_two) == 200
    assert not {r["original"] for r in train_one} & {r["original"] for r in val_one}


def test_a_smaller_total_is_inside_a_bigger_one_in_the_training_file(write_jsonl, tmp_path):
    src, cats = _categorized(write_jsonl, {"greetings": _lines("挨拶", 300), "chat": _lines("雑談", 300)})

    _, _, small = _run(src, cats, tmp_path / "small.jsonl", _recipe(total=40))
    _, _, big = _run(src, cats, tmp_path / "big.jsonl", _recipe(total=200))

    assert {r["original"] for r in small} < {r["original"] for r in big}


def test_a_reverse_row_is_written_only_with_its_selected_forward_row(write_jsonl, tmp_path):
    src, cats = _categorized(write_jsonl, {"greetings": _lines("挨拶", 20), "chat": _lines("雑談", 20), "trade": _lines("売ります", 20)})

    counts, _, rows = _run(src, cats, tmp_path / "out.jsonl", _recipe(total=8), reverse=True)

    forward = [r for r in rows if r["original"].startswith(("挨拶", "雑談"))]
    reverse = [r for r in rows if r["translated"].startswith(("挨拶", "雑談"))]  # Korean in, Japanese out
    assert len(forward) == len(reverse) == 8 and len(rows) == 16
    assert {r["translated"] for r in reverse} == {r["original"] for r in forward}
    assert counts["passed"] == 16


def test_the_recipe_decides_about_recruitment_walls(write_jsonl, tmp_path):
    groups = {"recruitment/guild": [(WALL, "모집 벽")] + _lines("ギルド", 5), "chat": _lines("雑談", 10)}
    src, cats = _categorized(write_jsonl, groups)
    recipe = {"recruitment": {"weight": 1}, "chat": {"weight": 1}}

    from dataset_recipe import parse_recipe

    counts, _, rows = _run(src, cats, tmp_path / "default.jsonl", parse_recipe({"categories": recipe}))
    assert counts["recruitment spam"] == 1 and not any(r["original"] == WALL for r in rows)

    counts, _, rows = _run(src, cats, tmp_path / "kept.jsonl", parse_recipe({"categories": recipe, "keep": ["recruitment spam"]}))
    assert counts["recruitment spam"] == 0 and any(r["original"] == WALL for r in rows)


def test_lines_the_recipe_leaves_out_do_not_trip_the_suspicious_drop_guard(write_jsonl, tmp_path):
    src, cats = _categorized(write_jsonl, {"greetings": _lines("挨拶", 50), "chat": _lines("雑談", 50)})

    counts, _, rows = _run(src, cats, tmp_path / "out.jsonl", _recipe(total=8), max_drop=0.05)

    assert len(rows) == 8 and counts["not selected (recipe)"] == 92


def test_a_recipe_that_selects_nothing_stops(write_jsonl, tmp_path):
    src, cats = _categorized(write_jsonl, {"trade": _lines("売ります", 10)})

    with pytest.raises(SystemExit) as stopped:
        _run(src, cats, tmp_path / "out.jsonl", _recipe())

    assert stopped.value.code == 1


def test_lines_without_a_category_are_uncategorized_and_can_be_weighted(write_jsonl, tmp_path):
    from dataset_recipe import parse_recipe

    src, cats = _categorized(write_jsonl, {"chat": _lines("雑談", 20)})
    with open(src, "a", encoding="utf-8") as f:
        for ja, ko in _lines("未分類", 20):
            f.write(json.dumps({"original": ja, "translated": ko}, ensure_ascii=False) + "\n")

    _, report, rows = _run(src, cats, tmp_path / "out.jsonl", parse_recipe({"categories": {"uncategorized": {"weight": 1}, "chat": {"weight": 1}}}))

    assert report["targets"] == {"chat": 20, "uncategorized": 20} and len(rows) == 40


# --- main: --recipe / --categories and the manifest ---------------------------------------------------------------------


def _main_setup(write_jsonl, tmp_path, monkeypatch, groups):
    src, cats = _categorized(write_jsonl, groups)
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", str(tmp_path / "no-eval.jsonl"))
    return src, cats, out


def test_main_applies_a_recipe_by_name_and_records_it_in_the_manifest(write_jsonl, tmp_path, monkeypatch):
    import manifest

    # configs/datasets/example.json: greetings 1, recruitment/party 2, chat 6, bot 1
    src, cats, out = _main_setup(write_jsonl, tmp_path, monkeypatch, {
        "greetings": _lines("挨拶", 30), "recruitment/party": _lines("募集", 30), "chat": _lines("雑談", 90), "bot": _lines("通知", 30)})

    preprocess.main(["--format", "pair", "--val-fraction", "0", "--recipe", "example", "--categories", cats])

    rows = read_jsonl(out)
    meta = manifest.read_manifest(manifest.manifest_path(str(out)))
    block = meta["recipe"]
    assert len(rows) == 150 and block["total"] == 150 and block["limited_by"] == "chat"
    assert block["selected"] == {"greetings": 15, "recruitment/party": 30, "chat": 90, "bot": 15}
    assert block["available"] == {"greetings": 30, "recruitment/party": 30, "chat": 90, "bot": 30}
    assert block["name"] == "example" and block["seed"] == 42 and block["keep"] == []
    assert block["weights"]["chat"] == "6"
    assert block["sha256"] and block["categories_sha256"] == manifest.file_sha256(cats)
    assert block["categories_file"] == os.path.basename(cats)
    assert meta["data_sha256"] == manifest.file_sha256(str(out))


def test_main_takes_a_recipe_file_by_path(write_jsonl, tmp_path, monkeypatch):
    import manifest

    src, cats, out = _main_setup(write_jsonl, tmp_path, monkeypatch, {"chat": _lines("雑談", 10), "bot": _lines("通知", 10)})
    recipe_file = tmp_path / "mine.json"
    recipe_file.write_text(json.dumps({"categories": {"chat": {"weight": 1}, "bot": {"weight": 1}}}), encoding="utf-8")

    preprocess.main(["--format", "pair", "--val-fraction", "0", "--recipe", str(recipe_file), "--categories", cats])

    assert len(read_jsonl(out)) == 20
    assert manifest.read_manifest(manifest.manifest_path(str(out)))["recipe"]["name"] == "mine"


def test_main_reads_the_recipe_from_the_environment_when_no_flag_is_given(write_jsonl, tmp_path, monkeypatch):
    src, cats, out = _main_setup(write_jsonl, tmp_path, monkeypatch, {"chat": _lines("雑談", 10), "bot": _lines("通知", 10)})
    recipe_file = tmp_path / "env.json"
    recipe_file.write_text(json.dumps({"categories": {"chat": {"weight": 1}}}), encoding="utf-8")
    monkeypatch.setenv("RESONANCE_RECIPE", str(recipe_file))

    preprocess.main(["--format", "pair", "--val-fraction", "0", "--categories", cats])

    assert len(read_jsonl(out)) == 10


def test_main_without_a_recipe_writes_no_recipe_block(write_jsonl, tmp_path, monkeypatch):
    import manifest

    src, cats, out = _main_setup(write_jsonl, tmp_path, monkeypatch, {"chat": _lines("雑談", 10)})
    monkeypatch.delenv("RESONANCE_RECIPE", raising=False)

    preprocess.main(["--format", "pair", "--val-fraction", "0"])

    assert "recipe" not in manifest.read_manifest(manifest.manifest_path(str(out)))


@pytest.mark.parametrize("problem", ["no categories file", "bad recipe", "no recipe file"])
def test_main_stops_with_a_message_when_the_recipe_cannot_be_used(write_jsonl, tmp_path, monkeypatch, capsys, problem):
    import manifest

    src, cats, out = _main_setup(write_jsonl, tmp_path, monkeypatch, {"chat": _lines("雑談", 10)})
    recipe_file = tmp_path / "r.json"
    recipe_file.write_text(json.dumps({"categories": {"chat": {"share": 1}} if problem == "bad recipe" else {"chat": {"weight": 1}}}), encoding="utf-8")
    argv = ["--format", "pair", "--val-fraction", "0", "--recipe", str(recipe_file), "--categories", cats]
    if problem == "no categories file":
        argv[-1] = str(tmp_path / "missing.jsonl")
    if problem == "no recipe file":
        argv[argv.index("--recipe") + 1] = str(tmp_path / "missing.json")

    with pytest.raises(SystemExit) as stopped:
        preprocess.main(argv)

    assert stopped.value.code == 1
    assert "[ERROR]" in capsys.readouterr().out
    assert not os.path.exists(manifest.manifest_path(str(out)))
