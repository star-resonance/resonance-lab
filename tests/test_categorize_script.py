import json
import os

import pytest

import categorize
from conftest import read_jsonl
from dataset_recipe import line_key, read_categories

SAMPLE = os.path.join(os.path.dirname(__file__), "fixtures", "gate1_sample.jsonl")


def run(argv):
    categorize.main(argv)


def test_it_writes_a_categories_file_keyed_by_the_line(write_jsonl, tmp_path, capsys):
    raw = write_jsonl("raw.jsonl", [
        {"pid": "p1", "original": "杖@2募集", "translated": "법사@2 모집"},
        {"pid": "p2", "original": "おはようございます", "translated": None},  # untranslated: still a message to categorise
        {"pid": "p3", "original": "お腹すいたなあ", "translated": "배고프다"},  # no rule: left out of the file
    ])
    out = tmp_path / "categories.jsonl"

    run(["--raw", raw, "--out", str(out)])

    rows = read_jsonl(out)
    assert {row["key"]: row["category"] for row in rows} == {line_key("杖@2募集"): "recruitment", line_key("おはようございます"): "social"}
    assert all(row["by"] == "regex-v1" for row in rows)
    assert read_categories(str(out))  # the file a recipe reads
    report = capsys.readouterr().out
    assert "recruitment" in report and "social" in report and "uncategorized" in report


def test_a_variant_of_a_line_shares_the_category_of_its_first_occurrence(write_jsonl, tmp_path):
    raw = write_jsonl("raw.jsonl", [{"original": "乙です", "translated": "x"}, {"original": "乙です！", "translated": "y"}, {"original": " 乙です", "translated": "z"}])
    out = tmp_path / "categories.jsonl"

    run(["--raw", raw, "--out", str(out)])

    assert len(read_jsonl(out)) == 1  # one key for the three variants


def test_damaged_and_empty_rows_are_skipped(write_jsonl, tmp_path, capsys):
    raw = write_jsonl("raw.jsonl", ["{broken", {"translated": "x"}, {"original": "   ", "translated": "x"}, {"original": "草", "translated": "ㅋ"}])
    out = tmp_path / "categories.jsonl"

    run(["--raw", raw, "--out", str(out)])

    assert [row["category"] for row in read_jsonl(out)] == ["chat"]
    assert "1 damaged" in capsys.readouterr().out


def test_the_output_folder_is_created(write_jsonl, tmp_path):
    raw = write_jsonl("raw.jsonl", [{"original": "草", "translated": "ㅋ"}])
    out = tmp_path / "new" / "folder" / "categories.jsonl"

    run(["--raw", raw, "--out", str(out)])

    assert out.exists()


def test_the_categories_file_ends_lines_with_lf_on_every_os(write_jsonl, tmp_path):
    # A recipe records its sha256 (categories_sha256): a CRLF file on Windows would hash differently from Linux.
    raw = write_jsonl("raw.jsonl", [{"original": "杖@2募集", "translated": "법사@2 모집"}, {"original": "おはよう", "translated": "안녕"}])
    out = tmp_path / "categories.jsonl"

    run(["--raw", raw, "--out", str(out)])

    data = out.read_bytes()
    assert data.count(b"\n") == 2 and b"\r" not in data


def test_a_missing_raw_log_stops_with_a_message(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        run(["--raw", str(tmp_path / "missing.jsonl"), "--out", str(tmp_path / "c.jsonl")])

    assert stopped.value.code == 1 and "[ERROR]" in capsys.readouterr().out
    assert not (tmp_path / "c.jsonl").exists()


def test_probe_scores_the_baseline_on_a_labelled_sample_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "categories.jsonl"

    run(["--probe", SAMPLE, "--out", str(out)])

    text = capsys.readouterr().out
    assert "regex-v1" in text and "accuracy" in text and "coverage" in text and "98 lines" in text
    assert not out.exists()


def test_probe_without_a_file_reads_the_configured_sample(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(categorize, "GATE1_SAMPLE", SAMPLE)

    run(["--probe"])

    assert "98 lines" in capsys.readouterr().out


def test_probe_stops_with_a_message_when_the_sample_is_missing_or_bad(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"original": "x", "category": "Nope"}) + "\n", encoding="utf-8")
    for path in (str(tmp_path / "missing.jsonl"), str(bad)):
        with pytest.raises(SystemExit) as stopped:
            run(["--probe", path])
        assert stopped.value.code == 1 and "[ERROR]" in capsys.readouterr().out
