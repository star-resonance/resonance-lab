import json
import os
from datetime import datetime, timezone

import pytest

import runs
from runs import RunError

T0 = datetime(2026, 10, 1, 16, 30, 5, tzinfo=timezone.utc)


def test_a_run_id_is_the_utc_time_to_the_second():
    assert runs.new_run_id(T0) == "20261001-163005"


def test_start_creates_a_run_dir_with_a_running_record_and_points_latest_at_it(tmp_path):
    base = str(tmp_path / "hy_lora")

    run = runs.start_run(base, "hy-mt2-1.8b", now=T0)

    assert run.id == "20261001-163005" and run.dir == os.path.join(base, run.id) and os.path.isdir(run.dir)
    record = runs.read_run(run.dir)
    assert record["profile"] == "hy-mt2-1.8b" and record["status"] == "running" and record["started"].startswith("2026-10-01T16:30:05")
    assert runs.latest_run(base) == run.dir


def test_two_runs_in_the_same_second_get_different_dirs(tmp_path):
    base = str(tmp_path)

    first = runs.start_run(base, "p", now=T0)
    second = runs.start_run(base, "p", now=T0)

    assert first.dir != second.dir and second.id == "20261001-163005-2"
    assert runs.latest_run(base) == second.dir


def test_a_new_run_never_reuses_the_old_runs_dir_so_nothing_is_resumed(tmp_path):
    base = str(tmp_path)
    old = runs.start_run(base, "p", now=T0)
    open(os.path.join(old.dir, "checkpoint-100"), "w").close()

    new = runs.start_run(base, "p", now=datetime(2026, 10, 2, tzinfo=timezone.utc))

    assert os.listdir(new.dir) == [runs.RUN_FILE]  # nothing of the old run for LLaMA-Factory to resume from


def test_finish_records_the_status_and_the_time(tmp_path):
    run = runs.start_run(str(tmp_path), "p", now=T0)

    runs.finish_run(run.dir, "complete")

    record = runs.read_run(run.dir)
    assert record["status"] == "complete" and "finished" in record


def test_finish_refuses_an_unknown_status(tmp_path):
    run = runs.start_run(str(tmp_path), "p", now=T0)

    with pytest.raises(ValueError):
        runs.finish_run(run.dir, "done-ish")


def test_latest_run_is_none_without_runs_or_with_a_dangling_pointer(tmp_path):
    assert runs.latest_run(str(tmp_path)) is None
    (tmp_path / runs.LATEST_FILE).write_text("gone", encoding="utf-8")
    assert runs.latest_run(str(tmp_path)) is None


def test_the_adapter_to_merge_is_the_latest_complete_run(tmp_path):
    base = str(tmp_path)
    run = runs.start_run(base, "p", now=T0)
    runs.finish_run(run.dir, "complete")

    assert runs.resolve_adapter(base) == run.dir


def test_a_run_that_is_still_running_or_failed_is_not_merged(tmp_path):
    base = str(tmp_path)
    run = runs.start_run(base, "p", now=T0)

    with pytest.raises(RunError, match="running"):
        runs.resolve_adapter(base)
    runs.finish_run(run.dir, "failed")
    with pytest.raises(RunError, match="failed"):
        runs.resolve_adapter(base)


def test_a_named_run_is_resolved_by_its_id_even_when_it_is_not_the_latest(tmp_path):
    base = str(tmp_path)
    first = runs.start_run(base, "p", now=T0)
    runs.finish_run(first.dir, "complete")
    second = runs.start_run(base, "p", now=datetime(2026, 10, 2, tzinfo=timezone.utc))
    runs.finish_run(second.dir, "complete")

    assert runs.resolve_adapter(base, run_id=first.id) == first.dir
    with pytest.raises(RunError, match="nope"):
        runs.resolve_adapter(base, run_id="nope")


def test_an_adapter_trained_before_runs_existed_is_still_merged(tmp_path):
    base = tmp_path
    (base / "adapter_config.json").write_text("{}", encoding="utf-8")  # what LLaMA-Factory writes into output_dir

    assert runs.resolve_adapter(str(base)) == str(base)


def test_without_any_adapter_the_error_names_the_directory(tmp_path):
    with pytest.raises(RunError, match="hy-mt2-1.8b_lora"):
        runs.resolve_adapter(str(tmp_path / "hy-mt2-1.8b_lora"))


def test_run_files_follow_the_latest_run_and_fall_back_to_the_legacy_adapter_dir(tmp_path):
    base = str(tmp_path)
    assert runs.run_files(base) == (os.path.join(base, "trainer_log.jsonl"), os.path.join(base, runs.TRAIN_LOG_NAME))

    run = runs.start_run(base, "p", now=T0)

    assert runs.run_files(base) == (os.path.join(run.dir, "trainer_log.jsonl"), os.path.join(run.dir, runs.TRAIN_LOG_NAME))


def test_the_merge_record_says_which_run_the_merged_model_came_from(tmp_path):
    run = runs.start_run(str(tmp_path / "lora"), "p", now=T0)
    merged = tmp_path / "merged"
    merged.mkdir()

    path = runs.write_merge_record(str(merged), "p", run.dir)

    assert json.loads(open(path, encoding="utf-8").read())["run"] == run.id
    assert os.path.dirname(path) == str(merged)


def test_the_merge_record_of_a_legacy_adapter_has_no_run(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    legacy = tmp_path / "lora"
    legacy.mkdir()
    (legacy / "adapter_config.json").write_text("{}", encoding="utf-8")

    path = runs.write_merge_record(str(merged), "p", str(legacy))

    assert json.loads(open(path, encoding="utf-8").read())["run"] is None


def test_run_json_ends_lines_with_lf_on_every_os(tmp_path):
    run = runs.start_run(str(tmp_path), "p", now=T0)
    runs.finish_run(run.dir, "complete")
    data = open(os.path.join(run.dir, runs.RUN_FILE), "rb").read()
    assert b"\n" in data and b"\r" not in data
