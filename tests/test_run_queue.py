import json
import os
import subprocess

import pytest

import run_queue
from run_queue import RunQueue


@pytest.fixture
def queue(tmp_path):
    return RunQueue(str(tmp_path / ".run.result.backup.jsonl"), str(tmp_path / ".run.result.backup.files"))


def _new(queue, name="hy-20261001", now=1_000):
    return queue.start_run("resonance-lab", name, now_ms=now)


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_started_run_is_on_disk_at_once_with_a_fresh_id(queue, tmp_path):
    first, second = _new(queue), _new(queue)

    assert first != second and len(first) >= 32
    on_disk = _lines(tmp_path / ".run.result.backup.jsonl")
    assert [r["t"] for r in on_disk] == ["run", "run"] and {r["local_run_id"] for r in on_disk} == {first, second}


def test_every_write_is_one_appended_line_and_nothing_already_written_is_touched(queue, tmp_path):
    path = tmp_path / ".run.result.backup.jsonl"
    local = _new(queue)
    seen = path.read_text(encoding="utf-8")

    queue.add_params(local, {"a": "1"})
    queue.set_remote_id(local, "r1")
    queue.mark_sent(local, 0)

    text = path.read_text(encoding="utf-8")
    assert text.startswith(seen) and [r["t"] for r in _lines(path)] == ["run", "event", "remote", "sent"]


def test_a_run_remembers_its_experiment_name_and_start(queue):
    run = queue.get(_new(queue, "my-run", now=1234))

    assert run["experiment"] == "resonance-lab" and run["run_name"] == "my-run" and run["start_time_ms"] == 1234
    assert run["remote_id"] is None and run["events"] == []


def test_a_given_local_id_is_kept_so_later_stages_can_resume_the_run(queue):
    local = queue.start_run("e", "n", local_run_id="abc", now_ms=1)

    assert local == "abc" and queue.has_run("abc") and not queue.has_run("zzz")
    assert queue.start_run("e", "n", local_run_id="abc", now_ms=2) == "abc" and len(queue.pending()) == 1  # not duplicated


def test_what_a_stage_records_is_appended_in_order_and_unsent(queue):
    local = _new(queue)

    queue.add_params(local, {"lr": "1e-05"})
    queue.add_tags(local, {"git.commit": "abc"})
    queue.add_metrics(local, {"eval.chrf": 66.7}, ts_ms=5000)
    queue.finish(local, "FINISHED", now_ms=9000)

    events = queue.get(local)["events"]
    assert [e["type"] for e in events] == ["params", "tags", "metrics", "status"]
    assert all(e["sent"] is False for e in events)
    assert events[0]["params"] == {"lr": "1e-05"} and events[2]["ts_ms"] == 5000 and events[3]["status"] == "FINISHED"


def test_get_hands_out_a_copy_not_the_stored_document(queue):
    local = _new(queue)

    queue.add_tags(local, {"a": "b"})
    doc = queue.get(local)
    doc["events"][0]["something_new"] = {"nested": [1, 2]}

    assert queue.get(local)["events"][0].get("something_new") is None  # get() hands out a copy, not the stored document


def test_files_are_copied_into_the_queue_so_the_originals_may_go(queue, tmp_path):
    source = tmp_path / "report.txt"
    source.write_text("chrF 66.7", encoding="utf-8")
    log = tmp_path / "trainer_log.jsonl"
    log.write_text('{"current_steps": 1}\n', encoding="utf-8")
    local = _new(queue)

    queue.add_artifact(local, str(source), artifact_path="eval")
    queue.add_step_log(local, str(log))
    source.unlink()
    log.unlink()

    artifact, step_log = queue.get(local)["events"]
    assert artifact["type"] == "artifact" and artifact["artifact_path"] == "eval"
    assert open(queue.file_path(local, artifact["file"]), encoding="utf-8").read() == "chrF 66.7"
    assert step_log["type"] == "step_log" and open(queue.file_path(local, step_log["file"]), encoding="utf-8").read() == '{"current_steps": 1}\n'


def test_two_files_with_the_same_name_do_not_overwrite_each_other(queue, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "report.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b" / "report.txt").write_text("B", encoding="utf-8")
    local = _new(queue)

    queue.add_artifact(local, str(tmp_path / "a" / "report.txt"))
    queue.add_artifact(local, str(tmp_path / "b" / "report.txt"))

    first, second = queue.get(local)["events"]
    assert open(queue.file_path(local, first["file"]), encoding="utf-8").read() == "A"
    assert open(queue.file_path(local, second["file"]), encoding="utf-8").read() == "B"
    assert first["artifact_path"] is None and first["file"] != second["file"]


def test_pending_lists_runs_oldest_first_and_only_those_with_work_left(queue):
    old, new = _new(queue, "old", now=1), _new(queue, "new", now=2)
    queue.add_params(old, {"a": "1"})
    queue.add_params(new, {"b": "2"})

    assert [r["run_name"] for r in queue.pending()] == ["old", "new"]

    queue.set_remote_id(old, "remote-old")
    queue.mark_sent(old, 0)
    assert [r["run_name"] for r in queue.pending()] == ["new"]  # old is created remotely and has nothing unsent


def test_a_run_not_yet_created_on_the_server_is_pending_even_without_events(queue):
    local = _new(queue)

    assert [r["local_run_id"] for r in queue.pending()] == [local]

    queue.set_remote_id(local, "r1")
    assert queue.pending() == []


def test_the_queue_survives_a_restart_and_only_what_is_unsent_is_kept(tmp_path):
    path, files = str(tmp_path / "q.jsonl"), str(tmp_path / "q.files")
    first = RunQueue(path, files)
    local = _new(first)
    first.add_params(local, {"a": "1"})
    first.add_tags(local, {"b": "2"})
    first.set_remote_id(local, "r1")
    first.mark_sent(local, 0)

    again = RunQueue(path, files).get(local)

    assert again["remote_id"] == "r1" and [e["type"] for e in again["events"]] == ["tags"] and again["events"][0]["sent"] is False


def test_sent_events_are_dropped_at_the_next_start_with_their_files_but_the_run_stays(tmp_path):
    path, files = str(tmp_path / "q.jsonl"), str(tmp_path / "q.files")
    first = RunQueue(path, files)
    local = _new(first)
    (tmp_path / "sent.txt").write_text("done", encoding="utf-8")
    (tmp_path / "waiting.txt").write_text("todo", encoding="utf-8")
    first.add_artifact(local, str(tmp_path / "sent.txt"))
    first.add_artifact(local, str(tmp_path / "waiting.txt"))
    first.set_remote_id(local, "r1")
    first.mark_sent(local, 0)
    sent_copy = first.file_path(local, first.get(local)["events"][0]["file"])
    assert os.path.exists(sent_copy)

    again = RunQueue(path, files)

    (event,) = again.get(local)["events"]
    assert open(again.file_path(local, event["file"]), encoding="utf-8").read() == "todo" and not os.path.exists(sent_copy)
    assert again.has_run(local) and again.get(local)["remote_id"] == "r1"  # a later stage can still resume the run
    assert len(_lines(tmp_path / "q.jsonl")) == 3  # the file itself shrank: run, remote, the one unsent event


def test_a_new_event_after_a_compaction_never_overwrites_a_kept_file(tmp_path):
    path, files = str(tmp_path / "q.jsonl"), str(tmp_path / "q.files")
    first = RunQueue(path, files)
    local = _new(first)
    for name, text in (("a.txt", "sent"), ("b.txt", "keep")):
        (tmp_path / name).write_text(text, encoding="utf-8")
        first.add_artifact(local, str(tmp_path / name))
    first.mark_sent(local, 0)
    again = RunQueue(path, files)
    (tmp_path / "b.txt").write_text("new", encoding="utf-8")

    again.add_artifact(local, str(tmp_path / "b.txt"))

    kept, new = again.get(local)["events"]
    assert kept["file"] != new["file"]
    assert open(again.file_path(local, kept["file"]), encoding="utf-8").read() == "keep"
    assert open(again.file_path(local, new["file"]), encoding="utf-8").read() == "new"


def test_a_run_that_was_closed_can_still_be_pruned_after_its_events_were_dropped(tmp_path):
    path, files = str(tmp_path / "q.jsonl"), str(tmp_path / "q.files")
    first = RunQueue(path, files)
    local = _new(first)
    first.finish(local, "FINISHED")
    first.set_remote_id(local, "r1")
    first.mark_sent(local, 0)

    again = RunQueue(path, files)  # the sent status event is gone now; the run still knows it was closed

    assert again.get(local)["events"] == [] and again.prune(keep=0) == 1 and not again.has_run(local)


def test_two_processes_appending_to_the_same_file_lose_nothing(tmp_path):
    path, files = str(tmp_path / "q.jsonl"), str(tmp_path / "q.files")
    one, two = RunQueue(path, files), RunQueue(path, files)  # e.g. a training and an eval at the same time
    a, b = one.start_run("e", "a", local_run_id="a", now_ms=1), two.start_run("e", "b", local_run_id="b", now_ms=2)

    one.add_params(a, {"x": "1"})
    two.add_params(b, {"y": "2"})
    one.add_tags(b, {"z": "3"})  # a stage continuing the other process's run

    seen = RunQueue(path, files)
    assert [r["run_name"] for r in seen.runs()] == ["a", "b"]
    assert [e["type"] for e in seen.get("a")["events"]] == ["params"] and [e["type"] for e in seen.get("b")["events"]] == ["params", "tags"]


# --- a crash or a bad line --------------------------------------------------------------------------------------------


def test_a_line_cut_off_by_a_crash_is_ignored_and_the_next_write_starts_a_fresh_line(tmp_path):
    path = tmp_path / "q.jsonl"
    queue = RunQueue(str(path), str(tmp_path / "q.files"))
    local = _new(queue)
    queue.add_params(local, {"a": "1"})
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"t": "event", "local_run_id": "' + local + '", "id": 1, "event": {"type": "tag')  # power cut mid-write, no newline

    again = RunQueue(str(path), str(tmp_path / "q.files"))
    again.add_tags(local, {"b": "2"})

    events = RunQueue(str(path), str(tmp_path / "q.files")).get(local)["events"]
    assert again.warning is None and [e["type"] for e in events] == ["params", "tags"]  # the torn line is gone, the next one is whole


def test_a_disk_error_while_appending_raises_and_the_queue_stays_readable(queue, tmp_path, monkeypatch):
    local = _new(queue)
    path = tmp_path / ".run.result.backup.jsonl"
    before = path.read_text(encoding="utf-8")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(run_queue.os, "fsync", boom)
    with pytest.raises(OSError):
        queue.add_params(local, {"a": "1"})
    monkeypatch.undo()

    assert path.read_text(encoding="utf-8").startswith(before)
    assert RunQueue(str(path), str(tmp_path / "f")).has_run(local)  # still a readable queue


def test_a_corrupt_queue_file_is_set_aside_not_lost_and_not_fatal(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"runs": {"1": {"local_run_id": "x", ', encoding="utf-8")  # not a journal at all

    queue = RunQueue(str(path), str(tmp_path / "q.files"))

    assert queue.pending() == [] and queue.warning and "corrupt" in queue.warning
    kept = [p for p in tmp_path.iterdir() if ".corrupt-" in p.name]
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8").startswith('{"runs"')
    queue.start_run("e", "n")  # and it works again
    assert len(queue.pending()) == 1


def test_a_failing_optional_event_is_counted_and_given_up_after_the_limit(queue, tmp_path):
    local = _new(queue, "r", now=1)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    queue.add_artifact(local, str(tmp_path / "f.txt"))

    assert queue.fail_event(local, 0, give_up_after=3) == 1 and queue.get(local)["events"][0]["sent"] is False
    assert queue.fail_event(local, 0, give_up_after=3) == 2
    assert queue.fail_event(local, 0, give_up_after=3) == 3
    event = queue.get(local)["events"][0]
    assert event["sent"] is True and event["gave_up"] is True and event["attempts"] == 3


def test_the_failed_tries_of_an_artifact_survive_a_restart(tmp_path):
    path, files = str(tmp_path / "q.jsonl"), str(tmp_path / "q.files")
    first = RunQueue(path, files)
    local = _new(first)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    first.add_artifact(local, str(tmp_path / "f.txt"))
    assert first.fail_event(local, 0, give_up_after=3) == 1

    second = RunQueue(path, files)

    assert second.get(local)["events"][0]["attempts"] == 1 and second.fail_event(local, 0, give_up_after=3) == 2
    assert RunQueue(path, files).get(local)["events"][0]["attempts"] == 2


def test_a_run_that_later_stages_continued_is_pruned_once_everything_is_sent(queue):
    """Merge, GGUF and eval add events after the training closed the run, so the last event is no longer its status."""
    done = _new(queue, "continued", now=1)
    queue.finish(done, "FINISHED")
    queue.add_tags(done, {"stage.merge": "done"})
    queue.set_remote_id(done, "r1")
    for i in range(len(queue.get(done)["events"])):
        queue.mark_sent(done, i)
    unfinished = _new(queue, "never closed", now=2)
    queue.add_params(unfinished, {"a": "1"})
    queue.set_remote_id(unfinished, "r2")
    queue.mark_sent(unfinished, 0)

    assert queue.prune(keep=0) == 1
    assert not queue.has_run(done) and queue.has_run(unfinished)  # a run no stage closed is kept


def test_synced_runs_are_pruned_to_the_newest_few_with_their_files(queue, tmp_path):
    ids = []
    for n in range(5):
        local = _new(queue, f"run{n}", now=n)
        queue.add_params(local, {"a": "1"})
        (tmp_path / f"f{n}.txt").write_text("x", encoding="utf-8")
        queue.add_artifact(local, str(tmp_path / f"f{n}.txt"))
        queue.finish(local, "FINISHED")
        queue.set_remote_id(local, f"r{n}")
        for i in range(len(queue.get(local)["events"])):
            queue.mark_sent(local, i)
        ids.append(local)
    waiting = _new(queue, "unsent", now=99)
    queue.add_params(waiting, {"a": "1"})

    removed = queue.prune(keep=2)

    assert removed == 3
    assert not queue.has_run(ids[0]) and not queue.has_run(ids[2]) and queue.has_run(ids[3]) and queue.has_run(ids[4])
    assert queue.has_run(waiting)  # never prune what has not been sent
    assert not os.path.exists(os.path.join(str(tmp_path / ".run.result.backup.files"), ids[0]))
    assert os.path.isdir(os.path.join(str(tmp_path / ".run.result.backup.files"), ids[4]))


def test_the_default_locations_are_the_gitignored_ones():
    import config

    assert config.RUN_QUEUE_PATH == os.path.join(config.BASE_DIR, ".run.result.backup.jsonl")
    assert config.RUN_QUEUE_LEGACY == os.path.join(config.BASE_DIR, ".run.result.backup.json")
    assert config.RUN_QUEUE_FILES == os.path.join(config.BASE_DIR, ".run.result.backup.files")
    for name in (".run.result.backup.jsonl", ".run.result.backup.json", ".run.result.backup.jsonl.tmp", ".run.result.backup.jsonl.corrupt-1700000000",
                 ".run.result.backup.json.migrated", ".run.result.backup.files/x"):
        assert subprocess.run(["git", "check-ignore", "-q", name], cwd=config.BASE_DIR).returncode == 0, name


@pytest.mark.parametrize("bad", ["a'b", "x y", "a;b", ""])
def test_a_given_local_id_must_be_safe_to_put_in_a_search_filter(queue, bad):
    with pytest.raises(ValueError, match="local_run_id"):
        queue.start_run("e", "n", local_run_id=bad)


# --- the old TinyDB file (.run.result.backup.json) is read once ---------------------------------------------------------


def _old_file(tmp_path, runs):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"runs": {str(i + 1): doc for i, doc in enumerate(runs)}}), encoding="utf-8")
    return path


def _old_run(local, remote=None, events=()):
    return {"local_run_id": local, "experiment": "resonance-lab", "run_name": local, "start_time_ms": 5, "remote_id": remote, "events": list(events)}


def test_a_queue_in_the_old_format_is_migrated_once_keeping_what_is_unsent(tmp_path):
    old = _old_file(tmp_path, [
        _old_run("stuck", None, [{"type": "params", "params": {"a": "1"}, "sent": False}, {"type": "tags", "tags": {"b": "2"}, "sent": False}]),
        _old_run("half", "r7", [{"type": "params", "params": {"a": "1"}, "sent": True}, {"type": "status", "status": "FINISHED", "ts_ms": 9, "sent": True},
                                {"type": "artifact", "file": "2-log.txt", "artifact_path": "train", "sent": False, "attempts": 1}]),
    ])
    path = tmp_path / "q.jsonl"

    queue = RunQueue(str(path), str(tmp_path / "q.files"), legacy_path=str(old))

    stuck, half = queue.get("stuck"), queue.get("half")
    assert [e["type"] for e in stuck["events"]] == ["params", "tags"] and stuck["remote_id"] is None
    assert [e["type"] for e in half["events"]] == ["artifact"] and half["events"][0]["attempts"] == 1 and half["remote_id"] == "r7"
    assert [r["local_run_id"] for r in queue.pending()] == ["stuck", "half"]
    assert not old.exists() and (tmp_path / "old.json.migrated").exists()  # kept, renamed: nothing lost
    assert RunQueue(str(path), str(tmp_path / "q.files"), legacy_path=str(old)).has_run("half")  # and not read again


def test_a_migrated_run_that_was_closed_can_be_pruned(tmp_path):
    old = _old_file(tmp_path, [_old_run("done", "r1", [{"type": "status", "status": "FINISHED", "ts_ms": 9, "sent": True}])])

    queue = RunQueue(str(tmp_path / "q.jsonl"), str(tmp_path / "f"), legacy_path=str(old))

    assert queue.prune(keep=0) == 1


def test_the_old_file_is_not_touched_when_the_new_queue_already_exists(tmp_path):
    old = _old_file(tmp_path, [_old_run("old-run")])
    path = tmp_path / "q.jsonl"
    RunQueue(str(path), str(tmp_path / "f")).start_run("e", "n", local_run_id="new-run", now_ms=1)

    queue = RunQueue(str(path), str(tmp_path / "f"), legacy_path=str(old))

    assert queue.has_run("new-run") and not queue.has_run("old-run") and old.exists()


def test_a_corrupt_old_file_is_set_aside_and_the_new_queue_starts_empty(tmp_path):
    old = tmp_path / "old.json"
    old.write_text('{"runs": {"1": {"local_run', encoding="utf-8")

    queue = RunQueue(str(tmp_path / "q.jsonl"), str(tmp_path / "f"), legacy_path=str(old))

    assert queue.pending() == [] and queue.warning and "corrupt" in queue.warning
    assert not old.exists() and len([p for p in tmp_path.iterdir() if ".corrupt-" in p.name]) == 1


def test_without_a_legacy_path_nothing_is_migrated_so_a_test_queue_never_reads_the_real_one(tmp_path):
    assert RunQueue(str(tmp_path / "q.jsonl"), str(tmp_path / "f")).runs() == []


def test_a_compacted_journal_ends_lines_with_lf_like_the_appended_ones(tmp_path):
    path, files = tmp_path / "q.jsonl", str(tmp_path / "q.files")
    first = RunQueue(str(path), files)
    local = _new(first)
    first.add_params(local, {"a": "1"})
    first.mark_sent(local, 0)

    RunQueue(str(path), files)  # opening it compacts: the sent event is dropped

    data = path.read_bytes()
    assert data.count(b"\n") == 1 and b"\r" not in data
