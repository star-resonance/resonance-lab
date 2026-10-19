import json
import os

import pytest

import runs
import stage_tracking
import tracker as tracker_module
from lf_tools import Profile
from run_queue import RunQueue
from test_tracker import FakeClient

PROFILE = Profile("tg", "t.yaml", "m.yaml", "google/translategemma-4b-it", "gemma3", "bp_translation", "/o", "/m")
RUN = runs.Run("20261001-185200", "/o/20261001-185200")
LOCAL = "tg-20261001-185200"


class Recorder:
    """A tracker that writes down its calls (the real one is tested in test_tracker.py)."""

    enabled = True

    def __init__(self, knows=(LOCAL,)):
        self.calls, self.run_id, self.knows = [], None, knows

    def _rec(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return call

    def begin(self, experiment, run_name, local_run_id=None):
        self.run_id = local_run_id
        self.calls.append(("begin", (experiment, run_name, local_run_id), {}))

    def resume(self, local_run_id):
        if local_run_id in self.knows:
            self.run_id = local_run_id
            return True
        return False

    def __getattr__(self, name):
        if name in ("params", "tags", "metrics", "step_log", "artifact", "finish", "flush"):
            return self._rec(name)
        raise AttributeError(name)

    def named(self, name):
        return [c for c in self.calls if c[0] == name]


def _write(path, data):
    with open(path, "w", encoding="utf-8") as f:
        f.write(data if isinstance(data, str) else json.dumps(data))


# --- the training ---------------------------------------------------------------------------------------------------


def test_a_training_starts_its_run_before_it_records_anything():
    t = Recorder()

    stage_tracking.start_training(t, PROFILE, RUN, {"learning_rate": 1e-5}, None, None, git={}, packages={})

    names = [c[0] for c in t.calls]
    assert names[0] == "begin" and set(names[1:]) == {"params", "tags"}
    assert t.calls[0][1] == ("resonance-lab", "tg-20261001-185200", LOCAL)
    assert t.named("params")[0][1][0]["profile"] == "tg" and t.named("tags")[0][1][0]["stage"] == "train"


def _training_files(tmp_path):
    _write(tmp_path / "trainer_log.jsonl", '{"current_steps": 1, "loss": 2.0, "elapsed_time": "0:00:01"}\n')
    _write(tmp_path / "train_results.json", {"train_runtime": 100.0})
    _write(tmp_path / "trainer_state.json", {"global_step": 9, "best_model_checkpoint": "o/checkpoint-8",
                                             "log_history": [{"step": 1, "loss": 2.0, "grad_norm": 7.5}]})
    _write(tmp_path / "train_stdout.log", "[INFO] trainable params: 13,434,880 || all params: 3,893,000,000 || trainable%: 0.3451\nstep 1\n")
    yaml_file, manifest = tmp_path / "train.yaml", tmp_path / "lora_train_data.meta.json"
    _write(yaml_file, "learning_rate: 1.0e-5\n")
    _write(manifest, {"style": "translategemma"})
    return str(yaml_file), str(manifest)


def test_a_finished_training_sends_its_curves_results_and_recipe_then_closes_the_run(tmp_path):
    train_yaml, manifest = _training_files(tmp_path)
    t = Recorder()

    stage_tracking.finish_training(t, str(tmp_path), "FINISHED", train_yaml=train_yaml, manifest_file=manifest)

    curves = tmp_path / "curves.jsonl"  # the trainer's log with the gradient norm of its state file joined in
    assert t.named("step_log")[0][1] == (str(curves),)
    assert json.loads(curves.read_text(encoding="utf-8").splitlines()[0])["grad_norm"] == 7.5
    assert t.named("metrics")[0][1][0]["train.runtime_s"] == 100.0
    tags = {k: v for call in t.named("tags") for k, v in call[1][0].items()}
    assert tags["train.best_checkpoint"] == "checkpoint-8" and tags["train.global_step"] == "9"
    assert tags["train.trainable_params"] == "13434880" and tags["train.all_params"] == "3893000000"
    sent = [os.path.basename(c[1][0]) for c in t.named("artifact")]
    assert sent == ["train.yaml", "lora_train_data.meta.json", "trainer_state.json"]  # the recipe and data record, not the noisy log
    assert all(c[1][1] == "train" for c in t.named("artifact"))
    assert t.calls[-1][0] == "flush" and t.named("finish")[0][1] == ("FINISHED",)


@pytest.mark.parametrize("status", ["FAILED", "KILLED"])
def test_a_failed_or_cancelled_training_also_sends_its_log_where_the_traceback_is(tmp_path, status):
    train_yaml, manifest = _training_files(tmp_path)
    t = Recorder()

    stage_tracking.finish_training(t, str(tmp_path), status, train_yaml=train_yaml, manifest_file=manifest)

    assert "train_stdout.log" in [os.path.basename(c[1][0]) for c in t.named("artifact")]
    assert t.named("finish")[0][1] == (status,)


def test_the_status_is_closed_before_the_artifacts_go_up(tmp_path):
    train_yaml, manifest = _training_files(tmp_path)
    t = Recorder()

    stage_tracking.finish_training(t, str(tmp_path), "FINISHED", train_yaml=train_yaml, manifest_file=manifest)

    names = [c[0] for c in t.calls]
    assert names.index("finish") < names.index("artifact")  # a slow or failing upload cannot keep the run RUNNING


def test_an_interrupted_training_is_killed_and_a_crashed_one_failed():
    assert stage_tracking.training_status(KeyboardInterrupt()) == "KILLED"
    assert stage_tracking.training_status(SystemExit(1)) == "FAILED"
    assert stage_tracking.training_status(RuntimeError("x")) == "FAILED"


def test_a_failed_training_with_no_result_files_still_closes_its_run(tmp_path):
    t = Recorder()

    stage_tracking.finish_training(t, str(tmp_path), "FAILED", manifest_file=str(tmp_path / "none.json"))

    assert not t.named("step_log") and not t.named("metrics") and not t.named("artifact")
    assert t.named("finish")[0][1] == ("FAILED",) and t.named("flush")


def test_the_training_context_reads_what_the_data_stages_left(tmp_path):
    fetch = tmp_path / "fetch_state.json"
    _write(fetch, {"repo": "r", "revision": "a" * 40})
    manifest = tmp_path / "lora_train_data.meta.json"
    _write(manifest, {"style": "translategemma", "reverse": False, "counts": {}})

    context = stage_tracking.training_context(str(fetch), str(manifest))

    assert context["fetch_state"]["repo"] == "r" and context["manifest"]["style"] == "translategemma"
    assert isinstance(context["git"], dict) and isinstance(context["packages"], dict)


def test_a_missing_data_record_is_none_not_an_error(tmp_path):
    context = stage_tracking.training_context(str(tmp_path / "x.json"), str(tmp_path / "y.json"))

    assert context["fetch_state"] is None and context["manifest"] is None


# --- merge, gguf, eval resume the training's run -----------------------------------------------------------------


def test_a_later_stage_resumes_the_run_its_record_names():
    t = Recorder()

    assert stage_tracking.resume_stage(t, {"run": "20261001-185200", "profile": "tg"}) is True
    assert t.run_id == LOCAL


@pytest.mark.parametrize("record", [None, {"run": None, "profile": "tg"}, {"run": "20990101-000000", "profile": "tg"}])
def test_a_stage_without_a_known_run_records_nothing(record):
    t = Recorder()

    assert stage_tracking.resume_stage(t, record) is False
    stage_tracking.record_stage(t, {"stage.merge": "done"})
    assert not t.calls


def test_a_stage_records_its_tags_metrics_and_artifacts_then_flushes(tmp_path):
    report = tmp_path / "r.txt"
    report.write_text("x", encoding="utf-8")
    t = Recorder()
    stage_tracking.resume_stage(t, {"run": "20261001-185200", "profile": "tg"})

    stage_tracking.record_stage(t, {"stage": "eval"}, {"eval.chrf": 45.2}, [(str(report), "eval")])

    assert [c[0] for c in t.calls] == ["tags", "metrics", "artifact", "flush"]
    assert t.named("artifact")[0][1] == (str(report), "eval")


def test_a_failing_stage_is_tagged_not_closed():
    t = Recorder()
    stage_tracking.resume_stage(t, {"run": "20261001-185200", "profile": "tg"})

    stage_tracking.fail_stage(t, "merge")

    assert t.named("tags")[0][1][0] == {"stage.merge": "failed"} and t.named("flush") and not t.named("finish")


# --- with the real tracker and the offline queue ----------------------------------------------------------------


def test_the_whole_flow_lands_in_one_run_even_with_the_nas_down(tmp_path):
    queue = RunQueue(str(tmp_path / "q.json"), str(tmp_path / "files"))
    client = FakeClient()
    client.down = True
    t = tracker_module.Tracker(queue, client, log=lambda *a: None)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    stage_tracking.start_training(t, PROFILE, RUN, {"learning_rate": 1e-5}, None, None, git={}, packages={})
    stage_tracking.finish_training(t, str(run_dir), "FINISHED")
    later = tracker_module.Tracker(queue, client, log=lambda *a: None)
    assert stage_tracking.resume_stage(later, {"run": RUN.id, "profile": "tg"})
    stage_tracking.record_stage(later, {"stage.merge": "done"})

    [run] = queue.runs()
    assert run["local_run_id"] == LOCAL and [e["type"] for e in run["events"]] == ["params", "tags", "status", "tags"]


# --- opening the tracker never stops a stage ---------------------------------------------------------------------


def test_a_broken_tracker_import_turns_tracking_off_with_a_message(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_tracker(name, *args, **kwargs):
        if name == "tracker":
            raise ImportError("No module named 'tinydb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tracker)
    lines = []

    t = stage_tracking.open_tracker(log=lines.append)

    assert t.enabled is False and t.run_id is None and t.resume("x") is False
    t.begin("e", "n", "id")
    t.flush()  # every call is a no-op
    assert any("off" in line and "tinydb" in line for line in lines)


def test_tracking_off_by_the_environment_gives_a_tracker_that_does_nothing(monkeypatch):
    monkeypatch.setenv("RESONANCE_MLFLOW", "0")

    t = stage_tracking.open_tracker(log=lambda *a: None)

    assert t.enabled is False and t.resume("x") is False


def test_the_curves_file_ends_lines_with_lf_on_every_os(tmp_path):
    train_yaml, manifest = _training_files(tmp_path)

    stage_tracking.finish_training(Recorder(), str(tmp_path), "FINISHED", train_yaml=train_yaml, manifest_file=manifest)

    data = (tmp_path / "curves.jsonl").read_bytes()
    assert data.count(b"\n") == 1 and b"\r" not in data
