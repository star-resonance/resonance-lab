"""The stages' use of the experiment tracker, so the model-part scripts stay a few lines each. No torch, no mlflow.

Every function takes the tracker (`tracker.from_environment()`: a `Tracker`, or a `NullTracker` when tracking is off) and
never raises into a stage: the tracker's own calls are guarded, and a stage that finds no run to continue records nothing.
Training starts a run; merge, GGUF and eval continue it (`resume_stage`) by the run id their record names (track_records.py).
"""

import os

import track_records
import tracking
from config import HF_DATA_DIR, MLFLOW_EXPERIMENT, PROCESSED_LOGS, TRAIN_LOG_NAME
from hf_data import read_state
from manifest import manifest_path

TRAINER_LOG = "trainer_log.jsonl"
TRAIN_RESULTS = "train_results.json"
TRAINER_STATE = "trainer_state.json"
CURVES = "curves.jsonl"  # trainer_log.jsonl with the gradient norm of trainer_state.json joined in; what the tracker sends as curves


class _Off:
    """Tracking is off because the tracker module could not even be imported (tracker.NullTracker needs it)."""

    enabled = False
    run_id = None

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def resume(self, local_run_id):
        return False


def open_tracker(log=print):
    """The tracker for this stage (`tracker.from_environment()`); a tracker that does nothing when tracking is off or
    cannot be set up (a missing package, a bad .env.mlflow). Never raises: a stage runs without tracking, not not at all."""
    try:
        import tracker
        return tracker.from_environment(log=log)
    except Exception as e:  # noqa: BLE001 - tracking must never stop a stage
        log(f"[tracking] off: {type(e).__name__}: {e}")
        return _Off()


def training_context(fetch_state_path=os.path.join(HF_DATA_DIR, "fetch_state.json"), manifest_file=manifest_path(PROCESSED_LOGS)):
    """What a training knew about its data and its code: the HF fetch record, the preprocess manifest, git, packages.
    A record that is missing is None (a local raw log has no fetch record)."""
    return {"fetch_state": read_state(fetch_state_path), "manifest": track_records.read_json(manifest_file),
            "git": tracking.git_info(), "packages": tracking.package_versions()}


def start_training(tracker, profile, run, train_cfg, fetch_state, manifest, git=None, packages=None, overrides=None):
    """Begin the tracker run of a training and record its recipe, data and code; returns the tracker run id. `overrides`
    (--lr / --epochs) replace the yaml's values in what is recorded, as they do in the training, and are tagged."""
    local_id, name = track_records.run_identity(profile.name, run.id)
    tracker.begin(MLFLOW_EXPERIMENT, name, local_id)
    params, tags = track_records.train_records(profile.name, profile.base_model, profile.template, {**train_cfg, **(overrides or {})},
                                               fetch_state, manifest, run.id, git=git, packages=packages, overrides=overrides)
    tracker.params(params)
    tracker.tags(tags)
    return local_id


def training_status(error):
    """The MLflow status a training that stopped with `error` is closed with: KILLED for Ctrl+C, else FAILED."""
    return "KILLED" if isinstance(error, KeyboardInterrupt) else "FAILED"


def _first_text(path, limit=1_000_000):
    """The start of a log (the trainer prints the model size early); '' when there is none."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return ""


def _curves_file(log, curves, state):
    """Write the curves file (the log + `grad_norm`) and return its path; the plain log when that cannot be done."""
    try:
        with open(log, encoding="utf-8") as f:
            lines = tracking.curve_lines(f.read().splitlines(), state)
        with open(curves, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(line + "\n" for line in lines))
        return curves
    except OSError:
        return log


def finish_training(tracker, run_dir, status, train_yaml=None, manifest_file=manifest_path(PROCESSED_LOGS)):
    """Send the curves, the result metrics and the model size of a training, close its run (`FINISHED` / `FAILED` /
    `KILLED`), then upload the files that say how it was made and flush.

    The run is closed BEFORE the artifacts: a slow or failing upload (MinIO down) must not leave it RUNNING. Artifacts:
    the recipe (`train.yaml`), the data record and the trainer state; the noisy log only for a run that did not finish,
    where it holds the traceback.
    """
    log = os.path.join(run_dir, TRAINER_LOG)
    results = track_records.read_json(os.path.join(run_dir, TRAIN_RESULTS))
    state = track_records.read_json(os.path.join(run_dir, TRAINER_STATE))
    if os.path.isfile(log):
        tracker.step_log(_curves_file(log, os.path.join(run_dir, CURVES), state))
    metrics, tags = track_records.train_result_records(results, state)
    stdout = os.path.join(run_dir, TRAIN_LOG_NAME)
    tags = {**tags, **tracking.model_size_tags(_first_text(stdout))}
    if metrics:
        tracker.metrics(metrics)
    if tags:
        tracker.tags(tags)
    tracker.finish(status)
    files = [train_yaml, manifest_file, os.path.join(run_dir, TRAINER_STATE)]
    if status != "FINISHED":
        files.append(stdout)
    for path in files:
        if path and os.path.isfile(path):
            tracker.artifact(path, "train")
    tracker.flush()


def resume_stage(tracker, record):
    """Continue the run `record` ({'run', 'profile'}) names; False (and nothing is recorded) when it names none or the
    tracker does not know it -- an adapter trained before runs existed, or tracking turned off."""
    local_id = track_records.stage_run_id(record)
    return bool(local_id) and bool(tracker.resume(local_id))


def record_stage(tracker, tags, metrics=None, artifacts=()):
    """Record a finished stage in the run `resume_stage` continued: tags, metrics, files `(path, folder)`; then flush."""
    if not tracker.run_id:
        return
    tracker.tags(tags)
    if metrics:
        tracker.metrics(metrics)
    for path, folder in artifacts:
        tracker.artifact(path, folder)
    tracker.flush()


def fail_stage(tracker, stage):
    """Tag a stage that failed; the run itself stays as the training closed it (its model may still be good)."""
    record_stage(tracker, {f"stage.{stage}": "failed"})
