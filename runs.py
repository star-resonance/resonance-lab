"""One training = one run: its own directory, status and log. Pure, no torch.

LLaMA-Factory resumes from the last checkpoint it finds in `output_dir` (hparams/parser.py), so a retrain into the
profile's fixed directory silently continued the previous run, with the old data. Every training now gets a fresh
directory `<profile output_dir>/<run id>/` (train.py passes it as an `output_dir=` override); `run.json` in it says
whether the run is running, failed or complete, and merge.py only merges a complete one. Adapters trained before runs
existed sit directly in the profile's output_dir and are still found.
"""

import json
import os
from datetime import datetime, timezone
from typing import NamedTuple

from config import TRAIN_LOG_NAME

RUN_FILE = "run.json"
LATEST_FILE = "latest_run.txt"  # in the profile's output_dir: the id of the newest run
MERGE_RECORD = "resonance_run.json"  # in a merged model's directory: the run it was merged from
STATUSES = ("running", "complete", "failed")


class RunError(Exception):
    """There is no (finished) run to use."""


class Run(NamedTuple):
    id: str
    dir: str


def _now(now=None):
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc)


def new_run_id(now=None):
    return _now(now).strftime("%Y%m%d-%H%M%S")


def _write_json(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_run(run_dir):
    """The run.json of a run directory (None when there is none)."""
    try:
        with open(os.path.join(run_dir, RUN_FILE), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None


def start_run(base_dir, profile_name, now=None):
    """Create a fresh run directory under `base_dir`, mark it running and make it the latest run."""
    os.makedirs(base_dir, exist_ok=True)
    run_id = new_run_id(now)
    suffix = 1
    while True:
        candidate = run_id if suffix == 1 else f"{run_id}-{suffix}"
        try:
            os.mkdir(os.path.join(base_dir, candidate))
            break
        except FileExistsError:
            suffix += 1
    run = Run(candidate, os.path.join(base_dir, candidate))
    _write_json(os.path.join(run.dir, RUN_FILE), {
        "run": run.id, "profile": profile_name, "status": "running", "started": _now(now).isoformat(timespec="seconds"),
    })
    with open(os.path.join(base_dir, LATEST_FILE), "w", encoding="utf-8") as f:
        f.write(run.id)
    return run


def finish_run(run_dir, status):
    if status not in STATUSES:
        raise ValueError(f"unknown run status {status!r}; choose one of: {', '.join(STATUSES)}")
    record = read_run(run_dir) or {}
    record.update(status=status, finished=_now().isoformat(timespec="seconds"))
    _write_json(os.path.join(run_dir, RUN_FILE), record)


def latest_run(base_dir):
    """The newest run's directory (None when there is no run, or the pointer is stale)."""
    try:
        with open(os.path.join(base_dir, LATEST_FILE), encoding="utf-8") as f:
            path = os.path.join(base_dir, f.read().strip())
    except FileNotFoundError:
        return None
    return path if os.path.isdir(path) else None


def _require_complete(run_dir):
    record = read_run(run_dir)
    status = record["status"] if record else "unknown (no run.json)"
    if status != "complete":
        raise RunError(f"run {os.path.basename(run_dir)} in {os.path.dirname(run_dir)} is {status}; "
                       "only a complete run is merged (--run picks another one)")


def resolve_adapter(base_dir, run_id=None):
    """The adapter directory to merge: run `run_id`, else the latest run, else an adapter trained before runs existed.

    Raises RunError when there is none, or the run did not finish.
    """
    if run_id:
        run_dir = os.path.join(base_dir, run_id)
        if not os.path.isdir(run_dir):
            raise RunError(f"no run {run_id!r} in {base_dir}")
        _require_complete(run_dir)
        return run_dir
    latest = latest_run(base_dir)
    if latest:
        _require_complete(latest)
        return latest
    if os.path.isfile(os.path.join(base_dir, "adapter_config.json")):
        return base_dir  # trained before runs existed
    raise RunError(f"No trained adapter in {base_dir}. Run the Fine-Tuning stage first.")


def run_files(base_dir):
    """(trainer_log.jsonl, train_stdout.log) of the latest run -- or of the adapter directory itself before any run."""
    folder = latest_run(base_dir) or base_dir
    return os.path.join(folder, "trainer_log.jsonl"), os.path.join(folder, TRAIN_LOG_NAME)


def write_merge_record(merged_dir, profile_name, adapter_dir):
    """Leave a note in the merged model's directory saying which run it came from (run None: a pre-runs adapter)."""
    record = read_run(adapter_dir)
    path = os.path.join(merged_dir, MERGE_RECORD)
    _write_json(path, {
        "run": record["run"] if record else None, "profile": profile_name, "adapter_dir": adapter_dir,
        "merged": _now().isoformat(timespec="seconds"),
    })
    return path


def read_merge_record(merged_dir):
    """The note write_merge_record left in a merged model's directory (None when there is none): which run it came from."""
    try:
        with open(os.path.join(merged_dir, MERGE_RECORD), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None
