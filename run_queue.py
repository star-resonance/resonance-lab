"""The local record of every run, written before anything is sent to the MLflow server. No network, no mlflow.

An append-only journal (`.run.result.backup.jsonl`, one JSON object per line) holds everything. A run is a `run` line (its
header: experiment, name, start), a `remote` line once the server has created it, and `event` lines in the order the
stages recorded them (params, tags, metrics, a trainer log, an artifact, the final status). Once the server has an event,
a `sent` line acknowledges it (a failed optional upload is a `fail` line with its try count). So a run made while the NAS
was off, or a sync that stopped halfway, is simply the events that are still unacknowledged: tracker.sync replays them
as a queue, oldest run first, without sending anything twice. Small files are copied into
`.run.result.backup.files/<run>/<event id>-<name>` so the originals may be deleted.

Every write appends one line (nothing already written is touched), so a crash leaves at worst one torn last line, which
is ignored, and two processes writing at once (a training and an eval) lose nothing. State is read by replaying the file.
At each start the journal is compacted: acknowledged events are dropped with their copied files; the run header and the
server's id stay (a later stage resumes the run by them, and the id stops a run from being created twice), until
`prune` forgets old finished runs. Compaction is an atomic temp-file replace and is skipped, to be retried next time,
when the file cannot be replaced (another process has it open on Windows).
"""

import copy
import json
import os
import re
import shutil
import time
import uuid

from config import RUN_QUEUE_FILES, RUN_QUEUE_LEGACY, RUN_QUEUE_PATH

_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")
_UNSET = object()
_INTERNAL = ("_id", "_closed")


def _now_ms():
    return int(time.time() * 1000)


def _public(run):
    """A run as callers see it: the header, the remote id and the events with their `sent` flags (a copy)."""
    doc = {k: v for k, v in run.items() if k not in _INTERNAL}
    doc["events"] = [{k: v for k, v in event.items() if k not in _INTERNAL} for event in run["events"]]
    return copy.deepcopy(doc)


class RunQueue:
    def __init__(self, path=RUN_QUEUE_PATH, files_dir=RUN_QUEUE_FILES, legacy_path=_UNSET):
        """`legacy_path`: the old TinyDB file (`.run.result.backup.json`), read once when there is no journal yet. Only the
        default queue looks for it: a queue made for a test never reads the maintainer's."""
        self.path, self.files_dir, self.warning = path, files_dir, None
        if legacy_path is _UNSET:
            legacy_path = RUN_QUEUE_LEGACY if path == RUN_QUEUE_PATH else None
        self._open(legacy_path)

    # --- opening: migrate once, set a corrupt file aside, compact ---

    def _set_aside(self, path, why):
        kept = f"{path}.corrupt-{int(time.time())}"
        os.replace(path, kept)  # never lose it silently, never stop a training because of it
        self.warning = f"queue file {path} was corrupt ({why}); kept as {kept}, starting a new one"

    def _open(self, legacy_path):
        if not os.path.exists(self.path) and legacy_path and os.path.exists(legacy_path):
            self._migrate(legacy_path)
        records, unreadable, tail_torn, has_text = self._read()
        if has_text and not records:
            self._set_aside(self.path, "no line of it is a journal record")
            return
        if unreadable - (1 if tail_torn else 0) > 0:
            self.warning = f"{unreadable - (1 if tail_torn else 0)} unreadable line(s) in {os.path.basename(self.path)} were skipped"
        if unreadable or any(record["t"] in ("sent", "fail") for record in records):
            self._compact(self._replay(records))

    def _migrate(self, legacy_path):
        """The old TinyDB file ({"runs": {"1": {run document}}}) becomes a journal holding what was still unsent; the old
        file is renamed `.migrated`, not deleted."""
        try:
            with open(legacy_path, encoding="utf-8") as f:
                docs = list((json.load(f).get("runs") or {}).values())
        except (ValueError, AttributeError) as e:
            self._set_aside(legacy_path, str(e))
            return
        runs = {}
        for doc in docs:
            events = [{**{k: v for k, v in event.items() if k != "sent"}, "sent": bool(event.get("sent")), "_id": number}
                      for number, event in enumerate(doc["events"])]
            runs[doc["local_run_id"]] = {
                "local_run_id": doc["local_run_id"], "experiment": doc["experiment"], "run_name": doc["run_name"],
                "start_time_ms": doc["start_time_ms"], "remote_id": doc.get("remote_id"), "events": events,
                "_closed": any(e["type"] == "status" and e["sent"] for e in events),
            }
        if self._compact(runs, force=True):
            os.replace(legacy_path, legacy_path + ".migrated")

    # --- the journal ---

    def _read(self):
        """(records, unreadable lines, whether the last line is one of them, whether there is any text)."""
        try:
            with open(self.path, encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            return [], 0, False, False
        records, unreadable, tail_torn = [], 0, False
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        for number, line in enumerate(lines):
            try:
                record = json.loads(line)
            except ValueError:
                record = None
            if isinstance(record, dict) and isinstance(record.get("t"), str):
                records.append(record)
            else:
                unreadable += 1
                tail_torn = number == len(lines) - 1
        return records, unreadable, tail_torn, bool(lines)

    @staticmethod
    def _replay(records):
        """The runs the records describe, in the order their headers were written."""
        runs = {}
        for record in records:
            kind, local = record["t"], record.get("local_run_id")
            if kind == "run":
                if local not in runs:
                    runs[local] = {"local_run_id": local, "experiment": record.get("experiment"), "run_name": record.get("run_name"),
                                   "start_time_ms": record.get("start_time_ms", 0), "remote_id": None, "events": [],
                                   "_closed": bool(record.get("closed"))}
                continue
            run = runs.get(local)
            if run is None:
                continue
            if kind == "remote":
                run["remote_id"] = record.get("remote_id")
            elif kind == "event":
                if all(e["_id"] != record.get("id") for e in run["events"]):
                    run["events"].append({**record["event"], "sent": False, "_id": record["id"]})
            elif kind in ("sent", "fail"):
                event = next((e for e in run["events"] if e["_id"] == record.get("id")), None)
                if event is None:
                    continue
                if kind == "fail":
                    event["attempts"] = record.get("attempts", 1)
                if kind == "sent" or record.get("gave_up"):
                    event["sent"] = True
                    event["gave_up"] = bool(record.get("gave_up"))
                    if not event["gave_up"]:
                        event.pop("gave_up")
                    if event["type"] == "status":
                        run["_closed"] = True
        return runs

    def _state(self):
        return self._replay(self._read()[0])

    def _append(self, record):
        """One line at the end of the file, on disk before this returns. A torn last line is closed off first."""
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a+b") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            prefix = b""
            if size:
                f.seek(size - 1)
                prefix = b"" if f.read(1) == b"\n" else b"\n"
            f.write(prefix + line.encode("utf-8"))  # append mode: always at the end, whatever the seek above
            f.flush()
            os.fsync(f.fileno())

    def _compact(self, runs, force=False):
        """Rewrite the journal with only what is still needed: headers, remote ids, unsent events (with their try counts).
        Drops the copied files of the acknowledged events. Returns False when the file could not be replaced."""
        lines, dropped = [], []
        for local, run in runs.items():
            header = {"t": "run", "local_run_id": local, "experiment": run["experiment"], "run_name": run["run_name"],
                      "start_time_ms": run["start_time_ms"]}
            if run["_closed"]:
                header["closed"] = True
            lines.append(header)
            if run["remote_id"] is not None:
                lines.append({"t": "remote", "local_run_id": local, "remote_id": run["remote_id"]})
            for event in run["events"]:
                if event["sent"]:
                    dropped.append((local, event))
                else:
                    payload = {k: v for k, v in event.items() if k not in ("sent", "_id", "gave_up")}
                    lines.append({"t": "event", "local_run_id": local, "id": event["_id"], "event": payload})
        temp = self.path + ".tmp"
        try:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(temp, "w", encoding="utf-8", newline="\n") as f:  # LF, like the lines _append writes as bytes
                f.write("".join(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n" for line in lines))
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, self.path)
        except OSError:
            return False
        finally:
            if os.path.exists(temp):
                os.remove(temp)
        for local, event in dropped:
            if event.get("file"):
                try:
                    os.remove(self.file_path(local, event["file"]))
                except OSError:
                    pass
        return True

    # --- reading ---

    def runs(self):
        """Every run, oldest start first."""
        return sorted((_public(run) for run in self._state().values()), key=lambda doc: doc["start_time_ms"])

    def _run(self, local_run_id):
        run = self._state().get(local_run_id)
        if run is None:
            raise KeyError(local_run_id)
        return run

    def has_run(self, local_run_id):
        return local_run_id in self._state()

    def get(self, local_run_id):
        """A copy of the run (changing it changes nothing)."""
        return _public(self._run(local_run_id))

    def pending(self):
        """The runs that still have something to send, oldest first: not created on the server yet, or with an unsent event."""
        return [run for run in self.runs() if run["remote_id"] is None or any(not e["sent"] for e in run["events"])]

    def file_path(self, local_run_id, name):
        return os.path.join(self.files_dir, local_run_id, name)

    # --- writing: every call is on disk before it returns ---

    def start_run(self, experiment, run_name, now_ms=None, local_run_id=None):
        """Record a new run (or return the id of the run that already has `local_run_id`)."""
        if local_run_id is not None:
            if not _SAFE_ID.fullmatch(local_run_id):
                raise ValueError(f"local_run_id {local_run_id!r} may only hold letters, digits, - and _")
            if self.has_run(local_run_id):
                return local_run_id
        local_run_id = local_run_id or uuid.uuid4().hex
        self._append({"t": "run", "local_run_id": local_run_id, "experiment": experiment, "run_name": run_name,
                      "start_time_ms": now_ms if now_ms is not None else _now_ms()})
        return local_run_id

    def _add_event(self, local_run_id, event, source_file=None):
        run = self._run(local_run_id)
        event_id = max((e["_id"] for e in run["events"]), default=-1) + 1
        if source_file:
            event = {**event, "file": self._copy_in(local_run_id, event_id, source_file)}
        self._append({"t": "event", "local_run_id": local_run_id, "id": event_id, "event": event})

    def add_params(self, local_run_id, params):
        self._add_event(local_run_id, {"type": "params", "params": dict(params)})

    def add_tags(self, local_run_id, tags):
        self._add_event(local_run_id, {"type": "tags", "tags": dict(tags)})

    def add_metrics(self, local_run_id, metrics, ts_ms=None):
        self._add_event(local_run_id, {"type": "metrics", "metrics": dict(metrics), "ts_ms": ts_ms if ts_ms is not None else _now_ms()})

    def _copy_in(self, local_run_id, event_id, path):
        name = f"{event_id}-{os.path.basename(path)}"
        os.makedirs(os.path.join(self.files_dir, local_run_id), exist_ok=True)
        shutil.copyfile(path, self.file_path(local_run_id, name))
        return name

    def add_step_log(self, local_run_id, path):
        """The trainer's curves file: its points are sent after training (tracking.step_metrics)."""
        self._add_event(local_run_id, {"type": "step_log"}, source_file=path)

    def add_artifact(self, local_run_id, path, artifact_path=None):
        self._add_event(local_run_id, {"type": "artifact", "artifact_path": artifact_path}, source_file=path)

    def finish(self, local_run_id, status, now_ms=None):
        self._add_event(local_run_id, {"type": "status", "status": status, "ts_ms": now_ms if now_ms is not None else _now_ms()})

    def set_remote_id(self, local_run_id, remote_id):
        self._run(local_run_id)
        self._append({"t": "remote", "local_run_id": local_run_id, "remote_id": remote_id})

    def mark_sent(self, local_run_id, index):
        event = self._run(local_run_id)["events"][index]
        self._append({"t": "sent", "local_run_id": local_run_id, "id": event["_id"]})

    def fail_event(self, local_run_id, index, give_up_after):
        """Count a failed send of an optional event (an artifact). After `give_up_after` tries it counts as sent and
        `gave_up`, so one broken upload stops being retried at every start. Returns the number of tries so far."""
        event = self._run(local_run_id)["events"][index]
        tries = event.get("attempts", 0) + 1
        self._append({"t": "fail", "local_run_id": local_run_id, "id": event["_id"], "attempts": tries, "gave_up": tries >= give_up_after})
        return tries

    def prune(self, keep=20):
        """Forget the oldest runs the server has completely (created, every event sent, closed by a status event),
        keeping the newest `keep`. Returns how many were forgotten."""
        runs = self._state()
        done = [run for run in sorted(runs.values(), key=lambda r: r["start_time_ms"])
                if run["remote_id"] is not None and run["_closed"] and all(e["sent"] for e in run["events"])]
        old = done[:-keep] if keep else done
        if old and self._compact({k: v for k, v in runs.items() if v not in old}):
            for run in old:
                shutil.rmtree(os.path.join(self.files_dir, run["local_run_id"]), ignore_errors=True)
            return len(old)
        return 0
