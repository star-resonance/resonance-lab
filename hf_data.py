"""The training data as it lives on Hugging Face. Pure, no network, no torch (the download is scripts/fetch_data.py).

The app (resonance-stream) exports one `dataset_<CHANNEL>.jsonl` per chat channel; they are stored in a dataset repo
and fetched at a pinned revision, so a run always knows which version of the data it trained on. The channel files are
merged into the one raw log that validate.py and preprocess.py read.
"""

import glob
import json
import os
import re
from datetime import datetime, timezone

import yaml

from manifest import file_sha256

DEFAULT_INCLUDE = "dataset_*.jsonl"
CHANNEL_FILE = re.compile(r"^dataset_(.+)\.jsonl$")
# How the merged raw log is made; a state written under another format is stale (is_current), so one `fetch_data.py` re-merges
# the cached download. 2: every row carries its `channel` (the part of the file name between `dataset_` and `.jsonl`).
MERGE_FORMAT = 2


class FetchError(Exception):
    """The dataset config or the downloaded files cannot be used."""


def load_config(path):
    """{repo, revision, include} from configs/hf_dataset.yaml (repo / revision are None while unset)."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {"repo": raw.get("repo"), "revision": raw.get("revision"), "include": raw.get("include") or DEFAULT_INCLUDE}


def download_command(cfg, local_dir):
    """The `hf download` argument list for the pinned revision; refuses an unpinned one."""
    if not cfg.get("revision"):
        raise FetchError("configs/hf_dataset.yaml has a repo but no revision: run `python scripts/fetch_data.py --pin`")
    return ["hf", "download", cfg["repo"], "--repo-type", "dataset", "--revision", cfg["revision"],
            "--include", cfg["include"], "--local-dir", local_dir]


def channel_files(local_dir, include=DEFAULT_INCLUDE):
    """The downloaded channel files, in name order (so the merged log does not depend on the file system)."""
    return sorted(glob.glob(os.path.join(local_dir, include)))


def channel_of(path):
    """The chat channel a `dataset_<CHANNEL>.jsonl` file holds (None for a file named otherwise)."""
    match = CHANNEL_FILE.match(os.path.basename(path))
    return match.group(1) if match else None


def _with_channel(line, channel):
    """The row with a `channel` field; a damaged line, a non-object or a row that has its own channel is returned untouched."""
    if channel is None:
        return line
    try:
        row = json.loads(line)
    except ValueError:
        return line  # validate.py counts the damaged lines
    if not isinstance(row, dict) or "channel" in row:
        return line
    return json.dumps({**row, "channel": channel}, ensure_ascii=False)


def merge_channels(files, out_path):
    """Concatenate the channel files' lines into `out_path`, each row tagged with its channel (a channel is a strong hint of
    what a line is: party chat recruits, world chat chats); returns {rows, files: {name: sha256}, raw_sha256, merge_format}."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for path in files:
            channel = channel_of(path)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(_with_channel(line.rstrip("\r\n"), channel) + "\n")  # a channel's last line may lack its newline
                        rows += 1
    return {"rows": rows, "files": {os.path.basename(p): file_sha256(p) for p in files}, "raw_sha256": file_sha256(out_path),
            "merge_format": MERGE_FORMAT}


def read_state(path):
    """What the last fetch wrote (None when there was none, or it is unreadable)."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None


def write_state(path, cfg, merged):
    state = {"repo": cfg["repo"], "revision": cfg["revision"], "include": cfg["include"], **merged,
             "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def is_current(state, cfg, raw_path):
    """True when the raw log is what the last fetch wrote for this repo and revision, in the current merge format, and has
    not been edited since."""
    if not state or state.get("repo") != cfg["repo"] or state.get("revision") != cfg["revision"]:
        return False
    if state.get("merge_format") != MERGE_FORMAT:
        return False
    try:
        return state.get("raw_sha256") == file_sha256(raw_path)
    except FileNotFoundError:
        return False


def with_revision(text, revision):
    """The yaml text with its `revision:` line set to `revision` (comments and the rest are kept)."""
    out, count = re.subn(r"(?m)^revision:.*$", f"revision: {revision}", text, count=1)
    if not count:
        raise FetchError("configs/hf_dataset.yaml has no `revision:` line")
    return out


def pin_report(repo, current, latest, written=False):
    """(message, is_current): whether the config's pinned revision `current` is the repo's `latest` commit.
    `written` words it for `--pin`, which is about to write `latest`, instead of `--check`, which only looks."""
    if not current:
        return (f"{repo} pinned at {latest[:12]} (was unset)" if written
                else f"{repo} not pinned (was unset); latest is {latest[:12]}"), False
    if current == latest:
        return f"{repo} already pinned at {latest[:12]}", True
    return f"{repo} pin moved {current[:12]} -> {latest[:12]}", False
