"""Fetch Data stage: the app's channel files from Hugging Face, at the pinned revision, merged into the raw log.

    python scripts/fetch_data.py            download (only when missing or the pinned revision changed) and merge
    python scripts/fetch_data.py --pin      write the dataset repo's latest commit into configs/hf_dataset.yaml, and say whether it moved
    python scripts/fetch_data.py --check    the same report, writes nothing; exit 1 when the pin is unset or behind the latest commit
    python scripts/fetch_data.py --force    download again, and replace a raw log this stage did not write

Skipped (the pipeline goes on with data/raw/ as it is) when no repo is configured or RESONANCE_RAW_LOGS names a
hand-made raw log. Needs the `hf` CLI (`pip install -U huggingface_hub`, `hf auth login` once).
"""

import argparse
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_DATA_DIR, HF_DATASET_CONFIG, RAW_LOGS
from hf_data import (FetchError, channel_files, download_command, is_current, load_config, merge_channels, pin_report, read_state,
                     with_revision, write_state)


def run_download(cmd):
    """Run `hf download`; exit 1 when it is missing or fails, so the pipeline stops."""
    print(f"[->] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd)
    except FileNotFoundError:
        print("[ERROR] the `hf` command was not found: pip install -U huggingface_hub, then `hf auth login`")
        sys.exit(1)
    if result.returncode != 0:
        print("[ERROR] hf download failed (not logged in? run `hf auth login`; wrong repo or revision?)")
        sys.exit(1)


def latest_revision(repo):
    """The commit SHA of the dataset repo's default branch."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] huggingface_hub is not installed: pip install -U huggingface_hub")
        sys.exit(1)
    return HfApi().dataset_info(repo).sha


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fetch the training data from Hugging Face.")
    parser.add_argument("--pin", action="store_true", help="pin the repo's latest commit in configs/hf_dataset.yaml; downloads nothing")
    parser.add_argument("--check", action="store_true", help="report whether the pin is the repo's latest commit; writes nothing, exit 1 if not")
    parser.add_argument("--force", action="store_true", help="download again, and replace a raw log this stage did not write")
    args = parser.parse_args(argv)

    if os.environ.get("RESONANCE_RAW_LOGS"):
        print(f"RESONANCE_RAW_LOGS is set ({os.environ['RESONANCE_RAW_LOGS']}): using that raw log, nothing fetched.")
        return
    cfg = load_config(HF_DATASET_CONFIG)
    if not cfg["repo"]:
        if args.pin or args.check:
            print(f"[ERROR] set `repo:` in {HF_DATASET_CONFIG} first")
            sys.exit(1)
        print(f"no HF dataset configured ({HF_DATASET_CONFIG}): using data/raw/ as it is.")
        return

    if args.pin or args.check:
        revision = latest_revision(cfg["repo"])
        message, current = pin_report(cfg["repo"], cfg["revision"], revision, written=args.pin)
        if args.check:
            print(message)
            if not current:
                sys.exit(1)
            return
        with open(HF_DATASET_CONFIG, encoding="utf-8") as f:
            text = f.read()
        try:
            text = with_revision(text, revision)
        except FetchError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        with open(HF_DATASET_CONFIG, "w", encoding="utf-8", newline="\n") as f:  # a file in git (eol=lf)
            f.write(text)
        print(f"{message} -> {HF_DATASET_CONFIG}")
        return

    state_file = os.path.join(HF_DATA_DIR, "fetch_state.json")
    state = read_state(state_file)
    try:
        cmd = download_command(cfg, HF_DATA_DIR)
    except FetchError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    if not args.force and is_current(state, cfg, RAW_LOGS):
        print(f"raw log is up to date: {cfg['repo']} @ {cfg['revision'][:12]}")
        return
    if not args.force and state is None and os.path.exists(RAW_LOGS):
        print(f"[ERROR] {RAW_LOGS} exists but this stage did not write it; --force replaces it "
              "(or set RESONANCE_RAW_LOGS to keep using it).")
        sys.exit(1)

    run_download(cmd)
    files = channel_files(HF_DATA_DIR, cfg["include"])
    if not files:
        print(f"[ERROR] the download holds no {cfg['include']} in {HF_DATA_DIR}")
        sys.exit(1)
    merged = merge_channels(files, RAW_LOGS)
    write_state(state_file, cfg, merged)
    print(f"{merged['rows']} rows from {len(files)} channel files ({cfg['repo']} @ {cfg['revision'][:12]}) -> {RAW_LOGS}")


if __name__ == "__main__":
    main()
