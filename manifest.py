"""The sidecar that says how `lora_train_data.jsonl` was made. Pure, no torch.

preprocess.py writes `lora_train_data.meta.json` next to its output. Every profile reads the same training file, so
without it a model can silently be trained on the wrong prompt (Hy trained on TranslateGemma's instruction), or on a
file edited after it was cleaned. update_dataset_info.py and train.py check it first; the same record is what the
experiment tracker logs as the run's data fingerprint.
"""

import hashlib
import json
import os
from datetime import datetime, timezone


class ManifestError(Exception):
    """The training file is missing its manifest, was changed since, or was made for another prompt style."""


def manifest_path(data_path):
    return os.path.splitext(data_path)[0] + ".meta.json"


def val_path(data_path):
    """The validation rows preprocess.py splits off `data_path` (lora_train_data.jsonl -> lora_train_data.val.jsonl)."""
    return os.path.splitext(data_path)[0] + ".val.jsonl"


def _count_rows(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(data_path, fmt, style, reverse, raw_path, counts, eval_set, eval_lines, val_fraction=0.0, recipe=None):
    """Record how `data_path` was made; returns the manifest's path.

    `eval_set` is the eval file whose lines were kept out (None when nothing was excluded), `eval_lines` how many.
    The validation file next to `data_path`, when there is one, is recorded too (`val_fraction` is what was asked for).
    `recipe`, when the lines were chosen by a dataset recipe, is the block that says which (name, hashes, what each
    category offered and gave); without a recipe the manifest has no such key.
    """
    val = val_path(data_path)
    has_val = os.path.exists(val)
    meta = {
        "format": fmt,
        "style": style,
        "reverse": bool(reverse),
        "raw_sha256": file_sha256(raw_path),
        "data_sha256": file_sha256(data_path),
        "counts": counts,
        "val_fraction": val_fraction,
        "val_rows": _count_rows(val) if has_val else 0,
        "val_sha256": file_sha256(val) if has_val else None,
        "eval_set": os.path.basename(eval_set) if eval_set else None,
        "eval_lines_excluded_from": eval_lines if eval_set else 0,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if recipe:
        meta["recipe"] = recipe
    path = manifest_path(data_path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


def remove_manifest(data_path):
    """Delete the manifest and the validation file that belong to an older version of the data file (preprocess is about
    to rewrite them; a failed run must leave neither behind)."""
    for path in (manifest_path(data_path), val_path(data_path)):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def require_for_style(data_path, style):
    """The manifest of `data_path`, after checking it was made for the prompt `style` and has not changed since."""
    path = manifest_path(data_path)
    try:
        meta = read_manifest(path)
    except (FileNotFoundError, ValueError):
        raise ManifestError(f"{path} is missing or unreadable: run the Preprocessing stage (preprocess.py) first") from None
    if meta.get("format") != "pair":
        raise ManifestError(f"{data_path} was made with --format {meta.get('format')}; LLaMA-Factory reads the pair format")
    made_for = meta.get("style")
    if made_for != style:
        shown = made_for if made_for else "none (the raw line, no instruction)"
        raise ManifestError(f"{data_path} was made with prompt style {shown}, this model trains on style {style}")
    try:
        unchanged = meta.get("data_sha256") == file_sha256(data_path)
    except FileNotFoundError:
        raise ManifestError(f"{data_path} not found: run the Preprocessing stage (preprocess.py) first") from None
    if not unchanged:
        raise ManifestError(f"{data_path} has changed since preprocess.py wrote it; run the Preprocessing stage again")
    _require_validation(data_path, meta)
    return meta


def _require_validation(data_path, meta):
    """LLaMA-Factory validates on the validation file (`eval_dataset`): it must exist, hold rows and be unchanged."""
    val = val_path(data_path)
    if not meta.get("val_rows"):
        raise ManifestError(
            f"{data_path} has no validation rows ({val} is empty or was never written); "
            "run the Preprocessing stage with --format pair (a --val-fraction above 0)"
        )
    try:
        unchanged = meta.get("val_sha256") == file_sha256(val)
    except FileNotFoundError:
        raise ManifestError(f"validation file {val} not found: run the Preprocessing stage (preprocess.py) first") from None
    if not unchanged:
        raise ManifestError(f"validation file {val} has changed since preprocess.py wrote it; run the Preprocessing stage again")
