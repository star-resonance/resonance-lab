import hashlib
import json

import pytest

import manifest
from manifest import ManifestError


def _data(tmp_path, text='{"original": "a", "translated": "b"}\n'):
    path = tmp_path / "lora_train_data.jsonl"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _write(tmp_path, val_text='{"original": "v", "translated": "w"}\n', **overrides):
    data = _data(tmp_path)
    if val_text is not None:
        with open(manifest.val_path(data), "w", encoding="utf-8") as f:
            f.write(val_text)
    args = dict(fmt="pair", style="hy", reverse=False, raw_path=data, counts={"total": 1, "passed": 1},
                eval_set=None, eval_lines=0, val_fraction=0.05)
    args.update(overrides)
    return data, manifest.write_manifest(data, **args)


def test_file_sha256_is_the_hex_digest_of_the_bytes(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"abc")

    assert manifest.file_sha256(str(path)) == hashlib.sha256(b"abc").hexdigest()


def test_the_manifest_lives_next_to_the_data_file():
    assert manifest.manifest_path("/x/processed/lora_train_data.jsonl") == "/x/processed/lora_train_data.meta.json"


def test_write_records_how_the_file_was_made(tmp_path):
    data, path = _write(tmp_path, reverse=True, eval_set=str(tmp_path / "eval.jsonl"), eval_lines=51)

    meta = json.loads(open(path, encoding="utf-8").read())
    assert path == manifest.manifest_path(data)
    assert meta["format"] == "pair" and meta["style"] == "hy" and meta["reverse"] is True
    assert meta["data_sha256"] == manifest.file_sha256(data)
    assert meta["raw_sha256"] == manifest.file_sha256(data)  # raw_path was the same file in this test
    assert meta["counts"] == {"total": 1, "passed": 1}
    assert meta["eval_lines_excluded_from"] == 51 and meta["eval_set"] == "eval.jsonl"


def test_a_missing_eval_set_is_recorded_as_none(tmp_path):
    _, path = _write(tmp_path)

    meta = manifest.read_manifest(path)
    assert meta["eval_set"] is None and meta["eval_lines_excluded_from"] == 0


def test_the_matching_data_passes(tmp_path):
    data, _ = _write(tmp_path, style="hy")

    meta = manifest.require_for_style(data, "hy")

    assert meta["style"] == "hy"


def test_a_missing_manifest_says_to_run_preprocessing(tmp_path):
    data = _data(tmp_path)

    with pytest.raises(ManifestError, match="preprocess"):
        manifest.require_for_style(data, "hy")


def test_a_style_the_profile_was_not_trained_with_is_refused(tmp_path):
    data, _ = _write(tmp_path, style="translategemma")

    with pytest.raises(ManifestError) as exc:
        manifest.require_for_style(data, "hy")
    assert "translategemma" in str(exc.value) and "hy" in str(exc.value)


def test_raw_lines_without_an_instruction_are_refused_for_a_styled_profile(tmp_path):
    data, _ = _write(tmp_path, style=None)

    with pytest.raises(ManifestError, match="raw"):
        manifest.require_for_style(data, "hy")


def test_the_instruction_format_is_not_what_llamafactory_reads(tmp_path):
    data, _ = _write(tmp_path, fmt="instruction", style=None)

    with pytest.raises(ManifestError, match="pair"):
        manifest.require_for_style(data, "hy")


def test_data_changed_after_preprocessing_is_refused(tmp_path):
    data, _ = _write(tmp_path)
    with open(data, "a", encoding="utf-8") as f:
        f.write('{"original": "c", "translated": "d"}\n')

    with pytest.raises(ManifestError, match="changed"):
        manifest.require_for_style(data, "hy")


def test_an_unreadable_manifest_is_a_manifest_error(tmp_path):
    data = _data(tmp_path)
    with open(manifest.manifest_path(data), "w", encoding="utf-8") as f:
        f.write("{not json")

    with pytest.raises(ManifestError, match="preprocess"):
        manifest.require_for_style(data, "hy")


def test_the_validation_file_lives_next_to_the_data_file():
    assert manifest.val_path("/x/processed/lora_train_data.jsonl") == "/x/processed/lora_train_data.val.jsonl"


def test_write_records_the_validation_file(tmp_path):
    data, path = _write(tmp_path)

    meta = manifest.read_manifest(path)
    assert meta["val_fraction"] == 0.05 and meta["val_rows"] == 1
    assert meta["val_sha256"] == manifest.file_sha256(manifest.val_path(data))


def test_without_a_validation_file_none_is_recorded(tmp_path):
    _, path = _write(tmp_path, val_text=None, val_fraction=0.0)

    meta = manifest.read_manifest(path)
    assert meta["val_rows"] == 0 and meta["val_sha256"] is None


def test_an_empty_validation_file_is_refused_because_training_needs_eval_data(tmp_path):
    data, _ = _write(tmp_path, val_text="")

    with pytest.raises(ManifestError, match="validation"):
        manifest.require_for_style(data, "hy")


def test_a_missing_validation_file_is_refused(tmp_path):
    data, _ = _write(tmp_path, val_text=None, val_fraction=0.0)

    with pytest.raises(ManifestError, match="validation"):
        manifest.require_for_style(data, "hy")


def test_a_validation_file_changed_after_preprocessing_is_refused(tmp_path):
    data, _ = _write(tmp_path)
    with open(manifest.val_path(data), "a", encoding="utf-8") as f:
        f.write('{"original": "x", "translated": "y"}\n')

    with pytest.raises(ManifestError, match="changed"):
        manifest.require_for_style(data, "hy")


def test_a_recipe_block_is_recorded_only_when_there_is_one(tmp_path):
    data = tmp_path / "out.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    raw = tmp_path / "raw.jsonl"
    raw.write_text("{}\n", encoding="utf-8")
    kwargs = dict(fmt="pair", style=None, reverse=False, raw_path=str(raw), counts={}, eval_set=None, eval_lines=0)

    plain = manifest.read_manifest(manifest.write_manifest(str(data), **kwargs))
    with_recipe = manifest.read_manifest(manifest.write_manifest(str(data), recipe={"name": "balanced", "sha256": "abc"}, **kwargs))

    assert "recipe" not in plain
    assert with_recipe["recipe"] == {"name": "balanced", "sha256": "abc"}


def test_the_manifest_ends_lines_with_lf_on_every_os(tmp_path):
    _, path = _write(tmp_path)
    data = open(path, "rb").read()
    assert b"\n" in data and b"\r" not in data
