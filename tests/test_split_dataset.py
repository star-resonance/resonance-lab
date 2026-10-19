import split_dataset
from conftest import read_jsonl


def row(i, original=None, output=None):
    return {
        "instruction": "inst",
        "input": original if original is not None else f"jp{i}",
        "output": output or f"ko{i}",
    }


def test_splits_deduplicated_rows_into_train_and_val(write_jsonl, tmp_path):
    rows = [row(i) for i in range(40)] + [row(0, output="dup")]
    src = write_jsonl("processed.jsonl", rows)

    split_dataset.prepare_lora_dataset(src, str(tmp_path / "out"), val_split=0.05, seed=42)

    train = read_jsonl(tmp_path / "out" / "train.jsonl")
    val = read_jsonl(tmp_path / "out" / "val.jsonl")
    assert len(val) == 2  # int(40 * 0.05)
    assert len(train) == 38
    inputs = [r["input"] for r in train + val]
    assert sorted(inputs) == sorted(f"jp{i}" for i in range(40))  # first occurrence kept, duplicate dropped
    assert {"instruction": "inst", "input": "jp0", "output": "ko0"} in train + val


def test_same_seed_gives_same_split(write_jsonl, tmp_path):
    src = write_jsonl("processed.jsonl", [row(i) for i in range(20)])

    split_dataset.prepare_lora_dataset(src, str(tmp_path / "a"), seed=7)
    split_dataset.prepare_lora_dataset(src, str(tmp_path / "b"), seed=7)

    assert read_jsonl(tmp_path / "a" / "val.jsonl") == read_jsonl(tmp_path / "b" / "val.jsonl")


def test_drops_empty_and_malformed_rows(write_jsonl, tmp_path):
    src = write_jsonl("processed.jsonl", ["{broken", row(1, original="  "), row(2, output=" "), row(3)])

    split_dataset.prepare_lora_dataset(src, str(tmp_path / "out"))

    # A single row is never split off into val.
    assert read_jsonl(tmp_path / "out" / "train.jsonl") == [row(3)]
    assert read_jsonl(tmp_path / "out" / "val.jsonl") == []


def test_train_and_val_end_lines_with_lf_on_every_os(write_jsonl, tmp_path):
    src = write_jsonl("processed.jsonl", [row(i) for i in range(40)])

    split_dataset.prepare_lora_dataset(src, str(tmp_path / "out"), val_split=0.05, seed=42)

    for name in ("train.jsonl", "val.jsonl"):
        data = (tmp_path / "out" / name).read_bytes()
        assert b"\n" in data and b"\r" not in data, name
