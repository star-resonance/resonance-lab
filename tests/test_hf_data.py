import json
import os

import pytest

import fetch_data
import hf_data
from manifest import file_sha256

CFG = {"repo": "someone/bp-chat", "revision": "a" * 40, "include": "dataset_*.jsonl"}


def _channel(tmp_path, name, rows):
    folder = tmp_path / "hf"
    folder.mkdir(exist_ok=True)
    (folder / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return str(folder)


def test_load_config_reads_repo_revision_and_include(tmp_path):
    path = tmp_path / "hf_dataset.yaml"
    path.write_text("repo: someone/bp-chat\nrevision: abc123\ninclude: 'dataset_*.jsonl'\n", encoding="utf-8")

    assert hf_data.load_config(str(path)) == {"repo": "someone/bp-chat", "revision": "abc123", "include": "dataset_*.jsonl"}


def test_an_unset_config_has_no_repo_and_a_default_include(tmp_path):
    path = tmp_path / "hf_dataset.yaml"
    path.write_text("repo: null\nrevision: null\n", encoding="utf-8")

    cfg = hf_data.load_config(str(path))

    assert cfg["repo"] is None and cfg["revision"] is None and cfg["include"] == "dataset_*.jsonl"


def test_the_download_command_pins_the_revision_and_only_fetches_the_channel_files(tmp_path):
    cmd = hf_data.download_command(CFG, str(tmp_path / "hf"))

    assert cmd[:3] == ["hf", "download", "someone/bp-chat"]
    assert cmd[cmd.index("--repo-type") + 1] == "dataset"
    assert cmd[cmd.index("--revision") + 1] == "a" * 40
    assert cmd[cmd.index("--include") + 1] == "dataset_*.jsonl"
    assert cmd[cmd.index("--local-dir") + 1] == str(tmp_path / "hf")


def test_download_command_refuses_a_config_without_a_pinned_revision():
    with pytest.raises(hf_data.FetchError, match="--pin"):
        hf_data.download_command({**CFG, "revision": None}, "x")


def test_channel_files_are_sorted_and_match_only_the_pattern(tmp_path):
    folder = _channel(tmp_path, "dataset_PARTY.jsonl", [{"original": "a"}])
    _channel(tmp_path, "dataset_GUILD.jsonl", [{"original": "b"}])
    _channel(tmp_path, "README.md", [])

    names = [os.path.basename(p) for p in hf_data.channel_files(folder, "dataset_*.jsonl")]

    assert names == ["dataset_GUILD.jsonl", "dataset_PARTY.jsonl"]


def test_merge_concatenates_the_channels_in_name_order_and_drops_blank_lines(tmp_path):
    folder = _channel(tmp_path, "dataset_PARTY.jsonl", [{"pid": 2, "original": "あ", "translated": "아"}])
    _channel(tmp_path, "dataset_GUILD.jsonl", [{"pid": 1, "original": "い", "translated": None}])
    (tmp_path / "hf" / "dataset_GUILD.jsonl").write_text(
        (tmp_path / "hf" / "dataset_GUILD.jsonl").read_text(encoding="utf-8") + "\n   \n", encoding="utf-8")
    out = tmp_path / "raw" / "raw.jsonl"

    summary = hf_data.merge_channels(hf_data.channel_files(folder, "dataset_*.jsonl"), str(out))

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["pid"] for r in rows] == [1, 2] and out.read_text(encoding="utf-8").endswith("\n")
    assert summary["rows"] == 2 and summary["raw_sha256"] == file_sha256(str(out))
    assert set(summary["files"]) == {"dataset_GUILD.jsonl", "dataset_PARTY.jsonl"}


def test_a_last_line_without_a_newline_does_not_glue_to_the_next_channel(tmp_path):
    folder = tmp_path / "hf"
    folder.mkdir()
    (folder / "dataset_A.jsonl").write_text('{"original": "a", "translated": "x"}', encoding="utf-8")  # no trailing newline
    (folder / "dataset_B.jsonl").write_text('{"original": "b", "translated": "y"}\n', encoding="utf-8")
    out = tmp_path / "raw.jsonl"

    hf_data.merge_channels(hf_data.channel_files(str(folder), "dataset_*.jsonl"), str(out))

    assert [json.loads(line)["original"] for line in out.read_text(encoding="utf-8").splitlines()] == ["a", "b"]


def test_a_state_matches_only_the_same_repo_revision_and_an_untouched_raw_file(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"original": "a"}\n', encoding="utf-8")
    state = {"repo": CFG["repo"], "revision": CFG["revision"], "raw_sha256": file_sha256(str(raw)), "merge_format": hf_data.MERGE_FORMAT}

    assert hf_data.is_current(state, CFG, str(raw))
    assert not hf_data.is_current(None, CFG, str(raw))
    assert not hf_data.is_current({**state, "revision": "b" * 40}, CFG, str(raw))
    assert not hf_data.is_current({**state, "repo": "other/repo"}, CFG, str(raw))
    raw.write_text("edited\n", encoding="utf-8")
    assert not hf_data.is_current(state, CFG, str(raw))
    assert not hf_data.is_current(state, CFG, str(tmp_path / "missing.jsonl"))


def test_state_round_trips_and_a_missing_or_broken_one_reads_as_none(tmp_path):
    path = str(tmp_path / "fetch_state.json")
    assert hf_data.read_state(path) is None

    hf_data.write_state(path, CFG, {"rows": 3, "files": {"dataset_A.jsonl": "x"}, "raw_sha256": "y"})
    state = hf_data.read_state(path)
    assert state["repo"] == CFG["repo"] and state["revision"] == CFG["revision"] and state["rows"] == 3

    with open(path, "w", encoding="utf-8") as f:
        f.write("{broken")
    assert hf_data.read_state(path) is None


def test_with_revision_replaces_only_the_revision_line_and_keeps_the_comments():
    text = "# the dataset\nrepo: someone/bp-chat\nrevision: null  # pinned by --pin\ninclude: 'dataset_*.jsonl'\n"

    out = hf_data.with_revision(text, "c" * 40)

    assert "revision: " + "c" * 40 in out and "# the dataset" in out and "repo: someone/bp-chat" in out
    assert out.count("revision:") == 1


def test_with_revision_needs_a_revision_line():
    with pytest.raises(hf_data.FetchError, match="revision"):
        hf_data.with_revision("repo: x\n", "c" * 40)


# --- the Fetch Data stage -----------------------------------------------------------------------------------


@pytest.fixture
def stage(tmp_path, monkeypatch):
    """fetch_data wired to a temp tree; `hf download` is faked: it writes the channel files into --local-dir."""
    cfg = tmp_path / "hf_dataset.yaml"
    cfg.write_text(f"repo: {CFG['repo']}\nrevision: {CFG['revision']}\ninclude: 'dataset_*.jsonl'\n", encoding="utf-8")
    raw = tmp_path / "raw" / "raw_translated_logs.jsonl"
    monkeypatch.setattr(fetch_data, "HF_DATASET_CONFIG", str(cfg))
    monkeypatch.setattr(fetch_data, "HF_DATA_DIR", str(tmp_path / "hf"))
    monkeypatch.setattr(fetch_data, "RAW_LOGS", str(raw))
    monkeypatch.delenv("RESONANCE_RAW_LOGS", raising=False)
    calls = []

    def fake_download(cmd):
        calls.append(cmd)
        _channel(tmp_path, "dataset_GUILD.jsonl", [{"pid": 1, "original": "あ", "translated": "아"}])

    monkeypatch.setattr(fetch_data, "run_download", fake_download)
    return type("Stage", (), {"cfg": cfg, "raw": raw, "calls": calls, "tmp": tmp_path})


def test_the_first_run_downloads_and_writes_the_raw_log(stage, capsys):
    fetch_data.main([])

    assert len(stage.calls) == 1 and stage.raw.exists()
    assert json.loads(stage.raw.read_text(encoding="utf-8"))["original"] == "あ"
    assert "1 rows" in capsys.readouterr().out


def test_a_second_run_with_the_same_revision_downloads_nothing(stage, capsys):
    fetch_data.main([])
    fetch_data.main([])

    assert len(stage.calls) == 1
    assert "up to date" in capsys.readouterr().out


def test_a_new_revision_downloads_again(stage):
    fetch_data.main([])
    stage.cfg.write_text(stage.cfg.read_text(encoding="utf-8").replace("a" * 40, "b" * 40), encoding="utf-8")

    fetch_data.main([])

    assert len(stage.calls) == 2 and stage.calls[1][stage.calls[1].index("--revision") + 1] == "b" * 40


def test_force_downloads_again(stage):
    fetch_data.main([])
    fetch_data.main(["--force"])

    assert len(stage.calls) == 2


def test_a_hand_edited_raw_log_is_not_overwritten_silently(stage):
    stage.raw.parent.mkdir(parents=True)
    stage.raw.write_text("hand made\n", encoding="utf-8")  # there is no fetch state: the stage did not write it

    with pytest.raises(SystemExit) as exc:
        fetch_data.main([])

    assert exc.value.code == 1 and stage.raw.read_text(encoding="utf-8") == "hand made\n" and not stage.calls


def test_force_overwrites_a_raw_log_the_stage_did_not_write(stage):
    stage.raw.parent.mkdir(parents=True)
    stage.raw.write_text("hand made\n", encoding="utf-8")

    fetch_data.main(["--force"])

    assert json.loads(stage.raw.read_text(encoding="utf-8"))["original"] == "あ"


def test_an_unconfigured_repo_skips_the_stage_so_local_files_keep_working(stage, capsys):
    stage.cfg.write_text("repo: null\nrevision: null\n", encoding="utf-8")

    fetch_data.main([])

    assert not stage.calls and "no HF dataset" in capsys.readouterr().out


def test_a_repo_without_a_pinned_revision_stops_the_pipeline(stage, capsys):
    stage.cfg.write_text("repo: someone/bp-chat\nrevision: null\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        fetch_data.main([])

    assert exc.value.code == 1 and "--pin" in capsys.readouterr().out


def test_a_raw_log_override_in_the_environment_is_never_replaced(stage, monkeypatch, capsys):
    monkeypatch.setenv("RESONANCE_RAW_LOGS", "data/raw/hand_curated.jsonl")

    fetch_data.main([])

    assert not stage.calls and "RESONANCE_RAW_LOGS" in capsys.readouterr().out


def test_no_channel_files_in_the_download_stops_the_pipeline(stage, monkeypatch, capsys):
    monkeypatch.setattr(fetch_data, "run_download", lambda cmd: None)

    with pytest.raises(SystemExit) as exc:
        fetch_data.main([])

    assert exc.value.code == 1 and "dataset_*.jsonl" in capsys.readouterr().out
    assert not stage.raw.exists()


def test_pin_writes_the_latest_revision_into_the_config(stage, monkeypatch, capsys):
    stage.cfg.write_text("# pinned by --pin\nrepo: someone/bp-chat\nrevision: null\n", encoding="utf-8")
    monkeypatch.setattr(fetch_data, "latest_revision", lambda repo: "d" * 40)

    fetch_data.main(["--pin"])

    text = stage.cfg.read_text(encoding="utf-8")
    assert "revision: " + "d" * 40 in text and "# pinned by --pin" in text
    assert not stage.calls  # pinning does not download


def test_pin_reports_whether_the_pin_moved(stage, monkeypatch, capsys):
    monkeypatch.setattr(fetch_data, "latest_revision", lambda repo: "d" * 40)

    fetch_data.main(["--pin"])  # the config holds "a" * 40
    moved = capsys.readouterr().out
    fetch_data.main(["--pin"])  # now it holds "d" * 40
    same = capsys.readouterr().out

    assert "pin moved" in moved and "a" * 12 in moved and "d" * 12 in moved
    assert "already pinned" in same and "d" * 12 in same


def test_pin_reports_a_first_pin(stage, monkeypatch, capsys):
    stage.cfg.write_text("repo: someone/bp-chat\nrevision: null\n", encoding="utf-8")
    monkeypatch.setattr(fetch_data, "latest_revision", lambda repo: "d" * 40)

    fetch_data.main(["--pin"])

    assert "was unset" in capsys.readouterr().out
    assert "revision: " + "d" * 40 in stage.cfg.read_text(encoding="utf-8")


def test_check_writes_nothing_and_exits_0_when_the_pin_is_current(stage, monkeypatch, capsys):
    monkeypatch.setattr(fetch_data, "latest_revision", lambda repo: CFG["revision"])
    before = stage.cfg.read_text(encoding="utf-8")

    fetch_data.main(["--check"])

    assert "already pinned" in capsys.readouterr().out
    assert stage.cfg.read_text(encoding="utf-8") == before and not stage.calls


def test_check_exits_1_and_writes_nothing_when_the_pin_moved(stage, monkeypatch, capsys):
    monkeypatch.setattr(fetch_data, "latest_revision", lambda repo: "d" * 40)
    before = stage.cfg.read_text(encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        fetch_data.main(["--check"])

    assert exc.value.code == 1 and "pin moved" in capsys.readouterr().out
    assert stage.cfg.read_text(encoding="utf-8") == before


def test_check_exits_1_when_nothing_is_pinned(stage, monkeypatch, capsys):
    stage.cfg.write_text("repo: someone/bp-chat\nrevision: null\n", encoding="utf-8")
    monkeypatch.setattr(fetch_data, "latest_revision", lambda repo: "d" * 40)

    with pytest.raises(SystemExit) as exc:
        fetch_data.main(["--check"])

    assert exc.value.code == 1 and "not pinned" in capsys.readouterr().out


def test_pin_needs_a_repo(stage):
    stage.cfg.write_text("repo: null\nrevision: null\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        fetch_data.main(["--pin"])
    assert exc.value.code == 1


def test_a_missing_hf_cli_stops_the_pipeline_with_the_fix(monkeypatch, capsys):
    def no_hf(cmd, **kwargs):
        raise FileNotFoundError("hf")

    monkeypatch.setattr(fetch_data.subprocess, "run", no_hf)

    with pytest.raises(SystemExit) as exc:
        fetch_data.run_download(["hf", "download", "x"])

    assert exc.value.code == 1 and "huggingface_hub" in capsys.readouterr().out


def test_a_failed_download_stops_the_pipeline(monkeypatch):
    class Result:
        returncode = 1

    monkeypatch.setattr(fetch_data.subprocess, "run", lambda cmd, **kwargs: Result())

    with pytest.raises(SystemExit) as exc:
        fetch_data.run_download(["hf", "download", "x"])
    assert exc.value.code == 1


def _merged_rows(tmp_path, files):
    out = tmp_path / "raw.jsonl"
    hf_data.merge_channels(files, str(out))
    return out.read_text(encoding="utf-8").splitlines()


def test_each_row_gets_the_channel_its_file_is_named_after(tmp_path):
    folder = _channel(tmp_path, "dataset_PARTY.jsonl", [{"pid": 2, "original": "2人募集", "translated": None}])
    _channel(tmp_path, "dataset_GUILD.jsonl", [{"pid": 1, "original": "おはよう", "translated": "안녕"}])

    lines = _merged_rows(tmp_path, hf_data.channel_files(folder, "dataset_*.jsonl"))

    assert [(json.loads(l)["pid"], json.loads(l)["channel"]) for l in lines] == [(1, "GUILD"), (2, "PARTY")]
    assert "2人募集" in lines[1]  # not escaped: the raw log stays readable, as the app wrote it
    assert json.loads(lines[1]) == {"pid": 2, "original": "2人募集", "translated": None, "channel": "PARTY"}


def test_a_channel_the_row_already_has_is_kept(tmp_path):
    folder = _channel(tmp_path, "dataset_PARTY.jsonl", [{"original": "a", "channel": "WORLD"}])

    (line,) = _merged_rows(tmp_path, hf_data.channel_files(folder, "dataset_*.jsonl"))

    assert json.loads(line)["channel"] == "WORLD"


def test_a_file_not_named_after_a_channel_adds_no_field(tmp_path):
    folder = _channel(tmp_path, "bp-training-dataset-2026.jsonl", [{"original": "a", "translated": "x"}])

    (line,) = _merged_rows(tmp_path, hf_data.channel_files(folder, "bp-training-dataset-*.jsonl"))

    assert json.loads(line) == {"original": "a", "translated": "x"}


def test_a_damaged_line_is_kept_as_it_is_for_the_validation_gate(tmp_path):
    folder = tmp_path / "hf"
    folder.mkdir()
    (folder / "dataset_PARTY.jsonl").write_text('{"original": "a"\n["not", "a", "row"]\n', encoding="utf-8")

    lines = _merged_rows(tmp_path, hf_data.channel_files(str(folder), "dataset_*.jsonl"))

    assert lines == ['{"original": "a"', '["not", "a", "row"]']


def test_the_summary_names_the_merge_format_and_an_older_state_is_not_current(tmp_path):
    folder = _channel(tmp_path, "dataset_PARTY.jsonl", [{"original": "a"}])
    out = tmp_path / "raw.jsonl"
    summary = hf_data.merge_channels(hf_data.channel_files(folder, "dataset_*.jsonl"), str(out))
    state = {"repo": CFG["repo"], "revision": CFG["revision"], **summary}

    assert hf_data.is_current(state, CFG, str(out))
    del state["merge_format"]  # a raw log merged before rows carried their channel
    assert not hf_data.is_current(state, CFG, str(out))


def test_the_fetch_state_and_a_pinned_config_end_lines_with_lf_on_every_os(stage, tmp_path, monkeypatch):
    path = tmp_path / "fetch_state.json"
    hf_data.write_state(str(path), CFG, {"rows": 3, "files": {"dataset_A.jsonl": "x"}, "raw_sha256": "y"})
    monkeypatch.setattr(fetch_data, "latest_revision", lambda repo: "d" * 40)

    fetch_data.main(["--pin"])  # rewrites the config, a file in git (eol=lf)

    for data in (path.read_bytes(), stage.cfg.read_bytes()):
        assert b"\n" in data and b"\r" not in data
