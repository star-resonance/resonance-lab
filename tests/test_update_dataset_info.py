import update_dataset_info


def test_dataset_info_ends_lines_with_lf_on_every_os(tmp_path, monkeypatch):
    path = tmp_path / "dataset_info.json"
    monkeypatch.setattr(update_dataset_info, "LF_DATASET_INFO_PATH", str(path))
    monkeypatch.setattr(update_dataset_info, "profile_from_args", lambda argv, description: (None, []))
    monkeypatch.setattr(update_dataset_info, "check_training_data", lambda profile: None)
    monkeypatch.setattr(update_dataset_info, "dataset_info_for_processed_logs",
                        lambda: {"bp_translation": {"file_name": "lora_train_data.jsonl"}})

    update_dataset_info.main([])

    data = path.read_bytes()
    assert b"\n" in data and b"\r" not in data
