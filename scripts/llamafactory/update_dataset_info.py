import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import LF_DATASET_INFO_PATH
from lf_tools import check_training_data, dataset_info_for_processed_logs, profile_from_args
from manifest import ManifestError


def main(argv=None):
    profile, _ = profile_from_args(argv, "Point LLaMA-Factory at the processed training file.")
    try:
        check_training_data(profile)
    except ManifestError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    info = dataset_info_for_processed_logs()
    with open(LF_DATASET_INFO_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"dataset_info.json written -> {LF_DATASET_INFO_PATH}")
    for name, entry in info.items():
        print(f"  {name}: {entry['file_name']}")


if __name__ == "__main__":
    main()
