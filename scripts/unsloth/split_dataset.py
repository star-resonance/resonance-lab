import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import PROCESSED_LOGS, LORA_DATASET_DIR


def prepare_lora_dataset(input_path, output_dir, val_split=0.05, seed=42):
    """
    Converts raw chat logs into a LoRA-ready Instruction format.
    Includes data deduplication and a train/val split.
    """
    random.seed(seed)

    processed_data = []
    seen_inputs = set()

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Processing {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line_data = json.loads(line)
                instruction = line_data.get("instruction", "").strip()
                original = line_data.get("input", "").strip()
                translated = line_data.get("output", "").strip()

                # Basic Data Cleaning
                if not original or not translated:
                    continue

                # Deduplication: Chat logs often have repetitive recruitment spam.
                # Keeping some is fine for style, but 17k of the exact same line can hurt.
                if original in seen_inputs:
                    continue
                seen_inputs.add(original)

                # Construct the LoRA object
                processed_data.append({
                    "instruction": instruction,
                    "input": original,
                    "output": translated
                })
            except json.JSONDecodeError:
                continue

    # Shuffle and Split
    random.shuffle(processed_data)
    total_count = len(processed_data)
    val_count = max(1, int(total_count * val_split)) if total_count > 1 else 0

    train_data = processed_data[val_count:]
    val_data = processed_data[:val_count]

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to JSONL
    def save_jsonl(data, filename):
        path = os.path.join(output_dir, filename)
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        return path

    train_path = save_jsonl(train_data, 'train.jsonl')
    val_path = save_jsonl(val_data, 'val.jsonl')

    print("--- Dataset Preparation Complete ---")
    print(f"Total Unique Entries: {total_count}")
    print(f"Training Set: {len(train_data)} samples -> {train_path}")
    print(f"Validation Set: {len(val_data)} samples -> {val_path}")

# Run the script
if __name__ == "__main__":
    # CRITICAL: Check if input exists and exit with error if not
    if not os.path.exists(PROCESSED_LOGS):
        print(f"[ERROR] Input file not found: {PROCESSED_LOGS}")
        print("Check if preprocess.py ran correctly and generated this file.")
        sys.exit(1) # This tells run_pipeline.py to STOP
    prepare_lora_dataset(
        input_path=PROCESSED_LOGS,
        output_dir=LORA_DATASET_DIR
    )