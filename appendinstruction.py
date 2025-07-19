import os
import json

def merge_jsonl_to_json(input_folder, output_file):
    all_data = []

    for fname in sorted(os.listdir(input_folder)):
        if fname.endswith(".jsonl"):
            filepath = os.path.join(input_folder, fname)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError:
                        print(f"⚠️ Skipping malformed line in '{fname}'.")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"✅ Merged {len(all_data)} items into '{output_file}'.")

# Example usage
merge_jsonl_to_json(
    input_folder="/home/coder/project/yunus/SummaryOutput/Processed",
    output_file="merged_instructions.json"
)
