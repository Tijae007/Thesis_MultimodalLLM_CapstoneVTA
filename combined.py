import os
import json

def combine_lectures_and_qa(paragraph_folder, instruction_folder, output_file):
    combined_data = []

    # Step 1: Load all lecture .txt files
    for idx, fname in enumerate(sorted(os.listdir(paragraph_folder))):
        if fname.endswith(".txt"):
            filepath = os.path.join(paragraph_folder, fname)
            with open(filepath, "r", encoding="utf-8") as f:
                paragraph = f.read().strip()
                if paragraph:
                    combined_data.append({
                        "conversations": [
                            {"role": "user", "content": paragraph}
                        ]
                    })

    # Step 2: Load and combine all instruction-format .jsonl files
    instruction_count = 0
    for fname in sorted(os.listdir(instruction_folder)):
        if fname.endswith(".jsonl"):
            filepath = os.path.join(instruction_folder, fname)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        question = item.get("instruction", "")
                        input_part = item.get("input", "")
                        full_question = question if not input_part else f"{question}\n\n{input_part}"
                        output = item.get("output", "").strip()
                        if full_question.strip() and output:
                            combined_data.append({
                                "conversations": [
                                    {"role": "user", "content": full_question.strip()},
                                    {"role": "assistant", "content": output}
                                ]
                            })
                            instruction_count += 1
                    except json.JSONDecodeError:
                        print(f"⚠️ Skipped malformed line in {fname}")

    # Step 3: Save all into one file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2)

    print(f"✅ Combined {len(combined_data)} total entries.")
    print(f"📚 Lectures from folder: {paragraph_folder}")
    print(f"❓ Instructions from folder: {instruction_folder} ({instruction_count} entries)")
    print(f"💾 Output written to: {output_file}")

# Example usage
combine_lectures_and_qa(
    paragraph_folder=r"/home/coder/project/yunus/Video_Transcript",             # Folder of .txt lectures
    instruction_folder=r"/home/coder/project/yunus/SummaryOutput/Processed",  # Folder of .jsonl instruction files
    output_file="combined_conversations2.json"      # Output
)
