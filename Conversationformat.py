import os
import json

def convert_folder_to_conversational(input_folder, output_file):
    all_conversations = []
    file_count = 0
    entry_count = 0

    for filename in os.listdir(input_folder):
        if filename.endswith(".jsonl"):
            file_count += 1
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        instruction = item.get("instruction", "")
                        input_text = item.get("input", "").strip()
                        user_msg = instruction if not input_text else f"{instruction}\n\n{input_text}"

                        convo = {
                            "conversations": [
                                {"role": "user", "content": user_msg.strip()},
                                {"role": "assistant", "content": item["output"].strip()}
                            ]
                        }
                        all_conversations.append(convo)
                        entry_count += 1
                    except Exception as e:
                        print(f"⚠️ Skipping malformed line in {filename}: {e}")

    # Save all combined conversations
    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(all_conversations, out_file, indent=2)

    print(f"✅ Converted {entry_count} entries from {file_count} files.")
    print(f"📄 Output saved to: {output_file}")

# Example usage:
convert_folder_to_conversational(r"/home/coder/project/SummaryOutput/Processed", "converted_conversations.json")
