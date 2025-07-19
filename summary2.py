import os
import json
import spacy
import pytextrank
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --- Configuration ---
INPUT_DIR = r"/home/coder/project/Video_Transcript/chunks"   # Folder with raw .txt transcripts
OUTPUT_DIR = r"/home/coder/project/SummaryOutput"  # Folder to save processed instruction data

# --- Ensure folders exist ---
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load NLP + Embedding Models ---
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_500_000  # Updated to handle long transcripts
nlp.add_pipe("textrank")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight, fast model

# --- Utility Functions ---
def get_semantic_passage(phrase, text, top_k=2):
    """Find semantically similar passage for the given phrase."""
    sentences = [s.strip() for s in text.split(".") if len(s.strip().split()) > 5]
    if not sentences:
        return ""
    phrase_vec = embedder.encode([phrase])
    sent_vecs = embedder.encode(sentences)
    sims = cosine_similarity(phrase_vec, sent_vecs)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return ". ".join(sentences[i] for i in top_indices)

def clean_phrase(phrase):
    return ' '.join(phrase.strip().lower().replace("\n", " ").split())

def is_valid_phrase(phrase):
    phrase = clean_phrase(phrase)
    return len(phrase.split()) >= 2 and phrase.isascii() and not phrase.startswith("you")

# --- Processing ---
def process_transcript(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text)
    seen = set()
    examples = []

    for phrase in doc._.phrases:
        keyword = clean_phrase(phrase.text)
        if keyword in seen or not is_valid_phrase(keyword):
            continue
        seen.add(keyword)

        instruction = f"What is '{keyword}' about?"
        output = get_semantic_passage(keyword, text)

        if len(output.strip().split()) < 10:
            continue  # skip too short answers

        print(f"✅ Valid phrase: {keyword} → {len(output.strip().split())} words in match")

        examples.append({
            "instruction": instruction,
            "input": "",
            "output": output.strip()
        })

    # Fallback if nothing found
    if not examples:
        fallback = text.strip().split("\n\n")[0]
        if len(fallback.split()) > 10:
            examples.append({
                "instruction": "Summarize the main idea of this segment.",
                "input": "",
                "output": fallback
            })

    return examples

# --- Run over folder ---
all_data = []
input_files = sorted(Path(INPUT_DIR).glob("*.txt"))

for file in tqdm(input_files, desc="Processing transcripts"):
    try:
        data = process_transcript(file)
        all_data.extend(data)

        print(f"📄 Processed {file.name}: {len(data)} examples")

        # Save per file
        output_file = Path(OUTPUT_DIR) / f"{file.stem}_qa.jsonl"
        with open(output_file, "w", encoding="utf-8") as out:
            for item in data:
                json.dump(item, out)
                out.write("\n")
    except Exception as e:
        print(f"⚠️ Error processing {file.name}: {e}")

# Optional: Save all combined
with open(Path(OUTPUT_DIR) / "all_instructions.jsonl", "w", encoding="utf-8") as out:
    for item in all_data:
        json.dump(item, out)
        out.write("\n")

print(f"✅ Done! {len(all_data)} instructions written to {OUTPUT_DIR}")
