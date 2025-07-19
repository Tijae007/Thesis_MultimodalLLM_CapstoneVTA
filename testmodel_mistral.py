# Optional: Uncomment and run this once
# !pip install bert_score

from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template
from tqdm import tqdm
import torch
import evaluate
import re
from bert_score import score as bertscore

# Load the fine-tuned Mistral model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs_mistral_final/checkpoint-1000",  # <-- adjust path if needed
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Use ChatML template for Mistral
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    map_eos_token = True,
)

FastLanguageModel.for_inference(model)
model.eval()

# Define a test set (prompt + expected answer)
test_data = [
    {"prompt": "Who were the previous TAs for the capstone course?", "expected_output": "Julia Kotovich and Muhammad Khalid"},
    {"prompt": "Is teamwork mandatory in the capstone?", "expected_output": "Yes, teamwork is a mandatory part of the capstone project."},
    {"prompt": "What are the previous projects developed in Capstone Course?", "expected_output": "Smart Laundry System, XOMO+PS, Neighborhood Connect, and Emotional Harmony"},
    {"prompt": "What is the name of the main lecturer for Capstone Course?", "expected_output": "Prof. Manuel Oriol"},
]

# Function to clean predictions
def clean_text(text):
    text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|.*?\|>', '', text)
    return text.strip()

# Run inference
predictions = []
references = []

for sample in tqdm(test_data):
    messages = [{"from": "human", "value": sample["prompt"]}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs, max_new_tokens=128)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        cleaned = clean_text(decoded)
    
    predictions.append(cleaned)
    references.append(sample["expected_output"])

# Load BLEU and ROUGE
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Compute BLEU
bleu_result = bleu.compute(
    predictions=predictions,
    references=[[r] for r in references]
)

# Compute ROUGE
rouge_result = rouge.compute(
    predictions=predictions,
    references=references,
    use_stemmer=True
)

# Compute BERTScore
P_clean = [p.strip() for p in predictions]
R_clean = [r.strip() for r in references]
P_, R_, F1 = bertscore(P_clean, R_clean, lang="en", verbose=True)

# Display results
print("\n📊 Evaluation Metrics")
print("-" * 40)
print("BLEU:", round(bleu_result["bleu"], 4))
print("ROUGE-1:", round(rouge_result["rouge1"], 4))
print("ROUGE-2:", round(rouge_result["rouge2"], 4))
print("ROUGE-L:", round(rouge_result["rougeL"], 4))
print("ROUGE-Lsum:", round(rouge_result["rougeLsum"], 4))
print("\n🧠 BERTScore:")
print("Precision:", round(P_.mean().item(), 4))
print("Recall:   ", round(R_.mean().item(), 4))
print("F1 Score: ", round(F1.mean().item(), 4))

# Show predictions vs references
print("\n🔍 Sample Outputs")
print("-" * 40)
for i in range(len(test_data)):
    print(f"\nPrompt:    {test_data[i]['prompt']}")
    print(f"Expected:  {references[i]}")
    print(f"Predicted: {predictions[i]}")
