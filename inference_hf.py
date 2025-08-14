from transformers import pipeline, AutoTokenizer
import torch
from tqdm import tqdm

model_id_local = "morit/gemma-3-4b-it-swissdial_dialect_base_5"

SYSTEM_PROMPT = """You are a person speaking different Swiss German dialects in many different varieties."""

PROMPT_BASE_ZÜRICH = """
Only output raw training data — do not explain, translate, or include any meta-commentary. Do not switch languages. Maintain a consistent dialect.

Generate one sentence in Swiss German using the Zürich dialect 
Each example should be realistic, diverse in phrasing, and aligned with how real people talk in daily life.

Follow these steps:
1. Generate a sentence in Swiss German using the Zürich dialect 
2. Check if the sentence is indeed in the #dialect_insert#; if not, generate a new one.
3. Compare the sentence with other dialects, and if it's too similar, generate a new one.

Do these steps one after the other.

Only output raw training data — do not explain, translate, or include any meta-commentary. Do not switch languages. Maintain a consistent dialect.

Output format:
1 sentence plain text on one line separated by a newline, no special formatting or JSON.

Begin now.
"""

# Load pipeline and tokenizer
pipe = pipeline(
    "text-generation",
    model=model_id_local,
    device="cpu",  # change to "cuda" if using GPU
)

tokenizer = AutoTokenizer.from_pretrained(model_id_local)

# Prepare system and user messages
chat_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": PROMPT_BASE_ZÜRICH},
]

# Apply chat template
formatted_prompt = tokenizer.apply_chat_template(
    chat_history,
    tokenize=False,
    add_generation_prompt=True
).strip()

# Collect 100 outputs
results = []
print("Generating 100 Zurich dialect sentences...\n")

for _ in tqdm(range(100)):
    output = pipe(
        formatted_prompt,
        max_new_tokens=64,
        do_sample=True,
        temperature=1.0,
        top_k=64,
        top_p=0.95,
        return_full_text=False,
    )
    sentence = output[0]["generated_text"].strip()
    results.append(sentence)

# Print all results
print("\nGenerated Sentences:\n")
for i, sentence in enumerate(results, 1):
    print(f"{i}. {sentence}")

with open("zurich_dialect_sentences.txt", "w", encoding="utf-8") as f:
    for sentence in results:
        f.write(sentence + "\n")
