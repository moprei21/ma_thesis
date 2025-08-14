import mauve
import random
import os
import sys

# Example human-written texts


folder = sys.argv[1] if len(sys.argv) > 1 else "synthetic_data/fewshot"
dialect = sys.argv[2] if len(sys.argv) > 2 else "zürich"

dialect_map = {
    "zürich": "__label__ch_zh",
    "bern": "__label__ch_be",
    "basel": "__label__ch_bs",
    "luzern": "__label__ch_lu"
}

with open("train_test_data/test.txt", "r", encoding="utf-8") as f:
    human_texts = [line.strip() for line in f if line.strip()]
    # split each text into label and content which are separated by whitespace
    human_texts = [
    text.split(maxsplit=1)[1]            # remove the label
    for text in human_texts
    if text.split(maxsplit=1)[0] == dialect_map[dialect]  # keep only POS
] 
    

folder_path = f"synthetic_data/{folder}"
file_path = f"{folder_path}/{dialect}.txt"
all_texts = []


with open(file_path, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]
    all_texts.extend(lines)


# Randomly shuffle to avoid ordering effects (optional but recommended)
random.shuffle(human_texts)
random.shuffle(all_texts)

# Run MAUVE
mauve_result = mauve.compute_mauve(
    p_text=human_texts[:200],
    q_text=all_texts,
    device_id=-1  # set to -1 to use CPU, or specify GPU device ID
)

# Print the MAUVE score
print(f"\nMAUVE score: {mauve_result.mauve:.4f} for {dialect} dialect in {folder} folder") 
