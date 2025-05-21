import json
import random
def create_data():
# Load your dataset
    with open("data/swissdial/sentences_ch_de_numerics.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    dialect_keys = ["ch_sg", "ch_be", "ch_zh", "ch_vs", "ch_bs", "ch_ag", "ch_lu"]

    lines = []
    for entry in data:
        for key in dialect_keys:
            if key in entry and entry[key].strip():
                text = entry[key].replace("\n", " ").strip()
                lines.append(f"__label__{key} {text}")

# Shuffle and split
    random.shuffle(lines)
    split_idx = int(len(lines) * 0.8)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]

# Save to files
    with open("train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines))

    with open("test.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines))

def train():
    import fasttext

# Train the dialect classifier
    model = fasttext.train_supervised(
    input="train.txt",
    lr=0.1,
    epoch=100,
    wordNgrams=2,
    verbose=2,
    minCount=1,
    loss='softmax'
)

    model.save_model("dialect_classifier.bin")

# Evaluate performance
    result = model.test("test.txt")
    print(f"Test samples: {result[0]}")
    print(f"Precision@1: {result[1]:.4f}")
    print(f"Recall@1: {result[2]:.4f}")

train()