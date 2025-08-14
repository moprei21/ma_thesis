import json
import random
import os

def create_data():
# Load your dataset
    with open("data/swissdial/sentences_ch_de_numerics.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    dialect_keys = ["ch_be", "ch_zh", "ch_bs", "ch_lu"]

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


def create_data_swissdial(data_dir, output_file):
    dialect_mapping = {'basel': 'ch_bs', 'zürich': 'ch_zh', 'bern': 'ch_be', 'luzern': 'ch_lu'}

    with open(output_file, "w", encoding="utf-8") as out_f:
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                label = filename.replace(".txt", "")  # e.g., "zürich"
                label = dialect_mapping[label]
                file_path = os.path.join(data_dir, filename)

                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        sentence = line.strip()
                        if sentence:
                            out_f.write(f"__label__{label} {sentence}\n")

    print(f"✅ test.txt generated successfully from dialect files in '{data_dir}'")

def eval_swissdial(test_file):
    import fasttext

    # Load the trained model
    model = fasttext.load_model("dialect_classifier.bin")

    # Evaluate the model on the test set
    result = model.test(test_file)
    print(result)
    print(f"Test samples: {result[0]}")
    print(f"Precision@1: {result[1]:.4f}")
    print(f"Recall@1: {result[2]:.4f}")


def train_fasttext_gswde(train_file, test_file_de, test_file_ch):
    with open(test_file_ch, 'r', encoding='utf-8') as f:
        for entry in f:
            label, text = entry.strip().split(maxsplit=1)
            label = "__label__ch"
            changed_entry = f"{label} {text}"
            with open("fasttext_fewshot_gswde.txt", 'a', encoding='utf-8') as f_out:
                f_out.write(changed_entry + '\n')
    with open(test_file_de, 'r', encoding='utf-8') as f:
        for entry in f:
            with open("fasttext_fewshot_gswde.txt", 'a', encoding='utf-8') as f_out:
                f_out.write(entry)
    import fasttext

    # Train the dialect classifier
    model = fasttext.train_supervised(
        input=train_file,
        lr=1.0,
        epoch=100,
        wordNgrams=3,
        verbose=2,
        minCount=1,
        loss='softmax'
    )

    model.save_model("dialect_classifier_gswde.bin")
    print("Model trained and saved as dialect_classifier.bin")

    # Evaluate performance
    result = model.test("fasttext_fewshot_gswde.txt")
    print(f"Test samples: {result[0]}")
    print(f"Precision@1: {result[1]:.4f}")
    print(f"Recall@1: {result[2]:.4f}")

def train_test(train_file, test_file):
    import fasttext

# Train the dialect classifier
    model = fasttext.train_supervised(
    input=train_file,
    lr=1.0,
    epoch=100,
    wordNgrams=3,
    verbose=2,
    minCount=1,
    loss='softmax'
)

    model.save_model(f"dialect_classifier_{train_file}.bin")
    print(f"Model trained and saved as dialect_classifier_{train_file}.bin")
# Evaluate performance
    result = model.test(test_file)
    print(f"Test samples: {result[0]}")
    print(f"Precision@1: {result[1]:.4f}")
    print(f"Recall@1: {result[2]:.4f}")

    return result


def create_binary_fasttext_file(input_file, target_label, output_file=None):
    """
    Reads FastText-labeled data from a file and writes/returns binary-labeled data
    for the given target label.

    Args:
        input_file (str): Path to the input file with FastText format (__label__X text).
        target_label (str): The dialect label to treat as positive (e.g., 'ch_bs').
        output_file (str, optional): Path to save the binary data. If None, returns list.

    Returns:
        list of str: Binary-labeled data lines (only if output_file is None).
    """
    target_fasttext_label = f"__label__{target_label}"
    binary_lines = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue  # skip empty lines

            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue  # malformed line

            original_label, text = parts
            binary_label = "__label__1" if original_label == target_fasttext_label else "__label__0"
            binary_lines.append(f"{binary_label} {text}")

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write("\n".join(binary_lines))
        print(f"Binary data written to {output_file}")
    else:
        return binary_lines
    
def evaluate_binary_dialects():
    overall_results = {}
    for strategy in ["fewshot_conversational", "fewshot", "zero_shot", "finetuned_zeroshot"]:
        print(f"##Evaluating {strategy} strategy...## \n ")

        overall_results[strategy] = {}
        for dialect in ["ch_bs", "ch_zh", "ch_be", "ch_lu"]:
            print(f"Creating binary FastText file for {dialect}...")
            create_binary_fasttext_file("train.txt",dialect, f"train_{dialect}.txt")
            create_binary_fasttext_file(f'fasttext_{strategy}.txt', dialect, f"test_{dialect}.txt")
            results = train_test(f"train_{dialect}.txt", f"test_{dialect}.txt")
            f1 = 2 * (results[1] * results[2]) / (results[1] + results[2]) if (results[1] + results[2]) > 0 else 0
            overall_results[strategy][dialect] = f1
    
    return overall_results



def main():
    create_data_swissdial(data_dir="synthetic_data/fewshot", output_file='fasttext_fewshot.txt')
    create_data_swissdial(data_dir="synthetic_data/fewshot_conversational", output_file='fasttext_fewshot_conversational.txt')
    create_data_swissdial(data_dir="synthetic_data/zero_shot", output_file='fasttext_zero_shot.txt')
    create_data_swissdial(data_dir="synthetic_data/finetuned_zeroshot", output_file='fasttext_finetuned_zeroshot.txt')
    results = evaluate_binary_dialects()
    print(f"\nOverall results: {results}")

if __name__ == "__main__":
    main()





#train_fasttext_gswde(train_file="fasttext_train_gswde.txt", test_file_de="fasttext_test_german.txt", test_file_ch="test_ch_be.txt")

