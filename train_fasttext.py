import json
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import fasttext

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


def train_fasttext_gswde(train_file):
    overall_results = {}
    for strategy in ["fewshot_conversational", "fewshot", "zero_shot", "finetuned_zeroshot", "zero_shot_gemma_3"]:
        print(f"##Evaluating {strategy} strategy...## \n ")
        data_dir = f"synthetic_data/{strategy}"
        overall_results[strategy] = {}

        ## open file in corresponding folder 
        with open(f"gswde_testfile_{strategy}.txt", "w", encoding="utf-8") as out_f:
            for filename in os.listdir(data_dir):
                ### create a file with all sentences from all dialects and label them with __label__gswde
                if filename.endswith(".txt"):
                    file_path = os.path.join(data_dir, filename)

                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            sentence = line.strip()
                            if sentence:
                                out_f.write(f"__label__ch {sentence}\n")
            
            test_de_file = "train_test_data/fasttext_test_german.txt"
            with open(test_de_file, "r", encoding="utf-8") as ge_file:
                # write all german sentences to the test file
                for line in ge_file:
                    sentence = line.strip()
                    out_f.write(line)
            
            


        


        
        

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
        result = model.test(f"gswde_testfile_{strategy}.txt")
        print(f"Test samples: {result[0]}")
        print(f"Precision@1: {result[1]:.4f}")
        print(f"Recall@1: {result[2]:.4f}")

        f1 = 2 * (result[1] * result[2]) / (result[1] + result[2]) if (result[1] + result[2]) > 0 else 0

        overall_results[strategy]['gswde'] = {'f1':f1}
        print(f"\nOverall results: {overall_results}")
    
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def train_test_v2(train_file, test_file):
    import fasttext
    from sklearn.metrics import confusion_matrix, classification_report
    
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
    label_vec = model.get_word_vector("__label__1")

# Find most similar words
    words = model.get_words()
    similarities = [(w, cosine_similarity(model.get_word_vector(w), label_vec)) for w in words]
    important_words = sorted(similarities, key=lambda x: x[1], reverse=True)[:20]
    print(important_words)
    
    model.save_model(f"dialect_classifier_{train_file}.bin")
    #print(f"Model trained and saved as dialect_classifier_{train_file}.bin")
    
    # Evaluate with FastText's built-in metrics
    result = model.test(test_file)
    #print(f"Test samples: {result[0]}")
    #print(f"Precision@1: {result[1]:.4f}")
    #print(f"Recall@1: {result[2]:.4f}")
    
    # Build confusion matrix manually
    y_true, y_pred = [], []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) < 2:
                continue
            label, text = parts[0], parts[1]
            y_true.append(label)
            labels, _ = model.predict(text, k=1)
            pred_label = labels[0]
            y_pred.append(pred_label)
    
    cm = confusion_matrix(y_true, y_pred, labels=list(set(y_true)))
    #print("\nConfusion Matrix:")
    #print(cm)
    
    # Optional: detailed per-class metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    return {
        "fasttext_result": result,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred
    }


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
        #print(f"Binary data written to {output_file}")
    else:
        return binary_lines
    
def plot_dialect_heatmaps(overall_results):
    """
    overall_results: dict[strategy][dialect] -> dict containing 'confusion_matrix' and optionally labels
    Example: overall_results['fewshot']['ch_bs']['confusion_matrix']
    """
    strategies = list(overall_results.keys())
    dialects = list(next(iter(overall_results.values())).keys())

    for strategy in strategies:
        num_dialects = len(dialects)
        fig, axes = plt.subplots(1, num_dialects, figsize=(5*num_dialects, 4))
        if num_dialects == 1:
            axes = [axes]  # ensure iterable

        fig.suptitle(f"Confusion Matrices for Strategy: {strategy}", fontsize=16)
        for ax, dialect in zip(axes, dialects):
            cm = overall_results[strategy][dialect]['confusion_matrix']
            cm = np.array(cm)  # convert list to array
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', ax=ax)
            
            # Use labels if you saved them
            labels = overall_results[strategy][dialect]['y_true']
            ax.set_xticks(np.arange(len(labels)) + 0.5)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticks(np.arange(len(labels)) + 0.5)
            ax.set_yticklabels(labels, rotation=0)
            
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Dialect: {dialect}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"confusion_matrices_{strategy}.png", dpi=300)
        plt.close()

    
def evaluate_binary_dialects():
    overall_results = {}
    for strategy in ["fewshot_conversational", "fewshot", "zero_shot", "finetuned_zeroshot", "zero_shot_gemma_3"]:
        print(f"##Evaluating {strategy} strategy...## \n ")

        overall_results[strategy] = {}
        for dialect in ["ch_bs", "ch_zh", "ch_be", "ch_lu"]:
            print(f"Creating binary FastText file for {dialect}...")
            create_binary_fasttext_file("train.txt",dialect, f"train_{dialect}.txt")
            create_binary_fasttext_file(f'fasttext_{strategy}.txt', dialect, f"test_{dialect}.txt")
            result = train_test_v2(f"train_{dialect}.txt", f"test_{dialect}.txt")
            conf_matrix = result["confusion_matrix"]
            y_true = result["y_true"]
            results = result["fasttext_result"]

            f1 = 2 * (results[1] * results[2]) / (results[1] + results[2]) if (results[1] + results[2]) > 0 else 0
            overall_results[strategy][dialect] = {'confusion_matrix': conf_matrix.tolist(), 'precision': results[1], 'recall': results[2], 'f1': f1, 'y_true': list(set(y_true))}
    
    return overall_results



def main():
    """
    create_data_swissdial(data_dir="synthetic_data/fewshot", output_file='fasttext_fewshot.txt')
    create_data_swissdial(data_dir="synthetic_data/fewshot_conversational", output_file='fasttext_fewshot_conversational.txt')
    create_data_swissdial(data_dir="synthetic_data/zero_shot", output_file='fasttext_zero_shot.txt')
    create_data_swissdial(data_dir="synthetic_data/finetuned_zeroshot", output_file='fasttext_finetuned_zeroshot.txt')
    """
    #create_data_swissdial(data_dir="synthetic_data/zero_shot_gemma_3", output_file='fasttext_zero_shot_gemma_3.txt')
    #results = evaluate_binary_dialects()
    #print(f"\nOverall results: {results}")

    #plot_dialect_heatmaps(results)
    train_fasttext_gswde(train_file="train_test_data/fasttext_train_gswde.txt")
if __name__ == "__main__":
    main()





#train_fasttext_gswde(train_file="fasttext_train_gswde.txt", test_file_de="fasttext_test_german.txt", test_file_ch="test_ch_be.txt")

