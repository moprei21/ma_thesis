import json
import pandas as pd
from datasets import Dataset



class SwissDialDataset():
    def __init__(self, file_path, dialect='Zürich'):
        self.file_path = file_path
        self.df = None
        self.dataset = None
        self.dialect = None
        self.dialect_mapping = {'Basel': 'ch_bs', 'Zürich': 'ch_zh', 'Bern': 'ch_be', 'Luzern': 'ch_lu', "Sankt Gallen":"ch_sg", 'Graubünden': 'ch_gr', 'Wallis': 'ch_vs', 'Aargau': 'ch_ag'}


    def load_from_json_file(self):
        """Load Swiss German dialect dataset from a JSON file"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
        print(f"Loaded {len(data)} entries from {self.file_path}")
    
    # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        return df

    def create_huggingface_dataset(self, df):
        """
        Convert a pandas DataFrame to a Hugging Face Dataset, 
        filtering by required columns and splitting into train/test.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            
        Returns:
            DatasetDict: A train/test split of the Hugging Face Dataset.
        """
        # Validate required columns
        required_columns = list(self.dialect_mapping.values())
        print(f"Required columns: {required_columns}")

        # Drop rows with missing values in required columns
        if required_columns:
            df = df.dropna(subset=required_columns)
            df = df[required_columns]

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        # Split into train/test
        train_test_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        self.dataset = train_test_dataset

        return train_test_dataset
    
    def create_fasttext_format(self, split='train', output_path='fasttext_train.txt'):
        """
        Convert the specified dataset split to FastText format and write to a file.

        Args:
            split (str): The split to convert ('train' or 'test').
            output_path (str): File path where the FastText-formatted data will be saved.
        """
        if not hasattr(self, 'dataset'):
            raise ValueError("Dataset not loaded. Call create_huggingface_dataset first.")

        data = self.dataset[split]

        with open(output_path, 'w', encoding='utf-8') as f:
            for example in data:
                print(example)
                # Map dialect to FastText label
                for dialect, text in example.items():
                    line = f"__label__ch {text.strip()}"
                    f.write(line + '\n')

    def sample_dataset(self, dataset, num_samples=10):
        """
        Sample a specified number of examples from the dataset.

        Args:
            dataset (Dataset): The Hugging Face Dataset to sample from.
            num_samples (int): Number of samples to return.

        Returns:
            Dataset: A new Dataset containing the sampled examples.
        """
        if num_samples > len(dataset):
            raise ValueError("Number of samples requested exceeds dataset size.")

        sampled_data = dataset.shuffle(seed=42).select(range(num_samples))

        return sampled_data[self.dialect_mapping[self.dialect]]
        
    

        
    

class GermanXNLI:
    def __init__(self):
        self.dataset =None
    

    def load_dataset(self, sample_sizes: dict = None, shuffle: bool = True, seed: int = 42):
        """
        Load the German XNLI dataset and optionally sample subsets from each split.

        Args:
            sample_sizes (dict, optional): Dictionary mapping split names ('train', 'test', ...) to sample sizes.
                                        If a split is not in the dict, the full split is loaded.
                                        Example: {'train': 1000, 'test': 200}
            shuffle (bool): Whether to shuffle before sampling.
            seed (int): Random seed for reproducibility.

        Returns:
            DatasetDict: The loaded (and possibly sampled) dataset.
        """
        from datasets import load_dataset

        self.dataset = load_dataset("facebook/xnli","de")

        sample_sizes = sample_sizes or {}

        for split in ['train', 'test']:
            if split in self.dataset:
                data_split = self.dataset[split]
                if split in sample_sizes:
                    size = min(sample_sizes[split], len(data_split))
                    if shuffle:
                        data_split = data_split.shuffle(seed=seed)
                    data_split = data_split.select(range(size))
                    print(f"Sampled {size} examples from the '{split}' set.")
                else:
                    print(f"Loaded full '{split}' set with {len(data_split)} examples.")
                self.dataset[split] = data_split
            else:
                print(f"Split '{split}' not found in dataset.")
            

    def convert_to_fasttext_format(self, split='train', output_path='fasttext_train_german.txt'):
        """
        Convert the specified dataset split to FastText format and write to a file.

        Args:
            split (str): The split to convert ('train', 'test', or 'validation').
            output_path (str): File path where the FastText-formatted data will be saved.
        """
        if not hasattr(self, 'dataset') or split not in self.dataset:
            raise ValueError(f"Split '{split}' not found. Load the dataset first and check the split name.")

        data = self.dataset[split]

        with open(output_path, 'w', encoding='utf-8') as f:
            for example in data:
                label = 'de'
                # XNLI uses 0=entailment, 1=neutral, 2=contradiction


                # You can choose how to format input text
                text = f"{example['premise']} {example['hypothesis']}".replace('\n', ' ')
                line = f"__label__{label} {text.strip()}"
                f.write(line + '\n')

        print(f"Saved FastText-formatted data to: {output_path}")

def main():
    # Load the dataset
    swiss_dial_dataset = SwissDialDataset(file_path="data/swissdial/sentences_ch_de_numerics.json")
    df = swiss_dial_dataset.load_from_json_file()
    
    # Create the Hugging Face dataset
    swiss_dial_dataset.create_huggingface_dataset(df)
    swiss_dial_dataset.create_fasttext_format(split='train', output_path='fasttext_train_swiss.txt')
    train_len = len(swiss_dial_dataset.dataset['train']) * 4

    sample_sizes = {'train': train_len, 'test': 800}
    # Save the dataset to disk

    german_xnli_dataset = GermanXNLI()
    german_xnli_dataset.load_dataset(sample_sizes=sample_sizes, shuffle=True, seed=42)
    german_xnli_dataset.convert_to_fasttext_format(split='train', output_path='fasttext_train_german.txt')
    german_xnli_dataset.convert_to_fasttext_format(split='test', output_path='fasttext_test_german.txt')
    print("\nGerman XNLI Dataset:")
    print(f"Train set size: {len(german_xnli_dataset.dataset['train'])}")
    print(f"Test set size: {len(german_xnli_dataset.dataset['test'])}") 
    

if __name__ == "__main__":
    main()