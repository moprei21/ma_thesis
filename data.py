import json
import pandas as pd
from datasets import Dataset



class SwissDialDataset():
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.dataset = None


    def load_from_json_file(self):
        """Load Swiss German dialect dataset from a JSON file"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
        print(f"Loaded {len(data)} entries from {self.file_path}")
    
    # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        return df

    def create_huggingface_dataset(self,df):
        """Convert DataFrame to a Hugging Face Dataset"""
        # Convert to Hugging Face Dataset
        required_columns = ['ch_bs']
        if required_columns is not None:
            df = df.dropna(subset=required_columns)
        dataset = Dataset.from_pandas(df)
    
        print("\nCreated Hugging Face Dataset:")
        print(f"Dataset features: {dataset.features}")
        print(f"Dataset shape: {dataset.shape}")
    
    # Split into train/test (for demonstration)
        train_test_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
        print(f"\nSplit into train/test:")
        print(f"Train: {train_test_dataset['train'].shape[0]} examples")
        print(f"Test: {train_test_dataset['test'].shape[0]} examples")
    
        return train_test_dataset
    

    def sample_dataset(self, dataset, num_samples=5):
        """Sample a few examples from the dataset"""
        sampled_data = dataset.shuffle(seed=42).select(range(num_samples))
       
        return sampled_data['ch_bs']


def main():
    # Load the dataset
    swiss_dial_dataset = SwissDialDataset(file_path="data/swissdial/sentences_ch_de_numerics.json")
    df = swiss_dial_dataset.load_from_json_file()
    
    # Create the Hugging Face dataset
    dataset = swiss_dial_dataset.create_huggingface_dataset(df)
    # Save the dataset to disk

    sampled_data = swiss_dial_dataset.sample_dataset(dataset['train'], num_samples=10)

if __name__ == "__main__":
    main()