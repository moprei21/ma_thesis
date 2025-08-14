from datasets import load_dataset
import pandas as pd

def load_and_inspect(dataset_id, config=None):
    """Load dataset from HF and return some inspection info."""
    print(f"Loading dataset `{dataset_id}`{f' config={config}' if config else ''} ...")
    ds = load_dataset(dataset_id, config) if config else load_dataset(dataset_id)
    info = {}
    info['splits'] = list(ds.keys())
    # Count samples per split
    info['split_counts'] = {split: ds[split].num_rows for split in info['splits']}
    # Languages: attempt to inspect language codes (heuristic)
    if hasattr(ds, 'info') and ds.info.features:
        # Look for columns that suggest language codes
        possible_lang_cols = [col for col in ds[list(ds.keys())[0]].features.keys()
                              if 'language' in col or 'lang' in col or 'code' in col]
        info['lang_columns'] = possible_lang_cols
    else:
        info['lang_columns'] = []
    return info, ds

def compare_datasets(info1, info2, id1, id2):
    """Compare basic dataset metadata."""
    rows = []
    keys = set(info1.keys()).union(info2.keys())
    for key in keys:
        rows.append({
            "Metric": key,
            id1: info1.get(key),
            id2: info2.get(key)
        })
    return pd.DataFrame(rows)

def main():
    # Dataset IDs
    id1 = "SEACrowd/ntrex_128"
    id2 = "facebook/flores"
    # Load and inspect
    info1, ds1 = load_and_inspect(id1)
    info2, ds2 = load_and_inspect(id2, config="all")  # using config "all" for full flores dataset
    print("\nDataset comparison:")
    df = compare_datasets(info1, info2, id1, id2)
    print(df.to_string(index=False))

    # Example: print a sample from each dataset
    print("\nSample entry from each dataset:")
    for ds, name in [(ds1, id1), (ds2, id2)]:
        print(f"\n{name}, split '{list(ds.keys())[0]}' — first item:")
        print(ds[list(ds.keys())[0]][0])

if __name__ == "__main__":
    main()