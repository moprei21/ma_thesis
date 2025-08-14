import pandas as pd
def clean_sentiment_data(filename: str = "data/SB-CH-full/SB-CH-full/sentiment.csv") -> pd.DataFrame:
    """
    Cleans the sentiment data by removing rows with ties in sentiment votes.
    Returns a DataFrame with unique sentiment labels.
    """
    # Load data
    df = pd.read_csv(filename)

    # Sentiment columns
    sentiment_cols = ["neut", "neg", "pos"]

    # Find the max vote per row
    df["max_votes"] = df[sentiment_cols].max(axis=1)

    # Count how many times this max occurs in each row
    df["max_count"] = df[sentiment_cols].eq(df["max_votes"], axis=0).sum(axis=1)

    # Keep only rows where the max is unique
    df_no_ties = df[df["max_count"] == 1].copy()

    # Assign label based on the unique max
    df_no_ties["label"] = df_no_ties[sentiment_cols].idxmax(axis=1)

    # Optional: map to readable names
    label_map = {
        "neut": "neutral",
        "neg": "negative",
        "pos": "positive"
    }
    df_no_ties["label"] = df_no_ties["label"].map(label_map)

    # Drop helper columns
    df_no_ties = df_no_ties.drop(columns=["max_votes", "max_count", "un", "unsure"])

    return df_no_ties

def merge_sentiment_data(sentiment_annotations) -> pd.DataFrame:
    """
    Merges two sentiment DataFrames, ensuring unique sentiment labels.
    """
    
    df1 = pd.read_csv("data/SB-CH-full/SB-CH-full/facebook.text.csv")
    df2 = pd.read_csv("data/SB-CH-full/SB-CH-full/noah.text.csv")
    df3 = pd.read_csv("data/SB-CH-full/SB-CH-full/sms4science.text.csv")
    df4 = pd.read_csv("data/SB-CH-full/SB-CH-full/chatmania.csv")


    # Combine the text CSVs into one DataFrame
    text_data = pd.concat([df1, df2, df3, df4], ignore_index=True)

    # Merge annotations with text using sentence_id
    merged_df = sentiment_annotations.merge(text_data, on="sentence_id")

    # Check results
    print(merged_df)

    # Save final dataset
    merged_df.to_csv("final_sentiment_dataset.csv", index=False)


def main():
    # Clean sentiment data
    cleaned_data = clean_sentiment_data()

    # Merge with text data
    merge_sentiment_data(cleaned_data)

if __name__ == "__main__":
    main()  

