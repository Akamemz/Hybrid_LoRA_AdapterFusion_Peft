
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_and_save_agnews_splits():
    """
    Loads AG News train and test data, splits train into train/validation, and saves all three splits as CSVs.
    """
    try:
        # Set up paths
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ag_news_dataset")
        train_csv = os.path.join(data_dir, "ag_news_train.csv")
        test_csv = os.path.join(data_dir, "ag_news_test.csv")
        val_csv = os.path.join(data_dir, "ag_news_validation.csv")

        # Load train and test data
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        # Split train into train/validation (90/10)
        train_split, val_split = train_test_split(
            train_df,
            test_size=0.1,
            random_state=42,
            stratify=train_df['label']
        )

        # Save splits
        train_split.to_csv(train_csv, index=False)
        val_split.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)  # Overwrite to ensure format

        print(f"Train split saved: {train_csv} ({len(train_split)})")
        print(f"Validation split saved: {val_csv} ({len(val_split)})")
        print(f"Test split saved: {test_csv} ({len(test_df)})")

        # Show sample rows
        print("\nSample train rows:")
        print(train_split.head(3).to_string())
        print("\nSample validation rows:")
        print(val_split.head(3).to_string())
        print("\nSample test rows:")
        print(test_df.head(3).to_string())

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_and_save_agnews_splits()