#%%
import os
from datasets import load_dataset


#%%
def download_and_save_sst2():
    """
    Downloads the GLUE SST-2 dataset and saves its splits to CSV files.
    """
    try:
        # Load the dataset from the Hugging Face Hub
        dataset_dict = load_dataset('glue', 'sst2', trust_remote_code=True)
        print("Dataset downloaded successfully.")
        print(f"Found splits: {list(dataset_dict.keys())}")

        # Create a directory to store the data
        output_dir = "sst2_dataset"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        # Iterate through each split then convert to CSV and save
        for split_name, dataset_split in dataset_dict.items():
            print(f"\nProcessing '{split_name}' split")

            # Convert the dataset split to a pandas Dataframe
            df = dataset_split.to_pandas()

            # Define the output file path
            file_path = os.path.join(output_dir, f"sst2_{split_name}.csv")

            # Save pandas Dataframe to a CSV file
            df.to_csv(file_path, index=False)

            print(f"'{split_name}' split saved to: {file_path}")
            print("First 3 rows of the saved data:")
            print(df.head(3).to_string())

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    download_and_save_sst2()
