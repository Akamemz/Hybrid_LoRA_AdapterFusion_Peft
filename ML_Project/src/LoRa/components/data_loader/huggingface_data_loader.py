import os
from typing import Dict
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer
from .base import BaseDataLoader


class LocalCsvDatasetLoader(BaseDataLoader):
    """
    Loads and prepares datasets from local CSV files.

    This class is designed to work with the output of the download_datasets.py
    script. It loads specific splits from CSVs, tokenizes them, and prepares
    them for the training pipeline.
    """

    def __init__(
            self,
            data_files: Dict[str, str],
            text_column: str,
            label_column: str,
            max_length: int = 128,
    ):
        """
        Initializes the dataset loader with paths to local CSV files.

        Args:
            data_files (Dict[str, str]): A dictionary mapping split names
                                        (e.g., 'train', 'validation') to
                                        their respective CSV file paths.
            text_column (str): The name of the column containing the text data.
            label_column (str): The name of the column containing the labels.
            max_length (int): The maximum sequence length for tokenization.
        """
        self.data_files = data_files
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        print(f"LocalCsvDatasetLoader initialized for files: {list(self.data_files.values())}")

    def load_and_prepare(self, tokenizer: PreTrainedTokenizer) -> DatasetDict:
        """
        Loads data from local CSVs, tokenizes, and prepares it.

        Returns:
            A DatasetDict with the tokenized data splits.
        """
        print(f"Loading dataset from local CSV files...")
        # Verify that all specified files exist before loading
        for split, path in self.data_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file for split '{split}' not found at: {path}")

        # Load the dataset from the provided CSV files
        dataset = load_dataset("csv", data_files=self.data_files)

        def tokenize_function(examples):
            """Helper function to apply tokenization to a batch of examples."""
            return tokenizer(
                examples[self.text_column],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        print("Tokenizing dataset...")
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Clean up columns to be model-friendly
        tokenized_datasets = tokenized_datasets.remove_columns([self.text_column])

        # Rename the label column to 'labels' if it's not already named that way
        # This is what HuggingFace models expect
        if self.label_column != "labels":
            tokenized_datasets = tokenized_datasets.rename_column(self.label_column, "labels")
            print(f"Renamed column '{self.label_column}' to 'labels'")

        tokenized_datasets.set_format("torch")

        print("Local dataset prepared successfully.")
        return tokenized_datasets