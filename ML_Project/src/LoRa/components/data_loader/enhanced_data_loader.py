"""
ML_Project/src/LoRa/components/data_loader/enhanced_data_loader.py

Enhanced dataset loader supporting multiple datasets with varying structures
Handles datasets with/without validation splits, different column names, etc.
"""

import os
from typing import Dict, Optional, Tuple
from datasets import load_dataset, DatasetDict, Dataset
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from .base import BaseDataLoader

# Dataset configurations - centralized metadata
DATASET_CONFIGS = {
    "sst2": {
        "text_column": "sentence",
        "label_column": "label",
        "num_labels": 2,
        "has_validation": True,
        "task_type": "sentiment"
    },
    "ag_news": {
        "text_column": "text",
        "label_column": "label",
        "num_labels": 4,
        "has_validation": False,  # Will create from train
        "task_type": "topic_classification"
    },
    "imdb": {
        "text_column": "text",
        "label_column": "label",
        "num_labels": 2,
        "has_validation": False,
        "task_type": "sentiment"
    },
    "tweet_eval": {
        "text_column": "text",
        "label_column": "label",
        "num_labels": 3,  # Varies by subtask
        "has_validation": True,
        "task_type": "sentiment",
        "subtasks": ["emotion", "sentiment", "offensive"]  # Multiple variants
    },
    "yelp": {
        "text_column": "text",
        "label_column": "label",
        "num_labels": 5,  # 1-5 stars
        "has_validation": False,
        "task_type": "sentiment"
    }
}


class UnifiedDatasetLoader(BaseDataLoader):
    """
    Unified dataset loader supporting multiple datasets with automatic handling of:
    - Missing validation splits (auto-creates from train)
    - Different column names
    - Few-shot scenarios
    - Custom CSV files
    """

    def __init__(
            self,
            dataset_name: str,
            data_dir: Optional[str] = None,
            max_length: int = 128,
            validation_split: float = 0.1,
            test_split: float = 0.1,
            few_shot_n: Optional[int] = None,
            seed: int = 42
    ):
        """
        Initialize unified dataset loader.

        Args:
            dataset_name: Name of dataset (e.g., 'sst2', 'ag_news', 'imdb')
            data_dir: Optional directory containing CSV files
            max_length: Maximum sequence length for tokenization
            validation_split: Proportion of train to use for validation if missing
            test_split: Proportion of train to use for test if missing
            few_shot_n: Number of examples per class for few-shot learning
            seed: Random seed for reproducibility
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: {list(DATASET_CONFIGS.keys())}"
            )

        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        self.data_dir = data_dir
        self.max_length = max_length
        self.validation_split = validation_split
        self.test_split = test_split
        self.few_shot_n = few_shot_n
        self.seed = seed

        print(f"UnifiedDatasetLoader initialized for: {dataset_name}")
        print(f"  Text column: {self.config['text_column']}")
        print(f"  Label column: {self.config['label_column']}")
        print(f"  Number of labels: {self.config['num_labels']}")
        print(f"  Has validation: {self.config['has_validation']}")
        if few_shot_n:
            print(f"  Few-shot mode: {few_shot_n} examples per class")

    def load_and_prepare(self, tokenizer: PreTrainedTokenizer) -> DatasetDict:
        """
        Load dataset, handle missing splits, tokenize, and prepare for training.

        Returns:
            DatasetDict with train/validation/test splits
        """
        # Step 1: Load raw dataset
        if self.data_dir:
            dataset = self._load_from_csv()
        else:
            dataset = self._load_from_hub()

        # Step 2: Create missing splits
        dataset = self._ensure_splits(dataset)

        # Step 3: Few-shot sampling if requested
        if self.few_shot_n:
            dataset = self._create_few_shot(dataset)

        # Step 4: Tokenize
        print("\nTokenizing dataset...")
        tokenized_dataset = self._tokenize_dataset(dataset, tokenizer)

        # Step 5: Format for training
        tokenized_dataset = self._format_for_training(tokenized_dataset)

        print("\nDataset preparation complete!")
        self._print_dataset_stats(tokenized_dataset)

        return tokenized_dataset

    def _load_from_csv(self) -> DatasetDict:
        """Load dataset from local CSV files."""
        print(f"\nLoading from CSV files in: {self.data_dir}")

        data_files = {}
        base_path = os.path.join(self.data_dir, f"{self.dataset_name}_dataset")

        # Check for standard files
        for split in ["train", "validation", "test"]:
            file_path = os.path.join(base_path, f"{self.dataset_name}_{split}.csv")
            if os.path.exists(file_path):
                data_files[split] = file_path
                print(f"  Found {split}: {file_path}")

        if not data_files:
            raise FileNotFoundError(
                f"No CSV files found in {base_path}. "
                f"Expected format: {self.dataset_name}_train.csv, etc."
            )

        dataset = load_dataset("csv", data_files=data_files)
        return dataset

    def _load_from_hub(self) -> DatasetDict:
        """Load dataset from Hugging Face Hub."""
        print(f"\nLoading from Hugging Face Hub...")

        # Map dataset names to Hub identifiers
        hub_names = {
            "sst2": ("glue", "sst2"),
            "ag_news": ("ag_news", None),
            "imdb": ("imdb", None),
            "tweet_eval": ("tweet_eval", "sentiment"),  # Default subtask
            "yelp": ("yelp_polarity", None)
        }

        dataset_path, config_name = hub_names.get(self.dataset_name, (self.dataset_name, None))

        if config_name:
            dataset = load_dataset(dataset_path, config_name)
        else:
            dataset = load_dataset(dataset_path)

        return dataset

    def _ensure_splits(self, dataset: DatasetDict) -> DatasetDict:
        """
        Ensure dataset has train/validation/test splits.
        Creates missing splits from train data.
        """
        # Check what splits exist
        has_train = "train" in dataset
        has_val = "validation" in dataset
        has_test = "test" in dataset

        if has_train and has_val and has_test:
            print("All splits present.")
            return dataset

        print("\nCreating missing splits...")

        if not has_train:
            raise ValueError("Dataset must have at least a 'train' split")

        # If missing validation and/or test, split from train
        if not has_val or not has_test:
            train_data = dataset["train"]

            if not has_val and not has_test:
                # Split train into train/val/test
                # First split: train vs (val+test)
                train_idx, temp_idx = train_test_split(
                    range(len(train_data)),
                    test_size=self.validation_split + self.test_split,
                    random_state=self.seed,
                    stratify=train_data[self.config["label_column"]]
                )

                # Second split: val vs test
                temp_data = train_data.select(temp_idx)
                val_size = self.validation_split / (self.validation_split + self.test_split)
                val_idx, test_idx = train_test_split(
                    range(len(temp_data)),
                    test_size=1 - val_size,
                    random_state=self.seed,
                    stratify=temp_data[self.config["label_column"]]
                )

                dataset = DatasetDict({
                    "train": train_data.select(train_idx),
                    "validation": temp_data.select(val_idx),
                    "test": temp_data.select(test_idx)
                })

                print(f"  Created validation split: {len(dataset['validation'])} examples")
                print(f"  Created test split: {len(dataset['test'])} examples")

            elif not has_val:
                # Only create validation
                train_idx, val_idx = train_test_split(
                    range(len(train_data)),
                    test_size=self.validation_split,
                    random_state=self.seed,
                    stratify=train_data[self.config["label_column"]]
                )

                dataset = DatasetDict({
                    "train": train_data.select(train_idx),
                    "validation": train_data.select(val_idx),
                    "test": dataset["test"]
                })

                print(f"  Created validation split: {len(dataset['validation'])} examples")

            elif not has_test:
                # Only create test
                train_idx, test_idx = train_test_split(
                    range(len(train_data)),
                    test_size=self.test_split,
                    random_state=self.seed,
                    stratify=train_data[self.config["label_column"]]
                )

                dataset = DatasetDict({
                    "train": train_data.select(train_idx),
                    "validation": dataset["validation"],
                    "test": train_data.select(test_idx)
                })

                print(f"  Created test split: {len(dataset['test'])} examples")

        return dataset

    def _create_few_shot(self, dataset: DatasetDict) -> DatasetDict:
        """
        Create few-shot version of dataset by sampling n examples per class.
        """
        print(f"\nCreating few-shot dataset: {self.few_shot_n} examples per class")

        train_data = dataset["train"]
        df = train_data.to_pandas()

        # Sample n examples per class
        few_shot_df = df.groupby(self.config["label_column"]).apply(
            lambda x: x.sample(n=min(self.few_shot_n, len(x)), random_state=self.seed)
        ).reset_index(drop=True)

        # Convert back to Dataset
        few_shot_dataset = Dataset.from_pandas(few_shot_df)

        print(f"  Original train size: {len(train_data)}")
        print(f"  Few-shot train size: {len(few_shot_dataset)}")
        print(f"  Reduction: {len(train_data) / len(few_shot_dataset):.1f}x")

        # Keep validation and test unchanged
        dataset = DatasetDict({
            "train": few_shot_dataset,
            "validation": dataset["validation"],
            "test": dataset["test"]
        })

        return dataset

    def _tokenize_dataset(self, dataset: DatasetDict,
                          tokenizer: PreTrainedTokenizer) -> DatasetDict:
        """Apply tokenization to all splits."""
        text_col = self.config["text_column"]

        def tokenize_function(examples):
            return tokenizer(
                examples[text_col],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        # ADD load_from_cache_file=False to disable caching
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            load_from_cache_file=False  # ADD THIS LINE
        )
        return tokenized

    def _format_for_training(self, dataset: DatasetDict) -> DatasetDict:
        """Format dataset for HuggingFace Trainer."""
        text_col = self.config["text_column"]
        label_col = self.config["label_column"]

        # Get all column names
        all_columns = dataset["train"].column_names

        # Columns to keep: input_ids, attention_mask, and labels
        columns_to_keep = ["input_ids", "attention_mask"]

        # Columns to remove: everything except what we need
        columns_to_remove = [
            col for col in all_columns
            if col not in columns_to_keep and col != label_col
        ]

        # Remove unwanted columns (including 'idx', original text column, etc.)
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)
            print(f"  Removed columns: {columns_to_remove}")

        # Rename label column to 'labels' (HF Trainer expectation)
        if label_col != "labels" and label_col in dataset["train"].column_names:
            dataset = dataset.rename_column(label_col, "labels")

        # Set format to PyTorch tensors
        dataset.set_format("torch")

        return dataset

    def _print_dataset_stats(self, dataset: DatasetDict):
        """Print dataset statistics."""
        print(f"\n{'=' * 60}")
        print("DATASET STATISTICS")
        print(f"{'=' * 60}")

        for split in dataset.keys():
            split_data = dataset[split]
            print(f"{split.capitalize():12s}: {len(split_data):>6,} examples")

            # Print label distribution
            if "labels" in split_data.features:
                labels = split_data["labels"]
                import torch
                if isinstance(labels[0], torch.Tensor):
                    labels = [label.item() for label in labels]

                from collections import Counter
                label_counts = Counter(labels)
                print(f"              Label distribution: {dict(label_counts)}")

        print(f"{'=' * 60}\n")

    @staticmethod
    def get_dataset_info(dataset_name: str) -> Dict:
        """Get configuration info for a dataset."""
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return DATASET_CONFIGS[dataset_name].copy()

    @staticmethod
    def list_available_datasets() -> list:
        """List all supported datasets."""
        return list(DATASET_CONFIGS.keys())