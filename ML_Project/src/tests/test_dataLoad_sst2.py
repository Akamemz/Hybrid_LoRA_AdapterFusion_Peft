import unittest
import os
from datasets import DatasetDict
from ..LoRa.components.huggingface_models.huggingface_model_loader import HuggingFaceModelLoader
from ..LoRa.components.data_loader.huggingface_data_loader import LocalCsvDatasetLoader


class TestSst2LocalCsvDatasetLoader(unittest.TestCase):
    """
    Integration test for the LocalCsvDatasetLoader using the local SST-2 dataset.

    This test assumes you have run the download_datasets.py script and have the
    SST-2 CSV files located in ../data/sst2_dataset/ relative to the src directory.
    """

    def test_load_sst2_from_local_csv(self):
        """
        Tests loading and preparing the SST-2 dataset from local CSV files.
        """
        print("\n--- Testing SST-2 local CSV dataset loading ---")

        # --- FIX START ---
        # Construct a robust path to the project's root directory.
        # This assumes the test file is in 'src/tests/'.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        base_path = os.path.join(project_root, "data", "sst2_dataset")
        # --- FIX END ---

        data_files = {
            "train": os.path.join(base_path, "sst2_train.csv"),
            "validation": os.path.join(base_path, "sst2_validation.csv"),
            "test": os.path.join(base_path, "sst2_test.csv"),
        }

        # Step 1: We need a tokenizer to prepare the data
        model_loader = HuggingFaceModelLoader()
        _, tokenizer = model_loader.load()

        # Step 2: Initialize and run the data loader
        # For SST-2, the columns are 'sentence' and 'label'
        data_loader = LocalCsvDatasetLoader(
            data_files=data_files,
            text_column="sentence",
            label_column="label"
        )
        processed_dataset = data_loader.load_and_prepare(tokenizer)

        # Step 3: Verify the output
        self.assertIsInstance(processed_dataset, DatasetDict)
        self.assertIn("train", processed_dataset)
        self.assertIn("validation", processed_dataset)
        self.assertIn("test", processed_dataset)
        self.assertTrue(len(processed_dataset["train"]) > 0)
        self.assertIn("input_ids", processed_dataset["train"].features)
        self.assertIn("label", processed_dataset["train"].features)

        print("SST-2 local CSV dataset test passed successfully.")


if __name__ == '__main__':
    unittest.main()


# run the following in the terminal ----> python -m src.tests.test_dataLoad_sst2
