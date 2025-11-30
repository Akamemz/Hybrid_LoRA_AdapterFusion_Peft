import unittest
from transformers import PreTrainedModel
from peft import PeftModel
from ..LoRa.components.huggingface_models.huggingface_model_loader import HuggingFaceModelLoader
from ..LoRa.components.peft.peft_model_builder import PeftModelBuilder


class TestPeftModelBuilder(unittest.TestCase):
    """
    Unit tests for the PeftModelBuilder.
    """

    @classmethod
    def setUpClass(cls):
        """Load a base model once for all tests."""
        print("\n--- Setting up base model for PEFT builder test ---")
        loader = HuggingFaceModelLoader()
        cls.base_model, _ = loader.load()

    def test_apply_lora(self):
        """
        Tests if the builder correctly applies a LoRA configuration.
        """
        print("\n--- Testing LoRA application ---")
        # Step 1: Initialize the builder with the base model
        builder = PeftModelBuilder(self.base_model)

        # Step 2: Define a LoRA configuration
        lora_config = {
            "method": "lora",
            "r": 4,
            "lora_alpha": 8,
        }

        # Step 3: Build the PEFT model
        peft_model = builder.build(lora_config)
        total_params, trainable_params = builder.count_parameters(peft_model)

        # Step 4: Verify the output
        self.assertIsInstance(peft_model, PeftModel)
        self.assertIsInstance(peft_model.base_model.model, PreTrainedModel)

        # Check that LoRA layers were added and are trainable
        self.assertTrue(trainable_params > 0)
        self.assertTrue(trainable_params < total_params)

        # A rough check to ensure trainable params are in a reasonable range for LoRA
        self.assertTrue(trainable_params < (total_params * 0.01))

        print("LoRA application test passed.")


if __name__ == '__main__':
    unittest.main()


# python -m src.tests.test_peft