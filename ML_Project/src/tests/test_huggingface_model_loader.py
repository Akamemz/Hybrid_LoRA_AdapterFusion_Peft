from ..LoRa.components.huggingface_models.huggingface_model_loader import HuggingFaceModelLoader
import torch


def test_default_model():
    """Test loading the default DistilBERT model."""
    print("=" * 60)
    print("Testing HuggingFaceModelLoader with default model")
    print("=" * 60)

    # Initialize loader with default model
    loader = HuggingFaceModelLoader()

    # Load model and tokenizer
    model, tokenizer = loader.load()

    # Verify they loaded correctly
    assert model is not None, "Model failed to load"
    assert tokenizer is not None, "Tokenizer failed to load"
    print("✓ Model and tokenizer loaded successfully")

    # Test with a simple sentence
    test_sentence = "This is a test sentence."
    print(f"\nTest input: '{test_sentence}'")

    # Tokenize
    inputs = tokenizer(test_sentence, return_tensors="pt")
    print(f"✓ Tokenization successful. Input shape: {inputs['input_ids'].shape}")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"✓ Model inference successful. Output shape: {outputs.last_hidden_state.shape}")

    # Print model info
    print(f"\nModel name: {model.config._name_or_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_default_model()

# python3 -m  src.tests.test_huggingface_model_loader.py