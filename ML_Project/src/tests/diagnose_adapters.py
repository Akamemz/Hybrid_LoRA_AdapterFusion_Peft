#!/usr/bin/env python
"""
Diagnostic script to identify adapter-transformers issues
Run this to diagnose the adapter problem and get specific fix instructions
"""

import sys


def check_imports():
    """Check if required libraries are installed."""
    print("=" * 70)
    print("STEP 1: Checking Library Installations")
    print("=" * 70)

    libraries = {}

    # Check transformers
    try:
        import transformers
        libraries['transformers'] = transformers.__version__
        print(f"✓ transformers: {transformers.__version__}")
    except ImportError:
        libraries['transformers'] = None
        print("✗ transformers: NOT INSTALLED")

    # Check adapters
    try:
        import adapters
        libraries['adapters'] = adapters.__version__
        print(f"✓ adapter-transformers: {adapters.__version__}")
    except ImportError:
        libraries['adapters'] = None
        print("✗ adapter-transformers: NOT INSTALLED")

    # Check PEFT
    try:
        import peft
        libraries['peft'] = peft.__version__
        print(f"✓ peft: {peft.__version__}")
    except ImportError:
        libraries['peft'] = None
        print("✗ peft: NOT INSTALLED")

    # Check torch
    try:
        import torch
        libraries['torch'] = torch.__version__
        print(f"✓ torch: {torch.__version__}")
    except ImportError:
        libraries['torch'] = None
        print("✗ torch: NOT INSTALLED")

    return libraries


def test_model_loading():
    """Test basic model loading."""
    print("\n" + "=" * 70)
    print("STEP 2: Testing Model Loading")
    print("=" * 70)

    try:
        from transformers import AutoModelForSequenceClassification

        print("Loading distilbert-base-uncased...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
        )
        print(f"✓ Model loaded successfully")
        print(f"  Type: {type(model).__name__}")
        print(f"  Config: {model.config.model_type}")

        return model

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None


def test_adapter_init(model):
    """Test adapter initialization."""
    print("\n" + "=" * 70)
    print("STEP 3: Testing Adapter Initialization")
    print("=" * 70)

    if model is None:
        print("✗ Skipping: No model available")
        return None

    try:
        from adapters import init, AdapterConfig
        print("✓ adapters module imported")

        # Try to initialize
        print("\nInitializing adapter support...")
        adapted_model = init(model)

        if adapted_model is None:
            print("✗ ERROR: init() returned None")
            print("\nThis means:")
            print("  - adapter-transformers doesn't support DistilBERT")
            print("  - OR there's a version incompatibility")
            return None

        print(f"✓ Model initialized")
        print(f"  Type: {type(adapted_model).__name__}")

        # Check for adapter methods
        has_add = hasattr(adapted_model, 'add_adapter')
        has_train = hasattr(adapted_model, 'train_adapter')
        has_set = hasattr(adapted_model, 'set_active_adapters')

        print(f"  Has add_adapter: {has_add}")
        print(f"  Has train_adapter: {has_train}")
        print(f"  Has set_active_adapters: {has_set}")

        if not (has_add and has_train and has_set):
            print("\n✗ ERROR: Model missing required adapter methods")
            return None

        return adapted_model

    except ImportError as e:
        print(f"✗ Failed to import adapters: {e}")
        print("\nFIX: Install adapter-transformers")
        print("  pip install adapter-transformers")
        return None
    except Exception as e:
        print(f"✗ Failed to initialize adapters: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_adapter_addition(adapted_model):
    """Test adding an adapter."""
    print("\n" + "=" * 70)
    print("STEP 4: Testing Adapter Addition")
    print("=" * 70)

    if adapted_model is None:
        print("✗ Skipping: No adapted model available")
        return False

    try:
        from adapters import AdapterConfig

        # Create config
        print("Creating adapter config...")
        config = AdapterConfig.load("houlsby", reduction_factor=16)
        print("✓ Config created")

        # Add adapter
        print("\nAdding adapter...")
        adapted_model.add_adapter("test_adapter", config=config)
        print("✓ Adapter added successfully!")

        # Set trainable
        print("\nSetting adapter to trainable...")
        adapted_model.train_adapter("test_adapter")
        print("✓ Adapter set to trainable")

        # Count parameters
        print("\nCounting parameters...")
        total = sum(p.numel() for p in adapted_model.parameters())
        trainable = sum(p.numel() for p in adapted_model.parameters() if p.requires_grad)

        print(f"  Total parameters: {total:,}")
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Trainable %: {trainable / total * 100:.2f}%")

        return True

    except Exception as e:
        print(f"✗ Failed to add adapter: {e}")
        import traceback
        traceback.print_exc()
        return False


def provide_recommendations(libraries, adapter_works):
    """Provide specific recommendations based on test results."""
    print("\n" + "=" * 70)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 70)

    if adapter_works:
        print("\n✅ ALL TESTS PASSED!")
        print("\nYour adapter setup is working correctly.")
        print("\nNext steps:")
        print("  1. Replace adapter_builder.py with the fixed version")
        print("  2. Replace peft_factory.py with the fixed version")
        print("  3. Run: python -m src.main.improved_experiment_runner \\")
        print("           --experiment_name test_adapter \\")
        print("           --dataset sst2 \\")
        print("           --peft_method adapter \\")
        print("           --adapter_reduction_factor 16 \\")
        print("           --epochs 1")
        return

    print("\n❌ ADAPTERS NOT WORKING")
    print("\nProblems detected:")

    # Check what's missing
    if libraries['adapters'] is None:
        print("\n1. adapter-transformers is NOT installed")
        print("   FIX: pip install adapter-transformers")

    if libraries['transformers'] is None:
        print("\n2. transformers is NOT installed")
        print("   FIX: pip install transformers")

    # Version compatibility
    if libraries['adapters'] and libraries['transformers']:
        print("\n3. Possible version incompatibility")
        print(f"   Current: transformers={libraries['transformers']}, adapters={libraries['adapters']}")
        print("   Recommended: transformers==4.36.0, adapter-transformers==3.2.1")
        print("\n   FIX:")
        print("   pip uninstall transformers adapter-transformers -y")
        print("   pip install transformers==4.36.0 adapter-transformers==3.2.1")

    print("\n" + "-" * 70)
    print("ALTERNATIVE SOLUTION: Use LoRA Only")
    print("-" * 70)
    print("\nLoRA is working fine. You can:")
    print("  1. Focus research on LoRA parameter efficiency")
    print("  2. Compare different LoRA configurations")
    print("  3. Switch to BERT-base-uncased (better adapter support)")
    print("\nExample LoRA command:")
    print("  python -m src.main.improved_experiment_runner \\")
    print("    --experiment_name sst2_lora_baseline \\")
    print("    --dataset sst2 \\")
    print("    --peft_method lora \\")
    print("    --lora_r 8 \\")
    print("    --lora_alpha 16 \\")
    print("    --epochs 3 \\")
    print("    --param_budget 75000")

    print("\n" + "-" * 70)
    print("ALTERNATIVE SOLUTION: Switch to BERT")
    print("-" * 70)
    print("\nBERT has better adapter support:")
    print("  1. Change --model_name to bert-base-uncased")
    print("  2. All other code stays the same")
    print("  3. BERT is slower but more reliable with adapters")


def main():
    """Run all diagnostic tests."""
    print("\n" + "#" * 70)
    print("# ADAPTER-TRANSFORMERS DIAGNOSTIC TOOL")
    print("#" * 70)
    print("\nThis script will:")
    print("  1. Check library installations")
    print("  2. Test model loading")
    print("  3. Test adapter initialization")
    print("  4. Test adapter addition")
    print("  5. Provide specific fix recommendations")
    print("\n" + "#" * 70 + "\n")

    # Run tests
    libraries = check_imports()
    model = test_model_loading()
    adapted_model = test_adapter_init(model)
    adapter_works = test_adapter_addition(adapted_model)

    # Provide recommendations
    provide_recommendations(libraries, adapter_works)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70 + "\n")

    return 0 if adapter_works else 1


if __name__ == "__main__":
    sys.exit(main())