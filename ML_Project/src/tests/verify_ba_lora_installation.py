"""
ML_Project/verify_ba_lora_installation.py

Quick verification script to test BA-LoRA implementation.
Run this after implementing all BA-LoRA components.

Usage:
    python verify_ba_lora_installation.py
"""

import sys
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def test_imports():
    """Test that all BA-LoRA modules can be imported."""
    print("=" * 70)
    print("TEST 1: Checking imports")
    print("=" * 70)

    try:
        from src.LoRa.components.peft.gradient_analyzer import GradientAnalyzer
        print("✓ GradientAnalyzer imported")
    except Exception as e:
        print(f"✗ Failed to import GradientAnalyzer: {e}")
        return False

    try:
        from src.LoRa.components.peft.rank_allocator import RankAllocator
        print("✓ RankAllocator imported")
    except Exception as e:
        print(f"✗ Failed to import RankAllocator: {e}")
        return False

    try:
        from src.LoRa.components.peft.ba_lora_builder import BALoRABuilder
        print("✓ BALoRABuilder imported")
    except Exception as e:
        print(f"✗ Failed to import BALoRABuilder: {e}")
        return False

    try:
        from src.LoRa.components.peft.peft_factory import PEFTFactory
        print("✓ PEFTFactory imported")

        # Check BA-LoRA is in supported methods
        if "ba_lora" in PEFTFactory.SUPPORTED_METHODS:
            print("✓ BA-LoRA is registered in PEFTFactory")
        else:
            print("✗ BA-LoRA not in PEFTFactory.SUPPORTED_METHODS")
            return False
    except Exception as e:
        print(f"✗ Failed to import PEFTFactory: {e}")
        return False

    print("\n✓ All imports successful!\n")
    return True


def test_gradient_analyzer():
    """Test GradientAnalyzer with tiny model and dataset."""
    print("=" * 70)
    print("TEST 2: Testing GradientAnalyzer")
    print("=" * 70)

    try:
        from src.LoRa.components.peft.gradient_analyzer import GradientAnalyzer

        # Load tiny model
        print("Loading model and tokenizer...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
        )
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Create tiny dataset
        print("Creating tiny dataset...")
        texts = ["This is positive.", "This is negative."] * 5
        labels = [1, 0] * 5

        dataset = Dataset.from_dict({
            "sentence": texts,
            "label": labels
        })

        # Tokenize
        def tokenize(examples):
            return tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True,
                max_length=128
            )

        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # Test gradient analyzer
        print("\nTesting GradientAnalyzer...")
        analyzer = GradientAnalyzer(
            model=model,
            tokenizer=tokenizer,
            target_modules=["q_lin", "v_lin"]
        )

        # Accumulate gradients
        gradients = analyzer.accumulate_gradients(
            train_dataset=dataset,
            num_samples=10,
            batch_size=2
        )

        assert len(gradients) > 0, "No gradients accumulated"
        print(f"✓ Accumulated {len(gradients)} gradients")

        # Compute importance
        importance = analyzer.compute_importance_scores()
        assert len(importance) > 0, "No importance scores computed"
        print(f"✓ Computed {len(importance)} importance scores")

        # Get layer importance
        layer_importance = analyzer.get_layer_importance()
        assert len(layer_importance) > 0, "No layer importance computed"
        print(f"✓ Computed {len(layer_importance)} layer importances")

        print("\n✓ GradientAnalyzer working correctly!\n")
        return True, layer_importance

    except Exception as e:
        print(f"\n✗ GradientAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def test_rank_allocator(importance_scores):
    """Test RankAllocator."""
    print("=" * 70)
    print("TEST 3: Testing RankAllocator")
    print("=" * 70)

    try:
        from src.LoRa.components.peft.rank_allocator import RankAllocator

        # Test with sample importance scores
        if not importance_scores:
            importance_scores = {
                f"layer_{i}_q_lin": 0.001 * (i + 1)
                for i in range(6)
            }
            print("Using dummy importance scores for testing")

        print(f"\nTesting RankAllocator with {len(importance_scores)} layers...")
        allocator = RankAllocator(
            importance_scores=importance_scores,
            param_budget=75000,
            base_rank=8,
            hidden_dim=768
        )

        # Allocate ranks
        rank_allocation = allocator.allocate_ranks()

        assert len(rank_allocation) > 0, "No ranks allocated"
        print(f"✓ Allocated ranks to {len(rank_allocation)} layers")

        # Check budget
        stats = allocator.get_allocation_stats()
        assert stats['total_params'] <= 75000, "Budget exceeded"
        print(f"✓ Budget respected: {stats['total_params']:,} ≤ 75,000")

        # Check rank variation
        unique_ranks = set(rank_allocation.values())
        if len(unique_ranks) > 1:
            print(f"✓ Ranks vary across layers: {min(unique_ranks)} to {max(unique_ranks)}")
        else:
            print(f"⚠ All ranks are the same: {list(unique_ranks)[0]}")

        print("\n✓ RankAllocator working correctly!\n")
        return True

    except Exception as e:
        print(f"\n✗ RankAllocator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test complete BA-LoRA pipeline."""
    print("=" * 70)
    print("TEST 4: Testing Full BA-LoRA Pipeline")
    print("=" * 70)

    try:
        from src.LoRa.components.peft.peft_factory import PEFTFactory

        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
        )
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Create tiny dataset
        print("Creating tiny dataset...")
        texts = ["This is positive.", "This is negative."] * 5
        labels = [1, 0] * 5

        dataset = Dataset.from_dict({
            "sentence": texts,
            "label": labels
        })

        # Tokenize
        def tokenize(examples):
            return tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True,
                max_length=128
            )

        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # Build BA-LoRA model
        print("\nBuilding BA-LoRA model...")
        factory = PEFTFactory(
            base_model=model,
            tokenizer=tokenizer,
            target_param_budget=75000
        )

        config = {
            "train_dataset": dataset,
            "base_rank": 4,
            "gradient_samples": 10,
            "use_warmstart": False,
            "target_modules": ["q_lin", "v_lin"],
            "lora_alpha": 8,
            "lora_dropout": 0.1,
        }

        ba_lora_model = factory.build("ba_lora", config)

        print("✓ BA-LoRA model built successfully!")

        # Verify it's trainable
        total_params = sum(p.numel() for p in ba_lora_model.parameters())
        trainable_params = sum(p.numel() for p in ba_lora_model.parameters() if p.requires_grad)

        print(f"\nModel statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable %: {trainable_params / total_params * 100:.2f}%")

        assert trainable_params > 0, "No trainable parameters!"
        assert trainable_params < total_params, "All parameters are trainable!"

        print("\n✓ Full BA-LoRA pipeline working correctly!\n")
        return True

    except Exception as e:
        print(f"\n✗ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "#" * 70)
    print("# BA-LORA INSTALLATION VERIFICATION")
    print("#" * 70 + "\n")

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: GradientAnalyzer
    success, importance = test_gradient_analyzer()
    results.append(("GradientAnalyzer", success))

    # Test 3: RankAllocator
    results.append(("RankAllocator", test_rank_allocator(importance)))

    # Test 4: Full pipeline
    results.append(("Full Pipeline", test_full_pipeline()))

    # Summary
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")

    all_passed = all(result[1] for result in results)

    print("=" * 70)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYour BA-LoRA implementation is ready to use.")
        print("\nNext steps:")
        print("  1. Run baseline LoRA experiments")
        print("  2. Run BA-LoRA experiments")
        print("  3. Compare results")
        print("\nExample command:")
        print("  python -m src.main.improved_experiment_runner \\")
        print("    --experiment_name ba_lora_test \\")
        print("    --dataset sst2 \\")
        print("    --peft_method ba_lora \\")
        print("    --param_budget 75000 \\")
        print("    --ba_lora_base_rank 8 \\")
        print("    --ba_lora_gradient_samples 1000 \\")
        print("    --ba_lora_use_warmstart \\")
        print("    --epochs 3")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease check the error messages above and fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())