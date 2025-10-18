"""
BA-LoRA Diagnostic Script
Run this to identify issues in your current implementation

Usage:
    python diagnose_ba_lora.py

Or from project root:
    python -m diagnose_ba_lora
"""

import torch
import numpy as np
from typing import Dict
import os
import sys

# Add src to path if running as standalone script
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Import your BA-LoRA components with correct paths
from ..LoRa.components.huggingface_models.huggingface_model_loader import HuggingFaceModelLoader
from ..LoRa.components.peft.ba_lora_builder import BALoRABuilder
from ..LoRa.components.data_loader.enhanced_data_loader import UnifiedDatasetLoader, DATASET_CONFIGS


def diagnose_ba_lora(use_small_sample=True):
    """
    Run comprehensive diagnostics on BA-LoRA implementation.

    Args:
        use_small_sample: If True, uses small dataset for faster testing
    """

    print("\n" + "=" * 80)
    print("BA-LORA DIAGNOSTIC TEST")
    print("=" * 80)

    # Setup
    print("\n[1/6] Loading model and data...")

    try:
        # Load model
        model_loader = HuggingFaceModelLoader(
            model_name="distilbert-base-uncased",
            num_labels=2
        )
        model, tokenizer = model_loader.load()

        # Load data using UnifiedDatasetLoader (FIXED - removed dataset_config parameter)
        data_loader = UnifiedDatasetLoader(
            dataset_name="sst2",
            max_length=128,
            validation_split=0.1,
            test_split=0.1,
            seed=42
        )

        # Load and prepare datasets (FIXED - correct method name)
        datasets = data_loader.load_and_prepare(tokenizer)
        train_data = datasets["train"]

        # Use small sample for faster testing
        if use_small_sample:
            train_data = train_data.select(range(min(500, len(train_data))))
            print(f"Using small sample: {len(train_data)} examples")

        print("✓ Model and data loaded")

    except Exception as e:
        print(f"❌ Failed to load model/data: {e}")
        print("\nTrying alternative data loading method...")

        try:
            # Alternative: Load from local CSV files
            from ..LoRa.components.data_loader.huggingface_data_loader import LocalCsvDatasetLoader

            # FIXED: Construct correct path to project root
            # Script is in src/main/, need to go up 2 levels to project root
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # If running as module (python -m src.main.diagnose_ba_lora)
            if 'src' in script_dir:
                # Go up from src/main to src, then to project root
                project_root = os.path.dirname(os.path.dirname(script_dir))
            else:
                # Running directly, assume current directory
                project_root = os.getcwd()

            data_files = {
                "train": os.path.join(project_root, "data", "sst2_dataset", "sst2_train.csv"),
            }

            print(f"Looking for CSV file at: {data_files['train']}")

            csv_loader = LocalCsvDatasetLoader(
                data_files=data_files,
                text_column="sentence",
                label_column="label",
                max_length=128
            )

            datasets = csv_loader.load_and_prepare(tokenizer)
            train_data = datasets["train"]

            if use_small_sample:
                train_data = train_data.select(range(min(500, len(train_data))))

            print("✓ Loaded data from local CSV files")

        except Exception as e2:
            print(f"❌ Alternative data loading also failed: {e2}")
            print("\nDEBUG INFO:")
            print(f"  Script directory: {os.path.dirname(os.path.abspath(__file__))}")
            print(f"  Current working directory: {os.getcwd()}")
            print(f"  Looking for file at: {data_files.get('train', 'N/A')}")
            print("\nPlease ensure you have:")
            print("  1. Internet connection for HuggingFace datasets, OR")
            print("  2. Local CSV files in <project_root>/data/sst2_dataset/")
            print("\nTo use local CSVs, your directory should look like:")
            print("  ML_Project/")
            print("  ├── data/")
            print("  │   └── sst2_dataset/")
            print("  │       └── sst2_train.csv")
            print("  └── src/")
            return

    # Build BA-LoRA model
    print("\n[2/6] Building BA-LoRA model...")
    builder = BALoRABuilder(
        model=model,
        tokenizer=tokenizer,
        param_budget=150000
    )

    gradient_samples = 3000 if use_small_sample else 3050
    config = {
        "train_dataset": train_data,
        "base_rank": 8,
        "gradient_samples": gradient_samples,  # Use appropriate sample size
        "use_warmstart": True,
        "target_modules": ["q_lin", "v_lin"],
    }

    try:
        ba_model = builder.build(config)
        print("✓ BA-LoRA model built")
    except Exception as e:
        print(f"❌ Failed to build BA-LoRA model: {e}")
        import traceback
        traceback.print_exc()
        return

    # DIAGNOSTIC 1: Check rank allocation
    print("\n[3/6] DIAGNOSTIC 1: Rank Allocation")
    print("-" * 80)

    if hasattr(builder, 'rank_allocation'):
        rank_alloc = builder.rank_allocation
        rank_values = list(rank_alloc.items())

        print(f"Allocated ranks ({len(rank_alloc)} layers):")
        for name, rank in sorted(rank_values, key=lambda x: x[1]):
            print(f"  {name:40s}: rank = {rank:2d}")

        unique_ranks = set(rank_alloc.values())
        print(f"\nRank statistics:")
        print(f"  Unique ranks: {len(unique_ranks)}")
        print(f"  Min rank: {min(rank_alloc.values())}")
        print(f"  Max rank: {max(rank_alloc.values())}")
        print(f"  Mean rank: {np.mean(list(rank_alloc.values())):.1f}")
        print(f"  Std rank: {np.std(list(rank_alloc.values())):.1f}")

        if len(unique_ranks) <= 2:
            print("\n❌ ISSUE DETECTED: Very few unique ranks allocated!")
            print("   Expected: Significant variation across layers")
            print("   This suggests rank allocation may not be working properly")
        else:
            print("\n✓ Rank allocation shows variation")
    else:
        print("❌ CRITICAL: No rank_allocation attribute found in builder!")

    # DIAGNOSTIC 2: Check applied ranks
    print("\n[4/6] DIAGNOSTIC 2: Applied Ranks in Model")
    print("-" * 80)

    applied_ranks = {}
    for name, module in ba_model.named_modules():
        if hasattr(module, 'r'):
            r_value = module.r
            print(f"  {name}: r = {r_value}")

            # Extract actual rank value (handle dict case)
            if isinstance(r_value, dict):
                actual_rank = r_value.get('default', list(r_value.values())[0] if r_value else 4)
            else:
                actual_rank = r_value

            applied_ranks[name] = actual_rank

    if applied_ranks:
        unique_applied = set(applied_ranks.values())  # ✅ Now works
        print(f"\nApplied ranks summary:")
        print(f"  Unique ranks: {sorted(unique_applied)}")
        print(f"  Total modules: {len(applied_ranks)}")

        # Check if matches allocation
        expected_ranks = set(builder.rank_allocation.values())
        if unique_applied == expected_ranks:
            print(f"  ✓ Applied ranks match allocation")
        else:
            print(f"  ⚠️  Applied ranks {unique_applied} != Expected {expected_ranks}")
    else:
        print("  ❌ No LoRA modules found!")

    # DIAGNOSTIC 3: Check parameter budget
    print("\n[5/6] DIAGNOSTIC 3: Parameter Budget")
    print("-" * 80)

    trainable_params = sum(p.numel() for p in ba_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in ba_model.parameters())
    budget = 150000

    print(f"Parameter counts:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Target budget: {budget:,}")
    print(f"  Budget usage: {trainable_params / budget:.1%}")

    if trainable_params > budget * 5:
        print("\n❌ CRITICAL ISSUE: Using MUCH more parameters than budget!")
        print(f"   Using {trainable_params / budget:.1f}x the budget")
        print("\n   DIAGNOSIS: Issue #3 - Parameter counting may be incorrect")
    elif trainable_params > budget * 1.2:
        print("\n⚠️  WARNING: Exceeding budget by >20%")
    elif trainable_params < budget * 0.8:
        print("\n⚠️  WARNING: Under-utilizing budget by >20%")
    else:
        print("\n✓ Parameter usage within acceptable range")

    lora_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'lora' in n.lower())
    print(f"Parameter counts:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")  # Keep for info
    print(f"  LoRA parameters: {lora_params:,}")  # New
    print(f"  Target budget (LoRA): {budget:,}")
    budget_usage = (lora_params / budget) * 100 if budget else 0
    print(f"  LoRA budget usage: {budget_usage:.1f}%")
    if budget_usage > 110 or budget_usage < 90:
        print("❌ ISSUE: LoRA budget mismatch")
    else:
        print("✓ LoRA budget OK")

    # DIAGNOSTIC 4: Check warm-start
    print("\n[6/6] DIAGNOSTIC 4: Warm-Start Initialization")
    print("-" * 80)

    lora_params = {}
    for name, param in ba_model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_params[name] = param

    if not lora_params:
        print("❌ CRITICAL: No LoRA parameters found!")
    else:
        print(f"Found {len(lora_params)} LoRA parameters")

        # Analyze parameter statistics
        zero_initialized = []
        warmstart_likely = []

        for name, param in lora_params.items():
            mean_val = param.data.mean().item()
            std_val = param.data.std().item()
            abs_mean = abs(mean_val)

            print(f"\n  {name}:")
            print(f"    Mean: {mean_val:+.6f}")
            print(f"    Std: {std_val:.6f}")

            # Check if looks like random initialization
            if abs_mean < 1e-6 and std_val < 0.02:
                zero_initialized.append(name)
                print(f"    Status: Appears zero-initialized (random)")
            elif abs_mean > 1e-4 or std_val > 0.05:
                warmstart_likely.append(name)
                print(f"    Status: Appears warm-started")
            else:
                print(f"    Status: Unclear")

        # Summary
        print(f"\nWarm-start analysis:")
        print(f"  Total LoRA params: {len(lora_params)}")
        print(f"  Likely random init: {len(zero_initialized)}")
        print(f"  Likely warm-started: {len(warmstart_likely)}")

        if len(zero_initialized) > len(lora_params) * 0.5:
            print("\n❌ ISSUE DETECTED: Most parameters appear randomly initialized!")
            print("   Expected: Warm-start weights with non-zero values")
            print("\n   DIAGNOSIS: Issue #2 - Warm-start weights not being applied")
        elif len(warmstart_likely) > len(lora_params) * 0.5:
            print("\n✓ Most parameters appear to be warm-started")
        else:
            print("\n⚠️  WARNING: Mixed initialization detected")

    # Final summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    issues_found = []

    # Check Issue #1
    if applied_ranks and len(set(applied_ranks.values())) == 1:
        issues_found.append({
            'id': 1,
            'severity': 'CRITICAL',
            'description': 'All layers have same rank - adaptive allocation not working',
            'impact': 'HIGH - Primary cause of underperformance'
        })

    # Check Issue #2
    if lora_params and len(zero_initialized) > len(lora_params) * 0.5:
        issues_found.append({
            'id': 2,
            'severity': 'CRITICAL',
            'description': 'Warm-start initialization not applied',
            'impact': 'MEDIUM - Missing ~20% of expected improvement'
        })

    # Check Issue #3
    if trainable_params > budget * 5:
        issues_found.append({
            'id': 3,
            'severity': 'MAJOR',
            'description': 'Parameter budget significantly exceeded',
            'impact': 'HIGH - Unfair comparison with baseline'
        })

    if issues_found:
        print(f"\n❌ Found {len(issues_found)} critical issue(s):\n")
        for issue in issues_found:
            print(f"Issue #{issue['id']} [{issue['severity']}]:")
            print(f"  Description: {issue['description']}")
            print(f"  Impact: {issue['impact']}")
            print()

        print("These issues explain why BA-LoRA is underperforming.")
        print("See the diagnostic report for detailed fixes.")
    else:
        print("\n✓ No critical issues detected!")
        print("If BA-LoRA is still underperforming, consider:")
        print("  - Increasing gradient samples (try 3000-5000)")
        print("  - Adjusting importance normalization range")
        print("  - Checking hyperparameters (learning rate, epochs)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose BA-LoRA implementation issues")
    parser.add_argument("--full", action="store_true",
                        help="Use full dataset instead of small sample (slower)")
    parser.add_argument("--gradient-samples", type=int, default=100,
                        help="Number of gradient samples for testing (default: 100)")

    args = parser.parse_args()

    try:
        diagnose_ba_lora(use_small_sample=not args.full)
    except Exception as e:
        print(f"\n❌ Diagnostic failed with error:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

        print("\n" + "=" * 80)
        print("TROUBLESHOOTING TIPS:")
        print("=" * 80)
        print("\n1. Make sure you're running from the project root directory")
        print("2. Verify all dependencies are installed:")
        print("   pip install torch transformers datasets peft")
        print("\n3. Check that the following files exist:")
        print("   - src/LoRa/components/huggingface_models/huggingface_model_loader.py")
        print("   - src/LoRa/components/peft/ba_lora_builder.py")
        print("   - src/LoRa/components/data_loader/enhanced_data_loader.py")
        print("\n4. Try running with --full flag for complete testing")
        print("\n5. If using local CSV data, ensure files are in data/sst2_dataset/")
        sys.exit(1)