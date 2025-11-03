"""
Enhanced BA-LoRA Diagnostic Script with Quality Checks

Usage:
    python -m src.main.diagnose_ba_lora_fixed
"""

import torch
import numpy as np
from typing import Dict
import os
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from ..LoRa.components.huggingface_models.huggingface_model_loader import HuggingFaceModelLoader
from ..LoRa.components.peft.ba_lora_builder import BALoRABuilder
from ..LoRa.components.data_loader.enhanced_data_loader import UnifiedDatasetLoader


def diagnose_ba_lora_fixed(use_small_sample=True, gradient_samples=3000):
    """
    Enhanced diagnostic with warm-start quality checks.

    Args:
        use_small_sample: Use small dataset for faster testing
        gradient_samples: Number of samples for gradient accumulation
    """

    print("\n" + "=" * 80)
    print("BA-LORA DIAGNOSTIC TEST (FIXED VERSION)")
    print("=" * 80)

    # Setup
    print("\n[1/7] Loading model and data...")

    try:
        model_loader = HuggingFaceModelLoader(
            model_name="roberta-base",
            num_labels=2
        )
        model, tokenizer = model_loader.load()

        data_loader = UnifiedDatasetLoader(
            dataset_name="sst2",
            max_length=128,
            validation_split=0.1,
            test_split=0.1,
            seed=42
        )

        datasets = data_loader.load_and_prepare(tokenizer)
        train_data = datasets["train"]

        if use_small_sample:
            sample_size = min(gradient_samples, len(train_data))
            train_data = train_data.select(range(sample_size))
            print(f"Using sample: {len(train_data)} examples")

        print("✓ Model and data loaded")

    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return

    # Build BA-LoRA model
    print("\n[2/7] Building BA-LoRA model with FIXED warm-start...")
    builder = BALoRABuilder(
        model=model,
        tokenizer=tokenizer,
        param_budget=150000
    )

    config = {
        "train_dataset": train_data,
        "base_rank": 8,
        "gradient_samples": len(train_data),  # Use actual dataset size
        "use_warmstart": True,
    }

    try:
        ba_model = builder.build(config)
        print("✓ BA-LoRA model built")
    except Exception as e:
        print(f"❌ Failed to build: {e}")
        import traceback
        traceback.print_exc()
        return

    # DIAGNOSTIC 1: Warm-start quality analysis
    print("\n[3/7] DIAGNOSTIC 1: Warm-Start Quality Analysis")
    print("-" * 80)

    if hasattr(builder.gradient_analyzer, 'gradients'):
        gradients = builder.gradient_analyzer.gradients
        print(f"\nGradient Statistics:")

        for name, grad in list(gradients.items())[:3]:
            grad_norm = torch.norm(grad).item()
            grad_mean = grad.abs().mean().item()
            grad_std = grad.std().item()
            print(f"  {name}:")
            print(f"    Norm: {grad_norm:.6f}")
            print(f"    Mean(|.|): {grad_mean:.6f}")
            print(f"    Std: {grad_std:.6f}")

        # Check if gradients are too small
        all_norms = [torch.norm(g).item() for g in gradients.values()]
        avg_norm = np.mean(all_norms)

        if avg_norm < 1e-5:
            print(f"\n⚠️  WARNING: Very small gradient magnitudes (avg={avg_norm:.2e})")
            print(f"   This may indicate:")
            print(f"   - Model is already well-initialized")
            print(f"   - Loss function has small gradients")
            print(f"   - Need more training samples")
        else:
            print(f"\n✓ Gradient magnitudes look reasonable (avg={avg_norm:.2e})")

    # DIAGNOSTIC 2: Rank allocation
    print("\n[4/7] DIAGNOSTIC 2: Rank Allocation")
    print("-" * 80)

    if hasattr(builder, 'rank_allocation'):
        rank_alloc = builder.rank_allocation
        rank_values = list(rank_alloc.values())

        print(f"\nAllocated ranks ({len(rank_alloc)} layers):")
        for name, rank in sorted(rank_alloc.items(), key=lambda x: x[1]):
            importance = builder.rank_allocator.importance_scores.get(name, 0)
            print(f"  {name:30s}: rank={rank:2d}, importance={importance:.6f}")

        print(f"\nStatistics:")
        print(f"  Unique ranks: {len(set(rank_values))}")
        print(f"  Range: [{min(rank_values)}, {max(rank_values)}]")
        print(f"  Mean: {np.mean(rank_values):.1f}")
        print(f"  Std: {np.std(rank_values):.1f}")

        if len(set(rank_values)) <= 2:
            print("\n❌ ISSUE: Very few unique ranks allocated!")
        else:
            print("\n✓ Good rank variation")

    # DIAGNOSTIC 3: Warm-start approximation quality
    print("\n[5/7] DIAGNOSTIC 3: Warm-Start Approximation Quality")
    print("-" * 80)

    # The builder now stores quality metrics in warmstart_weights
    # We can analyze them here
    print("\nChecking approximation errors from build phase...")
    print("(See Phase 3 output above for detailed quality metrics)")

    # DIAGNOSTIC 4: Applied ranks
    print("\n[6/7] DIAGNOSTIC 4: Applied Ranks in Model")
    print("-" * 80)

    applied_ranks = {}
    for name, module in ba_model.named_modules():
        if hasattr(module, 'r'):
            r_value = module.r
            if isinstance(r_value, dict):
                actual_rank = r_value.get('default', list(r_value.values())[0] if r_value else 4)
            else:
                actual_rank = r_value
            applied_ranks[name] = actual_rank

    if applied_ranks:
        unique_applied = set(applied_ranks.values())
        print(f"\nApplied ranks summary:")
        print(f"  Unique ranks: {sorted(unique_applied)}")
        print(f"  Total modules: {len(applied_ranks)}")

        expected_ranks = set(builder.rank_allocation.values())
        if unique_applied == expected_ranks:
            print(f"  ✓ Applied ranks match allocation perfectly")
        else:
            print(f"  ⚠️  Mismatch: applied={unique_applied}, expected={expected_ranks}")

    # DIAGNOSTIC 5: Parameter budget
    print("\n[7/7] DIAGNOSTIC 5: Parameter Budget")
    print("-" * 80)

    lora_params = sum(p.numel() for n, p in ba_model.named_parameters()
                      if p.requires_grad and 'lora' in n.lower())
    total_trainable = sum(p.numel() for p in ba_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in ba_model.parameters())
    budget = 150000

    print(f"\nParameter counts:")
    print(f"  Total parameters:     {total_params:>12,}")
    print(f"  LoRA parameters:      {lora_params:>12,}")
    print(f"  Other trainable:      {total_trainable - lora_params:>12,}")
    print(f"  Total trainable:      {total_trainable:>12,}")
    print(f"\nBudget analysis:")
    print(f"  Target budget:        {budget:>12,}")
    print(f"  LoRA usage:           {lora_params:>12,}")
    print(f"  LoRA usage %:         {lora_params/budget*100:>11.1f}%")

    if 90 <= lora_params/budget*100 <= 110:
        print(f"  ✓ LoRA budget usage is good")
    elif lora_params > budget * 1.1:
        print(f"  ⚠️  WARNING: Exceeding budget by {(lora_params/budget - 1)*100:.1f}%")
    else:
        print(f"  ⚠️  WARNING: Under-utilizing budget by {(1 - lora_params/budget)*100:.1f}%")

    # Final summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    issues_found = []

    # Check warm-start quality
    # Note: The actual quality metrics are printed during build phase
    # This is just a summary check

    # Check rank diversity
    if applied_ranks and len(set(applied_ranks.values())) <= 2:
        issues_found.append({
            'severity': 'CRITICAL',
            'description': 'Very few unique ranks - adaptive allocation may not be working',
            'fix': 'Check importance score calculation and rank allocation logic'
        })

    # Check parameter budget
    if lora_params > budget * 1.2:
        issues_found.append({
            'severity': 'MAJOR',
            'description': f'Parameter budget exceeded by {(lora_params/budget - 1)*100:.1f}%',
            'fix': 'Reduce base_rank or adjust budget constraint'
        })

    if issues_found:
        print(f"\n❌ Found {len(issues_found)} issue(s):\n")
        for i, issue in enumerate(issues_found, 1):
            print(f"Issue {i} [{issue['severity']}]:")
            print(f"  Description: {issue['description']}")
            print(f"  Fix: {issue['fix']}")
            print()
    else:
        print("\n✓ No critical issues detected!")
        print("\nNext steps:")
        print("  1. Check warm-start quality metrics from Phase 3 output")
        print("  2. If relative errors < 0.5, warm-start is working well")
        print("  3. Run full experiment and compare with/without warm-start")

    print("\n" + "=" * 80)
    print("\nTo run full experiment:")
    print("\n  # With fixed warm-start")
    print("  python -m src.main.improved_experiment_runner \\")
    print("    --experiment_name ba_lora_fixed_ws \\")
    print("    --dataset sst2 \\")
    print("    --peft_method ba_lora \\")
    print("    --param_budget 150000 \\")
    print("    --ba_lora_base_rank 8 \\")
    print("    --ba_lora_gradient_samples 5000 \\")
    print("    --ba_lora_use_warmstart \\")
    print("    --epochs 3")
    print("\n  # Without warm-start (for comparison)")
    print("  python -m src.main.improved_experiment_runner \\")
    print("    --experiment_name ba_lora_no_ws \\")
    print("    --dataset sst2 \\")
    print("    --peft_method ba_lora \\")
    print("    --param_budget 150000 \\")
    print("    --ba_lora_base_rank 8 \\")
    print("    --ba_lora_gradient_samples 5000 \\")
    print("    --epochs 3")
    print("\n" + "=" * 80)


def compare_warmstart_methods():
    """
    Helper function to compare old vs new warm-start initialization.
    """
    print("\n" + "=" * 80)
    print("WARM-START METHOD COMPARISON")
    print("=" * 80)

    # Create a synthetic gradient for testing
    torch.manual_seed(42)
    d, k, r = 768, 768, 8
    G = torch.randn(d, k) * 0.001  # Typical gradient scale

    print(f"\nTest setup:")
    print(f"  Gradient shape: {G.shape}")
    print(f"  Rank: {r}")
    print(f"  Gradient norm: {torch.norm(G):.6f}")

    # Method 1: Old method (Kaiming)
    print(f"\n--- Method 1: Kaiming Init (OLD) ---")
    A_old = torch.empty((r, k))
    torch.nn.init.kaiming_uniform_(A_old, a=np.sqrt(5))

    eps = 1e-6
    AAt_old = torch.matmul(A_old, A_old.T) + torch.eye(r) * eps
    AAt_inv_old = torch.inverse(AAt_old)
    GA_old = torch.matmul(G, A_old.T)
    B_old = -torch.matmul(GA_old, AAt_inv_old)

    AB_old = torch.matmul(B_old, A_old)
    error_old = torch.norm(AB_old + G) / torch.norm(G)

    print(f"  Relative error: {error_old:.4f}")

    # Method 2: New method (SVD)
    print(f"\n--- Method 2: SVD Init (NEW) ---")
    U, S, Vt = torch.svd(G)
    A_new = Vt[:r, :].clone()
    scale = torch.sqrt(S[:r].mean())
    A_new = A_new * scale

    grad_mean = G.abs().mean()
    eps_adaptive = max(1e-6, grad_mean.item() * 0.01)

    AAt_new = torch.matmul(A_new, A_new.T) + torch.eye(r) * eps_adaptive
    L = torch.linalg.cholesky(AAt_new)
    AAt_inv_new = torch.cholesky_inverse(L)
    GA_new = torch.matmul(G, A_new.T)
    B_new = -torch.matmul(GA_new, AAt_inv_new)

    AB_new = torch.matmul(B_new, A_new)
    error_new = torch.norm(AB_new + G) / torch.norm(G)

    print(f"  Relative error: {error_new:.4f}")

    # Comparison
    print(f"\n--- Comparison ---")
    print(f"  Old method error: {error_old:.4f}")
    print(f"  New method error: {error_new:.4f}")
    print(f"  Improvement: {(error_old - error_new)/error_old * 100:.1f}%")

    if error_new < error_old * 0.5:
        print(f"  ✓ NEW method is significantly better!")
    elif error_new < error_old:
        print(f"  ✓ NEW method is better")
    else:
        print(f"  ⚠️  WARNING: NEW method not better (may need tuning)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced BA-LoRA diagnostics")
    parser.add_argument("--full", action="store_true",
                        help="Use full dataset (slower)")
    parser.add_argument("--gradient-samples", type=int, default=3000,
                        help="Number of gradient samples")
    parser.add_argument("--compare-methods", action="store_true",
                        help="Compare old vs new warm-start methods")

    args = parser.parse_args()

    if args.compare_methods:
        compare_warmstart_methods()
    else:
        try:
            diagnose_ba_lora_fixed(
                use_small_sample=not args.full,
                gradient_samples=args.gradient_samples
            )
        except Exception as e:
            print(f"\n❌ Diagnostic failed:")
            print(f"   {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)