"""
ML_Project/src/LoRa/components/peft/ba_lora_builder.py

BA-LoRA Builder: Orchestrates all phases of Budget-Aware Adaptive LoRA
"""

from typing import Dict, Optional
import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

from .gradient_analyzer import GradientAnalyzer
from .rank_allocator import RankAllocator
from .lora_builder import LoRABuilder


class BALoRABuilder(LoRABuilder):
    """
    Budget-Aware Adaptive LoRA Builder.

    Extends LoRABuilder to support adaptive rank allocation.
    """

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            param_budget: Optional[int] = None
    ):
        """
        Initialize BA-LoRA builder.

        Args:
            model: Base transformer model
            tokenizer: Tokenizer
            param_budget: Maximum trainable parameters
        """
        super().__init__(model)
        self.tokenizer = tokenizer
        self.param_budget = param_budget

        self.gradient_analyzer = None
        self.rank_allocator = None
        self.rank_allocation = {}

        print(f"BALoRABuilder initialized")
        if param_budget:
            print(f"  Parameter budget: {param_budget:,}")

    def build(self, config: Dict) -> PreTrainedModel:
        """Build BA-LoRA model with adaptive rank allocation."""
        print(f"\n{'=' * 70}")
        print("BUILDING BA-LORA MODEL (FIXED VERSION)")
        print(f"{'=' * 70}")

        # FREEZE classifier/head parameters to respect budget
        print("\nFreezing classifier head parameters...")
        frozen_params = 0
        freeze_head = config.get("freeze_head", True)  # New: Configurable, default True for strict budget
        if freeze_head:
            print("\nFreezing classifier head parameters...")
            frozen_params = 0
            for name, param in self.model.named_parameters():
                if 'classifier' in name or 'pre_classifier' in name or 'head' in name:
                    param.requires_grad = False
                    frozen_params += param.numel()
                    print(f"  Frozen: {name}")
            print(f"  Total frozen: {frozen_params:,} parameters\n")
        else:
            print("\nKeeping classifier head trainable (note: increases total trainable params beyond LoRA budget)")
        print(f"  Total frozen: {frozen_params:,} parameters\n")

        # Extract configuration
        train_dataset = config.get("train_dataset")
        if train_dataset is None:
            raise ValueError("BA-LoRA requires 'train_dataset' for gradient analysis")

        base_rank = config.get("base_rank", 8)
        gradient_samples = config.get("gradient_samples", 3000)  # INCREASED from 1000
        use_warmstart = config.get("use_warmstart", True)
        target_modules = config.get("target_modules") or self._get_default_target_modules()
        lora_alpha = config.get("lora_alpha", 2 * base_rank)
        lora_dropout = config.get("lora_dropout", 0.1)

        print(f"\nConfiguration:")
        print(f"  Base rank: {base_rank}")
        print(f"  Gradient samples: {gradient_samples}")
        print(f"  Warm-start: {use_warmstart}")
        print(f"  Target modules: {target_modules}")
        print(f"  LoRA alpha: {lora_alpha}")
        print(f"  LoRA dropout: {lora_dropout}")

        # Phase 1: Estimate importance
        print(f"\n[Phase 1/4] Estimating layer importance...")
        importance_scores = self._estimate_importance(
            train_dataset, gradient_samples, target_modules
        )

        # Phase 2: Allocate ranks
        print(f"\n[Phase 2/4] Allocating ranks with budget constraint...")
        rank_allocation = self._allocate_ranks(
            importance_scores, base_rank, target_modules,
            min_rank=config.get("min_rank", 2),  # New
            max_rank=config.get("max_rank", 64)  # New
        )

        # Phase 3: Compute warm-start weights (but don't apply yet)
        warmstart_weights = None
        if use_warmstart:
            print(f"\n[Phase 3/4] Computing warm-start initialization...")
            warmstart_weights = self._compute_warmstart_weights(
                rank_allocation, target_modules
            )
        else:
            print(f"\n[Phase 3/4] Skipping warm-start (disabled)")

        # Phase 4: Apply LoRA with FIXED adaptive allocation
        print(f"\n[Phase 4/4] Applying LoRA with adaptive ranks...")
        peft_model = self._apply_adaptive_lora(
            rank_allocation,
            lora_alpha,
            lora_dropout,
            target_modules
        )

        # CRITICAL: Apply warm-start weights AFTER LoRA layers exist
        if use_warmstart and warmstart_weights is not None:
            print(f"\n[Post-Build] Injecting warm-start weights...")
            injection_success = self._inject_warmstart_weights(
                peft_model, warmstart_weights
            )

            if injection_success:
                print("  ✓ Warm-start weights successfully injected")
            else:
                print("  ⚠️  WARNING: Warm-start injection failed!")

        # Print final statistics and verification
        self._print_final_stats(peft_model)
        self._verify_model(peft_model, rank_allocation)

        return peft_model


    def _apply_warmstart(
            self,
            rank_allocation: Dict[str, int],
            target_modules: list
    ) -> None:
        """
        Phase 3: Warm-start initialization using accumulated gradients.

        Initializes LoRA matrices to approximate negative gradient direction:
        - A: Random initialization (Gaussian)
        - B: Pseudo-inverse approximation B = -G @ A.T @ (A @ A.T + εI)^(-1)

        This provides better starting point than random initialization.

        Args:
            rank_allocation: Dictionary mapping layer names to allocated ranks
            target_modules: List of target module names
        """
        if not hasattr(self.gradient_analyzer, 'gradients') or not self.gradient_analyzer.gradients:
            print("  WARNING: No gradients available for warm-start")
            print("  Falling back to standard initialization")
            return

        print("  Applying warm-start initialization...")

        # Regularization parameter for numerical stability
        eps = 1e-6

        initialized_count = 0
        skipped_count = 0

        # Iterate through all parameters in the model
        for name, param in self.model.named_parameters():
            # Check if this parameter is in our rank allocation
            if not any(target in name for target in target_modules):
                continue

            # Find matching gradient
            matching_grad = None
            for grad_name, grad_tensor in self.gradient_analyzer.gradients.items():
                if name in grad_name or grad_name in name:
                    matching_grad = grad_tensor
                    break

            if matching_grad is None:
                skipped_count += 1
                continue

            # Get allocated rank for this layer
            layer_rank = None
            for layer_name, rank in rank_allocation.items():
                if layer_name in name:
                    layer_rank = rank
                    break

            if layer_rank is None:
                skipped_count += 1
                continue

            # Get gradient shape [d, k] where d=output_dim, k=input_dim
            G = matching_grad.to(param.device)
            d, k = G.shape

            # Ensure rank is valid
            r = min(layer_rank, min(d, k))

            # Initialize A matrix randomly: A ∈ R^(r, k)
            # Using small random values (scaled by 0.01 for stability)
            A = torch.randn(r, k, device=param.device, dtype=param.dtype) * 0.01

            # Initialize B matrix using simplified pseudo-inverse: B ∈ R^(d, r)
            # Goal: AB ≈ -G (negative gradient direction for descent)
            # Formula: B = -G @ A.T @ (A @ A.T + εI)^(-1)

            # Compute A @ A.T + εI for regularization
            AAt = torch.matmul(A, A.T)  # Shape: [r, r]
            AAt_reg = AAt + torch.eye(r, device=param.device, dtype=param.dtype) * eps

            # Compute inverse (more stable than pseudo-inverse for small matrices)
            try:
                AAt_inv = torch.inverse(AAt_reg)  # Shape: [r, r]
            except RuntimeError:
                # If inverse fails, use pseudo-inverse as fallback
                print(f"    WARNING: Standard inverse failed for {name}, using pinv")
                AAt_inv = torch.pinverse(AAt_reg)

            # Compute B = -G @ A.T @ (A @ A.T + εI)^(-1)
            # Step 1: G @ A.T → [d, k] @ [k, r] = [d, r]
            GA = torch.matmul(G, A.T)

            # Step 2: (G @ A.T) @ (A @ A.T + εI)^(-1) → [d, r] @ [r, r] = [d, r]
            B = -torch.matmul(GA, AAt_inv)

            # Verify shapes
            assert A.shape == (r, k), f"A shape mismatch: {A.shape} vs ({r}, {k})"
            assert B.shape == (d, r), f"B shape mismatch: {B.shape} vs ({d}, {r})"

            # Store initialized matrices for later use
            # These will be injected into the model when LoRA layers are created
            if not hasattr(self, 'warmstart_init'):
                self.warmstart_init = {}

            self.warmstart_init[name] = {
                'A': A.detach().cpu(),
                'B': B.detach().cpu(),
                'rank': r
            }

            initialized_count += 1

            # Optional: Verify approximation quality
            if initialized_count <= 3:  # Only log first few for brevity
                AB_approx = torch.matmul(B, A)
                approx_error = torch.norm(AB_approx + G) / torch.norm(G)
                print(f"    {name}: rank={r}, approx_error={approx_error:.4f}")

        print(f"  ✓ Warm-start initialization complete")
        print(f"    Initialized: {initialized_count} layers")
        print(f"    Skipped: {skipped_count} layers")

        if initialized_count == 0:
            print("  WARNING: No layers were initialized with warm-start")
            print("  Model will use standard random initialization")

    def _compute_warmstart_weights(self, rank_allocation: Dict[str, int], target_modules: list) -> Dict[str, Dict]:
        if not hasattr(self.gradient_analyzer, 'gradients'):
            print("  WARNING: No gradients available")
            return None

        print("  Computing warm-start weights...")
        warmstart_weights = {}
        eps = 1e-6

        print(f"  Rank allocation keys: {list(rank_allocation.keys())[:3]}...")
        print(f"  Gradient keys: {list(self.gradient_analyzer.gradients.keys())[:3]}...")

        for layer_name, rank in rank_allocation.items():
            matching_grads = []
            for grad_name, grad_tensor in self.gradient_analyzer.gradients.items():
                if 'layer' in layer_name and 'layer' in grad_name:
                    try:
                        layer_num_alloc = layer_name.split('_')[1]
                        if f"layer.{layer_num_alloc}." in grad_name:
                            if grad_tensor.dim() == 2 and any(tm in grad_name for tm in target_modules):
                                matching_grads.append((grad_name, grad_tensor))
                    except (IndexError, ValueError):
                        continue

            if not matching_grads:
                print(f"  ⚠️  No gradients found for {layer_name}")
                continue

            # New: Loop over ALL matching (q_lin and v_lin)
            for grad_name, G in matching_grads:
                if G.dim() > 2:
                    G = G.view(G.size(0), -1)
                d, k = G.shape
                r = min(rank, min(d, k))  # Ensure valid rank

                # New: Use kaiming_uniform for A (PEFT default; better than randn*0.01)
                A = torch.empty((r, k), device=G.device, dtype=G.dtype)
                torch.nn.init.kaiming_uniform_(A, a=np.sqrt(5))

                # Compute B (unchanged)
                AAt = torch.matmul(A, A.T) + torch.eye(r, device=A.device, dtype=A.dtype) * eps
                try:
                    AAt_inv = torch.inverse(AAt)
                except RuntimeError:
                    print(f"  ⚠️  Inverse failed for {layer_name}, using pinv")
                    AAt_inv = torch.pinverse(AAt)
                GA = torch.matmul(G, A.T)
                B = -torch.matmul(GA, AAt_inv)

                approx_error = torch.norm(torch.matmul(B, A) + G) / torch.norm(G)
                print(f"    {layer_name} <- {grad_name}: rank={r}, error={approx_error:.4f}")

                warmstart_weights[grad_name] = {
                    'A': A.detach().cpu(),
                    'B': B.detach().cpu(),
                    'rank': r,
                    'layer_name': layer_name
                }

        print(len(warmstart_weights))
        print(f"  ✓ Computed {len(warmstart_weights)} warm-start weight sets")
        return warmstart_weights if warmstart_weights else None


    def _inject_warmstart_weights(
            self,
            peft_model: PreTrainedModel,
            warmstart_weights: Dict
    ) -> bool:
        """
        Inject pre-computed warm-start weights into PEFT model.
        Returns True if successful, False otherwise.
        """
        if not warmstart_weights:
            print("  No warm-start weights to inject")
            return False

        print("  Injecting warm-start weights into LoRA parameters...")

        injected_A = 0
        injected_B = 0
        failed = []

        # Iterate through all LoRA parameters in the PEFT model
        for name, param in peft_model.named_parameters():
            if 'lora' not in name.lower() or not param.requires_grad:
                continue

            # Determine if this is A or B matrix
            is_A = 'lora_A' in name or 'lora_a' in name
            is_B = 'lora_B' in name or 'lora_b' in name

            if not (is_A or is_B):
                continue

            # Find matching warm-start weights
            # name format: "base_model.model.distilbert.transformer.layer.0.attention.q_lin.lora_A.default.weight"
            # ws_key format: "distilbert.transformer.layer.0.attention.q_lin.weight"

            matched_weight = None
            matched_key = None

            for ws_key, ws_data in warmstart_weights.items():
                # Extract the core module path from both names
                # Remove "base_model.model." prefix and ".lora_A/B.default.weight" suffix
                core_name = name.replace('base_model.model.', '').replace('.lora_A.default.weight', '').replace(
                    '.lora_B.default.weight', '')
                core_name = core_name.replace('.lora_A.weight', '').replace('.lora_B.weight', '')

                # Remove ".weight" suffix from ws_key
                ws_core = ws_key.replace('.weight', '').replace('.bias', '')

                # Check if they match
                if ws_core in core_name or core_name in ws_core:
                    matched_weight = ws_data
                    matched_key = ws_key
                    break

            if matched_weight is None:
                continue

            # Inject the appropriate matrix
            if is_A and 'A' in matched_weight:
                A_init = matched_weight['A'].to(param.device, dtype=param.dtype)
                if param.shape == A_init.shape:
                    param.data.copy_(A_init)
                    injected_A += 1
                    print(f"    ✓ Injected A matrix: {matched_key} -> {name}")
                else:
                    failed.append((name, 'A', param.shape, A_init.shape))
                    print(f"    ✗ Shape mismatch A: {name} expected {param.shape}, got {A_init.shape}")

            elif is_B and 'B' in matched_weight:
                B_init = matched_weight['B'].to(param.device, dtype=param.dtype)
                if param.shape == B_init.shape:
                    param.data.copy_(B_init)
                    injected_B += 1
                    print(f"    ✓ Injected B matrix: {matched_key} -> {name}")
                else:
                    failed.append((name, 'B', param.shape, B_init.shape))
                    print(f"    ✗ Shape mismatch B: {name} expected {param.shape}, got {B_init.shape}")

        # Report results
        print(f"\n  Injection summary:")
        print(f"    A matrices injected: {injected_A}")
        print(f"    B matrices injected: {injected_B}")

        if failed:
            print(f"    Failed injections: {len(failed)}")
            for name, matrix, expected, actual in failed[:3]:  # Show first 3
                print(f"      {name} ({matrix}): shape mismatch {expected} vs {actual}")
            return False

        if injected_A == 0 and injected_B == 0:
            print("    ⚠️  WARNING: No weights were injected!")
            return False

        return True

    def _verify_model(self, peft_model: PreTrainedModel, rank_allocation: Dict[str, int]) -> None:
        print(f"\n{'=' * 70}")
        print("MODEL VERIFICATION")
        print(f"{'=' * 70}")

        # New: Compute LoRA-only params
        lora_params = sum(
            p.numel() for n, p in peft_model.named_parameters() if p.requires_grad and 'lora' in n.lower())
        print(f"\n1. Parameter Budget (LoRA only):")
        print(f"   Target: {self.param_budget:,}")
        print(f"   Actual: {lora_params:,}")
        print(f"   Ratio: {lora_params / self.param_budget:.2%}")

        if lora_params > self.param_budget * 1.1:
            print(f"   ⚠️  WARNING: Exceeds budget by {(lora_params / self.param_budget - 1) * 100:.1f}%")
        elif lora_params < self.param_budget * 0.9:
            print(f"   ⚠️  WARNING: Under budget by {(1 - lora_params / self.param_budget) * 100:.1f}%")
        else:
            print(f"   ✓ Within acceptable range")

        # Check 2: Rank allocation
        print(f"\n2. Rank Allocation:")
        layer_ranks_applied = {}
        for name, module in peft_model.named_modules():
            if hasattr(module, 'r'):
                rank = module.r
                # Handle dict-type ranks
                if isinstance(rank, dict):
                    rank = rank.get('default', list(rank.values())[0] if rank else 4)
                layer_ranks_applied[name] = rank

        if not layer_ranks_applied:
            print("   ❌ CRITICAL: No LoRA ranks found in model!")
        else:
            unique_ranks = set(layer_ranks_applied.values())  # ✅ Use extracted ranks
            expected_ranks = set(rank_allocation.values())

            print(f"   Expected rank range: [{min(expected_ranks)}, {max(expected_ranks)}]")
            print(f"   Applied rank range: [{min(layer_ranks_applied.values())}, {max(layer_ranks_applied.values())}]")
            print(f"   Unique ranks expected: {len(expected_ranks)}")
            print(f"   Unique ranks applied: {len(unique_ranks)}")

            if len(unique_ranks) == 1:
                print(f"   ❌ CRITICAL: All layers have SAME rank ({list(unique_ranks)[0]})")
                print(f"   This defeats the purpose of adaptive allocation!")
            else:
                print(f"   ✓ Multiple ranks applied successfully")

        # Check 3: Warm-start verification
        print(f"\n3. Warm-Start Initialization:")
        lora_params = {n: p for n, p in peft_model.named_parameters() if 'lora' in n.lower()}

        if not lora_params:
            print("   ❌ No LoRA parameters found")
        else:
            near_zero_count = 0
            for name, param in lora_params.items():
                mean_abs = torch.abs(param.data.mean()).item()
                if mean_abs < 1e-6:
                    near_zero_count += 1

            if near_zero_count == len(lora_params):
                print(f"   ⚠️  WARNING: All weights appear to be randomly initialized")
                print(f"   Warm-start may not have been applied")
            elif near_zero_count > len(lora_params) * 0.5:
                print(f"   ⚠️  WARNING: {near_zero_count}/{len(lora_params)} weights near zero")
                print(f"   Warm-start may be partially applied")
            else:
                print(f"   ✓ Weights appear to be warm-started")
                print(f"   ({len(lora_params) - near_zero_count}/{len(lora_params)} non-zero)")

        print(f"\n{'=' * 70}")

    def _estimate_importance(
            self, train_dataset: Dataset,
            num_samples: int, target_modules: list) -> Dict[str, float]:
        self.gradient_analyzer = GradientAnalyzer(
            model=self.model,
            tokenizer=self.tokenizer,
            target_modules=target_modules
        )
        importance_scores = self.gradient_analyzer.analyze(
            train_dataset=train_dataset,
            num_samples=num_samples,
            batch_size=8
        )
        # New: Filter out biases if included
        importance_scores = {k: v for k, v in importance_scores.items() if not k.endswith('.bias')}
        print(f"✓ Importance estimation complete (filtered to {len(importance_scores)} weight-only scores)")
        return importance_scores

    def _allocate_ranks(
            self,
            importance_scores: Dict[str, float],
            base_rank: int,
            target_modules: list,
            min_rank: int = 2,  # New param
            max_rank: int = 64  # New param
    ) -> Dict[str, int]:
        """Phase 2: Allocate ranks based on importance and budget."""
        if not self.param_budget:
            raise ValueError("BA-LoRA requires parameter budget")

        self.rank_allocator = RankAllocator(
            importance_scores=importance_scores,
            param_budget=self.param_budget,
            base_rank=base_rank,
            hidden_dim=self.model.config.hidden_size,
            min_rank=min_rank,  # Pass through
            max_rank=max_rank,  # Pass through
            num_target_modules=len(target_modules)
        )

        self.rank_allocation = self.rank_allocator.allocate_ranks()

        print(f"✓ Rank allocation complete")
        return self.rank_allocation

    def _apply_adaptive_lora(
            self,
            rank_allocation: Dict[str, int],
            lora_alpha: int,
            lora_dropout: float,
            target_modules: list
    ) -> PreTrainedModel:
        """
        Phase 4: Apply LoRA with per-layer rank allocation.
        """
        print(f"\nApplying adaptive LoRA...")
        print(f"  Rank allocation summary:")

        # Group layers by their allocated rank
        rank_groups = {}
        for layer_name, rank in rank_allocation.items():
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(layer_name)

        for rank, layers in sorted(rank_groups.items()):
            print(f"    Rank {rank:2d}: {len(layers)} layers")

        # Check PEFT version for rank_pattern support
        try:
            from peft import __version__
            peft_version = tuple(map(int, __version__.split('.')[:2]))
            supports_rank_pattern = peft_version >= (0, 5)
        except:
            supports_rank_pattern = False

        if supports_rank_pattern:
            print("  Using PEFT rank_pattern feature for per-layer ranks")

            # Create rank pattern with FULL module paths
            # We need to map simplified names to actual module names
            rank_pattern = {}

            for name, module in self.model.named_modules():
                # Check if this module matches any of our allocations
                for layer_name, rank in rank_allocation.items():
                    # Extract layer number from allocation
                    # "layer_1_attention" -> look for "layer.1"
                    if 'layer' in layer_name:
                        try:
                            layer_num = layer_name.split('_')[1]
                            # Check if this module is in that layer and is a target
                            if f"layer.{layer_num}." in name and any(tm in name for tm in target_modules):
                                rank_pattern[name] = rank
                                print(f"    Mapping {name} -> rank {rank}")
                        except (IndexError, ValueError):
                            continue

            if not rank_pattern:
                print("  ❌ CRITICAL: rank_pattern is empty! Using fallback.")
                # Fallback: just use target_modules with median rank
                base_rank = int(np.median(list(rank_allocation.values())))
            else:
                print(f"  ✓ Created rank_pattern with {len(rank_pattern)} entries")
                base_rank = int(np.median(list(rank_allocation.values())))

            lora_config = LoraConfig(
                r=base_rank,
                rank_pattern=rank_pattern if rank_pattern else {},
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                modules_to_save=["classifier", "pre_classifier"]  # New: Keep head trainable
            )

            peft_model = get_peft_model(self.model, lora_config)

        else:
            print("  ❌ PEFT rank_pattern not available - upgrade PEFT!")
            raise RuntimeError("PEFT >= 0.5.0 required for adaptive rank allocation")

        # Verification
        print("\n  Verification of applied ranks:")
        layer_ranks_applied = {}
        for name, module in peft_model.named_modules():
            if hasattr(module, 'r'):
                rank = module.r
                if isinstance(rank, dict):
                    rank = rank.get('default', list(rank.values())[0] if rank else 4)
                layer_ranks_applied[name] = rank
                print(f"    {name}: rank = {rank}")

        if len(layer_ranks_applied) == 0:
            print("  ⚠️  WARNING: No LoRA ranks found in model!")
        else:
            unique_applied = set(layer_ranks_applied.values())
            if len(unique_applied) == 1:
                print(f"  ⚠️  WARNING: All layers have same rank ({list(unique_applied)[0]})")
                print(f"  Expected different ranks: {set(rank_allocation.values())}")
            else:
                print(f"  ✓ Multiple ranks applied: {sorted(unique_applied)}")

        return peft_model

    def _print_final_stats(self, model: PreTrainedModel):
        """Print final model statistics."""
        # Count only LoRA parameters
        lora_params = sum(p.numel() for n, p in model.named_parameters()
                          if p.requires_grad and 'lora' in n.lower())
        other_trainable = sum(p.numel() for n, p in model.named_parameters()
                              if p.requires_grad and 'lora' not in n.lower())
        total = sum(p.numel() for p in model.parameters())

        print(f"\n{'=' * 70}")
        print("BA-LORA BUILD COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total parameters:        {total:>15,}")
        print(f"LoRA parameters:         {lora_params:>15,}")
        print(f"Other trainable params:  {other_trainable:>15,}")
        print(f"Total trainable:         {lora_params + other_trainable:>15,}")
        print(f"Trainable percentage:    {(lora_params + other_trainable) / total * 100:>14.3f}%")

        if self.param_budget:
            budget_usage = lora_params / self.param_budget * 100
            print(f"Budget allocated:        {self.param_budget:>15,}")
            print(f"LoRA budget usage:       {budget_usage:>14.1f}%")

        if self.rank_allocator:
            stats = self.rank_allocator.get_allocation_stats()
            print(f"\nRank allocation:")
            print(f"  Range: [{stats['min_rank']}, {stats['max_rank']}]")
            print(f"  Mean:  {stats['mean_rank']:.1f}")
            print(f"  Std:   {stats['std_rank']:.1f}")

        print(f"{'=' * 70}\n")

    def get_ba_lora_info(self) -> Dict:
        """Get detailed information about BA-LoRA configuration."""
        info = {
            "method": "ba_lora",
            "param_budget": self.param_budget,
            "rank_allocation": self.rank_allocation,
        }

        if self.rank_allocator:
            info["allocation_stats"] = self.rank_allocator.get_allocation_stats()

        if self.gradient_analyzer:
            info["importance_scores"] = self.gradient_analyzer.importance_scores

        return info