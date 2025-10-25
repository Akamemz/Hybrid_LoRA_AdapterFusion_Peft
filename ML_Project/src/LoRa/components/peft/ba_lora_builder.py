"""
ML_Project/src/LoRa/components/peft/ba_lora_builder.py

BA-LoRA Builder: Orchestrates all phases of Budget-Aware Adaptive LoRA
"""

from typing import Dict, Optional, Union
import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftMixedModel
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
            min_rank=2,
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
        self.min_rank = min_rank
        self.gradient_analyzer = None
        self.rank_allocator = None
        self.rank_allocation = {}

        print(f"BALoRABuilder initialized")
        if param_budget:
            print(f"  Parameter budget: {param_budget:,}")

    def build(self, config: Dict) -> Union[PreTrainedModel, PeftModel, PeftMixedModel]:
        """Build BA-LoRA model (main method - calls fixed warm-start)."""
        print(f"\n{'=' * 70}")
        print("BUILDING BA-LORA MODEL (FIXED WARM-START)")
        print(f"{'=' * 70}")

        # Freeze classifier head
        freeze_head = config.get("freeze_head", False)
        if freeze_head:
            print("\nFreezing classifier head parameters...")
            frozen_params = 0
            for name, param in self.model.named_parameters():
                if 'classifier' in name or 'pre_classifier' in name or 'head' in name:
                    param.requires_grad = False
                    frozen_params += param.numel()
            print(f"  Total frozen: {frozen_params:,} parameters\n")

        # Extract configuration
        train_dataset = config.get("train_dataset")
        if train_dataset is None:
            raise ValueError("BA-LoRA requires 'train_dataset' for gradient analysis")

        base_rank = config.get("base_rank", 8)
        gradient_samples = config.get("gradient_samples", 3000)
        use_warmstart = config.get("use_warmstart", True)
        target_modules = config.get("target_modules") or self._get_default_target_modules()
        lora_alpha = config.get("lora_alpha", 2 * base_rank)
        lora_dropout = config.get("lora_dropout", 0.1)

        print(f"Configuration:")
        print(f"  Base rank: {base_rank}")
        print(f"  Gradient samples: {gradient_samples}")
        print(f"  Warm-start: {use_warmstart}")
        print(f"  Target modules: {target_modules}")

        # Phase 1: Estimate importance
        print(f"\n[Phase 1/4] Estimating layer importance...")
        importance_scores = self._estimate_importance(
            train_dataset, gradient_samples, target_modules
        )

        # Phase 2: Allocate ranks
        print(f"\n[Phase 2/4] Allocating ranks with budget constraint...")
        rank_allocation = self._allocate_ranks(
            importance_scores, base_rank, target_modules,
            min_rank=config.get("min_rank", 5),
            max_rank=config.get("max_rank", 64)
        )

        # Phase 3: Compute warm-start weights (FIXED METHOD)
        warmstart_weights = None
        if use_warmstart:
            print(f"\n[Phase 3/4] Computing warm-start initialization (FIXED)...")
            warmstart_weights = self._compute_warmstart_weights(
                rank_allocation, target_modules
            )
        else:
            print(f"\n[Phase 3/4] Skipping warm-start (disabled)")

        # Phase 4: Apply LoRA
        print(f"\n[Phase 4/4] Applying LoRA with adaptive ranks...")
        peft_model = self._apply_adaptive_lora(
            rank_allocation,
            lora_alpha,
            lora_dropout,
            target_modules
        )

        # Inject warm-start weights
        if use_warmstart and warmstart_weights is not None:
            print(f"\n[Post-Build] Injecting warm-start weights...")
            injection_success = self._inject_warmstart_weights(
                peft_model, warmstart_weights
            )

            if injection_success:
                print("  ✓ Warm-start weights successfully injected")
            else:
                print("  ⚠️  WARNING: Warm-start injection had issues!")

        # Print final statistics
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

    def _compute_warmstart_weights(
            self,
            rank_allocation: Dict[str, int],
            target_modules: list
    ) -> Dict[str, Dict]:
        """
        FIXED: Compute warm-start weights using improved SVD-based approach.

        Changes from original:
        1. SVD-based initialization instead of Kaiming
        2. Adaptive regularization based on gradient magnitude
        3. Better error checking and reporting
        """
        if not hasattr(self.gradient_analyzer, 'gradients'):
            print("  WARNING: No gradients available")
            return None

        print("  Computing warm-start weights (FIXED METHOD)...")
        warmstart_weights = {}

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

            # Process each matching gradient
            for grad_name, G in matching_grads:
                if G.dim() > 2:
                    G = G.view(G.size(0), -1)

                d, k = G.shape
                r = min(rank, min(d, k))

                # ============================================================
                # FIX #1: SVD-based initialization for A
                # ============================================================
                # Use truncated SVD of gradient to initialize A
                # This ensures A captures the principal directions of G
                try:
                    # Normalize G for consistent magnitude
                    G_norm = G / (G.norm() + 1e-8)

                    # SVD on normalized G
                    U, S, Vt = torch.svd(G_norm)

                    # Higher scale for small grads
                    scale = torch.sqrt(S[:r].mean() * 2)  # Empirical boost
                    A = Vt[:r, :].clone() * scale

                    # Adaptive eps on normalized mean
                    grad_mean = G_norm.abs().mean()
                    eps_adaptive = max(1e-6, grad_mean.item() * 0.01)

                    AAt = torch.matmul(A, A.T) + torch.eye(r, device=A.device) * eps_adaptive

                    # Stable inverse with fallback
                    try:
                        L = torch.linalg.cholesky(AAt)
                        AAt_inv = torch.cholesky_inverse(L)
                    except RuntimeError:
                        print(f"    Cholesky failed for {grad_name}, using pinverse")
                        AAt_inv = torch.pinverse(AAt)

                    GA = torch.matmul(G_norm, A.T)  # Use G_norm
                    B = -torch.matmul(GA, AAt_inv)

                    # Quality metrics on original G scale (for reporting)
                    AB = torch.matmul(B, A) * G.norm()  # Rescale back
                    rel_error = torch.norm(AB + G) / torch.norm(G)
                    explained_var = (1 - rel_error ** 2).item()  # R^2 approx

                    print(f"    {layer_name} <- {grad_name}: Using SVD initialization")

                except RuntimeError as e:
                    print(f"    ⚠️  SVD failed for {grad_name}, using fallback")
                    # Fallback: normalized random initialization
                    A = torch.randn((r, k), device=G.device, dtype=G.dtype)
                    A = A / torch.norm(A, dim=1, keepdim=True) * np.sqrt(k)

                # ============================================================
                # FIX #2: Adaptive regularization
                # ============================================================
                # Adjust regularization based on gradient magnitude
                grad_norm = torch.norm(G)
                grad_mean = G.abs().mean()

                # Use relative regularization: eps = alpha * mean(|G|)
                # This adapts to the scale of gradients
                eps = max(1e-6, grad_mean.item() * 0.01)

                print(f"      Gradient stats: norm={grad_norm:.6f}, mean_abs={grad_mean:.6f}")
                print(f"      Using adaptive eps={eps:.8f}")

                # ============================================================
                # FIX #3: Improved B computation with condition number check
                # ============================================================
                # Compute A @ A^T with regularization
                AAt = torch.matmul(A, A.T)
                AAt_reg = AAt + torch.eye(r, device=A.device, dtype=A.dtype) * eps

                # Check condition number before inversion
                try:
                    # Try Cholesky decomposition for better numerical stability
                    L = torch.linalg.cholesky(AAt_reg)
                    AAt_inv = torch.cholesky_inverse(L)
                except RuntimeError:
                    # Fallback to regular inverse if Cholesky fails
                    try:
                        AAt_inv = torch.inverse(AAt_reg)
                    except RuntimeError:
                        print(f"      ⚠️  Inverse failed, using pinv")
                        AAt_inv = torch.pinverse(AAt_reg)

                # Compute B = -G @ A^T @ (A @ A^T + εI)^{-1}
                GA = torch.matmul(G, A.T)
                B = -torch.matmul(GA, AAt_inv)

                # ============================================================
                # FIX #4: Better approximation error computation
                # ============================================================
                AB_approx = torch.matmul(B, A)

                # Compute relative error (more meaningful than absolute)
                approx_error_abs = torch.norm(AB_approx + G)
                approx_error_rel = approx_error_abs / torch.norm(G)

                # Also compute explained variance (like R²)
                residual_variance = torch.var(AB_approx + G)
                gradient_variance = torch.var(G)
                explained_var = 1 - (residual_variance / gradient_variance)

                print(f"      rank={r}, rel_error={approx_error_rel:.4f}, "
                      f"explained_var={explained_var:.4f}")

                # ============================================================
                # FIX #5: Store with quality metrics
                # ============================================================
                warmstart_weights[grad_name] = {
                    'A': A.detach().cpu(),
                    'B': B.detach().cpu(),
                    'rank': r,
                    'layer_name': layer_name,
                    'rel_error': approx_error_rel.item(),
                    'explained_var': explained_var.item(),
                    'grad_norm': grad_norm.item()
                }

        if not warmstart_weights:
            print(f"  ❌ No warm-start weights computed!")
            return None

        # Print summary statistics
        errors = [w['rel_error'] for w in warmstart_weights.values()]
        exp_vars = [w['explained_var'] for w in warmstart_weights.values()]

        print(f"\n  Warm-start Quality Summary:")
        print(f"    Computed weights for: {len(warmstart_weights)} modules")
        print(f"    Relative error: mean={np.mean(errors):.4f}, "
              f"std={np.std(errors):.4f}, min={np.min(errors):.4f}, max={np.max(errors):.4f}")
        print(f"    Explained variance: mean={np.mean(exp_vars):.4f}, "
              f"std={np.std(exp_vars):.4f}")

        # Flag if quality is poor
        if np.mean(errors) > 0.5:
            print(f"  ⚠️  WARNING: High average error (>{50}%)")
            print(f"      Consider: increasing gradient_samples or base_rank")
        elif np.mean(exp_vars) < 0.3:
            print(f"  ⚠️  WARNING: Low explained variance (<30%)")
            print(f"      The warm-start may not help much")
        else:
            print(f"  ✓ Warm-start quality looks good!")

        return warmstart_weights


    def _inject_warmstart_weights(
            self,
            peft_model: PreTrainedModel,
            warmstart_weights: Dict
    ) -> bool:
        """Inject pre-computed warm-start weights (unchanged from original)."""
        if not warmstart_weights:
            print("  No warm-start weights to inject")
            return False

        print("  Injecting warm-start weights into LoRA parameters...")

        injected_A = 0
        injected_B = 0
        failed = []

        for name, param in peft_model.named_parameters():
            if 'lora' not in name.lower() or not param.requires_grad:
                continue

            is_A = 'lora_A' in name or 'lora_a' in name
            is_B = 'lora_B' in name or 'lora_b' in name

            if not (is_A or is_B):
                continue

            # Find matching warm-start weights
            matched_weight = None
            matched_key = None

            for ws_key, ws_data in warmstart_weights.items():
                core_name = name.replace('base_model.model.', '').replace('.lora_A.default.weight', '').replace(
                    '.lora_B.default.weight', '')
                core_name = core_name.replace('.lora_A.weight', '').replace('.lora_B.weight', '')
                ws_core = ws_key.replace('.weight', '').replace('.bias', '')

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
                else:
                    failed.append((name, 'A', param.shape, A_init.shape))

            elif is_B and 'B' in matched_weight:
                B_init = matched_weight['B'].to(param.device, dtype=param.dtype)
                if param.shape == B_init.shape:
                    param.data.copy_(B_init)
                    injected_B += 1
                else:
                    failed.append((name, 'B', param.shape, B_init.shape))

        # Report results
        print(f"\n  Injection summary:")
        print(f"    A matrices injected: {injected_A}")
        print(f"    B matrices injected: {injected_B}")

        if failed:
            print(f"    Failed injections: {len(failed)}")
            return False

        if injected_A == 0 and injected_B == 0:
            print("    ⚠️  WARNING: No weights were injected!")
            return False

        return True

    def _verify_model(self, peft_model: PreTrainedModel, rank_allocation: Dict[str, int]) -> None:
        """Verify model (simplified)."""
        print(f"\n{'=' * 70}")
        print("MODEL VERIFICATION")
        print(f"{'=' * 70}\n")

        unique_ranks = set(rank_allocation.values())
        print(f"Rank allocation: {len(unique_ranks)} unique ranks")
        print(f"  Range: [{min(unique_ranks)}, {max(unique_ranks)}]")

        if len(unique_ranks) <= 2:
            print("  ⚠️  WARNING: Very few unique ranks")
        else:
            print("  ✓ Good rank diversity")

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
            param_budget=self.param_budget ,
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
    ) -> Union[PreTrainedModel, PeftModel, PeftMixedModel]:

        if isinstance(self.model, (PeftModel, PeftMixedModel)):
            raise ValueError("Model is already a PEFT model. Cannot apply LoRA again.")

        """Phase 4: Apply LoRA (unchanged)."""
        print(f"\nApplying adaptive LoRA...")

        rank_pattern = {}
        for name, module in self.model.named_modules():
            for layer_name, rank in rank_allocation.items():
                if 'layer' in layer_name:
                    try:
                        layer_num = layer_name.split('_')[1]
                        if f"layer.{layer_num}." in name and any(tm in name for tm in target_modules):
                            rank_pattern[name] = rank
                    except (IndexError, ValueError):
                        continue

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
            modules_to_save=["classifier", "pre_classifier"]
        )

        return get_peft_model(self.model, lora_config)

    def _print_final_stats(self, model: PreTrainedModel):
        """Print final statistics (unchanged)."""
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

        if self.param_budget:
            budget_usage = lora_params / self.param_budget * 100
            print(f"Budget allocated:        {self.param_budget:>15,}")
            print(f"LoRA budget usage:       {budget_usage:>14.1f}%")


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