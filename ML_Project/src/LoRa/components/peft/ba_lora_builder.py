"""
ML_Project/src/LoRa/components/peft/ba_lora_builder.py

BA-LoRA Builder: Orchestrates all phases of Budget-Aware Adaptive LoRA
"""

from typing import Dict, Optional
import torch
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
        """
        Build BA-LoRA model with adaptive rank allocation.

        Config should contain:
        - train_dataset: Dataset for gradient analysis
        - base_rank: Base rank for scaling (default: 8)
        - gradient_samples: Samples for gradient accumulation (default: 1000)
        - use_warmstart: Use warm-start initialization (default: True)
        - target_modules: Target modules (optional, auto-detected)
        - lora_alpha: LoRA alpha (default: 2 × base_rank)
        - lora_dropout: LoRA dropout (default: 0.1)

        Args:
            config: Configuration dictionary

        Returns:
            Model with BA-LoRA applied
        """
        print(f"\n{'=' * 70}")
        print("BUILDING BA-LORA MODEL")
        print(f"{'=' * 70}")

        # Extract configuration
        train_dataset = config.get("train_dataset")
        if train_dataset is None:
            raise ValueError("BA-LoRA requires 'train_dataset' for gradient analysis")

        base_rank = config.get("base_rank", 8)
        gradient_samples = config.get("gradient_samples", 1000)
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
            importance_scores, base_rank, target_modules
        )

        # Phase 3: Warm-start (optional)
        if use_warmstart:
            print(f"\n[Phase 3/4] Warm-start initialization...")
            self._apply_warmstart(rank_allocation, target_modules)
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

        # Print final statistics
        self._print_final_stats(peft_model)

        if use_warmstart and hasattr(self, 'warmstart_init'):
            self._inject_warmstart_weights(peft_model)

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

    def _inject_warmstart_weights(self, peft_model):
        """
        Helper method to inject warm-started weights into the PEFT model.
        Call this after creating the LoRA model but before training.

        Args:
            peft_model: The PEFT model with LoRA adapters
        """
        if not hasattr(self, 'warmstart_init') or not self.warmstart_init:
            return

        print("\n  Injecting warm-start weights into LoRA adapters...")
        injected = 0

        for name, param in peft_model.named_parameters():
            if 'lora' not in name.lower():
                continue

            # Find matching warm-start initialization
            for base_name, init_data in self.warmstart_init.items():
                if base_name in name:
                    # Determine if this is A or B matrix
                    if 'lora_A' in name or 'lora_a' in name:
                        A_init = init_data['A'].to(param.device, dtype=param.dtype)
                        if param.shape == A_init.shape:
                            param.data.copy_(A_init)
                            injected += 1
                    elif 'lora_B' in name or 'lora_b' in name:
                        B_init = init_data['B'].to(param.device, dtype=param.dtype)
                        if param.shape == B_init.shape:
                            param.data.copy_(B_init)
                            injected += 1

        print(f"  ✓ Injected warm-start weights into {injected} parameters")

    def _estimate_importance(
            self,
            train_dataset: Dataset,
            num_samples: int,
            target_modules: list
    ) -> Dict[str, float]:
        """Phase 1: Estimate layer importance."""
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

        print(f"✓ Importance estimation complete")
        return importance_scores

    def _allocate_ranks(
            self,
            importance_scores: Dict[str, float],
            base_rank: int,
            target_modules: list
    ) -> Dict[str, int]:
        """Phase 2: Allocate ranks based on importance and budget."""
        if not self.param_budget:
            raise ValueError("BA-LoRA requires parameter budget")

        self.rank_allocator = RankAllocator(
            importance_scores=importance_scores,
            param_budget=self.param_budget,
            base_rank=base_rank,
            hidden_dim=self.model.config.hidden_size,
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
        """Phase 4: Apply LoRA with per-layer rank allocation."""

        # Check if all ranks are the same
        unique_ranks = set(rank_allocation.values())

        if len(unique_ranks) == 1:
            # All ranks same - use standard LoRA
            rank = list(unique_ranks)[0]
            print(f"  All layers have same rank ({rank}), using standard LoRA")

            lora_config = LoraConfig(
                r=rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
            )

            peft_model = get_peft_model(self.model, lora_config)

        else:
            # Different ranks - use average (limitation of PEFT library)
            avg_rank = int(sum(rank_allocation.values()) / len(rank_allocation))
            print(f"  Using average rank ({avg_rank}) as approximation")
            print(f"  Note: Per-layer ranks require custom PEFT implementation")

            lora_config = LoraConfig(
                r=avg_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
            )

            peft_model = get_peft_model(self.model, lora_config)

        return peft_model

    def _print_final_stats(self, model: PreTrainedModel):
        """Print final model statistics."""
        total, trainable = self.count_parameters(model)

        print(f"\n{'=' * 70}")
        print("BA-LORA BUILD COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total parameters:     {total:>15,}")
        print(f"Trainable parameters: {trainable:>15,}")
        print(f"Trainable percentage: {trainable / total * 100:>14.3f}%")

        if self.param_budget:
            budget_usage = trainable / self.param_budget * 100
            print(f"Budget allocated:     {self.param_budget:>15,}")
            print(f"Budget usage:         {budget_usage:>14.1f}%")

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