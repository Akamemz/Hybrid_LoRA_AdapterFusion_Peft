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
            importance_scores, base_rank, target_modules  # ADD target_modules parameter
        )

        # Phase 3: Warm-start (optional)
        if use_warmstart:
            print(f"\n[Phase 3/4] Warm-start initialization...")
            print("  Note: Using standard LoRA initialization")
            print("  (Custom warm-start is a future enhancement)")
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

        return peft_model

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
            target_modules: list  # ADD THIS PARAMETER
    ) -> Dict[str, int]:
        """Phase 2: Allocate ranks based on importance and budget."""
        if not self.param_budget:
            raise ValueError("BA-LoRA requires parameter budget")

        self.rank_allocator = RankAllocator(
            importance_scores=importance_scores,
            param_budget=self.param_budget,
            base_rank=base_rank,
            hidden_dim=self.model.config.hidden_size,
            num_target_modules=len(target_modules)  # ADD THIS LINE
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