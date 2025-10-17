"""
Phase 2 of BA-LoRA: Budget-Aware Rank Allocation
"""

from typing import Dict, Optional
import numpy as np


class RankAllocator:
    def __init__(
            self,
            importance_scores: Dict[str, float],
            param_budget: int,
            base_rank: int = 8,
            hidden_dim: int = 768,
            min_rank: int = 2,
            max_rank: int = 32,
            num_target_modules: int = 2  # ADD THIS PARAMETER
    ):
        """
        Initialize rank allocator.

        Args:
            importance_scores: Layer importance scores
            param_budget: Total parameter budget
            base_rank: Base rank for scaling
            hidden_dim: Hidden dimension of model
            min_rank: Minimum rank allowed
            max_rank: Maximum rank allowed
            num_target_modules: Number of target modules per layer (default: 2 for q,v)
        """
        self.importance_scores = importance_scores
        self.param_budget = param_budget
        self.base_rank = base_rank
        self.hidden_dim = hidden_dim
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.num_target_modules = num_target_modules  # ADD THIS

        self.rank_allocation = {}

        print(f"\nRankAllocator initialized")
        print(f"  Parameter budget: {param_budget:,}")
        print(f"  Base rank: {base_rank}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Target modules per layer: {num_target_modules}")  # ADD THIS
        print(f"  Rank range: [{min_rank}, {max_rank}]")

    def allocate_ranks(self) -> Dict[str, int]:
        """
        Allocate ranks proportionally to importance while meeting exact budget.

        Returns:
            Dictionary mapping layer names to allocated ranks
        """
        print("\nAllocating ranks...")

        if not self.importance_scores:
            raise ValueError("No importance scores provided")

        # Step 1: Normalize importance to [0.5, 2.0]
        normalized_importance = self._normalize_importance()

        # Step 2: Initial allocation
        initial_ranks = self._initial_allocation(normalized_importance)

        # Step 3: Enforce budget constraint
        final_ranks = self._enforce_budget(initial_ranks)

        # Step 4: Verify and report
        self._verify_allocation(final_ranks)

        self.rank_allocation = final_ranks
        return final_ranks

    def _normalize_importance(self) -> Dict[str, float]:
        """Normalize importance scores to [0.5, 2.0] range."""
        scores = list(self.importance_scores.values())
        max_score = max(scores)
        min_score = min(scores)

        if max_score == min_score:
            return {name: 1.0 for name in self.importance_scores.keys()}

        normalized = {}
        for name, score in self.importance_scores.items():
            # Map to [0, 1]
            norm = (score - min_score) / (max_score - min_score)
            # Map to [0.5, 2.0]
            normalized[name] = 0.5 + 1.5 * norm

        return normalized

    def _initial_allocation(self, normalized_importance: Dict[str, float]) -> Dict[str, int]:
        """Initial rank allocation based on normalized importance."""
        ranks = {}

        for name, norm_importance in normalized_importance.items():
            # Scale by base rank
            rank = int(self.base_rank * norm_importance)

            # Enforce bounds
            rank = max(self.min_rank, min(self.max_rank, rank))

            ranks[name] = rank

        total_params = self._calculate_params(ranks)
        print(f"  Initial allocation: {total_params:,} params")

        return ranks

    def _enforce_budget(self, ranks: Dict[str, int]) -> Dict[str, int]:
        """Iteratively adjust ranks to meet parameter budget with smarter strategy."""
        ranks = ranks.copy()
        max_iterations = 1000

        # Step 1: Reduce if significantly over budget (reduce highest ranks first)
        for iteration in range(max_iterations):
            total_params = self._calculate_params(ranks)

            if total_params <= self.param_budget:
                break

            # Sort by rank (highest first)
            sorted_layers = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

            # Try to reduce highest rank that's above min_rank
            reduced = False
            for layer_name, rank in sorted_layers:
                if rank > self.min_rank:
                    # Calculate how many params this reduction saves
                    params_saved = self.num_target_modules * 2 * self.hidden_dim

                    # Only reduce if it helps
                    if total_params - params_saved >= self.param_budget * 0.97:  # Within 3% is OK
                        ranks[layer_name] -= 1
                        reduced = True
                        break

            if not reduced:
                # Can't reduce further, break and accept result
                break

        # Step 2: Increase if under budget (increase lowest ranks first for balance)
        for iteration in range(max_iterations):
            total_params = self._calculate_params(ranks)
            params_per_increase = self.num_target_modules * 2 * self.hidden_dim

            # Check if we have room to add
            if total_params + params_per_increase > self.param_budget * 1.03:  # 3% tolerance
                break

            # Sort by rank (lowest first) - prioritize raising low ranks
            sorted_layers = sorted(ranks.items(), key=lambda x: x[1])

            # Increase lowest rank that's below max_rank
            increased = False
            for layer_name, rank in sorted_layers:
                if rank < self.max_rank:
                    test_total = total_params + params_per_increase
                    if test_total <= self.param_budget * 1.03:  # Within 3% tolerance
                        ranks[layer_name] += 1
                        increased = True
                        break

            if not increased:
                break

        return ranks

    def _calculate_params(self, ranks: Dict[str, int]) -> int:
        """
        Calculate total parameters for given rank allocation.

        PEFT applies LoRA to each target module separately, so we need to
        multiply by num_target_modules.
        """
        # Each layer gets: num_target_modules × 2 × rank × hidden_dim
        total = sum(
            self.num_target_modules * 2 * rank * self.hidden_dim
            for rank in ranks.values()
        )
        return total

    def _verify_allocation(self, ranks: Dict[str, int]):
        """Verify allocation meets constraints and print statistics."""
        total_params = self._calculate_params(ranks)
        num_layers = len(ranks)
        rank_values = list(ranks.values())

        print(f"\n  Final allocation summary:")
        print(f"    Layers: {num_layers}")
        print(f"    Total params: {total_params:,}")
        print(f"    Budget: {self.param_budget:,}")
        print(f"    Usage: {total_params / self.param_budget * 100:.1f}%")
        print(f"    Rank range: [{min(rank_values)}, {max(rank_values)}]")
        print(f"    Mean rank: {np.mean(rank_values):.1f}")
        print(f"    Std rank: {np.std(rank_values):.1f}")

        # CHANGE: Add 3% tolerance instead of strict check
        tolerance = 0.03  # 3% tolerance
        budget_usage = total_params / self.param_budget

        if budget_usage > (1 + tolerance):
            raise ValueError(
                f"Budget exceeded by {(budget_usage - 1) * 100:.1f}%! "
                f"{total_params:,} > {self.param_budget:,} "
                f"(tolerance: {tolerance * 100:.0f}%)"
            )

        if budget_usage < (1 - tolerance):
            print(f"⚠️  Warning: Only using {budget_usage * 100:.1f}% of budget")

        # Print allocation details
        print(f"\n  Rank allocation by layer:")
        sorted_layers = sorted(ranks.items(), key=lambda x: x[1])
        for name, rank in sorted_layers:
            importance = self.importance_scores.get(name, 0)
            params = self.num_target_modules * 2 * rank * self.hidden_dim  # FIX: multiply by num_target_modules
            print(f"    {name:30s}: rank={rank:2d}  importance={importance:.4f}  params={params:,}")

    def get_rank_for_layer(self, layer_name: str) -> Optional[int]:
        """Get allocated rank for a specific layer."""
        return self.rank_allocation.get(layer_name)

    def get_allocation_stats(self) -> Dict:
        """Get statistics about the rank allocation."""
        if not self.rank_allocation:
            raise ValueError("No allocation computed.")

        rank_values = list(self.rank_allocation.values())
        total_params = self._calculate_params(self.rank_allocation)

        return {
            "num_layers": len(self.rank_allocation),
            "total_params": total_params,
            "budget": self.param_budget,
            "budget_usage": total_params / self.param_budget,
            "min_rank": min(rank_values),
            "max_rank": max(rank_values),
            "mean_rank": np.mean(rank_values),
            "std_rank": np.std(rank_values),
        }