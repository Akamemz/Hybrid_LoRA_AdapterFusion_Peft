"""
Phase 1 of BA-LoRA: Gradient-Based Importance Estimation
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader, Subset
from datasets import Dataset


class GradientAnalyzer:
    """
    Analyzes gradient information to estimate layer importance.

    Based on GoRA's metric: I(W) = avg(|W ⊙ G|)
    """

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            target_modules: Optional[List[str]] = None,
            device: str = None
    ):
        """
        Initialize gradient analyzer.

        Args:
            model: Base model to analyze
            tokenizer: Tokenizer for the model
            target_modules: Module names to target (e.g., ['q_lin', 'v_lin'])
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Auto-detect target modules if not specified
        if target_modules is None:
            self.target_modules = self._auto_detect_target_modules()
        else:
            self.target_modules = target_modules

        self.importance_scores = {}
        self.gradients = {}

        print(f"GradientAnalyzer initialized")
        print(f"  Device: {self.device}")
        print(f"  Target modules: {self.target_modules}")

    def _auto_detect_target_modules(self) -> List[str]:
        """Auto-detect target modules based on model architecture."""
        model_type = self.model.config.model_type

        if model_type == "distilbert":
            return ["q_lin", "v_lin"]
        elif model_type in ["bert", "roberta"]:
            return ["query", "value"]
        else:
            return ["query", "value"]

    def accumulate_gradients(
            self,
            train_dataset: Dataset,
            num_samples: int = 1000,
            batch_size: int = 8
    ) -> Dict[str, torch.Tensor]:
        """
        Accumulate gradients over a subset of training data.

        Args:
            train_dataset: Training dataset
            num_samples: Number of samples for gradient accumulation
            batch_size: Batch size

        Returns:
            Dictionary mapping parameter names to accumulated gradients
        """
        print(f"\nAccumulating gradients over {num_samples} samples...")

        # Sample subset uniformly
        num_samples = min(num_samples, len(train_dataset))
        indices = np.random.choice(len(train_dataset), num_samples, replace=False)
        subset = Subset(train_dataset, indices)

        # Create dataloader
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)

        # Move model to device and set to train mode
        self.model.to(self.device)
        self.model.train()

        # Zero all gradients
        self.model.zero_grad()

        # Accumulate gradients
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss

            # Backward pass (accumulate gradients)
            loss.backward()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Processed {num_batches} batches")

        # Store accumulated gradients for target modules
        for name, param in self.model.named_parameters():
            if any(target in name for target in self.target_modules):
                if param.grad is not None:
                    self.gradients[name] = param.grad.clone().detach().cpu()

        print(f"  Stored gradients for {len(self.gradients)} parameters")

        return self.gradients

    def compute_importance_scores(self) -> Dict[str, float]:
        """
        Compute importance scores for each target module.

        Uses metric: I(W) = avg(|W ⊙ G|)

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        print("\nComputing importance scores...")

        if not self.gradients:
            raise ValueError("No gradients accumulated. Call accumulate_gradients() first.")

        self.importance_scores = {}

        for name, param in self.model.named_parameters():
            if name in self.gradients:
                W = param.data.cpu()
                G = self.gradients[name]

                # Compute sensitivity: avg(|W ⊙ G|)
                sensitivity = (W.abs() * G.abs()).mean().item()
                self.importance_scores[name] = sensitivity

        # Print statistics
        if self.importance_scores:
            scores = list(self.importance_scores.values())
            print(f"  Importance score statistics:")
            print(f"    Min:  {min(scores):.6f}")
            print(f"    Max:  {max(scores):.6f}")
            print(f"    Mean: {np.mean(scores):.6f}")
            print(f"    Std:  {np.std(scores):.6f}")

            # Print top 5 most important layers
            sorted_layers = sorted(
                self.importance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            print(f"\n  Top 5 most important layers:")
            for name, score in sorted_layers[:5]:
                print(f"    {name}: {score:.6f}")

        return self.importance_scores

    def get_layer_importance(self) -> Dict[str, float]:
        """
        Aggregate importance scores by layer.

        Returns:
            Dictionary mapping layer identifiers to aggregated importance
        """
        if not self.importance_scores:
            raise ValueError("No importance scores computed.")

        layer_importance = {}

        # Group by layer
        for param_name, score in self.importance_scores.items():
            # Extract layer identifier
            parts = param_name.split('.')

            # Find layer index
            layer_id = None
            for i, part in enumerate(parts):
                if part.isdigit():
                    layer_id = f"layer_{part}"
                    # Include module type
                    if i + 1 < len(parts):
                        layer_id += f"_{parts[i + 1]}"
                    break

            if layer_id:
                if layer_id not in layer_importance:
                    layer_importance[layer_id] = []
                layer_importance[layer_id].append(score)

        # Average scores for each layer
        layer_importance = {
            layer: np.mean(scores)
            for layer, scores in layer_importance.items()
        }

        return layer_importance

    def analyze(
            self,
            train_dataset: Dataset,
            num_samples: int = 1000,
            batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Complete analysis: accumulate gradients and compute importance.

        Args:
            train_dataset: Training dataset
            num_samples: Number of samples
            batch_size: Batch size

        Returns:
            Dictionary of layer importance scores
        """
        self.accumulate_gradients(train_dataset, num_samples, batch_size)
        self.compute_importance_scores()
        return self.get_layer_importance()