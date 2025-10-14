"""
ML_Project/src/LoRa/components/peft/peft_factory.py

FIXED: Proper parameter counting that accounts for classification head
"""

from typing import Dict, Optional, Tuple
from transformers import PreTrainedModel
from .lora_builder import LoRABuilder
from .adapter_builder import AdapterBuilder
from .hybrid_builder import HybridBuilder


class PEFTFactory:
    """
    Central factory for creating PEFT models with strict parameter budget control.

    This is the ONLY entry point for building PEFT models in experiments.
    """

    SUPPORTED_METHODS = ["lora", "adapter", "adapter_fusion", "hybrid"]

    def __init__(self, base_model: PreTrainedModel, target_param_budget: Optional[int] = None):
        """
        Initialize factory with base model and optional parameter budget.

        Args:
            base_model: Base transformer model
            target_param_budget: Maximum trainable parameters allowed (for fair comparison)
        """
        self.base_model = base_model
        self.target_param_budget = target_param_budget

        # Count parameters BEFORE freezing anything
        # Note: Classification head is typically trainable by default
        self.base_trainable, self.base_total = self._count_parameters(base_model)

        print(f"PEFTFactory initialized")
        print(f"  Base model: {base_model.config._name_or_path}")
        print(f"  Base total params: {self.base_total:,}")
        print(f"  Base trainable params: {self.base_trainable:,}")
        if target_param_budget:
            print(f"  Target param budget: {target_param_budget:,}")

        # Store model info
        self.model_type = base_model.config.model_type
        self.hidden_size = base_model.config.hidden_size
        self.num_layers = getattr(base_model.config, 'num_hidden_layers',
                                  getattr(base_model.config, 'n_layer', None))

    @staticmethod
    def _count_parameters(model: PreTrainedModel) -> Tuple[int, int]:
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total

    def build(self, method: str, config: Dict) -> PreTrainedModel:
        """
        Build PEFT model with specified method and configuration.

        Args:
            method: One of ['lora', 'adapter', 'adapter_fusion', 'hybrid']
            config: Method-specific configuration dictionary

        Returns:
            PEFT-configured model

        Raises:
            ValueError: If method is unsupported or budget is exceeded
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported: {self.SUPPORTED_METHODS}"
            )

        print(f"\n{'='*70}")
        print(f"Building PEFT Model: {method.upper()}")
        print(f"{'='*70}")

        # Build model based on method
        if method == "lora":
            builder = LoRABuilder(self.base_model)
            peft_model = builder.build(config)

        elif method == "adapter":
            builder = AdapterBuilder(self.base_model)
            # Ensure method is set for single adapter
            config["method"] = "adapter"
            peft_model = builder.build(config)

        elif method == "adapter_fusion":
            builder = AdapterBuilder(self.base_model)
            config["method"] = "adapter_fusion"
            peft_model = builder.build(config)

        elif method == "hybrid":
            builder = HybridBuilder(self.base_model)
            peft_model = builder.build(config)

        # Verify parameter budget
        trainable_after, total_after = self._count_parameters(peft_model)

        # Calculate PEFT parameters (excluding classification head which was already trainable)
        # The classification head is typically the last layer and remains trainable
        peft_specific_params = trainable_after - self.base_trainable

        print(f"\n{'='*70}")
        print("PARAMETER ANALYSIS")
        print(f"{'='*70}")
        print(f"Total parameters:           {total_after:>15,}")
        print(f"Trainable (before PEFT):    {self.base_trainable:>15,}")
        print(f"Trainable (after PEFT):     {trainable_after:>15,}")
        print(f"Added by PEFT:              {peft_specific_params:>15,}")
        print(f"Trainable percentage:       {trainable_after/total_after*100:>14.3f}%")

        if self.target_param_budget:
            budget_usage = peft_specific_params / self.target_param_budget * 100
            print(f"Budget allocated:           {self.target_param_budget:>15,}")
            print(f"Budget usage:               {budget_usage:>14.1f}%")
            print(f"{'='*70}")

            if peft_specific_params > self.target_param_budget:
                raise ValueError(
                    f"\n❌ PARAMETER BUDGET EXCEEDED!\n"
                    f"   Budget: {self.target_param_budget:,}\n"
                    f"   Used: {peft_specific_params:,}\n"
                    f"   Excess: {peft_specific_params - self.target_param_budget:,}\n"
                    f"\n   Reduce rank/increase reduction factor and try again."
                )

            if budget_usage < 50:
                print(f"\n⚠️  Warning: Only using {budget_usage:.1f}% of budget.")
                print(f"   Consider increasing model capacity for better performance.")
            elif budget_usage > 100:
                print(f"\n❌ Error: Budget exceeded by {budget_usage - 100:.1f}%!")
        else:
            print(f"No budget constraint specified")

        print(f"{'='*70}\n")

        return peft_model

    def get_model_info(self, model: PreTrainedModel) -> Dict:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with parameter counts and efficiency metrics
        """
        trainable, total = self._count_parameters(model)
        added_params = trainable - self.base_trainable

        info = {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "base_trainable_parameters": self.base_trainable,
            "added_parameters": added_params,
            "trainable_percentage": trainable / total * 100,
            "parameter_efficiency": (total - trainable) / total * 100,
        }

        if self.target_param_budget:
            info["budget_usage_percentage"] = added_params / self.target_param_budget * 100
            info["budget_remaining"] = self.target_param_budget - added_params
            info["within_budget"] = added_params <= self.target_param_budget

        return info

    @staticmethod
    def calculate_lora_params(hidden_size: int, num_layers: int, r: int,
                             target_modules: int = 2) -> int:
        """
        Calculate theoretical LoRA parameters.

        Args:
            hidden_size: Model hidden dimension
            num_layers: Number of transformer layers
            r: LoRA rank
            target_modules: Number of modules per layer (default: 2 for q,v)

        Returns:
            Approximate number of trainable parameters
        """
        # Each LoRA module adds 2 * r * hidden_size parameters
        params_per_layer = target_modules * 2 * r * hidden_size
        total_params = params_per_layer * num_layers
        return total_params

    @staticmethod
    def calculate_adapter_params(hidden_size: int, num_layers: int,
                                 reduction_factor: int) -> int:
        """
        Calculate theoretical adapter parameters.

        Args:
            hidden_size: Model hidden dimension
            num_layers: Number of transformer layers
            reduction_factor: Bottleneck reduction factor

        Returns:
            Approximate number of trainable parameters
        """
        bottleneck_size = hidden_size // reduction_factor
        # Adapter has: down projection + up projection + layer norms
        params_per_adapter = (hidden_size * bottleneck_size +
                             bottleneck_size * hidden_size +
                             2 * hidden_size)  # layer norm parameters
        total_params = params_per_adapter * num_layers * 2  # 2 adapters per layer
        return total_params

    def suggest_matching_configs(self, base_r: int = 8,
                                 base_reduction: int = 16) -> Dict:
        """
        Suggest configurations that roughly match in parameter count.

        Args:
            base_r: Base LoRA rank
            base_reduction: Base adapter reduction factor

        Returns:
            Dictionary of suggested configurations
        """
        if not self.num_layers:
            print("⚠️  Cannot suggest configs: number of layers unknown")
            return {}

        lora_params = self.calculate_lora_params(self.hidden_size, self.num_layers, base_r)
        adapter_params = self.calculate_adapter_params(self.hidden_size, self.num_layers, base_reduction)

        print(f"\n{'='*70}")
        print("PARAMETER BUDGET MATCHING SUGGESTIONS")
        print(f"{'='*70}")
        print(f"Model: {self.base_model.config._name_or_path}")
        print(f"Hidden size: {self.hidden_size}, Layers: {self.num_layers}")
        print(f"\nEstimated PEFT parameters (excluding base model):")
        print(f"  LoRA (r={base_r}):              ~{lora_params:,} parameters")
        print(f"  Adapter (rf={base_reduction}):  ~{adapter_params:,} parameters")

        # Suggest matching configs
        suggestions = {
            "lora": {
                "r": base_r,
                "lora_alpha": base_r * 2,
                "estimated_params": lora_params
            },
            "adapter": {
                "reduction_factor": base_reduction,
                "estimated_params": adapter_params
            },
            "hybrid_balanced": {
                "lora_config": {"r": base_r // 2, "lora_alpha": base_r},
                "adapter_config": {"reduction_factor": base_reduction * 2},
                "estimated_params": (lora_params // 2) + (adapter_params // 2)
            }
        }

        print(f"\nSuggested configurations for fair comparison:")
        for method, config in suggestions.items():
            est_params = config.pop('estimated_params', None)
            print(f"\n  {method}:")
            print(f"    Config: {config}")
            if est_params:
                print(f"    Estimated params: ~{est_params:,}")
        print(f"{'='*70}\n")

        return suggestions