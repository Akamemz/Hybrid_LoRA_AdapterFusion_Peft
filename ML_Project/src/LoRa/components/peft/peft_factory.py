"""
ML_Project/src/LoRa/components/peft/peft_factory.py

FIXED: Proper parameter counting that accounts for classification head
"""

from typing import Dict, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from .lora_builder import LoRABuilder

class PEFTFactory:
    """
    Central factory for creating PEFT models with strict parameter budget control.

    This is the ONLY entry point for building PEFT models in experiments.
    """

    SUPPORTED_METHODS = ["lora", "ba_lora"]

    def __init__(
            self,
            base_model: PreTrainedModel,
            tokenizer: Optional[PreTrainedTokenizer] = None,  # ADD THIS PARAMETER
            target_param_budget: Optional[int] = None
    ):
        """
        Initialize factory with base model and optional parameter budget.

        Args:
            base_model: Base transformer model
            tokenizer: Tokenizer (required for BA-LoRA)  # ADD THIS
            target_param_budget: Maximum trainable parameters allowed
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
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
    def _count_peft_parameters(model: PreTrainedModel) -> int:
        """
        Count only PEFT-specific parameters (LoRA, adapters, etc.).

        Excludes classification head and other task-specific layers.

        Returns:
            Number of trainable PEFT parameters
        """
        peft_params = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Count only parameters with PEFT-specific names
                if any(keyword in name.lower() for keyword in ['lora', 'adapter', 'prefix', 'prompt']):
                    peft_params += param.numel()

        return peft_params

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

        print(f"\n{'=' * 70}")
        print(f"Building PEFT Model: {method.upper()}")
        print(f"{'=' * 70}")

        # Build model based on method
        if method == "lora":
            builder = LoRABuilder(self.base_model)
            peft_model = builder.build(config)

        elif method == "ba_lora":
            # Import here to avoid circular dependency
            from .ba_lora_builder import BALoRABuilder

            if self.tokenizer is None:
                raise ValueError("BA-LoRA requires tokenizer. Pass it to PEFTFactory init.")

            # Add budget to config
            config["param_budget"] = self.target_param_budget

            builder = BALoRABuilder(
                model=self.base_model,
                tokenizer=self.tokenizer,
                param_budget=self.target_param_budget
            )
            peft_model = builder.build(config)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Verify parameter budget
        trainable_after, total_after = self._count_parameters(peft_model)

        # Calculate PEFT parameters (excluding classification head which was already trainable)
        peft_specific_params = self._count_peft_parameters(peft_model)

        # The classification head is typically the last layer and remains trainable
        classification_head_params = trainable_after - peft_specific_params

        print(f"\n{'=' * 70}")
        print("PARAMETER ANALYSIS")
        print(f"{'=' * 70}")
        print(f"Total parameters:           {total_after:>15,}")
        print(f"Frozen (base model):        {total_after - trainable_after:>15,}")
        print(f"Trainable (total):          {trainable_after:>15,}")
        print(f"  ├─ PEFT params:           {peft_specific_params:>15,}")
        print(f"  └─ Task head params:      {classification_head_params:>15,}")
        print(f"Trainable percentage:       {trainable_after / total_after * 100:>14.3f}%")

        if self.target_param_budget:
            budget_usage = peft_specific_params / self.target_param_budget * 100
            print(f"\nParameter Budget (PEFT only):")
            print(f"  Budget allocated:         {self.target_param_budget:>15,}")
            print(f"  PEFT params used:         {peft_specific_params:>15,}")
            print(f"  Budget usage:             {budget_usage:>14.1f}%")
            print(f"{'=' * 70}")

            if peft_specific_params > self.target_param_budget:
                raise ValueError(
                    f"\n❌ PARAMETER BUDGET EXCEEDED!\n"
                    f"   Budget: {self.target_param_budget:,}\n"
                    f"   PEFT params: {peft_specific_params:,}\n"
                    f"   Excess: {peft_specific_params - self.target_param_budget:,}\n"
                    f"\n   Reduce base_rank or increase budget."
                )

            if budget_usage < 90:
                print(f"\n⚠️  Note: Only using {budget_usage:.1f}% of PEFT budget.")
            elif budget_usage > 100:
                print(f"\n❌ PEFT budget exceeded by {budget_usage - 100:.1f}%!")
        else:
            print(f"\nNo budget constraint specified")

        print(f"{'=' * 70}\n")

        return peft_model

    def get_model_info(self, model: PreTrainedModel) -> Dict:
        """Get comprehensive model information."""
        trainable, total = self._count_parameters(model)

        # Count PEFT-specific parameters
        peft_params = self._count_peft_parameters(model)
        task_head_params = trainable - peft_params

        info = {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "peft_parameters": peft_params,
            "added_parameters": peft_params,
            "task_head_parameters": task_head_params,
            "trainable_percentage": trainable / total * 100,
            "parameter_efficiency": (total - trainable) / total * 100,
        }

        if self.target_param_budget:
            info["budget_usage_percentage"] = peft_params / self.target_param_budget * 100
            info["budget_remaining"] = self.target_param_budget - peft_params
            info["within_budget"] = peft_params <= self.target_param_budget

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

        print(f"\n{'='*70}")
        print("PARAMETER BUDGET MATCHING SUGGESTIONS")
        print(f"{'='*70}")
        print(f"Model: {self.base_model.config._name_or_path}")
        print(f"Hidden size: {self.hidden_size}, Layers: {self.num_layers}")
        print(f"\nEstimated PEFT parameters (excluding base model):")
        print(f"  LoRA (r={base_r}):              ~{lora_params:,} parameters")

        # Suggest matching configs
        suggestions = {
            "lora": {
                "r": base_r,
                "lora_alpha": base_r * 2,
                "estimated_params": lora_params
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