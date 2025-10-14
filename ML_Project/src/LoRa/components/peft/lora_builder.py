"""
LoRA Builder - Pure LoRA implementation using PEFT library
"""

from typing import Dict, Optional
from transformers import PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType
from .base import BasePeftBuilder


class LoRABuilder(BasePeftBuilder):
    """
    Implements LoRA (Low-Rank Adaptation) using the PEFT library.

    LoRA adds trainable low-rank matrices to transformer attention layers,
    allowing efficient fine-tuning with minimal parameters.
    """

    def __init__(self, model: PreTrainedModel):
        """
        Initialize with a base model.

        Args:
            model: Base transformer model (e.g., DistilBERT, BERT, RoBERTa)
        """
        self.model = model
        self.model_type = model.config.model_type
        print(f"LoRABuilder initialized with: {model.config._name_or_path}")
        print(f"Model type: {self.model_type}")

    @staticmethod
    def count_parameters(model: PreTrainedModel) -> tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def _get_default_target_modules(self) -> list[str]:
        """
        Get default target modules based on model architecture.

        Returns:
            List of module names to apply LoRA to
        """
        if self.model_type == "distilbert":
            # DistilBERT attention modules
            return ["q_lin", "v_lin"]

        elif self.model_type in ["bert", "roberta", "albert"]:
            # BERT-style attention modules
            return ["query", "value"]

        elif self.model_type == "gpt2":
            # GPT-2 combined attention module
            return ["c_attn"]

        elif self.model_type in ["t5", "mt5"]:
            # T5 attention modules
            return ["q", "v"]

        elif self.model_type == "llama":
            # LLaMA attention modules
            return ["q_proj", "v_proj"]

        else:
            # Default fallback - try common names
            print(f"Warning: Unknown model type '{self.model_type}', using default targets")
            return ["query", "value", "q_proj", "v_proj"]

    def build(self, config: Dict) -> PreTrainedModel:
        """
        Apply LoRA configuration to the model.

        Args:
            config: Dictionary with LoRA parameters:
                - r (int): Rank of the low-rank matrices (default: 8)
                - lora_alpha (int): Scaling factor (default: 16)
                - target_modules (list): Modules to apply LoRA to (optional)
                - lora_dropout (float): Dropout probability (default: 0.1)
                - bias (str): Bias configuration (default: "none")

        Returns:
            Model with LoRA applied
        """
        # Extract configuration parameters
        r = config.get("r", 8)
        lora_alpha = config.get("lora_alpha", 16)
        target_modules = config.get("target_modules") or self._get_default_target_modules()
        lora_dropout = config.get("lora_dropout", 0.1)
        bias = config.get("bias", "none")

        print(f"\nApplying LoRA configuration:")
        print(f"  Rank (r): {r}")
        print(f"  Alpha: {lora_alpha}")
        print(f"  Target modules: {target_modules}")
        print(f"  Dropout: {lora_dropout}")
        print(f"  Bias: {bias}")

        # Create LoRA configuration
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
        )

        # Apply LoRA to model
        try:
            peft_model = get_peft_model(self.model, lora_config)
            print("✓ LoRA applied successfully")
        except Exception as e:
            print(f"Error applying LoRA: {e}")
            print("Attempting with alternative target modules...")

            # Try alternative targets
            alt_targets = ["attention", "dense"]
            lora_config.target_modules = alt_targets
            peft_model = get_peft_model(self.model, lora_config)
            print(f"✓ LoRA applied with alternative targets: {alt_targets}")

        # Print parameter statistics
        total, trainable = self.count_parameters(peft_model)
        print(f"\nParameter Statistics:")
        print(f"  Total parameters: {total:,}")
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Trainable percentage: {trainable / total * 100:.3f}%")
        print(f"  Parameter reduction: {100 - (trainable / total * 100):.3f}%")

        # Calculate theoretical parameter count
        # For each target module: 2 * r * hidden_dim parameters
        hidden_size = self.model.config.hidden_size
        num_layers = getattr(self.model.config, 'num_hidden_layers',
                             getattr(self.model.config, 'n_layer', None))

        if num_layers:
            theoretical_params = len(target_modules) * num_layers * 2 * r * hidden_size
            print(f"  Theoretical LoRA params: ~{theoretical_params:,}")

        # Print trainable modules
        self._print_trainable_modules(peft_model)

        return peft_model

    def _print_trainable_modules(self, model: PreTrainedModel):
        """Print first few trainable module names."""
        trainable_modules = [
            name for name, param in model.named_parameters()
            if param.requires_grad
        ]

        print(f"\nTrainable modules ({len(trainable_modules)} total):")
        for module in trainable_modules[:5]:
            print(f"  - {module}")

        if len(trainable_modules) > 5:
            print(f"  ... and {len(trainable_modules) - 5} more")

    def get_lora_info(self, model: PreTrainedModel) -> Dict:
        """
        Get detailed information about LoRA configuration.

        Returns:
            Dictionary with LoRA statistics and configuration
        """
        from peft import PeftModel

        if not isinstance(model, PeftModel):
            return {"error": "Model is not a PEFT model"}

        config = model.peft_config
        total, trainable = self.count_parameters(model)

        return {
            "peft_type": "LoRA",
            "lora_config": {
                "r": config.get("default", {}).r if hasattr(config.get("default", {}), 'r') else "N/A",
                "lora_alpha": config.get("default", {}).lora_alpha if hasattr(config.get("default", {}),
                                                                              'lora_alpha') else "N/A",
                "target_modules": config.get("default", {}).target_modules if hasattr(config.get("default", {}),
                                                                                      'target_modules') else "N/A",
            },
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_percentage": trainable / total * 100,
        }