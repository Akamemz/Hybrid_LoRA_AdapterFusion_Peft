from typing import Dict, List, Optional
from transformers import PreTrainedModel
from peft import (
    get_peft_model,
    LoraConfig,
    AdapterConfig,
    TaskType,
    PeftModel,
    PeftConfig
)
from .base import BasePeftBuilder


class PeftModelBuilder(BasePeftBuilder):
    """
    Applies various PEFT configurations to a given transformer model.
    Supports LoRA, Adapters, AdapterFusion, and Hybrid approaches.
    """

    def __init__(self, model: PreTrainedModel):
        """
        Initializes the builder with a base model.

        Args:
            model (PreTrainedModel): The base transformer model to modify.
        """
        self.model = model
        self.base_model = model  # Keep reference to original model
        print(f"PeftModelBuilder initialized with base model: {model.config._name_or_path}")

    @staticmethod
    def count_parameters(model: PreTrainedModel) -> tuple[int, int]:
        """Helper function to count total and trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    def build(self, peft_config: Dict) -> PreTrainedModel:
        """
        Applies a specific PEFT configuration to the model.

        Args:
            peft_config (Dict): A dictionary specifying the PEFT method and its params.
                                Example: {'method': 'lora', 'r': 8, 'lora_alpha': 16}
                                     or: {'method': 'adapter', 'reduction_factor': 16}
                                     or: {'method': 'adapter_fusion', 'adapters': ['adapter1', 'adapter2']}
                                     or: {'method': 'hybrid', 'lora_config': {...}, 'adapter_configs': [...]}

        Returns:
            The modified model with the PEFT configuration applied.
        """
        method = peft_config.get("method")
        if not method:
            raise ValueError("PEFT configuration must specify a 'method'.")

        print(f"Applying PEFT method: {method}")

        if method == "lora":
            peft_model = self._apply_lora(peft_config)

        elif method == "adapter":
            peft_model = self._apply_adapter(peft_config)

        elif method == "adapter_fusion":
            peft_model = self._apply_adapter_fusion(peft_config)

        elif method == "hybrid":
            peft_model = self._apply_hybrid(peft_config)

        else:
            raise ValueError(f"Unknown PEFT method: {method}")

        # Print parameter counts for verification
        total, trainable = self.count_parameters(peft_model)
        print(f"Model configured. Total params: {total:,}, Trainable params: {trainable:,}")
        print(f"Percentage trainable: {trainable / total * 100:.2f}%")

        return peft_model

    def _apply_lora(self, peft_config: Dict) -> PreTrainedModel:
        """Apply LoRA configuration to the model."""
        # Default LoRA targets for different model architectures
        model_type = self.model.config.model_type
        if model_type == "distilbert":
            default_targets = ["q_lin", "v_lin"]
        elif model_type in ["bert", "roberta"]:
            default_targets = ["query", "value"]
        elif model_type == "gpt2":
            default_targets = ["c_attn"]
        else:
            # Fallback to common names
            default_targets = ["query", "value", "q_proj", "v_proj"]

        lora_config = LoraConfig(
            r=peft_config.get("r", 8),
            lora_alpha=peft_config.get("lora_alpha", 16),
            target_modules=peft_config.get("target_modules", default_targets),
            lora_dropout=peft_config.get("lora_dropout", 0.1),
            bias="none",
            task_type=TaskType.SEQ_CLS
        )

        peft_model = get_peft_model(self.model, lora_config)
        print(f"LoRA configuration applied successfully with r={lora_config.r}")
        return peft_model

    def _apply_adapter(self, peft_config: Dict) -> PreTrainedModel:
        """Apply Adapter configuration to the model."""
        # Adapter configuration (similar to Houlsby et al.)
        adapter_config = AdapterConfig(
            reduction_factor=peft_config.get("reduction_factor", 16),
            non_linearity=peft_config.get("non_linearity", "relu"),
            task_type=TaskType.SEQ_CLS,
            # adapter_type can be "houlsby" or "pfeiffer"
            adapter_type=peft_config.get("adapter_type", "houlsby")
        )

        adapter_name = peft_config.get("adapter_name", "default_adapter")
        peft_model = get_peft_model(self.model, adapter_config, adapter_name=adapter_name)

        print(f"Adapter '{adapter_name}' applied with reduction_factor={adapter_config.reduction_factor}")
        return peft_model

    def _apply_adapter_fusion(self, peft_config: Dict) -> PreTrainedModel:
        """
        Apply AdapterFusion by loading pre-trained adapters and adding fusion layer.

        Note: This requires pre-trained adapter checkpoints.
        """
        adapter_paths = peft_config.get("adapter_paths", [])
        fusion_config = peft_config.get("fusion_config", {})

        if not adapter_paths:
            raise ValueError("AdapterFusion requires 'adapter_paths' to be specified")

        # Start with the base model
        peft_model = self.model

        # Load each pre-trained adapter
        for i, adapter_path in enumerate(adapter_paths):
            adapter_name = f"adapter_{i}"
            print(f"Loading adapter from {adapter_path} as '{adapter_name}'")

            # Load adapter config and weights
            peft_model = PeftModel.from_pretrained(
                peft_model,
                adapter_path,
                adapter_name=adapter_name
            )

        # Apply fusion configuration
        # Note: The actual fusion layer implementation depends on the PEFT library version
        # This is a simplified version
        print(f"AdapterFusion configured with {len(adapter_paths)} adapters")

        # Set all adapters to eval mode except the fusion layer
        for adapter_name in peft_model.peft_config:
            peft_model.set_adapter(adapter_name)
            for param in peft_model.parameters():
                param.requires_grad = False

        # The fusion layer parameters should remain trainable
        # (Implementation depends on PEFT library support for fusion)

        return peft_model

    def _apply_hybrid(self, peft_config: Dict) -> PreTrainedModel:
        """
        Apply Hybrid approach: LoRA + Adapters/AdapterFusion

        This method combines multiple PEFT techniques on the same model.
        """
        print("Applying Hybrid PEFT configuration...")

        # First apply LoRA
        lora_config = peft_config.get("lora_config", {})
        if lora_config:
            lora_config["method"] = "lora"
            self.model = self._apply_lora(lora_config)
            print("LoRA component applied")

        # Then apply Adapters or AdapterFusion
        adapter_config = peft_config.get("adapter_config", {})
        if adapter_config:
            if adapter_config.get("use_fusion", False):
                # Apply AdapterFusion on top of LoRA
                adapter_config["method"] = "adapter_fusion"
                peft_model = self._apply_adapter_fusion(adapter_config)
            else:
                # Apply regular adapters on top of LoRA
                adapter_config["method"] = "adapter"
                peft_model = self._apply_adapter(adapter_config)
            print("Adapter component applied")
        else:
            peft_model = self.model

        print("Hybrid configuration complete")
        return peft_model

    def get_trainable_parameters_info(self, model: PreTrainedModel) -> Dict:
        """
        Get detailed information about trainable parameters.

        Returns:
            Dict with parameter counts and module information
        """
        total_params, trainable_params = self.count_parameters(model)

        # Get trainable module names
        trainable_modules = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_modules.append(name)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": trainable_params / total_params * 100,
            "trainable_modules": trainable_modules[:10],  # First 10 for brevity
            "num_trainable_modules": len(trainable_modules)
        }