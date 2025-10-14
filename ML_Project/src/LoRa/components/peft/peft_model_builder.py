from typing import Dict, List, Optional
from transformers import PreTrainedModel
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)

from .base import BasePeftBuilder


class PeftModelBuilder(BasePeftBuilder):
    """
    Applies various PEFT configurations to a given transformer model.
    Currently supports: LoRA and LoRA-based hybrid approaches.

    Note: This implementation uses the PEFT library which primarily supports LoRA.
    For true adapter-based methods and AdapterFusion, you would need the
    'adapter-transformers' library instead.
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
                                     or: {'method': 'bottleneck', 'r': 8}
                                     or: {'method': 'hybrid', 'lora_config': {...}, 'bottleneck_config': {...}}

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
            # Use bottleneck adapter implementation via LoRA on FFN layers
            print("Note: Using bottleneck adapter simulation via LoRA on FFN layers")
            peft_model = self._apply_bottleneck_adapter(peft_config)

        elif method == "adapter_fusion":
            raise NotImplementedError(
                "AdapterFusion requires the 'adapter-transformers' library. "
                "The PEFT library does not support AdapterFusion. "
                "Please install adapter-transformers: pip install adapter-transformers"
            )

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

    def _apply_bottleneck_adapter(self, peft_config: Dict) -> PreTrainedModel:
        """
        Simulate bottleneck adapters using LoRA on FFN layers.
        This is a workaround since PEFT doesn't natively support adapters.

        For true adapter implementation, use adapter-transformers library.
        """
        # Target FFN/MLP layers to simulate adapter behavior
        model_type = self.model.config.model_type

        if model_type == "distilbert":
            # DistilBERT has FFN in each transformer block
            target_modules = ["ffn.lin1", "ffn.lin2"]
        elif model_type in ["bert", "roberta"]:
            target_modules = ["intermediate.dense", "output.dense"]
        elif model_type == "gpt2":
            target_modules = ["mlp.c_fc", "mlp.c_proj"]
        else:
            # Fallback
            target_modules = ["intermediate", "output"]

        # Convert reduction_factor to LoRA rank
        # reduction_factor of 16 ≈ r of 48 for 768-dim models
        reduction_factor = peft_config.get("reduction_factor", 16)
        hidden_size = self.model.config.hidden_size
        r = max(4, hidden_size // reduction_factor)

        lora_config = LoraConfig(
            r=r,
            lora_alpha=r * 2,  # Common practice: alpha = 2*r
            target_modules=target_modules,
            lora_dropout=peft_config.get("dropout", 0.1),
            bias="none",
            task_type=TaskType.SEQ_CLS
        )

        peft_model = get_peft_model(self.model, lora_config)
        print(f"Bottleneck adapter (via LoRA on FFN) applied with r={r} (reduction_factor={reduction_factor})")
        return peft_model

    def _apply_hybrid(self, peft_config: Dict) -> PreTrainedModel:
        """
        Apply Hybrid approach: LoRA on attention + LoRA on FFN (simulating adapters)

        This simulates the hybrid LoRA+Adapter approach by applying LoRA to both
        attention and FFN layers simultaneously.
        """
        print("Applying Hybrid PEFT configuration (LoRA on attention + FFN)...")

        model_type = self.model.config.model_type

        # Get attention targets
        if model_type == "distilbert":
            attention_targets = ["q_lin", "v_lin"]
            ffn_targets = ["ffn.lin1", "ffn.lin2"]
        elif model_type in ["bert", "roberta"]:
            attention_targets = ["query", "value"]
            ffn_targets = ["intermediate.dense", "output.dense"]
        elif model_type == "gpt2":
            attention_targets = ["c_attn"]
            ffn_targets = ["mlp.c_fc", "mlp.c_proj"]
        else:
            attention_targets = ["query", "value"]
            ffn_targets = ["intermediate", "output"]

        # Combine targets
        all_targets = attention_targets + ffn_targets

        # Get configs
        lora_config_dict = peft_config.get("lora_config", {})
        adapter_config_dict = peft_config.get("adapter_config", {})

        # Use LoRA r for attention, derive FFN r from reduction_factor
        lora_r = lora_config_dict.get("r", 8)
        reduction_factor = adapter_config_dict.get("reduction_factor", 16)
        hidden_size = self.model.config.hidden_size
        adapter_r = max(4, hidden_size // reduction_factor)

        # Average the two ranks for a compromise
        combined_r = (lora_r + adapter_r) // 2

        lora_config = LoraConfig(
            r=combined_r,
            lora_alpha=combined_r * 2,
            target_modules=all_targets,
            lora_dropout=lora_config_dict.get("lora_dropout", 0.1),
            bias="none",
            task_type=TaskType.SEQ_CLS
        )

        peft_model = get_peft_model(self.model, lora_config)
        print(f"Hybrid configuration complete: r={combined_r}, targeting {len(all_targets)} module types")
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