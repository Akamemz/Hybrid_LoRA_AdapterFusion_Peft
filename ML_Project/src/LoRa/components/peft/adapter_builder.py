from typing import Dict, List, Optional
from transformers import PreTrainedModel
from adapters import AdapterConfig, AdapterFusionConfig
from .base import BasePeftBuilder


class AdapterBuilder(BasePeftBuilder):
    """
    Handles adapter and AdapterFusion using adapter-transformers library.
    This is separate from LoRA implementation.
    """

    def __init__(self, model: PreTrainedModel):
        """Initialize with a model that supports adapters."""
        # Convert model to support adapters
        from adapters import init
        self.model = init(model)
        print(f"Model initialized with adapter support")

    def build(self, config: Dict) -> PreTrainedModel:
        method = config.get("method")

        if method == "adapter":
            return self._add_adapter(config)
        elif method == "adapter_fusion":
            return self._add_adapter_fusion(config)
        else:
            raise ValueError(f"AdapterBuilder only supports 'adapter' and 'adapter_fusion', got {method}")

    def _add_adapter(self, config: Dict) -> PreTrainedModel:
        """Add a single adapter to the model."""
        adapter_name = config.get("adapter_name", "task_adapter")
        reduction_factor = config.get("reduction_factor", 16)

        # Configure adapter
        adapter_config = AdapterConfig.load(
            "houlsby",  # or "pfeiffer"
            reduction_factor=reduction_factor,
            non_linearity=config.get("non_linearity", "relu")
        )

        # Add adapter
        self.model.add_adapter(adapter_name, config=adapter_config)
        self.model.train_adapter(adapter_name)

        print(f"Added adapter '{adapter_name}' with reduction_factor={reduction_factor}")
        return self.model

    def _add_adapter_fusion(self, config: Dict) -> PreTrainedModel:
        """
        Add AdapterFusion layer that combines multiple pre-trained adapters.
        Requires pre-trained adapter checkpoints.
        """
        adapter_names = config.get("adapter_names", [])
        adapter_paths = config.get("adapter_paths", [])

        if not adapter_names or not adapter_paths:
            raise ValueError("AdapterFusion requires 'adapter_names' and 'adapter_paths'")

        # Load pre-trained adapters
        for name, path in zip(adapter_names, adapter_paths):
            self.model.load_adapter(path, load_as=name)
            print(f"Loaded adapter '{name}' from {path}")

        # Add fusion layer
        fusion_config = AdapterFusionConfig.load("dynamic")
        self.model.add_adapter_fusion(adapter_names, fusion_config)

        # Set which adapters and fusion to train
        self.model.train_adapter_fusion(adapter_names)

        print(f"AdapterFusion configured with adapters: {adapter_names}")
        return self.model


class LoRABuilder(BasePeftBuilder):
    """Handles LoRA using PEFT library."""

    def __init__(self, model: PreTrainedModel):
        self.model = model

    def build(self, config: Dict) -> PreTrainedModel:
        from peft import get_peft_model, LoraConfig, TaskType

        lora_config = LoraConfig(
            r=config.get("r", 8),
            lora_alpha=config.get("lora_alpha", 16),
            target_modules=config.get("target_modules", ["q_lin", "v_lin"]),
            lora_dropout=config.get("lora_dropout", 0.1),
            bias="none",
            task_type=TaskType.SEQ_CLS
        )

        peft_model = get_peft_model(self.model, lora_config)
        print(f"LoRA applied with r={lora_config.r}")
        return peft_model


class HybridBuilder(BasePeftBuilder):
    """
    Combines AdapterFusion + LoRA for your research.
    This is the core innovation of your project.
    """

    def __init__(self, model: PreTrainedModel):
        self.model = model

    def build(self, config: Dict) -> PreTrainedModel:
        """
        Apply hybrid approach:
        1. First set up AdapterFusion (adapter-transformers)
        2. Then apply LoRA on top (PEFT)
        """

        # Step 1: Apply AdapterFusion
        adapter_config = config.get("adapter_config", {})
        if adapter_config:
            adapter_builder = AdapterBuilder(self.model)
            self.model = adapter_builder.build(adapter_config)
            print("✓ AdapterFusion component applied")

        # Step 2: Apply LoRA on top
        lora_config = config.get("lora_config", {})
        if lora_config:
            lora_builder = LoRABuilder(self.model)
            self.model = lora_builder.build(lora_config)
            print("✓ LoRA component applied")

        print("✓ Hybrid AdapterFusion + LoRA configuration complete")
        return self.model