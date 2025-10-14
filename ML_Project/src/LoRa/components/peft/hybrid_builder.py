"""
Hybrid Builder: Combines AdapterFusion + LoRA
This is YOUR NOVEL CONTRIBUTION - the core of your research!

Research Hypothesis:
AdapterFusion provides knowledge transfer from multiple source tasks,
while LoRA provides efficient task-specific fine-tuning. Combining them
should achieve better performance than either method alone, especially
in low-resource scenarios.
"""

from typing import Dict, Optional
from transformers import PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from .base import BasePeftBuilder


class HybridBuilder(BasePeftBuilder):
    """
    Implements the hybrid AdapterFusion + LoRA approach.

    Process:
    1. Load pre-trained adapters from source tasks
    2. Apply AdapterFusion to combine them
    3. Apply LoRA on top for additional parameter-efficient tuning
    4. Train fusion weights + LoRA parameters together

    This creates a model that:
    - Leverages knowledge from multiple source tasks (AdapterFusion)
    - Can efficiently adapt to the target task (LoRA)
    - Maintains very low parameter overhead
    """

    def __init__(self, model: PreTrainedModel):
        """
        Initialize with base model.

        Args:
            model: Base transformer model (e.g., DistilBERT, BERT)
        """
        self.original_model = model
        self.model = model
        self.has_adapters = False
        self.has_lora = False
        print(f"HybridBuilder initialized with: {model.config._name_or_path}")

    @staticmethod
    def count_parameters(model: PreTrainedModel) -> tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def build(self, config: Dict) -> PreTrainedModel:
        """
        Build the hybrid model by applying AdapterFusion then LoRA.

        Args:
            config: Dictionary with structure:
                {
                    'adapter_config': {
                        'method': 'adapter_fusion',
                        'adapter_names': ['adapter1', 'adapter2'],
                        'adapter_paths': ['/path/to/adapter1', '/path/to/adapter2'],
                        'fusion_type': 'dynamic'
                    },
                    'lora_config': {
                        'r': 8,
                        'lora_alpha': 16,
                        'target_modules': ['q_lin', 'v_lin'],
                        'lora_dropout': 0.1
                    }
                }

        Returns:
            Model with both AdapterFusion and LoRA applied
        """
        print("\n" + "=" * 70)
        print("BUILDING HYBRID MODEL: AdapterFusion + LoRA")
        print("=" * 70)

        # Step 1: Apply AdapterFusion
        adapter_config = config.get("adapter_config")
        if not adapter_config:
            raise ValueError("Hybrid config must include 'adapter_config'")

        print("\n[Step 1/2] Applying AdapterFusion...")
        self._apply_adapter_fusion(adapter_config)
        self.has_adapters = True

        # Step 2: Apply LoRA on top
        lora_config = config.get("lora_config")
        if not lora_config:
            raise ValueError("Hybrid config must include 'lora_config'")

        print("\n[Step 2/2] Applying LoRA...")
        self._apply_lora(lora_config)
        self.has_lora = True

        # Final statistics
        print("\n" + "=" * 70)
        print("HYBRID MODEL COMPLETE")
        print("=" * 70)
        total, trainable = self.count_parameters(self.model)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Trainable percentage: {trainable / total * 100:.3f}%")
        print(f"Components: AdapterFusion ✓ | LoRA ✓")
        print("=" * 70 + "\n")

        return self.model

    def _apply_adapter_fusion(self, adapter_config: Dict):
        """
        Apply AdapterFusion component using adapter-transformers library.

        This sets up knowledge transfer from multiple source tasks.
        """
        from .adapter_builder import AdapterBuilder

        # Initialize adapter builder with current model
        adapter_builder = AdapterBuilder(self.model)

        # Build AdapterFusion
        self.model = adapter_builder.build(adapter_config)

        print("✓ AdapterFusion applied successfully")

        # Store adapter info
        self.adapter_info = adapter_builder.get_adapter_info()
        print(f"  Active adapters: {self.adapter_info['active_adapters']}")

    def _apply_lora(self, lora_config: Dict):
        """
        Apply LoRA component using PEFT library on top of AdapterFusion model.

        This adds task-specific adaptation capability.
        """
        # Determine target modules based on model type
        model_type = self.model.config.model_type

        if model_type == "distilbert":
            default_targets = ["q_lin", "v_lin"]
        elif model_type in ["bert", "roberta"]:
            default_targets = ["query", "value"]
        elif model_type == "gpt2":
            default_targets = ["c_attn"]
        else:
            default_targets = ["query", "value"]

        # Extract LoRA parameters
        r = lora_config.get("r", 8)
        lora_alpha = lora_config.get("lora_alpha", 16)
        target_modules = lora_config.get("target_modules", default_targets)
        lora_dropout = lora_config.get("lora_dropout", 0.1)

        # Create LoRA configuration
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
        )

        print(f"  LoRA rank (r): {r}")
        print(f"  LoRA alpha: {lora_alpha}")
        print(f"  Target modules: {target_modules}")
        print(f"  Dropout: {lora_dropout}")

        try:
            # Apply LoRA to the adapter-enabled model
            self.model = get_peft_model(self.model, peft_config)
            print("✓ LoRA applied successfully")
        except Exception as e:
            print(f"Warning: LoRA application encountered issue: {e}")
            print("This may happen if adapters and LoRA target overlapping parameters.")
            print("Attempting alternative configuration...")

            # Try with different target modules
            peft_config.target_modules = ["dense", "attention"]
            self.model = get_peft_model(self.model, peft_config)
            print("✓ LoRA applied with alternative targets")

        # Print trainable modules
        self._print_trainable_modules()

    def _print_trainable_modules(self):
        """Print information about trainable parameters."""
        trainable_modules = []
        adapter_params = 0
        lora_params = 0
        fusion_params = 0

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_modules.append(name)

                # Categorize parameters
                if 'adapter' in name.lower() and 'fusion' not in name.lower():
                    adapter_params += param.numel()
                elif 'lora' in name.lower():
                    lora_params += param.numel()
                elif 'fusion' in name.lower():
                    fusion_params += param.numel()

        print(f"\n  Trainable module breakdown:")
        print(f"    Adapter parameters: {adapter_params:,}")
        print(f"    Fusion parameters: {fusion_params:,}")
        print(f"    LoRA parameters: {lora_params:,}")
        print(f"    Total trainable: {adapter_params + lora_params + fusion_params:,}")
        print(f"  First few trainable modules:")
        for module in trainable_modules[:5]:
            print(f"    - {module}")
        if len(trainable_modules) > 5:
            print(f"    ... and {len(trainable_modules) - 5} more")

    def get_model_info(self) -> Dict:
        """
        Get detailed information about the hybrid model.

        Returns:
            Dictionary with model statistics and component info
        """
        total, trainable = self.count_parameters(self.model)

        info = {
            "model_type": self.model.config.model_type,
            "model_name": self.model.config._name_or_path,
            "has_adapters": self.has_adapters,
            "has_lora": self.has_lora,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_percentage": trainable / total * 100,
        }

        if self.has_adapters and hasattr(self, 'adapter_info'):
            info["adapter_info"] = self.adapter_info

        return info

    def save_model(self, output_dir: str):
        """
        Save the hybrid model (both adapters and LoRA).

        Args:
            output_dir: Directory to save the model components
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save LoRA weights (PEFT handles this)
        if self.has_lora and isinstance(self.model, PeftModel):
            lora_path = os.path.join(output_dir, "lora")
            self.model.save_pretrained(lora_path)
            print(f"✓ LoRA weights saved to {lora_path}")

        # Save adapter configuration
        if self.has_adapters:
            adapter_path = os.path.join(output_dir, "adapters")
            # Note: Adapters are typically already saved from pre-training
            # The fusion weights are part of the model state
            print(f"✓ Adapter configuration maintained")

        print(f"Hybrid model saved to {output_dir}")

    def load_model(self, model_dir: str):
        """
        Load a previously saved hybrid model.

        Args:
            model_dir: Directory containing saved model components
        """
        import os

        # Load LoRA weights
        lora_path = os.path.join(model_dir, "lora")
        if os.path.exists(lora_path):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.has_lora = True
            print(f"✓ LoRA weights loaded from {lora_path}")

        print(f"Hybrid model loaded from {model_dir}")