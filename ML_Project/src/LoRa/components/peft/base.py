# src/LoRa/components/peft/base.py
from abc import ABC, abstractmethod
from typing import Dict, Union
from transformers import PreTrainedModel
from peft import PeftModel, PeftMixedModel  # Add this import


class BasePeftBuilder(ABC):
    """Abstract base class for applying PEFT configurations."""

    @abstractmethod
    def build(self, config: Dict) -> Union[PreTrainedModel, PeftModel, PeftMixedModel]:
        """
        Applies a PEFT configuration to the model and returns it.

        Args:
            config: Configuration dictionary for the PEFT method

        Returns:
            Model with PEFT applied (can be PeftModel, PeftMixedModel, or PreTrainedModel)
        """
        pass