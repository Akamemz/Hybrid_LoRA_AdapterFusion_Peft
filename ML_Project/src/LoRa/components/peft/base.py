# src/components/peft/base.py
from abc import ABC, abstractmethod
from typing import Dict
from transformers import PreTrainedModel


class BasePeftBuilder(ABC):
    """Abstract base class for applying PEFT configurations."""

    @abstractmethod
    def build(self, config: Dict) -> PreTrainedModel:
        """Applies a PEFT configuration to the model and returns it."""
        pass