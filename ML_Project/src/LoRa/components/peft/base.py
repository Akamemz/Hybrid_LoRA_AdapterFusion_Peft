from abc import ABC, abstractmethod
from typing import Dict
from datasets import DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizer

class BaseModelLoader(ABC):
    """Abstract base class for all model loaders."""
    @abstractmethod
    def load(self) -> (PreTrainedModel, PreTrainedTokenizer):
        """Loads a model and its tokenizer."""
        pass

class BaseDataLoader(ABC):
    """Abstract base class for all data loaders."""
    @abstractmethod
    def load_and_prepare(self, tokenizer: PreTrainedTokenizer) -> DatasetDict:
        """Loads and prepares a dataset for training/evaluation."""
        pass

class BasePeftBuilder(ABC):
    """Abstract base class for applying PEFT configurations."""
    @abstractmethod
    def build(self, config: Dict) -> PreTrainedModel:
        """Applies a PEFT configuration to the model and returns it."""
        pass

class BaseTrainer(ABC):
    """Abstract base class for experiment trainers."""
    @abstractmethod
    def train(self):
        """Runs the training and evaluation loop."""
        pass

