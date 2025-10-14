from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, List


# --- BaseModelLoader ---
class BaseModelLoader(ABC):
    """Abstract base class for loading transformer models."""

    @abstractmethod
    def load(self, num_labels: int) -> Tuple[Any, Any]:
        """
        Loads a model and its corresponding tokenizer.

        Args:
            num_labels (int): The number of labels for the classification task.

        Returns:
            A tuple containing the loaded model and tokenizer.
        """
        pass


# --- BaseDataLoader ---
class BaseDataLoader(ABC):
    """Abstract base class for loading and preparing datasets."""

    @abstractmethod
    def load_and_prepare(self, tokenizer: Any) -> Any:
        """
        Loads the dataset and prepares it for training by tokenizing.

        Args:
            tokenizer: The tokenizer to use for processing the text data.

        Returns:
            The processed dataset, ready for the Trainer.
        """
        pass


# --- BasePeftModelBuilder ---
class BasePeftModelBuilder(ABC):
    """Abstract base class for applying PEFT configurations to a model."""

    @abstractmethod
    def build(self, peft_config: Dict[str, Any]) -> Any:
        """
        Applies a PEFT configuration to the base model.

        Args:
            peft_config: A dictionary containing the PEFT method and its parameters.

        Returns:
            The modified PEFT model.
        """
        pass


# --- BaseTrainer ---
class BaseTrainer(ABC):
    """Abstract base class for an experiment trainer."""

    @abstractmethod
    def train(self):
        """
        Executes the training and evaluation loop.
        """
        pass

