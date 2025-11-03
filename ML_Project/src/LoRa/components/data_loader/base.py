from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, List


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

