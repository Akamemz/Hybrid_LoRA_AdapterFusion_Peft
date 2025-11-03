from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseModelLoader(ABC):
    """Abstract base class for model loaders."""

    @abstractmethod
    def load(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Loads a model and its tokenizer."""
        pass