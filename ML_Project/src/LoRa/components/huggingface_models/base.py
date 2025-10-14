from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizer

class BaseModelLoader(ABC):
    """
    Abstract Base Class for model loaders.

    This class defines the interface that all model loader components must
    adhere to. It ensures that any loader we create will have a consistent
    way of loading a model and its tokenizer.
    """

    @abstractmethod
    def load(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Loads a pre-trained model and its corresponding tokenizer.

        This method must be implemented by any concrete subclass.

        Returns:
            A tuple containing the loaded pre-trained model and tokenizer.
        """
        pass
