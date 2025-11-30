from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from .base import BaseModelLoader, PreTrainedModel, PreTrainedTokenizer
from typing import Optional, Dict, Any


class HuggingFaceModelLoader(BaseModelLoader):
    """
    A robust implementation for loading transformer models from Hugging Face.

    Supports various model architectures and configuration options.
    """

    def __init__(
            self,
            model_name: str = "distilbert-base-uncased",
            model_class: Optional[type] = None,
            trust_remote_code: bool = False,
            model_kwargs: Optional[Dict[str, Any]] = None,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
            num_labels: Optional[int] = None  # Added parameter for classification
    ):
        """
        Initializes the loader with configuration options.

        Args:
            model_name: The identifier of the model on Hugging Face Hub
            model_class: Specific AutoModel class (e.g., AutoModelForSequenceClassification)
            trust_remote_code: Whether to trust remote code for custom models
            model_kwargs: Additional arguments for model loading
            tokenizer_kwargs: Additional arguments for tokenizer loading
            num_labels: Number of labels for classification tasks (default: 2 for binary)
        """
        self.model_name = model_name
        # Default to sequence classification model if not specified
        self.model_class = model_class or AutoModelForSequenceClassification
        self.trust_remote_code = trust_remote_code
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.num_labels = num_labels

        # Set trust_remote_code in kwargs if specified
        if self.trust_remote_code:
            self.model_kwargs.setdefault('trust_remote_code', True)
            self.tokenizer_kwargs.setdefault('trust_remote_code', True)

        # Add num_labels to model_kwargs if specified and using classification model
        if self.num_labels is not None and self.model_class == AutoModelForSequenceClassification:
            self.model_kwargs['num_labels'] = self.num_labels

        print(f"HuggingFaceModelLoader initialized for model: {self.model_name}")

    def load(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Loads the specified model and tokenizer from Hugging Face Hub.

        Returns:
            A tuple containing the loaded model and tokenizer.

        Raises:
            ValueError: If model or tokenizer cannot be loaded
        """
        print(f"Loading model and tokenizer for '{self.model_name}'...")

        try:
            # Load tokenizer first (faster, helps validate model exists)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **self.tokenizer_kwargs
            )
            print("Tokenizer loaded successfully.")
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer for '{self.model_name}': {e}")

        try:
            # Load model
            model = self.model_class.from_pretrained(
                self.model_name,
                **self.model_kwargs
            )
            print("Model loaded successfully.")
            print(f"Model type: {type(model).__name__}")

            # Print number of labels if it's a classification model
            if hasattr(model, 'config') and hasattr(model.config, 'num_labels'):
                print(f"Number of labels: {model.config.num_labels}")

        except Exception as e:
            raise ValueError(f"Failed to load model '{self.model_name}': {e}")

        return model, tokenizer