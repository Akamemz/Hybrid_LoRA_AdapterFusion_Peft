import evaluate
import numpy as np
from typing import Dict, Optional
from datasets import DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from .base import BaseTrainer

class ExperimentTrainer(BaseTrainer):
    """
    Handles the training and evaluation of a PEFT-configured model.
    """

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 dataset: DatasetDict,
                 training_args_dict: Dict,
                 compute_metrics_type: str = "full"):
        """
        Initializes the ExperimentTrainer.

        Args:
            model (PreTrainedModel): The model to be trained (already PEFT-configured).
            tokenizer (PreTrainedTokenizer): The tokenizer for the model.
            dataset (DatasetDict): The dataset containing 'train' and 'validation' splits.
            training_args_dict (Dict): Dictionary of arguments for HF TrainingArguments.
            compute_metrics_type (str): Type of metrics to compute ("accuracy_only" or "full")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.training_args_dict = training_args_dict
        self.compute_metrics_type = compute_metrics_type
        print("ExperimentTrainer initialized.")

    def _compute_metrics_accuracy_only(self, eval_preds):
        """Computes only accuracy metric for evaluation."""
        metric = evaluate.load("accuracy")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def _compute_metrics_full(self, eval_preds):
        """Computes comprehensive metrics including accuracy, F1, precision, and recall."""
        # Load multiple metrics
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")

        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Compute all metrics
        metrics = {}
        metrics.update(accuracy.compute(predictions=predictions, references=labels))

        # For binary classification, we can use 'binary' average
        # For multi-class, we should use 'weighted' or 'macro'
        num_labels = logits.shape[-1]
        average_type = 'binary' if num_labels == 2 else 'weighted'

        metrics.update(f1.compute(predictions=predictions, references=labels, average=average_type))
        metrics.update(precision.compute(predictions=predictions, references=labels, average=average_type))
        metrics.update(recall.compute(predictions=predictions, references=labels, average=average_type))

        # Round all metrics to 4 decimal places for cleaner output
        metrics = {k: round(v, 4) for k, v in metrics.items()}

        return metrics

    def train(self) -> Dict:
        """
        Configures and runs the Hugging Face Trainer.

        Returns:
            Dictionary containing evaluation results
        """
        # Select compute metrics function based on type
        compute_metrics_fn = (
            self._compute_metrics_full if self.compute_metrics_type == "full"
            else self._compute_metrics_accuracy_only
        )

        # Define TrainingArguments
        training_args = TrainingArguments(**self.training_args_dict)
        print("TrainingArguments configured.")

        # Data collator will dynamically pad the inputs to the max length in a batch
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Initialize the Trainer
        hf_trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            compute_metrics=compute_metrics_fn,
            data_collator=data_collator,
        )

        print("Hugging Face Trainer initialized. Starting training...")
        hf_trainer.train()
        print("Training complete.")

        # Evaluate the model
        print("Evaluating model...")
        eval_results = hf_trainer.evaluate()
        print("Evaluation results:", eval_results)

        # Save the final model
        hf_trainer.save_model()
        print(f"Model saved to {training_args.output_dir}")

        return eval_results