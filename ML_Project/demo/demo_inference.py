"""
Live Inference Demo Script for BA-LoRA Presentation

This script loads a trained BA-LoRA model and performs inference on test examples.
Can be used as part of the live demonstration.

Usage:
    python demo_inference.py --checkpoint <path_to_checkpoint> --examples "text1" "text2" "text3"
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from pathlib import Path
import sys


class SentimentPredictor:
    """Loads a trained PEFT model and performs sentiment inference."""

    def __init__(self, checkpoint_path: str, base_model: str = "distilbert-base-uncased"):
        """
        Initialize the predictor with a trained checkpoint.

        Args:
            checkpoint_path: Path to the PEFT checkpoint
            base_model: Base model name (default: distilbert-base-uncased)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print(f"Loading tokenizer: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        print(f"Loading base model: {base_model}")
        base_model_obj = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=2  # SST-2 is binary classification
        )

        print(f"Loading PEFT adapter from: {checkpoint_path}")
        self.model = PeftModel.from_pretrained(base_model_obj, checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        self.labels = ["NEGATIVE", "POSITIVE"]
        print("✓ Model loaded successfully!\n")

    def predict(self, text: str):
        """
        Predict sentiment for a single text.

        Args:
            text: Input text to classify

        Returns:
            tuple: (predicted_label, confidence_score)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        return self.labels[predicted_class], confidence

    def predict_batch(self, texts: list):
        """
        Predict sentiment for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            list: List of (text, predicted_label, confidence) tuples
        """
        results = []
        for text in texts:
            label, confidence = self.predict(text)
            results.append((text, label, confidence))
        return results


def display_predictions(results):
    """Display predictions in a nice table format."""
    print("\n" + "="*100)
    print("SENTIMENT CLASSIFICATION RESULTS")
    print("="*100 + "\n")

    print("┌─" + "─"*95 + "┐")
    print("│ {:<60} │ {:<15} │ {:<15} │".format("Text", "Prediction", "Confidence"))
    print("├─" + "─"*95 + "┤")

    for text, label, confidence in results:
        truncated = text[:60] + "..." if len(text) > 60 else text
        color = "\033[92m" if label == "POSITIVE" else "\033[91m"  # Green for positive, red for negative
        reset = "\033[0m"

        print("│ {:<60} │ {}{:<15}{} │ {:<15} │".format(
            truncated,
            color,
            label,
            reset,
            f"{confidence:.2%}"
        ))

    print("└─" + "─"*95 + "┘\n")


def find_latest_checkpoint(results_dir: str = "results/results_sst2"):
    """Find the most recent checkpoint in the results directory."""
    results_path = Path(results_dir)

    if not results_path.exists():
        return None

    # Look for checkpoint directories
    checkpoints = list(results_path.glob("**/checkpoint-*"))

    if not checkpoints:
        return None

    # Return the most recent one (by modification time)
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def main():
    parser = argparse.ArgumentParser(description="Live Inference Demo for BA-LoRA")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the PEFT checkpoint directory"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="distilbert-base-uncased",
        help="Base model name (default: distilbert-base-uncased)"
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        default=None,
        help="Custom examples to classify (space-separated)"
    )

    args = parser.parse_args()

    # Find checkpoint if not specified
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        print("No checkpoint specified, searching for latest checkpoint...")
        checkpoint_path = find_latest_checkpoint()

        if checkpoint_path is None:
            print("\n⚠ Error: No checkpoint found!")
            print("\nPlease specify a checkpoint path using --checkpoint")
            print("Example: python demo_inference.py --checkpoint results/results_sst2/checkpoint-1000")
            sys.exit(1)

        print(f"Found checkpoint: {checkpoint_path}")

    # Default examples if none provided
    if args.examples is None:
        examples = [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "Terrible experience. Complete waste of time and money.",
            "The film had some good moments but overall was disappointing.",
            "An masterpiece of cinema with brilliant performances throughout.",
            "Not the worst, but definitely not worth watching again.",
            "Outstanding! One of the best films I've ever seen.",
            "Boring and predictable. Fell asleep halfway through."
        ]
    else:
        examples = args.examples

    print("\n" + "="*100)
    print("BA-LORA LIVE INFERENCE DEMO")
    print("="*100 + "\n")

    # Load model
    try:
        predictor = SentimentPredictor(checkpoint_path, args.base_model)
    except Exception as e:
        print(f"\n⚠ Error loading model: {e}")
        print("\nMake sure the checkpoint path is correct and contains:")
        print("  - adapter_config.json")
        print("  - adapter_model.safetensors (or adapter_model.bin)")
        sys.exit(1)

    # Perform predictions
    print("Processing examples...\n")
    results = predictor.predict_batch(examples)

    # Display results
    display_predictions(results)

    print("✓ Demo complete!")
    print(f"\nProcessed {len(examples)} examples using checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
