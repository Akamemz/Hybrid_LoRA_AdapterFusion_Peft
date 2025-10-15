"""
Use this in terminal to run the script

python -m src.main.improved_experiment_runner \
  --experiment_name ba_lora_test \
  --dataset sst2 \
  --peft_method ba_lora \
  --param_budget 75000 \
  --ba_lora_base_rank 4 \
  --ba_lora_gradient_samples 1000 \
  --ba_lora_use_warmstart \
  --epochs 3

"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from transformers import AutoModelForSequenceClassification, set_seed

# Updated imports using new components
from ..LoRa.components.huggingface_models.huggingface_model_loader import HuggingFaceModelLoader
from ..LoRa.components.data_loader.enhanced_data_loader import UnifiedDatasetLoader, DATASET_CONFIGS
from ..LoRa.components.peft.peft_factory import PEFTFactory
from ..LoRa.components.trainer.experiment_trainer import ExperimentTrainer


class ImprovedExperimentRunner:
    """
    Orchestrates PEFT experiments with strict parameter budget control.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dataset_config = DATASET_CONFIGS[args.dataset]
        self.results_dir = Path(args.output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed for reproducibility
        set_seed(args.seed)

        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT: {args.experiment_name}")
        print(f"{'=' * 80}")
        print(f"Method: {args.peft_method}")
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model_name}")
        print(f"Seed: {args.seed}")
        if args.param_budget:
            print(f"Parameter Budget: {args.param_budget:,}")
        print(f"{'=' * 80}\n")

    def run(self) -> Dict:
        """Execute complete experiment pipeline."""
        start_time = datetime.now()

        # Step 1: Load base model and tokenizer
        print("[1/5] Loading base model and tokenizer...")
        base_model, tokenizer = self._load_model()


        # Step 2: Load and prepare dataset
        print("\n[2/5] Loading and preparing dataset...")
        self.dataset = self._load_dataset(tokenizer)  # CHANGE: add self.

        # Step 3: Build PEFT model with budget enforcement
        print("\n[3/5] Building PEFT model...")
        peft_model = self._build_peft_model(base_model, tokenizer)  # ADD tokenizer

        # Step 4: Train model
        print("\n[4/5] Training model...")
        eval_results = self._train_model(peft_model, tokenizer, self.dataset)  # Use self.dataset

        # Step 5: Save results
        print("\n[5/5] Saving results...")
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        results = self._compile_results(
            peft_model, eval_results, start_time, duration
        )

        self._save_results(results)

        print(f"\n{'=' * 80}")
        print("EXPERIMENT COMPLETE")
        print(f"{'=' * 80}")
        print(f"Duration: {duration:.2f}s ({duration / 60:.2f} minutes)")
        print(f"Best {self.args.metric_for_best_model}: "
              f"{eval_results.get(f'eval_{self.args.metric_for_best_model}', 'N/A'):.4f}")
        print(f"Results saved to: {self.results_dir}")
        print(f"{'=' * 80}\n")

        return results

    def _load_model(self):
        """Load base model and tokenizer."""
        model_loader = HuggingFaceModelLoader(
            model_name=self.args.model_name,
            model_class=AutoModelForSequenceClassification,
            num_labels=self.dataset_config["num_labels"]
        )
        return model_loader.load()

    def _load_dataset(self, tokenizer):
        """Load and prepare dataset."""
        data_loader = UnifiedDatasetLoader(
            dataset_name=self.args.dataset,
            data_dir=self.args.data_dir,
            max_length=self.args.max_length,
            validation_split=self.args.validation_split,
            test_split=self.args.test_split,
            few_shot_n=self.args.few_shot_n,
            seed=self.args.seed
        )
        return data_loader.load_and_prepare(tokenizer)

    def _build_peft_model(self, base_model, tokenizer):  # ADD tokenizer parameter
        """Build PEFT model with configuration and budget enforcement."""
        # Initialize factory with parameter budget AND tokenizer
        factory = PEFTFactory(
            base_model=base_model,
            tokenizer=tokenizer,  # ADD THIS
            target_param_budget=self.args.param_budget
        )

        # If no budget specified, suggest matching configurations
        if self.args.peft_method == "lora":
            config = {
                "r": self.args.lora_r,
                "lora_alpha": self.args.lora_alpha,
                "target_modules": self.args.lora_target_modules or None,
                "lora_dropout": self.args.lora_dropout,
            }

        elif self.args.peft_method == "ba_lora":  # ADD THIS ENTIRE BLOCK
            # BA-LoRA requires the dataset for gradient analysis
            # We need to pass it in the config
            config = {
                "train_dataset": self.dataset["train"],  # From self.dataset loaded earlier
                "base_rank": self.args.ba_lora_base_rank,
                "gradient_samples": self.args.ba_lora_gradient_samples,
                "use_warmstart": self.args.ba_lora_use_warmstart,
                "target_modules": self.args.lora_target_modules or None,
                "lora_alpha": self.args.ba_lora_alpha or (2 * self.args.ba_lora_base_rank),
                "lora_dropout": self.args.lora_dropout,
            }

        elif self.args.peft_method == "adapter":
            # REMOVE THIS ENTIRE BLOCK (adapters no longer supported)
            raise ValueError("Adapter method no longer supported. Use 'lora' or 'ba_lora'")

        # Remove adapter_fusion and hybrid blocks too

        else:
            raise ValueError(f"Unknown PEFT method: {self.args.peft_method}")

        # Build model (will enforce budget if specified)
        peft_model = factory.build(self.args.peft_method, config)

        # Store model info for results
        self.model_info = factory.get_model_info(peft_model)

        return peft_model

    def _train_model(self, model, tokenizer, dataset):
        """Train and evaluate model."""
        output_dir = self.results_dir / "checkpoints"

        training_args = {
            "output_dir": str(output_dir),
            "num_train_epochs": self.args.epochs,
            "per_device_train_batch_size": self.args.batch_size,
            "per_device_eval_batch_size": self.args.batch_size,
            "learning_rate": self.args.learning_rate,
            "warmup_steps": self.args.warmup_steps,
            "weight_decay": self.args.weight_decay,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "logging_dir": str(self.results_dir / "logs"),
            "logging_steps": self.args.logging_steps,
            "fp16": self.args.fp16,
            "remove_unused_columns": True,
            "label_names": ["labels"],
            "metric_for_best_model": self.args.metric_for_best_model,
            "greater_is_better": True,
            "save_total_limit": 2,
            "seed": self.args.seed,
        }

        trainer = ExperimentTrainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            training_args_dict=training_args,
            compute_metrics_type="full"
        )

        return trainer.train()

    def _compile_results(self, model, eval_results, start_time, duration):
        """Compile all experiment results."""
        results = {
            "experiment_name": self.args.experiment_name,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "config": {
                "model": self.args.model_name,
                "dataset": self.args.dataset,
                "peft_method": self.args.peft_method,
                "epochs": self.args.epochs,
                "batch_size": self.args.batch_size,
                "learning_rate": self.args.learning_rate,
                "seed": self.args.seed,
                "max_length": self.args.max_length,
        },
        "model_info": self.model_info,
        "eval_results": eval_results,
    }

        # Add PEFT-specific config
        if self.args.peft_method == "lora":
            results["config"]["lora_config"] = {
                "r": self.args.lora_r,
                "lora_alpha": self.args.lora_alpha,
                "lora_dropout": self.args.lora_dropout,
            }
        elif self.args.peft_method == "ba_lora":
            results["config"]["ba_lora_config"] = {
                "base_rank": self.args.ba_lora_base_rank,
                "gradient_samples": self.args.ba_lora_gradient_samples,
                "use_warmstart": self.args.ba_lora_use_warmstart,
                "alpha": self.args.ba_lora_alpha or (2 * self.args.ba_lora_base_rank),
            }

        # Add parameter budget info if specified
        if self.args.param_budget:
            results["parameter_budget"] = {
                "target": self.args.param_budget,
                "used": self.model_info.get("peft_parameters",
                                            self.model_info.get("added_parameters",
                                                                self.model_info.get("trainable_parameters", 0))),
                "usage_percentage": self.model_info.get("budget_usage_percentage", 0),
                "within_budget": self.model_info.get("within_budget", True)
            }

        return results

    def _save_results(self, results):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.args.experiment_name}_{timestamp}.json"
        filepath = self.results_dir / filename

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {filepath}")

        # Also save a summary CSV for easy comparison
        summary_file = self.results_dir / "experiment_summary.csv"
        summary_row = {
            "experiment": self.args.experiment_name,
            "method": self.args.peft_method,
            "dataset": self.args.dataset,
            "accuracy": results["eval_results"].get("eval_accuracy", 0),
            "f1": results["eval_results"].get("eval_f1", 0),
            "trainable_params": self.model_info["trainable_parameters"],
            "duration_min": results["duration_seconds"] / 60,
        }

        import pandas as pd
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        else:
            df = pd.DataFrame([summary_row])

        df.to_csv(summary_file, index=False)
        print(f"Summary updated in: {summary_file}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run PEFT experiments with parameter budget control"
    )

    # Experiment identification
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Unique name for this experiment")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")

    # Model and data
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Base model from Hugging Face")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to use")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing CSV files (optional)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")

    # Dataset splitting
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="Proportion for validation if not present")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Proportion for test if not present")
    parser.add_argument("--few_shot_n", type=int, default=None,
                        help="Number of examples per class for few-shot")

    # PEFT method
    parser.add_argument("--peft_method", type=str, required=True,
                        choices=["lora", "ba_lora"],  # CHANGE: removed adapter options
                        help="PEFT method to use")

    # Parameter budget
    parser.add_argument("--param_budget", type=int, default=None,
                        help="Maximum trainable parameters (for fair comparison)")
    parser.add_argument("--suggest_configs", action="store_true",
                        help="Suggest configurations with matching parameters")

    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", nargs="+", default=None,
                        help="Target modules for LoRA")

    # BA-LoRA configuration (ADD THIS ENTIRE SECTION)
    parser.add_argument("--ba_lora_base_rank", type=int, default=8,
                        help="BA-LoRA base rank for scaling")
    parser.add_argument("--ba_lora_gradient_samples", type=int, default=1000,
                        help="Number of samples for gradient accumulation")
    parser.add_argument("--ba_lora_use_warmstart", action="store_true",
                        help="Use warm-start initialization for BA-LoRA")
    parser.add_argument("--ba_lora_alpha", type=int, default=None,
                        help="BA-LoRA alpha (default: 2 × base_rank)")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--metric_for_best_model", type=str, default="f1",
                        choices=["accuracy", "f1", "precision", "recall"],
                        help="Metric to use for selecting best model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def main():
    args = parse_arguments()
    runner = ImprovedExperimentRunner(args)
    results = runner.run()
    return results


if __name__ == "__main__":
    main()


# Tl:DR -->
#
# --> ### `epoch: 0.43` - Why So Small?
#
# **An epoch = one complete pass through ALL training data**
#
# SST-2 has **67,349 training examples**. With `batch_size=32`:
# - **Total batches per epoch** = 67,349 ÷ 32 = **2,105 batches**
# - **Your current progress**: 0.43 epochs = **~905 batches processed**
#
# #### Why Show Fractional Epochs?
#
# The Trainer logs progress every `logging_steps=100` batches by default:
# ```
# Step 100:  epoch = 100/2105 = 0.05
# Step 200:  epoch = 200/2105 = 0.10
# Step 300:  epoch = 300/2105 = 0.14
# ...
# Step 905:  epoch = 905/2105 = 0.43  ← You are here!
# ```
#
# This lets you track progress **within** an epoch, not just between epochs.
#
# ## What to Expect Next
#
# Your training will progress through **3 full epochs**:
# ```
# Epoch 0.0 → 1.0:  First complete pass (2,105 batches)
# Epoch 1.0 → 2.0:  Second pass
# Epoch 2.0 → 3.0:  Third pass
#
# Total: ~6,315 batches (3 epochs × 2,105 batches/epoch)
# ```