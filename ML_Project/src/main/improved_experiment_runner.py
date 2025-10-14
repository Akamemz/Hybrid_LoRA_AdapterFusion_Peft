"""
ML_Project/src/main/improved_experiment_runner.py

Improved experiment runner with:
- Unified PEFT factory usage
- Enhanced dataset support
- Parameter budget enforcement
- Comprehensive logging
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
        """
        Execute complete experiment pipeline.

        Returns:
            Dictionary containing all experiment results
        """
        start_time = datetime.now()

        # Step 1: Load base model and tokenizer
        print("[1/5] Loading base model and tokenizer...")
        base_model, tokenizer = self._load_model()

        # Step 2: Load and prepare dataset
        print("\n[2/5] Loading and preparing dataset...")
        dataset = self._load_dataset(tokenizer)

        # Step 3: Build PEFT model with budget enforcement
        print("\n[3/5] Building PEFT model...")
        peft_model = self._build_peft_model(base_model)

        # Step 4: Train model
        print("\n[4/5] Training model...")
        eval_results = self._train_model(peft_model, tokenizer, dataset)

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

    def _build_peft_model(self, base_model):
        """Build PEFT model with configuration and budget enforcement."""
        # Initialize factory with parameter budget
        factory = PEFTFactory(
            base_model=base_model,
            target_param_budget=self.args.param_budget
        )

        # If no budget specified, suggest matching configurations
        if not self.args.param_budget and self.args.suggest_configs:
            factory.suggest_matching_configs(
                base_r=self.args.lora_r,
                base_reduction=self.args.adapter_reduction_factor
            )

        # Build PEFT configuration based on method
        if self.args.peft_method == "lora":
            config = {
                "r": self.args.lora_r,
                "lora_alpha": self.args.lora_alpha,
                "target_modules": self.args.lora_target_modules or None,
                "lora_dropout": self.args.lora_dropout,
            }

        elif self.args.peft_method == "adapter":
            config = {
                "method": "adapter",
                "adapter_name": self.args.adapter_name,
                "reduction_factor": self.args.adapter_reduction_factor,
                "adapter_type": self.args.adapter_type,
                "non_linearity": self.args.adapter_nonlinearity,
            }

        elif self.args.peft_method == "adapter_fusion":
            if not self.args.adapter_names or not self.args.adapter_paths:
                raise ValueError(
                    "AdapterFusion requires --adapter_names and --adapter_paths"
                )
            config = {
                "method": "adapter_fusion",
                "adapter_names": self.args.adapter_names,
                "adapter_paths": self.args.adapter_paths,
                "fusion_type": self.args.fusion_type,
            }

        elif self.args.peft_method == "hybrid":
            if not self.args.adapter_names or not self.args.adapter_paths:
                raise ValueError(
                    "Hybrid method requires --adapter_names and --adapter_paths"
                )
            config = {
                "adapter_config": {
                    "method": "adapter_fusion",
                    "adapter_names": self.args.adapter_names,
                    "adapter_paths": self.args.adapter_paths,
                    "fusion_type": self.args.fusion_type,
                },
                "lora_config": {
                    "r": self.args.lora_r,
                    "lora_alpha": self.args.lora_alpha,
                    "target_modules": self.args.lora_target_modules or None,
                    "lora_dropout": self.args.lora_dropout,
                }
            }

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
        elif self.args.peft_method == "adapter":
            results["config"]["adapter_config"] = {
                "reduction_factor": self.args.adapter_reduction_factor,
                "adapter_type": self.args.adapter_type,
            }
        elif self.args.peft_method == "hybrid":
            results["config"]["hybrid_config"] = {
                "lora_r": self.args.lora_r,
                "adapter_reduction_factor": self.args.adapter_reduction_factor,
            }

        # Add parameter budget info if specified
        if self.args.param_budget:
            results["parameter_budget"] = {
                "target": self.args.param_budget,
                "used": self.model_info["added_parameters"],
                "usage_percentage": self.model_info["budget_usage_percentage"],
                "within_budget": self.model_info["added_parameters"] <= self.args.param_budget
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
                        choices=["lora", "adapter", "adapter_fusion", "hybrid"],
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

    # Adapter configuration
    parser.add_argument("--adapter_name", type=str, default="default_adapter",
                        help="Name for the adapter")
    parser.add_argument("--adapter_reduction_factor", type=int, default=16,
                        help="Adapter bottleneck reduction factor")
    parser.add_argument("--adapter_type", type=str, default="houlsby",
                        choices=["houlsby", "pfeiffer"],
                        help="Adapter architecture type")
    parser.add_argument("--adapter_nonlinearity", type=str, default="relu",
                        choices=["relu", "gelu", "swish"],
                        help="Adapter non-linearity function")

    # AdapterFusion configuration
    parser.add_argument("--adapter_names", nargs="+", default=None,
                        help="Names of adapters to fuse")
    parser.add_argument("--adapter_paths", nargs="+", default=None,
                        help="Paths to pre-trained adapters")
    parser.add_argument("--fusion_type", type=str, default="dynamic",
                        choices=["dynamic", "static"],
                        help="Fusion mechanism type")

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