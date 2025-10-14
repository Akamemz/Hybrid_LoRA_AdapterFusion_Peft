"""
Updated Main Experiment Runner
Supports: LoRA, Adapters, AdapterFusion, and Hybrid (AdapterFusion + LoRA)
"""

import os
import argparse
from transformers import AutoModelForSequenceClassification

# Import loaders
from ..LoRa.components.huggingface_models.huggingface_model_loader import HuggingFaceModelLoader
from ..LoRa.components.data_loader.huggingface_data_loader import LocalCsvDatasetLoader
from LoRa.components.trainer.experiment_trainer import ExperimentTrainer

# --- Dataset Configuration ---
DATASET_CONFIG = {
    "sst2": {"text_col": "sentence", "label_col": "label", "num_labels": 2},
    "ag_news": {"text_col": "text", "label_col": "label", "num_labels": 4},
    "imdb": {"text_col": "text", "label_col": "label", "num_labels": 2},
    "yelp": {"text_col": "text", "label_col": "label", "num_labels": 2},
}


def build_peft_model(base_model, args):
    """
    Build PEFT model based on specified method.
    Supports: lora, adapter, adapter_fusion, hybrid
    """

    if args.peft_method == "lora":
        # Pure LoRA approach
        from ..LoRa.components.peft.lora_builder import LoRABuilder

        builder = LoRABuilder(base_model)
        config = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.lora_target_modules if args.lora_target_modules else None,
            "lora_dropout": args.lora_dropout,
        }
        return builder.build(config)

    elif args.peft_method == "adapter":
        # Single adapter approach
        from ..LoRa.components.peft.adapter_builder import AdapterBuilder

        builder = AdapterBuilder(base_model)
        config = {
            "method": "adapter",
            "adapter_name": args.adapter_name,
            "reduction_factor": args.adapter_reduction_factor,
            "adapter_type": args.adapter_type,
            "non_linearity": args.adapter_nonlinearity,
        }
        return builder.build(config)

    elif args.peft_method == "adapter_fusion":
        # AdapterFusion approach (knowledge transfer from multiple tasks)
        from ..LoRa.components.peft.adapter_builder import AdapterBuilder

        if not args.adapter_names or not args.adapter_paths:
            raise ValueError(
                "AdapterFusion requires --adapter_names and --adapter_paths. "
                "Example: --adapter_names sst2_adapter agnews_adapter "
                "--adapter_paths ./results/sst2_adapter ./results/agnews_adapter"
            )

        builder = AdapterBuilder(base_model)
        config = {
            "method": "adapter_fusion",
            "adapter_names": args.adapter_names,
            "adapter_paths": args.adapter_paths,
            "fusion_type": args.fusion_type,
        }
        return builder.build(config)

    elif args.peft_method == "hybrid":
        # HYBRID: Your novel contribution! AdapterFusion + LoRA
        from ..LoRa.components.peft.hybrid_builder import HybridBuilder

        if not args.adapter_names or not args.adapter_paths:
            raise ValueError(
                "Hybrid method requires pre-trained adapters. "
                "Use --adapter_names and --adapter_paths"
            )

        builder = HybridBuilder(base_model)
        config = {
            "adapter_config": {
                "method": "adapter_fusion",
                "adapter_names": args.adapter_names,
                "adapter_paths": args.adapter_paths,
                "fusion_type": args.fusion_type,
            },
            "lora_config": {
                "r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "target_modules": args.lora_target_modules if args.lora_target_modules else None,
                "lora_dropout": args.lora_dropout,
            }
        }
        return builder.build(config)

    else:
        raise ValueError(f"Unknown PEFT method: {args.peft_method}")


def main(args):
    """
    Main function to orchestrate the experiment pipeline.
    """
    print("\n" + "=" * 80)
    print(f"STARTING EXPERIMENT: {args.peft_method.upper()}")
    print("=" * 80 + "\n")

    # Get dataset configuration
    dataset_info = DATASET_CONFIG.get(args.dataset)
    if not dataset_info:
        raise ValueError(
            f"Configuration for dataset '{args.dataset}' not found. "
            f"Available: {list(DATASET_CONFIG.keys())}"
        )

    # 1. Load Model and Tokenizer
    print("[1/4] Loading base model and tokenizer...")
    model_loader = HuggingFaceModelLoader(
        model_name=args.model_name,
        model_class=AutoModelForSequenceClassification,
        num_labels=dataset_info["num_labels"]
    )
    base_model, tokenizer = model_loader.load()

    # 2. Load and Prepare Dataset
    print("\n[2/4] Loading and preparing dataset...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    data_path = os.path.join(project_root, "data", f"{args.dataset}_dataset")

    # Allow custom train file (for few-shot experiments)
    train_file = args.train_file if hasattr(args, 'train_file') and args.train_file else f"{args.dataset}_train.csv"

    data_files = {
        "train": os.path.join(data_path, train_file),
        "validation": os.path.join(data_path, f"{args.dataset}_validation.csv"),
    }

    data_loader = LocalCsvDatasetLoader(
        data_files=data_files,
        text_column=dataset_info["text_col"],
        label_column=dataset_info["label_col"],
        max_length=args.max_length,
    )
    processed_dataset = data_loader.load_and_prepare(tokenizer)

    # 3. Build PEFT Model
    print("\n[3/4] Building PEFT model...")
    peft_model = build_peft_model(base_model, args)

    # Print parameter efficiency
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

    print(f"\n{'=' * 60}")
    print("PARAMETER EFFICIENCY SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total parameters:      {total_params:>15,}")
    print(f"Trainable parameters:  {trainable_params:>15,}")
    print(f"Trainable percentage:  {trainable_params / total_params * 100:>14.3f}%")
    print(f"{'=' * 60}\n")

    # 4. Configure and Run Trainer
    print("[4/4] Setting up trainer and starting training...")

    output_dir = os.path.join(
        project_root,
        "results",
        f"{args.dataset}_{args.model_name.replace('/', '-')}_{args.peft_method}"
    )

    training_args = {
        "output_dir": output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "logging_dir": os.path.join(project_root, "logs"),
        "logging_steps": args.logging_steps,
        "fp16": args.fp16,
        "remove_unused_columns": True,
        "label_names": ["labels"],
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": True,
        "save_total_limit": 2,  # Keep only best 2 checkpoints
    }

    trainer = ExperimentTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        dataset=processed_dataset,
        training_args_dict=training_args,
        compute_metrics_type="full"
    )

    eval_results = trainer.train()

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(
        f"Best model metric ({args.metric_for_best_model}): {eval_results.get(f'eval_{args.metric_for_best_model}', 'N/A')}")
    print("=" * 80 + "\n")

    return eval_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run PEFT experiments with LoRA, Adapters, AdapterFusion, or Hybrid"
    )

    # Model and Data
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Base model from Hugging Face")
    parser.add_argument("--dataset", type=str, default="sst2",
                        choices=list(DATASET_CONFIG.keys()),
                        help="Dataset to use")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--train_file", type=str, default=None,
                        help="Custom train file (for few-shot experiments)")

    # PEFT Method Selection
    parser.add_argument("--peft_method", type=str, default="lora",
                        choices=["lora", "adapter", "adapter_fusion", "hybrid"],
                        help="PEFT method to use")

    # LoRA Configuration
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", nargs="+", default=None,
                        help="Target modules for LoRA (e.g., q_lin v_lin)")

    # Adapter Configuration
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

    # AdapterFusion Configuration
    parser.add_argument("--adapter_names", nargs="+", default=None,
                        help="Names of adapters to fuse (e.g., sst2_adapter agnews_adapter)")
    parser.add_argument("--adapter_paths", nargs="+", default=None,
                        help="Paths to pre-trained adapters")
    parser.add_argument("--fusion_type", type=str, default="dynamic",
                        choices=["dynamic", "static"],
                        help="Fusion mechanism type")

    # Training Configuration
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

    args = parser.parse_args()
    main(args)