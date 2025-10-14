import os
import argparse
from transformers import AutoModelForSequenceClassification
from ..LoRa.components.huggingface_models.huggingface_model_loader import HuggingFaceModelLoader
from ..LoRa.components.data_loader.huggingface_data_loader import LocalCsvDatasetLoader
from ..LoRa.components.peft.peft_model_builder import PeftModelBuilder
from ..LoRa.components.peft.experiment_trainer import ExperimentTrainer

# --- Dataset Configuration ---
# Maps dataset names to their specific column names for text and labels.
DATASET_CONFIG = {
    "sst2": {"text_col": "sentence", "label_col": "label", "num_labels": 2},
    # Add other datasets here as you download them, e.g.:
    # "ag_news": {"text_col": "text", "label_col": "label", "num_labels": 4},
    # "imdb": {"text_col": "text", "label_col": "label", "num_labels": 2},
}


def main(args):
    """
    Main function to orchestrate the experiment pipeline.
    """
    print("--- Starting Experiment ---")

    # Get the dataset configuration
    dataset_info = DATASET_CONFIG.get(args.dataset)
    if not dataset_info:
        raise ValueError(f"Configuration for dataset '{args.dataset}' not found in DATASET_CONFIG.")

    # 1. Load Model and Tokenizer
    model_loader = HuggingFaceModelLoader(
        model_name=args.model_name,
        model_class=AutoModelForSequenceClassification,  # Explicitly specify classification model
        num_labels=dataset_info["num_labels"]  # Pass the number of labels
    )
    base_model, tokenizer = model_loader.load()

    # 2. Load and Prepare Dataset
    # Construct path to data files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    data_path = os.path.join(project_root, "data", f"{args.dataset}_dataset")

    data_files = {
        "train": os.path.join(data_path, f"{args.dataset}_train.csv"),
        "validation": os.path.join(data_path, f"{args.dataset}_validation.csv"),
    }

    # Pass the column names to the loader
    data_loader = LocalCsvDatasetLoader(
        data_files=data_files,
        text_column=dataset_info["text_col"],
        label_column=dataset_info["label_col"]
    )
    processed_dataset = data_loader.load_and_prepare(tokenizer)

    # 3. Build PEFT Model
    peft_config = {
        "method": "lora",
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    peft_builder = PeftModelBuilder(base_model)
    peft_model = peft_builder.build(peft_config)

    # 4. Configure and Run Trainer
    training_args = {
        "output_dir": os.path.join(project_root, "results", f"{args.dataset}_{args.model_name}_lora_r{args.lora_r}"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "logging_dir": os.path.join(project_root, "logs"),
        "logging_steps": 100,
        "fp16": False,  # Disabled for Apple Silicon (mps) compatibility
        "remove_unused_columns": True,
        "label_names": ["labels"],  # Explicitly specify label column name
    }

    trainer = ExperimentTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        dataset=processed_dataset,
        training_args_dict=training_args
    )
    trainer.train()

    print("--- Experiment Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PEFT experiments.")

    # Model and Data args
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Base model from Hugging Face.")
    parser.add_argument("--dataset", type=str, default="sst2", help="Dataset to use (e.g., 'sst2').")

    # LoRA args
    parser.add_argument("--lora_r", type=int, default=8, help="Rank for LoRA.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha for LoRA.")

    # Training args
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")

    args = parser.parse_args()
    main(args)


# python -m src.main.main_experiment_runner

# python -m src.main.main_experiment_runner