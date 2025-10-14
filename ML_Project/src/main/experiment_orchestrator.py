#!/usr/bin/env python3
"""
Master experiment runner for PEFT comparison study.
Executes all experiments defined in the research plan.
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List
import pandas as pd
from pathlib import Path

# Import main runner
from main.main_experiment_runner import main as run_single_experiment


class ExperimentOrchestrator:
    """Orchestrates running multiple PEFT experiments and collecting results."""

    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_few_shot_datasets(self, dataset_name: str, shot_counts: List[int]):
        """Create few-shot versions of datasets for Experiment 2."""
        print(f"Creating few-shot datasets for {dataset_name}...")

        data_path = Path("data") / f"{dataset_name}_dataset"
        train_file = data_path / f"{dataset_name}_train.csv"

        if not train_file.exists():
            print(f"Warning: {train_file} not found. Skipping few-shot creation.")
            return

        # Read full training data
        df = pd.read_csv(train_file)

        for n_shots in shot_counts:
            # Sample n_shots per class (assumes binary classification)
            few_shot_df = df.groupby('label').sample(n=n_shots, random_state=42)
            few_shot_file = data_path / f"{dataset_name}_train_{n_shots}shot.csv"
            few_shot_df.to_csv(few_shot_file, index=False)
            print(f"Created {few_shot_file} with {len(few_shot_df)} samples")

    def run_experiment_suite(self, experiment_type: str):
        """Run a specific experiment suite based on the research plan."""

        if experiment_type == "E1_single_task":
            self._run_e1_single_task()
        elif experiment_type == "E2_few_shot":
            self._run_e2_few_shot()
        elif experiment_type == "E3_transfer":
            self._run_e3_transfer()
        elif experiment_type == "all":
            self._run_e1_single_task()
            self._run_e2_few_shot()
            self._run_e3_transfer()
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

    def _run_e1_single_task(self):
        """E1: Compare PEFT methods on single task with matched parameters."""
        print("\n" + "=" * 80)
        print("EXPERIMENT 1: Single-Task Performance Comparison")
        print("=" * 80 + "\n")

        # Configuration for each PEFT method with matched parameters
        peft_configs = [
            {
                "name": "LoRA_r8",
                "method": "lora",
                "config": {"r": 8, "lora_alpha": 16}
            },
            {
                "name": "LoRA_r16",
                "method": "lora",
                "config": {"r": 16, "lora_alpha": 32}
            },
            {
                "name": "Adapter_rf16",
                "method": "adapter",
                "config": {"reduction_factor": 16}
            },
            {
                "name": "Adapter_rf8",
                "method": "adapter",
                "config": {"reduction_factor": 8}
            },
            {
                "name": "Hybrid_LoRA8_Adapter16",
                "method": "hybrid",
                "config": {
                    "lora_config": {"r": 8, "lora_alpha": 16},
                    "adapter_config": {"reduction_factor": 16}
                }
            }
        ]

        # Run experiments
        for peft_config in peft_configs:
            print(f"\nRunning {peft_config['name']}...")

            args = argparse.Namespace(
                model_name="distilbert-base-uncased",
                dataset="sst2",
                epochs=3,
                batch_size=16,
                peft_method=peft_config["method"],
                peft_config=peft_config["config"]
            )

            try:
                # Run the experiment
                results = self._run_single_config(args, peft_config["name"])
                self.results.append(results)

            except Exception as e:
                print(f"Error in {peft_config['name']}: {e}")
                continue

    def _run_e2_few_shot(self):
        """E2: Few-shot learning comparison."""
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: Few-Shot Learning Performance")
        print("=" * 80 + "\n")

        # Create few-shot datasets
        shot_counts = [8, 16, 32, 64, 128]
        self.create_few_shot_datasets("sst2", shot_counts)

        # Test configurations
        peft_configs = [
            {"name": "LoRA_fewshot", "method": "lora", "config": {"r": 8}},
            {"name": "Adapter_fewshot", "method": "adapter", "config": {"reduction_factor": 16}},
            {"name": "Hybrid_fewshot", "method": "hybrid",
             "config": {"lora_config": {"r": 8}, "adapter_config": {"reduction_factor": 16}}}
        ]

        for n_shots in shot_counts:
            for peft_config in peft_configs:
                print(f"\nRunning {peft_config['name']} with {n_shots} shots...")

                args = argparse.Namespace(
                    model_name="distilbert-base-uncased",
                    dataset="sst2",
                    epochs=5,  # More epochs for few-shot
                    batch_size=8,  # Smaller batch for few samples
                    peft_method=peft_config["method"],
                    peft_config=peft_config["config"],
                    train_file=f"sst2_train_{n_shots}shot.csv"
                )

                try:
                    results = self._run_single_config(args, f"{peft_config['name']}_{n_shots}shot")
                    results["n_shots"] = n_shots
                    self.results.append(results)
                except Exception as e:
                    print(f"Error in {peft_config['name']} with {n_shots} shots: {e}")

    def _run_e3_transfer(self):
        """E3: Transfer learning with adapter fusion."""
        print("\n" + "=" * 80)
        print("EXPERIMENT 3: Transfer Learning with AdapterFusion")
        print("=" * 80 + "\n")

        # This is more complex and requires pre-training adapters
        # Simplified version for demonstration

        print("E3 requires pre-trained adapter checkpoints.")
        print("Please run adapter training on source tasks first.")

        # Placeholder for transfer learning experiments
        # Would involve:
        # 1. Training adapters on source tasks (AG News, etc.)
        # 2. Loading them with AdapterFusion
        # 3. Training fusion layer + LoRA on target task

    def _run_single_config(self, args: argparse.Namespace, exp_name: str) -> Dict:
        """Run a single configuration and return results."""
        start_time = datetime.now()

        # Modify args to work with your existing main function
        # This is a simplified version - you'd need to update main_experiment_runner.py
        # to accept these additional parameters

        try:
            # Call your main training function
            eval_results = run_single_experiment(args)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Compile results
            results = {
                "experiment_name": exp_name,
                "timestamp": start_time.isoformat(),
                "duration_seconds": duration,
                "model": args.model_name,
                "dataset": args.dataset,
                "peft_method": args.peft_method,
                "peft_config": args.peft_config,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "eval_results": eval_results
            }

            # Save individual result
            result_file = self.output_dir / f"{exp_name}_{self.timestamp}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)

            return results

        except Exception as e:
            print(f"Error in experiment {exp_name}: {e}")
            raise

    def save_summary(self):
        """Save summary of all experiments."""
        if not self.results:
            print("No results to save.")
            return

        # Create summary DataFrame
        summary_data = []
        for result in self.results:
            row = {
                "experiment": result["experiment_name"],
                "method": result["peft_method"],
                "accuracy": result.get("eval_results", {}).get("eval_accuracy", None),
                "f1": result.get("eval_results", {}).get("eval_f1", None),
                "duration": result["duration_seconds"]
            }

            # Add config details
            if isinstance(result["peft_config"], dict):
                row.update(result["peft_config"])

            summary_data.append(row)

        df = pd.DataFrame(summary_data)

        # Save summary
        summary_file = self.output_dir / f"experiment_summary_{self.timestamp}.csv"
        df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to {summary_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(df.to_string())


def main():
    parser = argparse.ArgumentParser(description="Run PEFT comparison experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["E1_single_task", "E2_few_shot", "E3_transfer", "all"],
                        help="Which experiment set to run")
    parser.add_argument("--output_dir", type=str, default="experiment_results",
                        help="Directory to save results")

    args = parser.parse_args()

    # Run experiments
    orchestrator = ExperimentOrchestrator(output_dir=args.output_dir)
    orchestrator.run_experiment_suite(args.experiment)
    orchestrator.save_summary()


if __name__ == "__main__":
    main()