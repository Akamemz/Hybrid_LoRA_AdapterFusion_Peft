"""
Analyze and visualize experimental results from PEFT comparison study.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
import numpy as np


class ResultsAnalyzer:
    """Analyzes experimental results and generates plots/tables."""

    def __init__(self, results_dir: str = "experiment_results"):
        self.results_dir = Path(results_dir)
        self.results_data = []
        self.load_results()

    def load_results(self):
        """Load all JSON result files from the results directory."""
        if not self.results_dir.exists():
            print(f"Results directory {self.results_dir} not found.")
            return

        # Load all JSON files
        for json_file in self.results_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    self.results_data.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        print(f"Loaded {len(self.results_data)} result files.")

    def create_summary_table(self) -> pd.DataFrame:
        """Create a summary table of all experiments."""
        rows = []

        for result in self.results_data:
            eval_results = result.get("eval_results", {})

            row = {
                "experiment": result.get("experiment_name", ""),
                "method": result.get("peft_method", ""),
                "model": result.get("model", ""),
                "dataset": result.get("dataset", ""),
                "epochs": result.get("epochs", 0),
                "accuracy": eval_results.get("eval_accuracy", 0) * 100,  # Convert to percentage
                "f1": eval_results.get("eval_f1", 0) * 100,
                "precision": eval_results.get("eval_precision", 0) * 100,
                "recall": eval_results.get("eval_recall", 0) * 100,
                "duration_min": result.get("duration_seconds", 0) / 60
            }

            # Extract config details
            peft_config = result.get("peft_config", {})
            if isinstance(peft_config, dict):
                if "r" in peft_config:
                    row["lora_r"] = peft_config["r"]
                if "reduction_factor" in peft_config:
                    row["adapter_rf"] = peft_config["reduction_factor"]

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def plot_method_comparison(self, df: pd.DataFrame):
        """Create bar plots comparing different PEFT methods."""
        # Filter for single-task experiments
        single_task_df = df[~df['experiment'].str.contains('shot')]

        if single_task_df.empty:
            print("No single-task experiments found.")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('PEFT Methods Comparison on SST-2', fontsize=16)

        # Metrics to plot
        metrics = ['accuracy', 'f1', 'precision', 'recall']

        for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
            # Group by method and get mean scores
            method_scores = single_task_df.groupby('method')[metric].mean()

            # Create bar plot
            bars = ax.bar(method_scores.index, method_scores.values)
            ax.set_title(f'{metric.capitalize()} by Method')
            ax.set_ylabel(f'{metric.capitalize()} (%)')
            ax.set_xlabel('PEFT Method')
            ax.set_ylim(0, 100)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_few_shot_learning(self, df: pd.DataFrame):
        """Plot few-shot learning curves."""
        # Filter for few-shot experiments
        few_shot_df = df[df['experiment'].str.contains('shot')]

        if few_shot_df.empty:
            print("No few-shot experiments found.")
            return

        # Extract shot count
        few_shot_df['n_shots'] = few_shot_df['experiment'].str.extract(r'(\d+)shot').astype(int)

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot lines for each method
        for method in few_shot_df['method'].unique():
            method_data = few_shot_df[few_shot_df['method'] == method]
            method_data = method_data.sort_values('n_shots')

            plt.plot(method_data['n_shots'], method_data['accuracy'],
                     marker='o', label=f'{method} (Accuracy)', linewidth=2)
            plt.plot(method_data['n_shots'], method_data['f1'],
                     marker='s', label=f'{method} (F1)', linestyle='--', alpha=0.7)

        plt.xlabel('Number of Training Examples per Class')
        plt.ylabel('Performance (%)')
        plt.title('Few-Shot Learning Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'few_shot_learning.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_parameter_efficiency(self, df: pd.DataFrame):
        """Create scatter plot of performance vs parameters."""
        # This would require parameter count data from experiments
        # Placeholder for now

        # Approximate parameter counts for different configs
        param_counts = {
            'lora': lambda r: 2 * r * 768,  # Approximate for BERT-base
            'adapter': lambda rf: 2 * 768 * (768 // rf),
            'hybrid': lambda r, rf: (2 * r * 768) + (2 * 768 * (768 // rf))
        }

        plt.figure(figsize=(10, 6))

        # Plot would show accuracy vs trainable parameters
        # Implementation depends on actual parameter counts from experiments

        plt.xlabel('Trainable Parameters')
        plt.ylabel('Accuracy (%)')
        plt.title('Parameter Efficiency: Accuracy vs Trainable Parameters')
        plt.xscale('log')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'parameter_efficiency.png', dpi=300, bbox_inches='tight')

    def generate_latex_table(self, df: pd.DataFrame):
        """Generate LaTeX table for the paper."""
        # Select relevant columns and format
        table_df = df[['method', 'accuracy', 'f1', 'precision', 'recall']].copy()

        # Group by method and compute mean ± std
        grouped = table_df.groupby('method').agg(['mean', 'std'])

        # Format as mean ± std
        latex_rows = []
        for method in grouped.index:
            row = [method.capitalize()]
            for metric in ['accuracy', 'f1', 'precision', 'recall']:
                mean = grouped.loc[method, (metric, 'mean')]
                std = grouped.loc[method, (metric, 'std')]
                if pd.notna(std) and std > 0:
                    row.append(f"${mean:.1f} \pm {std:.1f}$")
                else:
                    row.append(f"${mean:.1f}$")
            latex_rows.append(' & '.join(row) + ' \\\\')

        # Create LaTeX table
        latex_table = [
            "\\begin{table}[ht]",
            "\\centering",
            "\\caption{PEFT Methods Performance Comparison on SST-2}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Method & Accuracy & F1 & Precision & Recall \\\\",
            "\\midrule"
        ]
        latex_table.extend(latex_rows)
        latex_table.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        # Save to file
        with open(self.results_dir / 'results_table.tex', 'w') as f:
            f.write('\n'.join(latex_table))

        print("LaTeX table saved to results_table.tex")
        return '\n'.join(latex_table)

    def analyze_all(self):
        """Run all analyses and generate all plots."""
        print("\n=== Analyzing Experimental Results ===\n")

        # Create summary table
        df = self.create_summary_table()

        if df.empty:
            print("No results to analyze.")
            return

        # Save summary
        summary_file = self.results_dir / 'results_summary.csv'
        df.to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file}")

        # Print summary statistics
        print("\n--- Summary Statistics ---")
        print(df.groupby('method')[['accuracy', 'f1']].agg(['mean', 'std']))

        # Generate plots
        print("\n--- Generating Plots ---")
        self.plot_method_comparison(df)
        self.plot_few_shot_learning(df)
        self.plot_parameter_efficiency(df)

        # Generate LaTeX table
        print("\n--- Generating LaTeX Table ---")
        latex_table = self.generate_latex_table(df)

        print("\n=== Analysis Complete ===")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze PEFT experiment results")
    parser.add_argument("--results_dir", type=str, default="experiment_results",
                        help="Directory containing result JSON files")

    args = parser.parse_args()

    analyzer = ResultsAnalyzer(results_dir=args.results_dir)
    analyzer.analyze_all()


if __name__ == "__main__":
    main()