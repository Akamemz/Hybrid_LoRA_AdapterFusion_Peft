"""
Multi-Dataset Comprehensive Analysis for BA-LoRA vs LoRA
Generates publication-ready plots and statistical comparisons

"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class MultiDatasetAnalyzer:
    """Comprehensive analysis across multiple datasets"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.datasets = ['sst2', 'ag_news', 'imdb', 'tweet_eval']
        self.results_df = None

    def load_all_results(self) -> pd.DataFrame:
        """Load all experiment results into a DataFrame"""
        all_results = []

        # 🔹 Scan recursively across all subfolders for JSON files
        for json_file in self.results_dir.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Skip non-eval/trainer files
                if not isinstance(data, dict):
                    continue
                if 'eval_results' not in data or 'config' not in data:
                    continue
                if 'eval_accuracy' not in data.get('eval_results', {}):
                    continue

                # Extract key information
                row = {
                    'experiment': data.get('experiment_name', ''),
                    'method': data.get('config', {}).get('peft_method', ''),
                    'dataset': data.get('config', {}).get('dataset', '').lower(),
                    'model': data.get('config', {}).get('model', ''),
                    'accuracy': data.get('eval_results', {}).get('eval_accuracy', 0),
                    'f1': data.get('eval_results', {}).get('eval_f1', 0),
                    'precision': data.get('eval_results', {}).get('eval_precision', 0),
                    'recall': data.get('eval_results', {}).get('eval_recall', 0),
                    'train_time': data.get('duration_seconds', 0) / 60,  # minutes
                    'trainable_params': data.get('model_info', {}).get('peft_parameters',
                                                                       data.get('model_info', {}).get('trainable_parameters', 0)),
                    'total_params': data.get('model_info', {}).get('total_parameters', 0),
                }

                # Extract rank info for BA-LoRA
                if row['method'] == 'ba_lora':
                    ba_config = data.get('config', {}).get('ba_lora_config', {})
                    row['base_rank'] = ba_config.get('base_rank', None)
                    row['use_warmstart'] = ba_config.get('use_warmstart', False)
                elif row['method'] == 'lora':
                    lora_config = data.get('config', {}).get('lora_config', {})
                    row['rank'] = lora_config.get('r', None)

                all_results.append(row)

            except Exception as e:
                print(f"⚠️  Error loading {json_file}: {e}")

        if not all_results:
            print("⚠️  No valid experiment results found!")
            self.results_df = pd.DataFrame()
            return self.results_df

        self.results_df = pd.DataFrame(all_results)
        print(f"✓ Loaded {len(self.results_df)} results across {self.results_df['dataset'].nunique()} datasets")
        return self.results_df

    def statistical_comparison(self, df: pd.DataFrame, dataset: str) -> Dict:
        """Perform statistical tests comparing BA-LoRA vs LoRA"""
        ba_lora = df[(df['dataset'] == dataset) & (df['method'] == 'ba_lora')]
        lora = df[(df['dataset'] == dataset) & (df['method'] == 'lora')]

        if len(ba_lora) == 0 or len(lora) == 0:
            return None

        results = {}

        # Accuracy comparison
        acc_diff = ba_lora['accuracy'].mean() - lora['accuracy'].mean()
        results['accuracy_delta'] = acc_diff

        # F1 comparison
        f1_diff = ba_lora['f1'].mean() - lora['f1'].mean()
        results['f1_delta'] = f1_diff

        # Training time comparison
        time_diff = ba_lora['train_time'].mean() - lora['train_time'].mean()
        results['time_overhead_pct'] = (time_diff / lora['train_time'].mean()) * 100

        # Welch's t-test (doesn't assume equal variance)
        if len(ba_lora) > 1 and len(lora) > 1:
            t_stat, p_val = stats.ttest_ind(ba_lora['accuracy'], lora['accuracy'],
                                            equal_var=False)
            results['welch_t_statistic'] = t_stat
            results['welch_p_value'] = p_val
            results['significant'] = p_val < 0.05

        return results

    def create_overall_comparison(self):
        """Create overall performance comparison across datasets"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('BA-LoRA vs LoRA: Multi-Dataset Performance', fontsize=16, fontweight='bold')

        metrics = ['accuracy', 'f1', 'precision', 'recall']

        for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
            # Prepare data for grouped bar chart
            plot_data = []
            for dataset in self.datasets:
                dataset_df = self.results_df[self.results_df['dataset'] == dataset]
                if len(dataset_df) == 0:
                    continue

                for method in ['ba_lora', 'lora']:
                    method_data = dataset_df[dataset_df['method'] == method]
                    if len(method_data) > 0:
                        mean_val = method_data[metric].mean()
                        std_val = method_data[metric].std() if len(method_data) > 1 else 0
                        plot_data.append({
                            'dataset': dataset.upper(),
                            'method': 'BA-LoRA' if method == 'ba_lora' else 'LoRA',
                            'mean': mean_val,
                            'std': std_val
                        })

            if not plot_data:
                continue

            plot_df = pd.DataFrame(plot_data)

            # Create grouped bar chart
            x = np.arange(len(plot_df['dataset'].unique()))
            width = 0.35

            ba_lora_data = plot_df[plot_df['method'] == 'BA-LoRA']
            lora_data = plot_df[plot_df['method'] == 'LoRA']

            bars1 = ax.bar(x - width / 2, ba_lora_data['mean'], width,
                           yerr=ba_lora_data['std'], label='BA-LoRA',
                           color='#FF6B6B', alpha=0.8, capsize=5)
            bars2 = ax.bar(x + width / 2, lora_data['mean'], width,
                           yerr=lora_data['std'], label='LoRA',
                           color='#4ECDC4', alpha=0.8, capsize=5)

            ax.set_xlabel('Dataset')
            ax.set_ylabel(f'{metric.capitalize()} Score')
            ax.set_title(f'{metric.capitalize()} Comparison', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(ba_lora_data['dataset'].unique(), rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0.5, 1.0])

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'multi_dataset_comparison.png',
                    bbox_inches='tight', dpi=300)
        print("✓ Saved: multi_dataset_comparison.png")
        plt.close()

    def create_statistical_summary(self):
        """Create statistical summary table with significance tests"""
        summary_data = []

        for dataset in self.datasets:
            stats_result = self.statistical_comparison(self.results_df, dataset)
            if stats_result is None:
                continue

            dataset_df = self.results_df[self.results_df['dataset'] == dataset]
            ba_lora = dataset_df[dataset_df['method'] == 'ba_lora']
            lora = dataset_df[dataset_df['method'] == 'lora']

            summary_data.append({
                'Dataset': dataset.upper(),
                'BA-LoRA Acc': f"{ba_lora['accuracy'].mean():.4f} ± {ba_lora['accuracy'].std():.4f}",
                'LoRA Acc': f"{lora['accuracy'].mean():.4f} ± {lora['accuracy'].std():.4f}",
                'Δ Acc': f"{stats_result['accuracy_delta']:+.4f}",
                'p-value': f"{stats_result.get('welch_p_value', np.nan):.4f}",
                'Significant': '✓' if stats_result.get('significant', False) else '✗',
                'Time Overhead': f"{stats_result['time_overhead_pct']:+.1f}%"
            })

        summary_df = pd.DataFrame(summary_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, len(summary_data) * 0.8 + 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=summary_df.values,
                         colLabels=summary_df.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(len(summary_df.columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#4ECDC4')
            cell.set_text_props(weight='bold', color='white')

        # Highlight significant results
        for i in range(len(summary_data)):
            if summary_data[i]['Significant'] == '✓':
                for j in range(len(summary_df.columns)):
                    table[(i + 1, j)].set_facecolor('#FFE5E5')

        plt.title('Statistical Comparison: BA-LoRA vs LoRA Across Datasets',
                  fontsize=14, fontweight='bold', pad=20)
        plt.savefig(self.results_dir / 'statistical_summary.png',
                    bbox_inches='tight', dpi=300)
        print("✓ Saved: statistical_summary.png")
        plt.close()

        # Also save as CSV
        summary_df.to_csv(self.results_dir / 'statistical_summary.csv', index=False)
        print("✓ Saved: statistical_summary.csv")

        return summary_df

    def create_efficiency_analysis(self):
        """Analyze parameter efficiency and training time"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Accuracy vs Training Time
        for method in ['ba_lora', 'lora']:
            method_df = self.results_df[self.results_df['method'] == method]
            for dataset in self.datasets:
                dataset_method = method_df[method_df['dataset'] == dataset]
                if len(dataset_method) > 0:
                    ax1.scatter(dataset_method['train_time'],
                                dataset_method['accuracy'],
                                label=f"{method.upper().replace('_', '-')} ({dataset.upper()})",
                                s=100, alpha=0.7)

        ax1.set_xlabel('Training Time (minutes)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Training Time', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Training Time Comparison by Dataset
        plot_data = []
        for dataset in self.datasets:
            dataset_df = self.results_df[self.results_df['dataset'] == dataset]
            for method in ['ba_lora', 'lora']:
                method_data = dataset_df[dataset_df['method'] == method]
                if len(method_data) > 0:
                    plot_data.append({
                        'dataset': dataset.upper(),
                        'method': 'BA-LoRA' if method == 'ba_lora' else 'LoRA',
                        'time': method_data['train_time'].mean()
                    })

        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            plot_df_pivot = plot_df.pivot(index='dataset', columns='method', values='time')
            plot_df_pivot.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
            ax2.set_xlabel('Dataset')
            ax2.set_ylabel('Training Time (minutes)')
            ax2.set_title('Training Time Comparison', fontweight='bold')
            ax2.legend(title='Method')
            ax2.grid(True, alpha=0.3, axis='y')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'efficiency_analysis.png',
                    bbox_inches='tight', dpi=300)
        print("✓ Saved: efficiency_analysis.png")
        plt.close()

    def create_rank_distribution_analysis(self):
        """Analyze rank distributions for BA-LoRA if data available"""
        ba_lora_df = self.results_df[self.results_df['method'] == 'ba_lora']

        if len(ba_lora_df) == 0:
            print("⚠️  No BA-LoRA results found for rank analysis")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('BA-LoRA: Rank Allocation Analysis', fontsize=16, fontweight='bold')

        for idx, dataset in enumerate(self.datasets[:4]):
            ax = axes.flat[idx]
            dataset_df = ba_lora_df[ba_lora_df['dataset'] == dataset]

            if len(dataset_df) > 0:
                # This would require storing rank allocation in results
                # For now, show base rank used
                base_ranks = dataset_df['base_rank'].dropna()
                if len(base_ranks) > 0:
                    ax.bar(range(len(base_ranks)), base_ranks)
                    ax.set_title(f'{dataset.upper()}', fontweight='bold')
                    ax.set_xlabel('Experiment')
                    ax.set_ylabel('Base Rank')
                    ax.grid(True, alpha=0.3, axis='y')
                else:
                    ax.text(0.5, 0.5, 'No rank data available',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{dataset.upper()}', fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'No data for {dataset.upper()}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{dataset.upper()}', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'rank_distribution.png',
                    bbox_inches='tight', dpi=300)
        print("✓ Saved: rank_distribution.png")
        plt.close()

    def generate_latex_table(self):
        """Generate LaTeX table for paper"""
        latex_lines = [
            "\\begin{table}[ht]",
            "\\centering",
            "\\caption{BA-LoRA vs LoRA: Multi-Dataset Performance Comparison}",
            "\\label{tab:multi_dataset_results}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "\\textbf{Dataset} & \\textbf{Method} & \\textbf{Accuracy} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{Time (min)} \\\\",
            "\\midrule"
        ]

        for dataset in self.datasets:
            dataset_df = self.results_df[self.results_df['dataset'] == dataset]
            if len(dataset_df) == 0:
                continue

            latex_lines.append(f"\\multirow{{2}}{{*}}{{{dataset.upper()}}}")

            for method in ['ba_lora', 'lora']:
                method_df = dataset_df[dataset_df['method'] == method]
                if len(method_df) > 0:
                    method_name = "BA-LoRA" if method == 'ba_lora' else "LoRA"
                    acc = method_df['accuracy'].mean()
                    f1 = method_df['f1'].mean()
                    prec = method_df['precision'].mean()
                    rec = method_df['recall'].mean()
                    time = method_df['train_time'].mean()

                    latex_lines.append(
                        f" & {method_name} & {acc:.4f} & {f1:.4f} & {prec:.4f} & {rec:.4f} & {time:.1f} \\\\"
                    )

            latex_lines.append("\\midrule")

        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        latex_table = "\n".join(latex_lines)

        with open(self.results_dir / 'results_table.tex', 'w') as f:
            f.write(latex_table)

        print("✓ Saved: results_table.tex")
        return latex_table

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 80)
        print("MULTI-DATASET COMPREHENSIVE ANALYSIS")
        print("=" * 80 + "\n")

        # Load data
        print("[1/6] Loading results...")
        self.load_all_results()

        if self.results_df is None or len(self.results_df) == 0:
            print("❌ No results found!")
            return

        print(f"\nDatasets found: {self.results_df['dataset'].unique()}")
        print(f"Methods found: {self.results_df['method'].unique()}")
        print(f"Total experiments: {len(self.results_df)}")

        # Generate analyses
        print("\n[2/6] Creating overall comparison...")
        self.create_overall_comparison()

        print("\n[3/6] Generating statistical summary...")
        summary_df = self.create_statistical_summary()
        print("\nStatistical Summary:")
        print(summary_df.to_string(index=False))

        print("\n[4/6] Analyzing efficiency...")
        self.create_efficiency_analysis()

        print("\n[5/6] Analyzing rank distributions...")
        self.create_rank_distribution_analysis()

        print("\n[6/6] Generating LaTeX table...")
        self.generate_latex_table()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nAll outputs saved to: {self.results_dir}")
        print("\nGenerated files:")
        print("  - multi_dataset_comparison.png")
        print("  - statistical_summary.png")
        print("  - statistical_summary.csv")
        print("  - efficiency_analysis.png")
        print("  - rank_distribution.png")
        print("  - results_table.tex")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-dataset analysis for PEFT methods")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing result JSON files")
    args = parser.parse_args()

    analyzer = MultiDatasetAnalyzer(results_dir=args.results_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()