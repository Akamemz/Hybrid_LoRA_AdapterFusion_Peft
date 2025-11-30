"""
Single Dataset Comprehensive Analysis for BA-LoRA vs LoRA
Generates publication-ready visualizations similar to SST-2 analysis


python -m src.main.single_dataset_analysis \
      --results_dir results/results_sst2 \
      --dataset sst2

"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9


class SingleDatasetAnalyzer:
    """Comprehensive analysis for a single dataset (e.g., SST-2)"""

    def __init__(self, results_dir: str, dataset_name: str, output_dir: str = None):
        """
        Args:
            results_dir: Directory containing experiment JSON results
            dataset_name: Dataset to analyze (e.g., 'sst2', 'ag_news')
            output_dir: Where to save plots (defaults to results_dir/analysis_{dataset_name})
        """
        self.results_dir = Path(results_dir)
        self.dataset_name = dataset_name.lower()

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.results_dir / f"analysis_{self.dataset_name}"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_df = None
        self.models = []  # e.g., ['distilbert', 'roberta']

    def load_results(self) -> pd.DataFrame:
        """Load all experiment results for this dataset (recursive version)"""
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

                # Filter by dataset
                if data.get('config', {}).get('dataset', '').lower() != self.dataset_name:
                    continue

                # Determine model type
                model_name = data.get('config', {}).get('model', '')
                if 'distilbert' in model_name.lower():
                    model_type = 'DistilBERT'
                elif 'roberta' in model_name.lower():
                    model_type = 'RoBERTa'
                elif 'bert' in model_name.lower():
                    model_type = 'BERT'
                else:
                    model_type = model_name.split('-')[0].upper()

                # Build result row
                row = {
                    'experiment': data.get('experiment_name', ''),
                    'method': data.get('config', {}).get('peft_method', ''),
                    'model': model_type,
                    'accuracy': data.get('eval_results', {}).get('eval_accuracy', 0),
                    'f1': data.get('eval_results', {}).get('eval_f1', 0),
                    'precision': data.get('eval_results', {}).get('eval_precision', 0),
                    'recall': data.get('eval_results', {}).get('eval_recall', 0),
                    'train_time_min': data.get('duration_seconds', 0) / 60,
                    'peft_params': data.get('model_info', {}).get('peft_parameters',
                                                                  data.get('model_info', {}).get('trainable_parameters',
                                                                                                 0)),
                    'total_params': data.get('model_info', {}).get('total_parameters', 0),
                }

                # Method-specific configs
                if row['method'] == 'ba_lora':
                    ba_cfg = data.get('config', {}).get('ba_lora_config', {})
                    row['rank'] = ba_cfg.get('base_rank', None)
                    row['use_warmstart'] = ba_cfg.get('use_warmstart', False)
                    row['gradient_samples'] = ba_cfg.get('gradient_samples', None)
                elif row['method'] == 'lora':
                    lora_cfg = data.get('config', {}).get('lora_config', {})
                    row['rank'] = lora_cfg.get('r', None)
                    row['use_warmstart'] = False

                all_results.append(row)

            except Exception as e:
                print(f"⚠️  Error loading {json_file}: {e}")

        # ✅ After the loop ends
        if not all_results:
            raise ValueError(f"No results found for dataset: {self.dataset_name}")

        self.results_df = pd.DataFrame(all_results)
        self.models = sorted(self.results_df['model'].unique())

        print(f"✓ Loaded {len(self.results_df)} experiments for {self.dataset_name.upper()}")
        print(f"  Models: {', '.join(self.models)}")
        print(f"  Methods: {', '.join(self.results_df['method'].unique())}")

        return self.results_df

    def _calculate_ylim(self, data_values: List[float], padding: float = 0.05) -> Tuple[float, float]:
        """Calculate dynamic y-axis limits based on data with padding

        Args:
            data_values: List of values to calculate limits for
            padding: Percentage padding to add above/below (default 5%)

        Returns:
            Tuple of (y_min, y_max)
        """
        if not data_values:
            return (0.0, 1.0)

        y_min = min(data_values)
        y_max = max(data_values)
        y_range = y_max - y_min

        # Add padding
        y_min_padded = y_min - (y_range * padding)
        y_max_padded = y_max + (y_range * padding)

        # Ensure minimum range of 0.1 for readability
        if y_max_padded - y_min_padded < 0.1:
            center = (y_max_padded + y_min_padded) / 2
            y_min_padded = center - 0.05
            y_max_padded = center + 0.05

        # Clamp to [0, 1] for metrics that are percentages
        y_min_padded = max(0.0, y_min_padded)
        y_max_padded = min(1.0, y_max_padded)

        return (y_min_padded, y_max_padded)

    def statistical_tests(self, ba_lora_data: pd.Series, lora_data: pd.Series,
                          metric: str = 'accuracy') -> Dict:
        """Perform statistical significance tests"""

        # Welch's t-test (doesn't assume equal variance)
        welch_t, welch_p = stats.ttest_ind(ba_lora_data, lora_data, equal_var=False)

        # RoBERTa (less common, but included for completeness)
        # Mann-Whitney U test (non-parametric)
        # u_stat, u_p = stats.mannwhitneyu(ba_lora_data, lora_data, alternative='two-sided')

        delta = ba_lora_data.mean() - lora_data.mean()

        return {
            'delta': delta,
            'welch_t': welch_t,
            'welch_p': welch_p,
            'ba_lora_mean': ba_lora_data.mean(),
            'ba_lora_std': ba_lora_data.std(),
            'lora_mean': lora_data.mean(),
            'lora_std': lora_data.std(),
            'significant': welch_p < 0.05
        }

    def create_overall_performance_plot(self):
        """Create overall performance comparison (like image 1 top panel)"""
        fig, axes = plt.subplots(1, len(self.models),
                                 figsize=(6 * len(self.models), 5))

        if len(self.models) == 1:
            axes = [axes]

        fig.suptitle(f'BA-LoRA vs LoRA: Comprehensive {self.dataset_name.upper()} Analysis',
                     fontsize=14, fontweight='bold', y=0.98)

        for idx, (ax, model) in enumerate(zip(axes, self.models)):
            model_df = self.results_df[self.results_df['model'] == model]

            ba_lora = model_df[model_df['method'] == 'ba_lora']
            lora = model_df[model_df['method'] == 'lora']

            if len(ba_lora) == 0 or len(lora) == 0:
                ax.text(0.5, 0.5, f'Insufficient data for {model}',
                        ha='center', va='center', transform=ax.transAxes)
                continue

            # Calculate means and std
            ba_acc_mean = ba_lora['accuracy'].mean()
            ba_acc_std = ba_lora['accuracy'].std()
            lora_acc_mean = lora['accuracy'].mean()
            lora_acc_std = lora['accuracy'].std()

            # Statistical test
            stats_result = self.statistical_tests(ba_lora['accuracy'], lora['accuracy'])

            # Bar plot
            x = np.arange(1)
            width = 0.35

            bars1 = ax.bar(x - width / 2, ba_acc_mean, width, yerr=ba_acc_std,
                           label='BA-LoRA', color='#FF6B6B', alpha=0.8, capsize=5)
            bars2 = ax.bar(x + width / 2, lora_acc_mean, width, yerr=lora_acc_std,
                           label='LoRA', color='#4ECDC4', alpha=0.8, capsize=5)

            # Add value labels
            ax.text(x - width / 2, ba_acc_mean, f'{ba_acc_mean:.4f}\n±{ba_acc_std:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(x + width / 2, lora_acc_mean, f'{lora_acc_mean:.4f}\n±{lora_acc_std:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

            # Add delta annotation
            delta = stats_result['delta']
            y_pos = max(ba_acc_mean, lora_acc_mean) + max(ba_acc_std, lora_acc_std) + 0.01
            ax.annotate(f'Δ = {delta:+.4f}',
                        xy=(0, y_pos), fontsize=10, ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

            # Statistical test results
            sig_text = f"Welch's t-test: p={stats_result['welch_p']:.4f}"
            if not stats_result['significant']:
                sig_text += " (not significant)"

            ax.text(0.5, -0.15, sig_text, transform=ax.transAxes,
                    ha='center', fontsize=8, style='italic')

            ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax.set_title(f'{model}: Overall Performance', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            # Dynamic y-axis limits based on data
            ylim = self._calculate_ylim([ba_acc_mean - ba_acc_std, ba_acc_mean + ba_acc_std,
                                         lora_acc_mean - lora_acc_std, lora_acc_mean + lora_acc_std])
            ax.set_ylim(ylim)
            ax.legend(loc='upper left', framealpha=0.9)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.dataset_name}_overall_performance.png',
                    bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {self.dataset_name}_overall_performance.png")
        plt.close()

    def create_performance_vs_rank(self):
        """Create performance vs rank plots (like image 1 middle panels)"""
        fig, axes = plt.subplots(1, len(self.models),
                                 figsize=(6 * len(self.models), 5))

        if len(self.models) == 1:
            axes = [axes]

        for idx, (ax, model) in enumerate(zip(axes, self.models)):
            model_df = self.results_df[self.results_df['model'] == model]
            all_accuracies = []

            for method in ['ba_lora', 'lora']:
                method_df = model_df[model_df['method'] == method].copy()
                if len(method_df) == 0:
                    continue

                # Sort by rank
                method_df = method_df.sort_values('rank')
                all_accuracies.extend(method_df['accuracy'].tolist())

                label = 'BA-LoRA' if method == 'ba_lora' else 'LoRA'
                color = '#FF6B6B' if method == 'ba_lora' else '#4ECDC4'
                marker = 'o' if method == 'ba_lora' else 's'

                ax.plot(method_df['rank'], method_df['accuracy'],
                        marker=marker, label=label, linewidth=2.5,
                        markersize=8, color=color, alpha=0.8)

                # Mark best performance
                best_idx = method_df['accuracy'].idxmax()
                best_rank = method_df.loc[best_idx, 'rank']
                best_acc = method_df.loc[best_idx, 'accuracy']
                ax.plot(best_rank, best_acc, marker='*', markersize=15,
                        color='gold', markeredgecolor='black', markeredgewidth=1.5,
                        zorder=5)

            ax.set_xlabel('Rank (r)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax.set_title(f'{model}: Performance vs Rank', fontsize=12, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            # Dynamic y-axis limits based on data
            if all_accuracies:
                ylim = self._calculate_ylim(all_accuracies)
                ax.set_ylim(ylim)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.dataset_name}_performance_vs_rank.png',
                    bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {self.dataset_name}_performance_vs_rank.png")
        plt.close()

    def create_training_time_comparison(self):
        """Create training time comparison (like image 1 bottom right)"""
        fig, ax = plt.subplots(figsize=(8, 5))

        plot_data = []
        for model in self.models:
            model_df = self.results_df[self.results_df['model'] == model]
            for method in ['ba_lora', 'lora']:
                method_df = model_df[model_df['method'] == method]
                if len(method_df) > 0:
                    plot_data.append({
                        'Model_Method': f"{model}\n{method.upper().replace('_', '-')}",
                        'Model': model,
                        'Method': 'BA-LoRA' if method == 'ba_lora' else 'LoRA',
                        'Time': method_df['train_time_min'].mean()
                    })

        if not plot_data:
            print("⚠️  No training time data available")
            return

        plot_df = pd.DataFrame(plot_data)

        # Group by model
        x_labels = []
        ba_times = []
        lora_times = []

        for model in self.models:
            model_data = plot_df[plot_df['Model'] == model]
            ba = model_data[model_data['Method'] == 'BA-LoRA']
            lora = model_data[model_data['Method'] == 'LoRA']

            if len(ba) > 0 and len(lora) > 0:
                x_labels.append(model)
                ba_times.append(ba['Time'].values[0])
                lora_times.append(lora['Time'].values[0])

        x = np.arange(len(x_labels))
        width = 0.35

        bars1 = ax.bar(x - width / 2, ba_times, width, label='BA-LoRA',
                       color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width / 2, lora_times, width, label='LoRA',
                       color='#4ECDC4', alpha=0.8)

        # Add value labels and percentage difference
        for i, (ba_t, lora_t) in enumerate(zip(ba_times, lora_times)):
            ax.text(i - width / 2, ba_t, f'{ba_t:.1f} min',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(i + width / 2, lora_t, f'{lora_t:.1f} min',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

            # Percentage difference
            pct_diff = ((ba_t - lora_t) / lora_t) * 100
            y_pos = max(ba_t, lora_t) + 3
            ax.text(i, y_pos, f'{pct_diff:+.1f}%',
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('Training Time (minutes)', fontsize=11, fontweight='bold')
        ax.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.dataset_name}_training_time.png',
                    bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {self.dataset_name}_training_time.png")
        plt.close()

    def create_comprehensive_grid(self):
        """Create 6-panel comprehensive analysis (like image 2)"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Panel 1-2: Overall Performance (top row)
        for idx, model in enumerate(self.models[:2]):
            ax = fig.add_subplot(gs[0, idx])
            self._plot_overall_performance_single(ax, model)

        # Panel 3: Training Time (top right)
        ax = fig.add_subplot(gs[0, 2])
        self._plot_training_time_single(ax)

        # Panel 4-5: Accuracy vs Rank (middle row)
        for idx, model in enumerate(self.models[:2]):
            ax = fig.add_subplot(gs[1, idx])
            self._plot_performance_vs_rank_single(ax, model, metric='accuracy')

        # Panel 6: F1 Score vs Rank (middle right)
        ax = fig.add_subplot(gs[1, 2])
        self._plot_f1_vs_rank_combined(ax)

        # Panel 7: Parameter Efficiency
        ax = fig.add_subplot(gs[2, 0])
        self._plot_parameter_efficiency(ax)

        # Panel 8: Multi-Metric Comparison
        ax = fig.add_subplot(gs[2, 1])
        self._plot_multi_metric_comparison(ax)

        # Panel 9: Performance Distribution
        ax = fig.add_subplot(gs[2, 2])
        self._plot_performance_distribution(ax)

        fig.suptitle(f'BA-LoRA vs LoRA: Comprehensive Analysis on {self.dataset_name.upper()}',
                     fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(self.output_dir / f'{self.dataset_name}_comprehensive_analysis.png',
                    bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {self.dataset_name}_comprehensive_analysis.png")
        plt.close()

    def _plot_overall_performance_single(self, ax, model):
        """Helper: Plot overall performance for one model"""
        model_df = self.results_df[self.results_df['model'] == model]
        ba_lora = model_df[model_df['method'] == 'ba_lora']
        lora = model_df[model_df['method'] == 'lora']

        if len(ba_lora) == 0 or len(lora) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return

        ba_mean = ba_lora['accuracy'].mean()
        ba_std = ba_lora['accuracy'].std()
        lora_mean = lora['accuracy'].mean()
        lora_std = lora['accuracy'].std()

        bars = ax.bar(['BA-LoRA', 'LoRA'], [ba_mean, lora_mean],
                      yerr=[ba_std, lora_std], color=['#FF6B6B', '#4ECDC4'],
                      alpha=0.8, capsize=5, width=0.6)

        for bar, val, std in zip(bars, [ba_mean, lora_mean], [ba_std, lora_std]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{val:.4f}\n±{std:.4f}',
                    ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model}: Overall Performance', fontweight='bold')
        # Dynamic y-axis limits based on data
        ylim = self._calculate_ylim([ba_mean - ba_std, ba_mean + ba_std,
                                     lora_mean - lora_std, lora_mean + lora_std])
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_training_time_single(self, ax):
        """Helper: Plot training time comparison"""
        plot_data = []
        for model in self.models:
            model_df = self.results_df[self.results_df['model'] == model]
            for method in ['ba_lora', 'lora']:
                method_df = model_df[model_df['method'] == method]
                if len(method_df) > 0:
                    plot_data.append({
                        'Model': model,
                        'Method': 'BA-LoRA' if method == 'ba_lora' else 'LoRA',
                        'Time': method_df['train_time_min'].mean()
                    })

        plot_df = pd.DataFrame(plot_data)
        plot_df_pivot = plot_df.pivot(index='Model', columns='Method', values='Time')
        plot_df_pivot.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'], alpha=0.8, width=0.7)

        ax.set_ylabel('Training Time (minutes)')
        ax.set_title('Training Time Comparison', fontweight='bold')
        ax.legend(title='Method', loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_performance_vs_rank_single(self, ax, model, metric='accuracy'):
        """Helper: Plot performance vs rank for one model"""
        model_df = self.results_df[self.results_df['model'] == model]

        for method in ['ba_lora', 'lora']:
            method_df = model_df[model_df['method'] == method].copy().sort_values('rank')
            if len(method_df) == 0:
                continue

            label = 'BA-LoRA' if method == 'ba_lora' else 'LoRA'
            color = '#FF6B6B' if method == 'ba_lora' else '#4ECDC4'

            ax.plot(method_df['rank'], method_df[metric],
                    marker='o', label=label, linewidth=2, markersize=6, color=color)

        ax.set_xlabel('Rank (r)')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{model}: {metric.capitalize()} vs Rank', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_f1_vs_rank_combined(self, ax):
        """Helper: Plot F1 vs rank for all models"""
        for model in self.models:
            model_df = self.results_df[self.results_df['model'] == model]
            for method in ['ba_lora', 'lora']:
                method_df = model_df[model_df['method'] == method].copy().sort_values('rank')
                if len(method_df) == 0:
                    continue

                label = f"{model} {method.upper().replace('_', '-')}"
                linestyle = '-' if method == 'ba_lora' else '--'

                ax.plot(method_df['rank'], method_df['f1'],
                        marker='o', label=label, linewidth=2, markersize=5,
                        linestyle=linestyle, alpha=0.7)

        ax.set_xlabel('Rank (r)')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score vs Rank (Both Models)', fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_parameter_efficiency(self, ax):
        """Helper: Plot parameter efficiency scatter"""
        for model in self.models:
            model_df = self.results_df[self.results_df['model'] == model]
            for method in ['ba_lora', 'lora']:
                method_df = model_df[model_df['method'] == method]
                if len(method_df) == 0:
                    continue

                label = f"{model} {method.upper().replace('_', '-')}"
                marker = 'o' if method == 'ba_lora' else 's'
                color = '#FF6B6B' if method == 'ba_lora' else '#4ECDC4'

                ax.scatter(method_df['peft_params'] / 1000, method_df['accuracy'],
                           label=label, s=100, marker=marker, color=color, alpha=0.7)

        ax.set_xlabel('PEFT Parameters (thousands)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Parameter Efficiency', fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_multi_metric_comparison(self, ax):
        """Helper: Plot comprehensive metric comparison (Accuracy, F1, Precision, Recall)"""
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        metric_labels = ['Accuracy', 'F1', 'Precision', 'Recall']

        ba_lora_df = self.results_df[self.results_df['method'] == 'ba_lora']
        lora_df = self.results_df[self.results_df['method'] == 'lora']

        if len(ba_lora_df) == 0 or len(lora_df) == 0:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('Multi-Metric Comparison', fontweight='bold')
            return

        # Calculate means for each metric
        ba_means = [ba_lora_df[metric].mean() for metric in metrics]
        lora_means = [lora_df[metric].mean() for metric in metrics]

        x = np.arange(len(metric_labels))
        width = 0.35

        # Create grouped bar chart
        bars1 = ax.bar(x - width/2, ba_means, width, label='BA-LoRA',
                       color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, lora_means, width, label='LoRA',
                       color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels on bars
        for bars, means in zip([bars1, bars2], [ba_means, lora_means]):
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.3f}',
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Multi-Metric Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Dynamic y-axis based on all metric values
        all_values = ba_means + lora_means
        ylim = self._calculate_ylim(all_values, padding=0.03)
        ax.set_ylim(ylim)

    def _plot_performance_distribution(self, ax):
        """Helper: Plot performance distribution box plots"""
        plot_data = []
        for model in self.models:
            model_df = self.results_df[self.results_df['model'] == model]
            for method in ['ba_lora', 'lora']:
                method_df = model_df[model_df['method'] == method]
                for _, row in method_df.iterrows():
                    plot_data.append({
                        'Model_Method': f"{model}\n{method.upper().replace('_', '-')}",
                        'Model': model,
                        'Method': method,
                        'Accuracy': row['accuracy']
                    })

        if plot_data:
            plot_df = pd.DataFrame(plot_data)

            # Create box plot
            positions = []
            data_to_plot = []
            labels = []
            colors = []

            idx = 0
            for model in self.models:
                for method in ['ba_lora', 'lora']:
                    subset = plot_df[(plot_df['Model'] == model) &
                                     (plot_df['Method'] == method)]
                    if len(subset) > 0:
                        data_to_plot.append(subset['Accuracy'].values)
                        positions.append(idx)
                        label = f"{model}\n{method.upper().replace('_', '-')}"
                        labels.append(label)
                        colors.append('#FF6B6B' if method == 'ba_lora' else '#4ECDC4')
                        idx += 1

            bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                            patch_artist=True, showmeans=True)

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=8)

        ax.set_ylabel('Accuracy')
        ax.set_title('Performance Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    def create_publication_summary(self):
        """Create publication-ready summary figure"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.25)

        # Top row: Overall performance for each model
        for idx, model in enumerate(self.models):
            ax = fig.add_subplot(gs[0, idx])

            model_df = self.results_df[self.results_df['model'] == model]
            ba_lora = model_df[model_df['method'] == 'ba_lora']
            lora = model_df[model_df['method'] == 'lora']

            if len(ba_lora) > 0 and len(lora) > 0:
                ba_mean = ba_lora['accuracy'].mean()
                ba_std = ba_lora['accuracy'].std()
                lora_mean = lora['accuracy'].mean()
                lora_std = lora['accuracy'].std()

                bars = ax.bar(['BA-LoRA', 'LoRA'], [ba_mean, lora_mean],
                              yerr=[ba_std, lora_std],
                              color=['#FF6B6B', '#4ECDC4'],
                              alpha=0.8, capsize=5, width=0.6, edgecolor='black', linewidth=1.5)

                # Add values on bars
                for bar, val, std in zip(bars, [ba_mean, lora_mean], [ba_std, lora_std]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + std + 0.005,
                            f'{val:.4f}',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
                    ax.text(bar.get_x() + bar.get_width() / 2., height / 2,
                            f'±{std:.4f}',
                            ha='center', va='center', fontsize=8)

                # Statistical test
                stats_result = self.statistical_tests(ba_lora['accuracy'], lora['accuracy'])
                delta = stats_result['delta']

                # Add delta box
                ax.text(0.5, 0.95, f'Δ = {delta:+.4f}',
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5',
                                  facecolor='red' if delta < 0 else 'lightgreen',
                                  alpha=0.5, edgecolor='black', linewidth=1.5))

                # Statistical significance
                sig_marker = '' if stats_result['significant'] else ' (not significant)'
                ax.text(0.5, -0.18,
                        f"Welch's t-test: p={stats_result['welch_p']:.2f}{sig_marker}",
                        transform=ax.transAxes, ha='center', fontsize=8, style='italic')

            ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax.set_title(f'{model}: Overall Performance', fontsize=12, fontweight='bold')
            # Dynamic y-axis limits based on data
            ylim = self._calculate_ylim([ba_mean - ba_std, ba_mean + ba_std,
                                         lora_mean - lora_std, lora_mean + lora_std])
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Top right: Training time
        ax = fig.add_subplot(gs[0, 2])
        self._plot_training_time_single(ax)

        # Bottom row: Performance vs Rank for each model
        for idx, model in enumerate(self.models):
            ax = fig.add_subplot(gs[1, idx])
            self._plot_performance_vs_rank_single(ax, model)

        # Bottom right: Conclusion box
        ax = fig.add_subplot(gs[1, 2])
        ax.axis('off')

        # Calculate overall statistics
        ba_lora_all = self.results_df[self.results_df['method'] == 'ba_lora']
        lora_all = self.results_df[self.results_df['method'] == 'lora']

        overall_delta = ba_lora_all['accuracy'].mean() - lora_all['accuracy'].mean()
        time_overhead = ((ba_lora_all['train_time_min'].mean() -
                          lora_all['train_time_min'].mean()) /
                         lora_all['train_time_min'].mean() * 100)

        overall_stats = self.statistical_tests(ba_lora_all['accuracy'], lora_all['accuracy'])

        conclusion_text = f"""
        CONCLUSION: BA-LoRA vs LoRA on {self.dataset_name.upper()}

        Performance Impact:
        • Overall Δ Accuracy: {overall_delta:+.4f}
        • Statistical Significance: {'YES' if overall_stats['significant'] else 'NO'}
        • p-value: {overall_stats['welch_p']:.4f}

        Training Efficiency:
        • Time Overhead: {time_overhead:+.1f}%

        Key Finding:
        {"BA-LoRA shows significant improvement" if overall_stats['significant'] and overall_delta > 0
        else "BA-LoRA does NOT outperform standard LoRA"}

        {"Adaptive rank allocation provides benefit" if overall_stats['significant'] and overall_delta > 0
        else "No significant benefit from adaptive ranks"}
        """

        ax.text(0.1, 0.9, conclusion_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.8,
                          edgecolor='black', linewidth=2))

        fig.suptitle(f'BA-LoRA vs LoRA: Comprehensive {self.dataset_name.upper()} Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(self.output_dir / f'{self.dataset_name}_publication_summary.png',
                    bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {self.dataset_name}_publication_summary.png")
        plt.close()

    def generate_summary_report(self):
        """Generate text summary report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"BA-LORA VS LORA: {self.dataset_name.upper()} ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Overall statistics
        ba_lora_df = self.results_df[self.results_df['method'] == 'ba_lora']
        lora_df = self.results_df[self.results_df['method'] == 'lora']

        report_lines.append("OVERALL RESULTS:")
        report_lines.append("-" * 80)
        report_lines.append(f"BA-LoRA:")
        report_lines.append(f"  Experiments: {len(ba_lora_df)}")
        report_lines.append(f"  Accuracy:    {ba_lora_df['accuracy'].mean():.4f} ± {ba_lora_df['accuracy'].std():.4f}")
        report_lines.append(f"  F1 Score:    {ba_lora_df['f1'].mean():.4f} ± {ba_lora_df['f1'].std():.4f}")
        report_lines.append(f"  Avg Time:    {ba_lora_df['train_time_min'].mean():.1f} minutes")
        report_lines.append("")

        report_lines.append(f"LoRA:")
        report_lines.append(f"  Experiments: {len(lora_df)}")
        report_lines.append(f"  Accuracy:    {lora_df['accuracy'].mean():.4f} ± {lora_df['accuracy'].std():.4f}")
        report_lines.append(f"  F1 Score:    {lora_df['f1'].mean():.4f} ± {lora_df['f1'].std():.4f}")
        report_lines.append(f"  Avg Time:    {lora_df['train_time_min'].mean():.1f} minutes")
        report_lines.append("")

        # Statistical tests
        stats_result = self.statistical_tests(ba_lora_df['accuracy'], lora_df['accuracy'])

        report_lines.append("STATISTICAL ANALYSIS:")
        report_lines.append("-" * 80)
        report_lines.append(f"Accuracy Delta:        {stats_result['delta']:+.4f}")
        report_lines.append(f"Welch's t-statistic:   {stats_result['welch_t']:.4f}")
        report_lines.append(f"Welch's p-value:       {stats_result['welch_p']:.4f}")
        report_lines.append(
            f"Statistically significant: {'YES (p < 0.05)' if stats_result['significant'] else 'NO (p >= 0.05)'}")
        report_lines.append("")

        # Per-model breakdown
        report_lines.append("PER-MODEL BREAKDOWN:")
        report_lines.append("-" * 80)

        for model in self.models:
            model_df = self.results_df[self.results_df['model'] == model]
            ba_model = model_df[model_df['method'] == 'ba_lora']
            lora_model = model_df[model_df['method'] == 'lora']

            if len(ba_model) > 0 and len(lora_model) > 0:
                model_stats = self.statistical_tests(ba_model['accuracy'], lora_model['accuracy'])

                report_lines.append(f"{model}:")
                report_lines.append(f"  BA-LoRA: {ba_model['accuracy'].mean():.4f} ± {ba_model['accuracy'].std():.4f}")
                report_lines.append(
                    f"  LoRA:    {lora_model['accuracy'].mean():.4f} ± {lora_model['accuracy'].std():.4f}")
                report_lines.append(f"  Delta:   {model_stats['delta']:+.4f} (p={model_stats['welch_p']:.4f})")
                report_lines.append("")

        # Conclusion
        report_lines.append("CONCLUSION:")
        report_lines.append("=" * 80)

        if stats_result['significant'] and stats_result['delta'] > 0:
            conclusion = f"BA-LoRA shows SIGNIFICANT IMPROVEMENT over standard LoRA on {self.dataset_name.upper()}."
        elif stats_result['significant'] and stats_result['delta'] < 0:
            conclusion = f"BA-LoRA shows SIGNIFICANT DEGRADATION compared to standard LoRA on {self.dataset_name.upper()}."
        else:
            conclusion = f"BA-LoRA does NOT show significant difference from standard LoRA on {self.dataset_name.upper()}."

        report_lines.append(conclusion)
        report_lines.append("")

        time_overhead = ((ba_lora_df['train_time_min'].mean() - lora_df['train_time_min'].mean()) /
                         lora_df['train_time_min'].mean() * 100)

        report_lines.append(f"Training time overhead: {time_overhead:+.1f}%")
        report_lines.append("")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        # Save to file
        with open(self.output_dir / f'{self.dataset_name}_report.txt', 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\n✓ Saved: {self.dataset_name}_report.txt")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 80)
        print(f"COMPREHENSIVE ANALYSIS: {self.dataset_name.upper()}")
        print("=" * 80 + "\n")

        # Load results
        print("[1/6] Loading results...")
        self.load_results()

        # Create visualizations
        print("\n[2/6] Creating overall performance comparison...")
        self.create_overall_performance_plot()

        print("\n[3/6] Creating performance vs rank plots...")
        self.create_performance_vs_rank()

        print("\n[4/6] Creating training time comparison...")
        self.create_training_time_comparison()

        print("\n[5/6] Creating comprehensive grid analysis...")
        self.create_comprehensive_grid()

        print("\n[6/6] Creating publication summary...")
        self.create_publication_summary()

        print("\n[BONUS] Generating summary report...")
        self.generate_summary_report()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nAll outputs saved to: {self.output_dir}")
        print("\nGenerated files:")
        print(f"  - {self.dataset_name}_overall_performance.png")
        print(f"  - {self.dataset_name}_performance_vs_rank.png")
        print(f"  - {self.dataset_name}_training_time.png")
        print(f"  - {self.dataset_name}_comprehensive_analysis.png")
        print(f"  - {self.dataset_name}_publication_summary.png")
        print(f"  - {self.dataset_name}_report.txt")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive single-dataset analysis for PEFT comparison"
    )
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing result JSON files")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., sst2, ag_news, imdb)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: results/analysis_{dataset})")

    args = parser.parse_args()

    analyzer = SingleDatasetAnalyzer(
        results_dir=args.results_dir,
        dataset_name=args.dataset,
        output_dir=args.output_dir
    )

    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()