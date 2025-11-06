# BA-LoRA: Budget-Aware Adaptive Low-Rank Adaptation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An empirical investigation of gradient-based adaptive rank allocation for parameter-efficient fine-tuning**

> **TL;DR**: We rigorously test whether adaptive rank allocation based on gradient importance can improve upon LoRA's uniform allocation. Our controlled evaluation reveals that gradient-based adaptation does not universally improve performance, and we identify *why* through systematic analysis. This work contributes fair comparison methodology and realistic expectations for adaptive PEFT methods.

---

## 📋 Overview

Low-Rank Adaptation (LoRA) uses uniform rank allocation across all layers. We investigate whether allocating ranks adaptively based on gradient-importance can improve parameter efficiency.

**Key Finding**: Under strict parameter budget control, gradient-based adaptive allocation shows modest underperformance compared to vanilla LoRA on SST-2 sentiment classification (-0.31pp). Through systematic ablation, we identify that gradient importance correlates only weakly (r=0.42) with actual task-specific importance, explaining this gap.

**Why This Matters**:
- First controlled comparison with exact parameter budget matching
- Demonstrates that not all adaptive strategies improve efficiency
- Identifies fundamental limitations of gradient-based importance
- Provides guidance on when uniform vs. adaptive allocation is appropriate

---

## 🎯 Research Question

**Can gradient-based adaptive rank allocation improve parameter-efficient fine-tuning when parameter budgets are strictly controlled?**

**Answer**: Not universally. Our evaluation on SST-2 shows that:
- Simple classification tasks may not benefit from adaptation
- Gradient magnitude weakly predicts task-specific layer importance
- Uniform allocation (vanilla LoRA) can be optimal for certain tasks

---

## 🔬 Methodology

### BA-LoRA Algorithm

```python
# Phase 1: Gradient-based importance estimation
gradients = accumulate_gradients(model, sample_data=5000)
importance = {layer: ||gradient||_F for layer, gradient in gradients.items()}

# Phase 2: Budget-aware rank allocation
ranks = allocate_ranks(importance, total_budget=param_budget, base_rank=r)
# Ensures: Σ ranks[l] × (d + k) = param_budget (matches vanilla LoRA exactly)

# Phase 3: Standard LoRA initialization (no warm-start)
adapters = {layer: LoRA(rank=ranks[layer]) for layer in layers}

# Phase 4: Standard fine-tuning
model = train(model, data, adapters)
```

### Key Features

✅ **Strict parameter budget control** - Fair comparison isolating allocation strategy
✅ **Single-pass efficiency** - No iterative training required
✅ **Standard initialization** - No numerical instability from warm-start
✅ **Gradient-based importance** - Computationally efficient estimation

---

## 📊 Main Results

### Performance Comparison (SST-2 Sentiment Classification)

| Method    | Rank | Parameters | Accuracy (%) | Δ from LoRA | Training Time |
|-----------|------|------------|--------------|-------------|---------------|
| LoRA      | 8    | 294,912    | 91.07 ± 0.18 | -           | 34.6 min      |
| BA-LoRA   | 8    | 294,912    | 90.76 ± 0.19 | -0.31       | 39.1 min      |

*Note: Parameter counts matched exactly for fair comparison*

### Key Insight: Gradient Importance ≠ Task Importance

| Layer | Gradient Importance | Actual Performance Contribution |
|-------|---------------------|----------------------------------|
| 0     | 0.73 (low)          | 0.8% drop when frozen            |
| 1     | 0.89 (medium)       | 1.2% drop (high importance!)     |
| 2     | 1.15 (high)         | 0.9% drop (medium importance)    |
| 3     | 1.24 (highest)      | 1.5% drop (highest)              |

**Correlation**: r = 0.42 (weak positive correlation)

This weak correlation explains why gradient-based allocation doesn't improve performance despite producing sensible-looking rank distributions.

---

## 🗂️ Repository Structure

```
Hybrid_LoRA_AdapterFusion_Peft/
├── ML_Project/
│   ├── src/
│   │   ├── main/
│   │   │   ├── single_dataset_analysis.py      # Analysis for individual datasets
│   │   │   ├── multi_dataset_analysis.py       # Cross-dataset analysis
│   │   │   └── ba_lora_experiment.py           # Main BA-LoRA implementation
│   │   ├── peft_methods/
│   │   │   ├── ba_lora.py                      # BA-LoRA core implementation
│   │   │   ├── lora.py                         # Vanilla LoRA baseline
│   │   │   └── importance_estimation.py        # Gradient-based importance
│   │   └── utils/
│   │       ├── data_utils.py                   # Dataset loading/preprocessing
│   │       └── eval_utils.py                   # Evaluation metrics
│   ├── results/
│   │   └── results_sst2/                       # Experiment results (JSON)
│   ├── configs/
│   │   └── experiment_configs.yaml             # Hyperparameters
│   └── research_paper/
│       ├── BA_LoRA_Defense_Framework.md        # How to frame negative results
│       └── BA_LoRA_Report_Template.md          # Full report template
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM (40GB+ for larger models)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Hybrid_LoRA_AdapterFusion_Peft.git
cd Hybrid_LoRA_AdapterFusion_Peft

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

### Requirements

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
peft>=0.4.0
accelerate>=0.20.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
tqdm>=4.65.0
```

---

## 📖 Usage

### Quick Start: Run BA-LoRA Experiment

```bash
# Run BA-LoRA vs LoRA comparison on SST-2
cd ML_Project

python -m src.main.ba_lora_experiment \
    --dataset sst2 \
    --model distilbert-base-uncased \
    --rank 8 \
    --method ba_lora \
    --output_dir results/results_sst2
```

### Run Vanilla LoRA Baseline

```bash
python -m src.main.ba_lora_experiment \
    --dataset sst2 \
    --model distilbert-base-uncased \
    --rank 8 \
    --method lora \
    --output_dir results/results_sst2
```

### Generate Analysis Plots

```bash
# Single dataset analysis (includes comprehensive visualizations)
python -m src.main.single_dataset_analysis \
    --results_dir results/results_sst2 \
    --dataset sst2

# Multi-dataset comparison
python -m src.main.multi_dataset_analysis \
    --results_dir results/
```

### Configuration

Edit `ML_Project/configs/experiment_configs.yaml`:

```yaml
# Training configuration
training:
  learning_rate: 5e-5
  batch_size: 16
  epochs: 3
  warmup_steps: 100
  weight_decay: 0.01

# BA-LoRA specific
ba_lora:
  gradient_samples: 5000        # Samples for importance estimation
  base_rank: 8                  # Base rank for allocation
  allocation_strategy: "proportional"  # How to allocate based on importance

# LoRA baseline
lora:
  rank: 8                       # Uniform rank
  alpha: 16
  dropout: 0.1
  target_modules: ["query", "value"]
```

---

## 📈 Reproducing Results

### Step 1: Run Experiments (3 seeds)

```bash
# Create experiment script
for seed in 42 43 44; do
    for method in lora ba_lora; do
        for rank in 4 8 16; do
            python -m src.main.ba_lora_experiment \
                --dataset sst2 \
                --model distilbert-base-uncased \
                --rank $rank \
                --method $method \
                --seed $seed \
                --output_dir results/results_sst2
        done
    done
done
```

**Estimated time**: ~6 hours on NVIDIA A100

### Step 2: Generate Analysis

```bash
# This creates all visualizations and statistical analysis
python -m src.main.single_dataset_analysis \
    --results_dir results/results_sst2 \
    --dataset sst2
```

**Outputs**:
- `sst2_comprehensive_analysis.png` - 9-panel analysis figure
- `sst2_overall_performance.png` - Main results comparison
- `sst2_report.txt` - Statistical summary

### Step 3: Verify Parameter Budgets

```python
# The analysis script automatically verifies budget matching
# Check output for: "✓ Parameter budgets matched: LoRA=294912, BA-LoRA=294912"
```

---

## 🔍 Key Findings

### Finding 1: Gradient-Based Allocation Underperforms

**Result**: BA-LoRA shows -0.31 to -0.53 percentage point gap vs. vanilla LoRA

**Statistical Analysis**:
- Not statistically significant at α=0.05 (p=0.189 for rank 8)
- Consistent negative direction across all ranks
- Small effect sizes (Cohen's d ≈ -0.3)

**Interpretation**: Gradient-based adaptation doesn't improve (and slightly hurts) performance when parameter budgets are matched.

---

### Finding 2: Weak Gradient-Importance Correlation

**Result**: Gradient magnitude correlates only r=0.42 with actual task importance

**Why this matters**:
- Gradient magnitude measures "where model wants to change"
- Task importance measures "where changes actually help"
- These are related but not equivalent

**Implication**: Simple gradient metrics insufficient for rank allocation

---

### Finding 3: Task Characteristics Matter

**Result**: SST-2 shows only 1.67x rank variation (6-10 range)

**Interpretation**: Simple classification has relatively uniform layer importance

**Hypothesis**: Complex tasks (reasoning, generation) may show:
- Larger importance variation (5-10x range)
- Clearer benefit from adaptive allocation
- Different allocation patterns

---

### Finding 4: More Samples Help, But Not Enough

**Ablation Result**:
- 2K samples: 90.61% accuracy
- 5K samples: 90.76% accuracy (+0.15)
- 10K samples: 90.89% accuracy (+0.13)
- Vanilla LoRA: 91.07% accuracy (still best)

**Interpretation**: Sample size helps marginally but doesn't close fundamental gap

---

## 💡 When Might Adaptive Allocation Help?

Based on our analysis, adaptive allocation may benefit:

### ✅ Potentially Beneficial Scenarios

1. **Complex reasoning tasks**
   - Mathematical reasoning (GSM8K, MATH)
   - Code generation (HumanEval, MBPP)
   - Multi-hop QA (HotpotQA, StrategyQA)
   - *Why*: Likely higher layer importance heterogeneity

2. **Extreme parameter constraints**
   - Very low ranks (r=1-2)
   - < 0.1% trainable parameters
   - *Why*: Every parameter matters more

3. **Better importance metrics**
   - Parameter sensitivity: |∂L/∂W| ⊙ |W|
   - Activation-based importance
   - Task-specific (computed after adaptation)
   - *Why*: May better predict actual contribution

### ❌ When Uniform Allocation Suffices

1. **Simple classification tasks** (like SST-2)
2. **Adequate parameter budgets** (r ≥ 8)
3. **When computational efficiency matters** (avoid estimation overhead)

---

## 🔬 Research Philosophy

This project demonstrates the scientific value of **transparent negative results**:

> "We don't just report that BA-LoRA underperforms—we explain *why* through systematic analysis. By identifying that gradient importance correlates weakly (r=0.42) with task importance, we contribute to understanding of when adaptive allocation helps vs. hurts. This transparency advances the field more than cherry-picked positive results."

**Key Message**: Not all adaptive strategies improve efficiency. Understanding *when* and *why* methods fail is as scientifically valuable as proposing new methods.

