# BA-LoRA: Budget-Aware Adaptive Low-Rank Adaptation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An empirical investigation of gradient-based adaptive rank allocation for parameter-efficient fine-tuning**

> **TL;DR**: We rigorously test whether adaptive rank allocation based on gradient importance can improve upon LoRA's uniform allocation. Our controlled evaluation reveals that gradient-based adaptation does not universally improve performance, and we identify *why* through systematic analysis. This work contributes a fair comparison methodology and realistic expectations for adaptive PEFT methods.

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
│   ├── research_paper                          # TODO: Manuscript
│   └── demo                                    # TODO: Polish demo
├── requirements.txt
└── README.md
```
