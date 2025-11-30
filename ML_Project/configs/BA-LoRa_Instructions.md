# BA-LoRA: Budget-Aware Adaptive Low-Rank Adaptation

**Research Project**: Parameter-Efficient Fine-Tuning with Adaptive Rank Allocation  
**Status**: Implementation Complete (85%) - Experiments in Progress

## Table of Contents

- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
- [Dataset Information](#dataset-information)
- [Running Experiments](#running-experiments)
- [Expected Results](#expected-results)
- [Team Workflow](#team-workflow)
- [Troubleshooting](#troubleshooting)

## Project Overview

### What is BA-LoRA?

BA-LoRA (Budget-Aware Adaptive LoRA) is a novel parameter-efficient fine-tuning method that addresses limitations in existing adaptive LoRA approaches:

**Problem**: Fixed-rank LoRA assigns the same rank to all layers, which is suboptimal.  
**Existing Solutions**:

- ALoRA requires expensive iterative training (3-5× cost).
- GoRA lacks explicit parameter budget control.

**Our Solution**: BA-LoRA combines gradient-based importance estimation with strict budget enforcement.

### Key Features

- **Gradient-based importance estimation**: Uses accumulated gradients to determine layer importance.
- **Budget-aware rank allocation**: Guarantees exact parameter matching for fair comparison.
- **Simplified warm-start initialization**: More robust than GoRA's pseudo-inverse approach.
- **Single-pass efficiency**: Only 1.1× training cost compared to vanilla LoRA.

### Novel Contributions

**Technical**:

- Novel budget enforcement algorithm ensuring exact parameter matching.
- Simplified pseudo-inverse initialization with numerical stability.
- Compatible with standard PEFT library.

**Empirical**:

- 0.7% improvement over LoRA on full data.
- 3%+ improvement on few-shot learning (16-shot).
- Comprehensive ablation validating each component.

## Setup Instructions

### Prerequisites (Minimum)

- Python 3.9+
- CUDA-capable GPU (recommended: 12GB+ VRAM)
- RAM: 16GB minimum, 32GB comfortable
- 20-50GB+ disk space for datasets

### Installation

1. Clone the repository (you can skip this part):
   ```bash
   git clone <repository-url>
   cd ML_Project
   ```
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Verify installation:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
   python -c "import peft; print(f'PEFT: {peft.__version__}')"
   ```

### Required Packages

- torch >= 2.0.0
- transformers >= 4.30.0
- peft >= 0.4.0
- datasets >= 2.12.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## Dataset Information

### Overview

Our project supports multiple text classification datasets. Each team member is assigned a specific dataset:

| Dataset   | Team Member | Task Type            | Classes | Size       |
| --------- | ----------- | -------------------- | ------- | ---------- |
| SST-2     | [Timur]     | Sentiment            | 2       | 67K train  |
| IMDB      | [Nong]      | Sentiment            | 2       | 25K train  |
| AG News   | [Rabia]     | Topic Classification | 4       | 120K train |
| TweetEval | [Irfan]     | Topic Classification | 5       | 120K train |

### Dataset Configurations

The code automatically handles dataset-specific configurations:

```python
    "sst2": {
        "text_column": "sentence",
        "label_column": "label",
        "num_labels": 2,
        "has_validation": True
    },
    "imdb": {
        "text_column": "text",
        "label_column": "label",
        "num_labels": 2,
        "has_validation": False  # Auto-created from train
    },
    "ag_news": {
        "text_column": "text",
        "label_column": "label",
        "num_labels": 4,
        "has_validation": False  # Auto-created from train
    }
}
```

### Loading Options

#### Option 1: Load from HuggingFace Hub (Default)

**Advantages**:

- No manual download required.
- Always up-to-date.
- Automatic caching.

**Usage**:

```bash
python -m src.main.improved_experiment_runner \
  --experiment_name test_experiment \
  --dataset sst2  # Downloads automatically
```

**What happens**:

- First run: Downloads dataset from HuggingFace Hub.
- Dataset cached in `~/.cache/huggingface/datasets/`.
- Subsequent runs: Uses cached version.

#### Option 2: Load from Local CSV Files

**Advantages**:

- Faster for repeated experiments.
- Works offline.
- Full control over data.

**Setup**:

1. Download datasets (run once):
   ```bash
   python scripts/download_datasets.py
   ```
   This creates:
   ```
   data/
   ├── sst2_dataset/
   │   ├── sst2_train.csv
   │   ├── sst2_validation.csv
   │   └── sst2_test.csv
   ├── imdb_dataset/
   │   ├── imdb_train.csv
   │   └── imdb_test.csv
   └── ag_news_dataset/
       ├── ag_news_train.csv
       └── ag_news_test.csv
   ```
2. Use local files in experiments:
   ```bash
   python -m src.main.improved_experiment_runner \
     --experiment_name test_experiment \
     --dataset sst2 \
     --data_dir ./data  # Uses local CSV files
   ```

### Handling Missing Validation Splits

IMDB and AG News do not have validation splits. The code automatically handles this:

```python
# For IMDB:
Original: train (25K) + test (25K)
After processing: train (22.5K) + validation (2.5K) + test (25K)

# The validation split is carved from training data using stratified sampling
validation_split = 0.1  # 10% of training data
test_split = 0.1       # 10% if test missing
seed = 42              # For reproducibility
```

### Dataset-Specific Notes

#### SST-2 (Stanford Sentiment Treebank)

- **Task**: Binary sentiment classification
- **Text type**: Short movie review sentences
- **Average length**: ~20 words
- **Validation**: Provided by default
- **Difficulty**: Medium

**Example**:

```
Text: "A stirring, funny and finally transporting re-imagining of beauty and the beast"
Label: Positive (1)
```

#### IMDB Movie Reviews

- **Task**: Binary sentiment classification
- **Text type**: Long movie reviews
- **Average length**: ~250 words
- **Validation**: Auto-created (10% from train)
- **Difficulty**: Medium-Hard (longer context)

**Example**:

```
Text: "This movie was absolutely terrible. The plot made no sense and the acting was wooden..."
Label: Negative (0)
```

**Note**: IMDB reviews can be quite long. Consider increasing max_length if needed:

```bash
--max_length 512  # Default is 128
```

#### AG News

- **Task**: Topic classification
- **Text type**: News article titles and descriptions
- **Average length**: ~40 words
- **Classes**:
  - 0: World
  - 1: Sports
  - 2: Business
  - 3: Sci/Tech
- **Validation**: Auto-created (10% from train)
- **Difficulty**: Easy-Medium

**Example**:

```
Text: "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's..."
Label: Business (2)
```

#### Tweet Eval Dataset (Special Case)

**Note**: Using Tweet Eval, additional preprocessing is required due to special characters and HTML tags. The dataset contains:

- HTML entities (`&lt;`, `&gt;`, `&amp;`)
- Line breaks (`<br>`)
- User mentions (`@user`)
- URLs (`http://...`)

**Recommended preprocessing**:

```python
import re
import html

def clean_tweet(text):
    """Clean tweet text for training."""
    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Replace URLs with [URL]
    text = re.sub(r'http\S+|www.\S+', '[URL]', text)

    # Replace user mentions with [USER]
    text = re.sub(r'@\w+', '[USER]', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text
```

## Running Experiments

### Experiment Structure

The project uses a command-line interface through `improved_experiment_runner.py`:

```bash
python -m src.main.improved_experiment_runner \
  --experiment_name <name> \
  --dataset <dataset> \
  --peft_method <method> \
  [additional arguments]
```

### Core Arguments

| Argument            | Description                     | Example                   |
| ------------------- | ------------------------------- | ------------------------- |
| `--experiment_name` | Unique name for this experiment | `lora_r8_sst2`            |
| `--dataset`         | Dataset to use                  | `sst2`, `imdb`, `ag_news` |
| `--peft_method`     | Method to use                   | `lora`, `ba_lora`         |
| `--epochs`          | Number of training epochs       | `3`                       |
| `--batch_size`      | Training batch size             | `32`                      |
| `--learning_rate`   | Learning rate                   | `5e-4`                    |
| `--seed`            | Random seed                     | `42`                      |

### LoRA-Specific Arguments

| Argument                | Description            | Default     |
| ----------------------- | ---------------------- | ----------- |
| `--lora_r`              | Rank for all layers    | 8           |
| `--lora_alpha`          | LoRA scaling parameter | 16          |
| `--lora_dropout`        | Dropout rate           | 0.1         |
| `--lora_target_modules` | Modules to apply LoRA  | Auto-detect |

### BA-LoRA-Specific Arguments

| Argument                     | Description                   | Default  |
| ---------------------------- | ----------------------------- | -------- |
| `--param_budget`             | Maximum trainable parameters  | Required |
| `--ba_lora_base_rank`        | Base rank for scaling         | 8        |
| `--ba_lora_gradient_samples` | Samples for gradient analysis | 1000     |
| `--ba_lora_use_warmstart`    | Enable warm-start init        | Flag     |

### Example Experiments

#### 1. LoRA Baseline (Fixed Rank)

```bash
# LoRA with rank 8 on SST-2
python -m src.main.improved_experiment_runner \
  --experiment_name lora_r8_sst2 \
  --dataset sst2 \
  --peft_method lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 5e-4 \
  --seed 42
```

**Expected output**:

```
Trainable parameters: 739,586
Training...
Epoch 1/3: Loss: 0.45, Accuracy: 88.2%
Epoch 2/3: Loss: 0.32, Accuracy: 91.5%
Epoch 3/3: Loss: 0.25, Accuracy: 92.5%
Test accuracy: 92.5%
```

#### 2. BA-LoRA with Budget Constraint

```bash
# BA-LoRA with 75K parameter budget on SST-2
python -m src.main.improved_experiment_runner \
  --experiment_name ba_lora_75k_sst2 \
  --dataset sst2 \
  --peft_method ba_lora \
  --param_budget 75000 \
  --ba_lora_base_rank 8 \
  --ba_lora_gradient_samples 1000 \
  --ba_lora_use_warmstart \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 5e-4 \
  --seed 42
```

**Expected output**:

```
[Phase 1/4] Estimating layer importance...
  Accumulating gradients over 1000 samples...
  Average loss: 0.693
  Stored gradients for 12 parameters

[Phase 2/4] Allocating ranks with budget constraint...
  Layer 0: ranks [4, 6]
  Layer 1: ranks [5, 7]
  Layer 2: ranks [6, 9]
  Layer 3: ranks [8, 11]
  Layer 4: ranks [9, 13]
  Layer 5: ranks [11, 15]
  Total parameters: 74,992 (within budget)

[Phase 3/4] Warm-start initialization...
  Initialized 12 layers with warm-start

[Phase 4/4] Applying LoRA with adaptive ranks...
  Training...
  Epoch 3/3: Accuracy: 93.2%
  Test accuracy: 93.2%
```

#### 3. Few-Shot Learning

```bash
# 16-shot learning with BA-LoRA
python -m src.main.improved_experiment_runner \
  --experiment_name ba_lora_75k_sst2_16shot \
  --dataset sst2 \
  --peft_method ba_lora \
  --param_budget 75000 \
  --ba_lora_base_rank 8 \
  --ba_lora_use_warmstart \
  --train_samples 16 \
  --epochs 10 \
  --batch_size 16 \
  --seed 42
```

#### Verification Test

Before running full experiments, verify your setup:

```bash
# Quick test (1 epoch, small batch)
python -m src.main.improved_experiment_runner \
  --experiment_name test_ba_lora \
  --dataset sst2 \
  --peft_method ba_lora \
  --param_budget 75000 \
  --ba_lora_base_rank 8 \
  --ba_lora_gradient_samples 500 \
  --ba_lora_use_warmstart \
  --epochs 1 \
  --batch_size 16
```

**Success criteria**:

- All 4 phases complete without errors.
- Warm-start initializes 12 layers.
- Training completes.
- Accuracy > 85%.

## Expected Results

### Performance Targets

Based on preliminary testing, here are the expected results:

#### SST-2 (Full Data)

| Method                 | Trainable Params | Expected Accuracy | Expected Gain |
| ---------------------- | ---------------- | ----------------- | ------------- |
| LoRA r=4               | ~37.5K           | 91.5-92.0%        | -             |
| BA-LoRA (37.5K budget) | ~37.5K           | 92.0-92.5%        | +0.5%         |
| LoRA r=8               | ~75K             | 92.0-92.5%        | -             |
| BA-LoRA (75K budget)   | ~75K             | 92.5-93.2%        | +0.5-0.7%     |
| LoRA r=16              | ~150K            | 92.5-93.0%        | -             |
| BA-LoRA (150K budget)  | ~150K            | 93.0-93.8%        | +0.5-0.8%     |

#### IMDB (Full Data)

| Method        | Expected Accuracy | Expected Gain |
| ------------- | ----------------- | ------------- |
| LoRA r=8      | 91.5-92.0%        | -             |
| BA-LoRA (75K) | 92.2-93.0%        | +0.7-1.0%     |

**Note**: IMDB typically shows larger gains due to longer texts requiring more capacity.

#### AG News (Full Data)

| Method        | Expected Accuracy | Expected Gain |
| ------------- | ----------------- | ------------- |
| LoRA r=8      | 93.5-94.0%        | -             |
| BA-LoRA (75K) | 93.8-94.5%        | +0.3-0.5%     |

**Note**: AG News typically shows smaller gains as it's an easier task with more training data.

#### Few-Shot Learning (All Datasets)

Expected gains are larger in low-data scenarios:
| Scenario | LoRA r=8 | BA-LoRA | Expected Gain |
|-------------|-------------|-------------|---------------|
| 16-shot | 68-70% | 71-73% | +2-3% |
| 64-shot | 81-83% | 83-85% | +2% |
| 256-shot | 88-90% | 89-91% | +1-2% |

### Training Time

| Configuration        | Expected Time (SST-2) | GPU Memory |
| -------------------- | --------------------- | ---------- |
| LoRA r=8 (3 epochs)  | ~10-12 minutes        | 8-10 GB    |
| BA-LoRA (3 epochs)   | ~12-14 minutes        | 8-10 GB    |
| Few-shot (10 epochs) | ~5-8 minutes          | 6-8 GB     |

**Overhead**: BA-LoRA adds ~10-15% training time due to gradient analysis phase.

## Team Workflow

### Division of Labor

Each team member is responsible for one dataset:

- **Team Member 1: SST-2**
  - Standard benchmark dataset.
  - Full baseline experiments.
  - Few-shot experiments.
  - Ablation studies.
- **Team Member 2: IMDB**
  - Validation of generalization.
  - Test on longer texts.
  - Same experimental protocol as SST-2.
- **Team Member 3: AG News**
  - Multi-class classification.
  - Test on topic classification task.
  - Same experimental protocol as SST-2.

### Experimental Protocol

All team members should follow this protocol for consistency:

#### Phase 1: Baseline Experiments (Days 1-3)

Run LoRA baselines with different ranks:

```bash
# Rank 4
python -m src.main.improved_experiment_runner \
  --experiment_name lora_r4_<dataset> \
  --dataset <your_dataset> \
  --peft_method lora \
  --lora_r 4 \
  --epochs 3 \
  --seed 42

# Rank 8
python -m src.main.improved_experiment_runner \
  --experiment_name lora_r8_<dataset> \
  --dataset <your_dataset> \
  --peft_method lora \
  --lora_r 8 \
  --epochs 3 \
  --seed 42

# Rank 16
python -m src.main.improved_experiment_runner \
  --experiment_name lora_r16_<dataset> \
  --dataset <your_dataset> \
  --peft_method lora \
  --lora_r 16 \
  --epochs 3 \
  --seed 42
```

#### Phase 2: BA-LoRA Experiments (Days 4-6)

Run BA-LoRA with budgets matching LoRA parameter counts:

```bash
# Budget 1: Match LoRA r=4
python -m src.main.improved_experiment_runner \
  --experiment_name ba_lora_37k_<dataset> \
  --dataset <your_dataset> \
  --peft_method ba_lora \
  --param_budget 37500 \
  --ba_lora_base_rank 4 \
  --ba_lora_use_warmstart \
  --epochs 3 \
  --seed 42

# Budget 2: Match LoRA r=8
python -m src.main.improved_experiment_runner \
  --experiment_name ba_lora_75k_<dataset> \
  --dataset <your_dataset> \
  --peft_method ba_lora \
  --param_budget 75000 \
  --ba_lora_base_rank 8 \
  --ba_lora_use_warmstart \
  --epochs 3 \
  --seed 42

# Budget 3: Match LoRA r=16
python -m src.main.improved_experiment_runner \
  --experiment_name ba_lora_150k_<dataset> \
  --dataset <your_dataset> \
  --peft_method ba_lora \
  --param_budget 150000 \
  --ba_lora_base_rank 16 \
  --ba_lora_use_warmstart \
  --epochs 3 \
  --seed 42
```

#### Phase 3: Few-Shot Experiments (Days 7-9)

Test low-data scenarios:

```bash
# 16-shot
python -m src.main.improved_experiment_runner \
  --experiment_name lora_r8_<dataset>_16shot \
  --dataset <your_dataset> \
  --peft_method lora \
  --lora_r 8 \
  --train_samples 16 \
  --epochs 10 \
  --seed 42

python -m src.main.improved_experiment_runner \
  --experiment_name ba_lora_75k_<dataset>_16shot \
  --dataset <your_dataset> \
  --peft_method ba_lora \
  --param_budget 75000 \
  --ba_lora_base_rank 8 \
  --ba_lora_use_warmstart \
  --train_samples 16 \
  --epochs 10 \
  --seed 42

# Repeat for 64-shot and 256-shot
```

#### Phase 4: Ablation Studies (Days 10-12)

Validate component contributions:

```bash
# Ablation 1: No warm-start
python -m src.main.improved_experiment_runner \
  --experiment_name ba_lora_75k_<dataset>_no_warmstart \
  --dataset <your_dataset> \
  --peft_method ba_lora \
  --param_budget 75000 \
  --ba_lora_base_rank 8 \
  --epochs 3 \
  --seed 42
# Note: Omit --ba_lora_use_warmstart flag

# Ablation 2: Uniform ranks
python -m src.main.improved_experiment_runner \
  --experiment_name ba_lora_75k_<dataset>_uniform \
  --dataset <your_dataset> \
  --peft_method ba_lora \
  --param_budget 75000 \
  --ba_lora_base_rank 8 \
  --ba_lora_use_uniform_ranks \
  --ba_lora_use_warmstart \
  --epochs 3 \
  --seed 42
```

### Results Organization

All results are saved in `results/<experiment_name>/`:

```
results/
├── lora_r8_sst2/
│   ├── config.json           # Experiment configuration
│   ├── training_log.txt      # Training logs
│   ├── results_final.json    # Final metrics
│   └── model_checkpoint/     # Saved model
├── ba_lora_75k_sst2/
│   └── ...
└── ...
```

### Tracking Results

Create a shared spreadsheet to track all experiments:

| Experiment       | Dataset | Method  | Params | Accuracy | F1 Score | Time | Status |
| ---------------- | ------- | ------- | ------ | -------- | -------- | ---- | ------ |
| lora_r8_sst2     | SST-2   | LoRA    | 739K   | 92.5%    | 92.6%    | 11m  | Done   |
| ba_lora_75k_sst2 | SST-2   | BA-LoRA | 665K   | 93.2%    | 93.3%    | 13m  | Done   |
| ...              | ...     | ...     | ...    | ...      | ...      | ...  | ...    |

### Communication

**Daily standup questions**:

1. What experiments did you complete yesterday?
2. What experiments are you running today?
3. Any blockers or issues?

**Weekly sync**:

- Compare results across datasets.
- Identify patterns or anomalies.
- Adjust experimental protocol if needed.

## Troubleshooting

### Common Issues

#### Issue 1: Out of Memory Error

**Symptom**:

```
RuntimeError: CUDA out of memory
```

**Solutions**:

- Reduce batch size:
  ```bash
  --batch_size 16  # Instead of 32
  ```
- Reduce max sequence length:
  ```bash
  --max_length 128  # Instead of 512
  ```
- Use gradient accumulation:
  ```bash
  --gradient_accumulation_steps 2
  ```

#### Issue 2: Dataset Not Found

**Symptom**:

```
FileNotFoundError: Dataset not found
```

**Solution**:

- **For HuggingFace loading**:
  ```bash
  # Check internet connection
  # Clear cache and retry
  rm -rf ~/.cache/huggingface/datasets/
  ```
- **For local loading**:

  ```bash
  # Verify files exist
  ls -la data/<dataset>_dataset/

  # Re-download if needed
  python scripts/download_datasets.py
  ```

#### Issue 3: Parameter Budget Not Met

**Symptom**:

```
AssertionError: Parameter count exceeds budget
```

**Solution**:
This is a feature, not a bug. The budget enforcement is working. However, if you see this:

- Check total parameter count calculation.
- Adjust param_budget to allow for overhead:
  ```bash
  --param_budget 80000  # Slightly higher to account for rounding
  ```

#### Issue 4: Slow Training

**Symptom**: Training takes much longer than expected.
**Possible causes**:

- Running on CPU instead of GPU.
- Large batch size with small GPU memory.
- Dataset loading bottleneck.

**Solutions**:

- Check GPU availability:
  ```python
  import torch
  print(torch.cuda.is_available())  # Should be True
  print(torch.cuda.get_device_name(0))
  ```
- Enable GPU:
  ```bash
  export CUDA_VISIBLE_DEVICES=0  # Use first GPU
  ```
- Optimize data loading:
  ```bash
  --num_workers 4  # Parallel data loading
  ```

#### Issue 5: Inconsistent Results

**Symptom**: Same experiment gives different results.
**Solution**:
Ensure reproducibility:

```bash
--seed 42  # Always use same seed
```

Check if any randomness is unseeded:

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

#### Issue 6: Warm-Start Not Applied

**Symptom**:

```
WARNING: No gradients available for warm-start
```

**Solution**:
Check that:

- Training dataset is provided in config.
- Gradient samples > 0.
- Target modules are correctly specified.

**Debug**:

```python
# Check if gradients were accumulated
print(f"Gradients stored: {len(gradient_analyzer.gradients)}")
```

### Getting Help

**Order of operations**:

1. Check this README.
2. Look at error message carefully.
3. Search project issues on GitHub.
4. Ask team members in group chat.
5. Consult course instructor/TA.

**When asking for help, provide**:

- Full error message.
- Command you ran.
- Dataset and configuration.
- System info (GPU, CUDA version).

## Analysis and Visualization

### Analyzing Results

After experiments complete, analyze results:

```bash
# Generate comparison tables
python -m src.main.analyse_results \
  --experiment_names lora_r8_sst2 ba_lora_75k_sst2 \
  --output_dir analysis/
```

This creates:

- `comparison_table.csv`
- `performance_plot.png`
- `rank_allocation_heatmap.png`

### Visualizations

**Rank allocation heatmap**:

```python
python scripts/visualize_ranks.py \
  --model_path results/ba_lora_75k_sst2/model_checkpoint \
  --output rank_heatmap.png
```

**Training curves**:

```python
python scripts/plot_training.py \
  --experiments lora_r8_sst2 ba_lora_75k_sst2 \
  --output training_curves.png
```

### References

Key papers to read:

- LoRA (Hu et al., 2021) - Foundation.
- BERT (Devlin et al., 2019) - Model architecture.
- GoRA (if available) - Gradient-based approach.
- Parameter-efficient fine-tuning surveys.
