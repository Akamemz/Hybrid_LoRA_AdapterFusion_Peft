# Introduction

**Large language models** are very costly computationally to fine-tune the full model. Modern LLMs like GPT, BERT, and LLaMA are designed with billions of parameters. Adaptation of these large models to specific tasks (like medical chatbot, legal assistance, or sentiment analysis) through traditional fine-tuning is prohibitively expensive for most researchers and organizations.

## LoRA (Low Rank Adaptation)

Instead of updating all billions of parameters of a pretrained model (GPT-3), LoRA freezes the original model and adds small “adapter” matrices that capture the task-specific changes. This reduces trainable parameters 1000 times from billions to thousands.

Pretrained models have billions of parameters stored in weight matrix *W*, and full fine-tuning requires adapting the matrix for a new task. It works like:

```
W_new = W_pretrained + ΔW   ;   (ΔW is huge as W_pretrained)
```

LoRA represents weight updates in a low intrinsic dimension. Instead of updating the (12,288×12,288 = 150M), LoRA uses a smart trick of two tiny matrices *B (12,288×8)* and *A (8×12,288)*, turning the billions of trainable parameters into just 196K. LoRA works as:

```
W_new = W_pretrained + B×A   ;   (B & A are very small W)
```

### Comparison of Fine-Tuning Methods

| **Method** | **Trainable Params** | **Storage** | **Training Time** |
|-------------|----------------------|--------------|-------------------|
| Full Fine-Tuning | 150M | 350 GB | 8 hours |
| LoRA | 196K | 35 MB | 1.5 hours |

*Table 1: Real-world comparison of full fine-tuning and LoRA models.*

LoRA reduces the trainable parameters and GPU usage significantly. However, it has some drawbacks:

- **Suboptimal rank allocation:** Different layers have different roles; rank allocation is not optimal in LoRA.  
- **Expensive manual tuning:** Finding the right rank needs grid search over *r ∈ {2,4,8,16,32,64}*, which is costly.  
- **Task-dependent sweet spot:** Sentiment analysis may work best for *r = 4*, question answering might be better for *r = 16*, and domain adaptation may need *r = 32*.

## BA-LoRA

**BA-LoRA’s** adaptive allocation strategy assigns higher ranks to critical layers and reduces ranks for less important ones. This smart distribution keeps the total parameter budget identical to existing methods.

### BA-LoRA vs Other Approaches

| **Feature** | **LoRA** | **ALoRA** | **GoRA** | **BA-LoRA** |
|--------------|-----------|------------|-----------|-------------|
| Adaptive Ranks | ✗ | ✓ | ✓ | ✓ |
| Efficient | ✓ | ✗ | ✗ | ✓ |
| Budget Control | ✗ | ✗ | ✓ | ✓ |
| Easy to Use | ✓ | ✗ | ✗ | ✓ |

*Table 2: Features eligibility of LoRA, ALoRA, GoRA, and BA-LoRA.*
