## Instruction for code folder

- All classes and functions need to be in component folder
- Main loop code needs to be in the main loop code folder.
- All codes need to have docstrings and hints.

---

# Research Proposals

## 1) LoRA-DC: Data-Centric Curricula and Augmentation for Parameter-Efficient Fine-Tuning

**Core Idea**  
- Standard LoRA focuses on *model-side efficiency* (low-rank adapters).  
- LoRA-DC introduces *data-centric improvements*:  
  - Curriculum learning (train from easy → hard examples).  
  - Lightweight augmentation (paraphrasing, mixup).  
  - Robust loss functions (label smoothing, noise-aware training).  

**Contribution**  
- Shows that **data quality + smart sampling** improves performance without increasing parameters.  
- Boosts generalization and data efficiency, especially on low-resource datasets.  

**Novelty**  
- “LoRA + Data Quality” has been underexplored.  
- Most work tweaks adapters; novelty here lies in **combining LoRA with data-handling strategies**.  

---

## 2) Hybrid PEFT: LoRA + AdapterFusion for Efficient and Transferable Fine-Tuning

**Core Idea**  
- LoRA is efficient but usually task-specific (one adapter per task).  
- AdapterFusion learns how to combine multiple adapters across tasks.  
- Hybrid PEFT merges **LoRA adapters with AdapterFusion**, making LoRA more transferable and multi-task capable.  

**Contribution**  
- Enables **cross-task transfer**: instead of retraining for each new dataset, the model fuses knowledge from multiple LoRA adapters.  
- Maintains efficiency (small parameter overhead) while improving transferability.  

**Novelty**  
- Existing LoRA research doesn’t address multi-task/generalization.  
- This hybrid approach bridges **LoRA’s efficiency** with **AdapterFusion’s transfer power**.  

---

## 🔹 One-Line Contrast
- **LoRA-DC** → “Improve LoRA by making the *data smarter*.”  
- **Hybrid PEFT (LoRA + AdapterFusion)** → “Improve LoRA by making the *adapters transferable across tasks*.”  
