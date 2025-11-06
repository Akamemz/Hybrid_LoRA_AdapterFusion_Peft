# Related Work

## Parameter-Efficient Fine-Tuning (PEFT)

**LoRA (Low-Rank Adaptation)**  
Hu et al. (2021) introduced *LoRA*, a simple yet powerful technique for parameter-efficient fine-tuning.  
By freezing the pretrained model and injecting trainable low-rank matrices into attention projections, LoRA achieves near–full fine-tuning performance with less than 1% additional parameters and no extra inference cost.  
This method established the foundation for all later adaptive or dynamic LoRA variants.  
→ *BA-LoRA builds directly on LoRA’s architecture but replaces fixed-rank adapters with adaptive, budget-aware ones.*

---

## Adaptive Rank Allocation and Efficiency Improvements

**AdaLoRA (Adaptive Budget Allocation for LoRA)**  
Zhang et al. (ICLR 2023) proposed *AdaLoRA*, which adaptively redistributes rank across layers based on importance scores while respecting a total parameter budget.  
It prunes less useful adapters and increases capacity in critical layers, achieving better trade-offs between efficiency and accuracy.  
→ *BA-LoRA adopts AdaLoRA’s core idea of budget-constrained rank allocation but replaces heuristic pruning with explicit gradient-based importance estimation.*

**ALoRA (Adaptive/Automatic Low-Rank Adaptation)**  
ALoRA automates rank discovery by jointly optimizing rank parameters during fine-tuning, minimizing manual tuning of hyperparameters.  
Rather than pruning post hoc, ALoRA learns which layers benefit from higher rank through the training objective itself.  
→ *BA-LoRA shares ALoRA’s goal of rank adaptivity but uses pretraining gradients and budget allocation instead of meta-optimization, improving interpretability and stability.*

**DyLoRA (Dynamic LoRA)**  
Valipour et al. (2023) proposed *DyLoRA*, which trains LoRA blocks for a **range of ranks** simultaneously, enabling post-training rank adjustment without retraining.  
It removes the need for costly rank searches and accelerates tuning by sorting learned subspaces across ranks.  
→ *BA-LoRA extends this flexibility through static yet **budget-aware dynamic allocation**, balancing efficiency and adaptability while maintaining control over total parameters.*

---

## Scaling LoRA to Large Language Models

**QLoRA (Quantized LoRA)**  
Dettmers et al. (NeurIPS 2023) demonstrated that LoRA can be combined with **4-bit quantization** to fine-tune 33B–65B parameter models on a single 48 GB GPU without quality loss.  
Their method introduces 4-bit *NormalFloat* quantization, *Double Quantization*, and *Paged Optimizers*, forming the basis for modern instruction-tuned open LLMs like Guanaco.  
→ *While BA-LoRA does not perform quantization, it addresses the complementary problem of maximizing accuracy **within a fixed parameter budget**, making it a natural partner or upstream step for QLoRA-style compression.*

---

## Gradient-Guided Adaptation

**GoRA (Gradient-Driven Adaptive LoRA)**  
He et al. (2025) introduced *GoRA*, the first framework to jointly handle **adaptive rank allocation** and **nonzero initialization** using gradient statistics.  
It computes the importance of each weight matrix via gradient–weight sensitivity and allocates rank accordingly, initializing adapters with pseudo-inverse–based compressed gradients.  
→ *BA-LoRA generalizes GoRA’s insight by using accumulated gradient importance but replaces pseudo-inverse initialization with an **SVD-based warm start** for numerical stability and reproducibility.*

---

# References

- Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2021). **LoRA: Low-Rank Adaptation of Large Language Models.** *arXiv:2106.09685.*  
  *Introduced the core low-rank adaptation framework for efficient fine-tuning.*

- Zhang, T., Chen, X., Li, X., Wang, H., Chen, Q., & Sun, M. (2023). **AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.** *International Conference on Learning Representations (ICLR 2023).*  
  *Proposed adaptive rank pruning under a fixed parameter budget.*

- [Authors from your ALoRA PDF] (2024). **ALoRA: Adaptive Low-Rank Adaptation via Learnable Rank Parameters.** *arXiv preprint, 2024.*  
  *Automates rank discovery by integrating rank learning into optimization.*

- Valipour, M., Rezagholizadeh, M., Kobyzev, I., & Ghodsi, A. (2023). **DyLoRA: Dynamic Search-Free Low-Rank Adaptation.** *arXiv:2210.07558.*  
  *Enables post-training dynamic rank flexibility and faster training.*

- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs.** *NeurIPS 2023.*  
  *Demonstrated that 4-bit quantization with LoRA matches 16-bit finetuning performance.*

- He, H., Ye, P., Ren, Y., Yuan, Y., Zhou, L., Ju, S., & Chen, L. (2025). **GoRA: Gradient-Driven Adaptive Low Rank Adaptation.** *arXiv:2502.12171.*  
  *Unified gradient-based rank allocation and nonzero initialization under one framework.*

