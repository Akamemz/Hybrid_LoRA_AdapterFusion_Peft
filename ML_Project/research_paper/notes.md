# Project at a Glance

**Working title:** “LoRA-DC: Data-Centric Curricula and Augmentation for Parameter-Efficient Fine-Tuning”  
**Core claim:** With the same adapter budget, data quality + curriculum yields better accuracy/robustness than model-only tweaks, while keeping PEFT’s efficiency.

---

## 1. Hypotheses (testable)

- **H1 (Curriculum):** Easy-to-hard sampling (by confidence/entropy) improves final accuracy vs. uniform sampling for LoRA at fixed rank.  
- **H2 (Augmentation):** Lightweight paraphrase augmentation + mixup stabilizes LoRA training on small datasets, improving generalization.  
- **H3 (Robustness):** A noise-aware loss (label smoothing or generalized cross-entropy) with LoRA reduces overfitting to noisy/augmented data.  
- **H4 (Efficiency):** LoRA-DC reaches same accuracy with fewer steps vs. vanilla LoRA (better data efficiency).  

---

## 2. Datasets (manageable, comparative)

- **SST-2:** binary sentiment, short
  - ```Python
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
- **AG News:** 4-class topic, medium 
  - ```Python
    from datasets import load_dataset
    ds = load_dataset("sh0416/ag_news")
- **IMDB:** binary sentiment, long reviews 
  - ```Python
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb")
- **Optional robustness check:** TweetEval – irony (noisy, short) 
  - ```Python
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/tweet_eval", "emoji")

---

## 3. Models & Baselines

**Backbone(s):**  
- roberta-base (primary)  
- distilbert-base-uncased (light)  
- optional mistral-7B-instruct via QLoRA (if GPU allows)  

**Baselines:**  
- Full fine-tune (upper bound, report cost)  
- LoRA (query+value; r∈{4,8,16}, α=16r, dropout 0.1)  
- QLoRA (optional, for one dataset)  

---

## 4. Method: LoRA-DC

A simple, reproducible pipeline that adds:  
(A) curriculum, (B) augmentation + mixup, (C) noise-aware loss on top of standard LoRA.  
No extra trainable params beyond adapters.  

### A) Confidence-ranked curriculum (easy → hard)
- **Scoring:**  
  - Freeze backbone + attach a linear head; train 5 epochs on 20% bootstrap split.  
  - Run probe → per-example entropy H(x) (lower = easier).  
- **Binning:** Tertiles → Easy (E), Medium (M), Hard (H).  
- **Schedule:**  
  - Phase 1: Train on E (20% steps)  
  - Phase 2: E+M (next 40%)  
  - Phase 3: E+M+H (final 40%), cosine LR decay  

### B) Paraphrase augmentation + mixup
- **Paraphrase:** With prob 0.3, replace by T5-base paraphrase (or EDA synonyms).  
- **Mixup:** Mix embeddings at CLS layer with λ~Beta(0.4).  
- Apply only in Phase 2 & 3.  

### C) Noise-aware loss
- Label Smoothing (ε=0.1)  
- Optional: Generalized Cross-Entropy (q=0.7)  

---

## 5. Training Config (solid defaults)

- Optimizer: AdamW (LoRA params only)  
- LR: 2e-4; weight decay 0.01; linear warmup 6%; cosine decay  
- Batch sizes: 32 (SST-2/AG), 16 (IMDB)  
- Epoch caps: ~5–10  
- Seeds: 5 runs per setting  
- LoRA targets: attention q,v (opt. o/proj)  
- Precision: bf16/fp16 if available  

---

## 6. Experiments & Ablations

**Main table (per dataset):**  
- Full FT, LoRA, LoRA-DC, (optional QLoRA)  
- Metrics: Accuracy, Macro-F1, steps to target acc, wall-clock, VRAM, params  

**Ablations (SST-2 & IMDB):**  
- LoRA vs LoRA + Curriculum  
- LoRA vs LoRA + Aug+Mixup  
- LoRA vs LoRA + Label Smoothing  
- LoRA-DC (all three)  
- Curriculum scoring: entropy-probe vs heuristic length  
- LoRA rank r∈{4,8,16} under LoRA-DC  

**Robustness:**  
- Label noise: flip 10% → report drop  
- Text noise: typos/word-drop 10% → OOD acc  

---

## 7. Success Criteria

- +0.8 to +1.5 Accuracy vs vanilla LoRA on IMDB  
- Same accuracy as LoRA with ≥20% fewer steps  
- Smaller robustness drop under label/typo noise  

---

## 8. Implementation Roadmap

**Stack:** HF transformers, datasets, peft, accelerate, bitsandbytes, wandb/TensorBoard  

- Data loaders + noise toggle  
- Probe trainer → entropy JSON  
- Curriculum sampler (E→M→H)  
- LoRA config (peft.LoraConfig)  
- Paraphrase cache + EDA fallback  
- Mixup hook (Phase 2–3)  
- Loss: LS or GCE  
- Eval loop with logging  

---

## 9. Writing the Paper

- **Abstract:** PEFT works, but data-centric curricula + augmentation push further.  
- **Intro:** Problem, motivation, contributions.  
- **Related Work:** LoRA, QLoRA, Dy/ALoRA/GoRA, curricula, mixup, noise losses.  
- **Method:** scoring, phases, aug, mixup math, loss.  
- **Experiments:** setup, baselines, ablations, robustness.  
- **Results:** tables, curves, efficiency plots.  
- **Discussion:** why curriculum helps; limits.  
- **Conclusion:** summary + future work.  

---

## 10. Risks & Mitigation

- Paraphrase quality varies → cache & SBERT filter (cos > 0.85)  
- Probe too weak → fallback: sentence length + OOV ratio  
- Compute limits → skip QLoRA, stick to roberta-base  

---

## 11. Concrete Checklists

**Code tasks (week 1–2):**  
- Load datasets + baseline trainer  
- Probe + entropy scoring  
- Curriculum sampler  
- Paraphrase cache + mixup hook  
- Label smoothing/GCE  
- Seeds + logging  

**Experiment grid (week 3–5):**  
- Baselines (LoRA r=8)  
- LoRA-DC (all datasets)  
- Ablations on SST-2 & IMDB  
- Robustness tests  
- Ranks {4,8,16}  

**Writing (finalize by Nov 20):**  
- Methods & diagrams early  
- Tables auto-export  
- Draft Related Work (10 papers)  
- Final polish + appendix with configs  

---
Short note to self:

-Yes: We still get awarded if paper is just in submittable form or in process.

-No: We don’t have to wait for actual conference acceptance.
