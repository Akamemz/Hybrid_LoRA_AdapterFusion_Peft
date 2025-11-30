# Working title

**Hybrid PEFT: LoRA + AdapterFusion for Efficient and Transferable Fine-Tuning**

---

## Core claim

With the same adapter budget, combining LoRA (low-rank updates) with AdapterFusion (learned mixing of task adapters) yields better accuracy and cross-task transfer than either method alone—without increasing inference latency.

---

## 1) Hypotheses (testable)

- **H1 (Performance):** LoRA+AdapterFusion ≥ max(LoRA, Adapters, Prefix/BitFit) on single tasks at a fixed parameter budget.
- **H2 (Transfer):** Adapters trained on related source tasks, fused with LoRA for the target task, improve few-shot and domain-shift performance.
- **H3 (Budget fairness):** When total trainable parameters are matched, LoRA+Fusion remains superior (parameter-efficiency, no inference overhead).
- **H4 (Stability):** Fusion stabilizes low-rank adaptation under low-data regimes (e.g., 1–5k examples) vs. LoRA-only.

---

## 2) Datasets (manageable, comparative)

Pick 2–3 from below (NLP focus for speed):

- **SST-2 (sentiment, small)** – single-task performance and low-data tests.
- **AG News (4-class topic)** – medium, clean baseline.
- **IMDB (long reviews)** – robustness to long sequences.
- **TweetEval (sentiment / irony)** – mild domain shift; good for transfer.
- _Optional:_ Amazon Reviews polarity or Yelp for cross-domain sentiment.

---

## 3) Baselines (fair, parameter-matched)

- Full fine-tuning (upper bound; compute heavy, maybe SST-2 only).
- Adapters (Houlsby or Pfeiffer).
- LoRA (fixed rank _r_).
- Prefix-tuning or BitFit (very light PEFT).
- AdapterFusion only (fuse multiple task adapters, no LoRA).
- **Ours:** LoRA + AdapterFusion (train LoRA on target; fuse pre-trained adapters from source tasks).

**Parameter budget rule:** keep total trainable parameters within ±5% across methods.

---

## 4) Methods (what you actually build)

- **Backbone:** bert-base-uncased (or DistilBERT for speed).
- **Adapters:** Train task-specific adapters (e.g., SST-2, AG News).
- **Fusion:** Learn AdapterFusion weights on the target task to combine these adapters.
- **LoRA:** Inject LoRA modules into attention (and optionally FFN) for the target task.
- **Hybrid:** Freeze backbone; enable Fusion + LoRA (Fusion mixes knowledge; LoRA provides task-specific low-rank adaptation).

**Implementation sketch (HuggingFace-friendly):**

- Build a PEFT config loader that toggles: LoRA, Adapters, AdapterFusion, LoRA+Fusion.
- CLI flags: `--method`, `--rank`, `--lora_alpha`, `--fusion_sources`, `--param_budget_target`.

---

## 5) Experiments

- **E1: Single-task accuracy (SST-2, AG News, IMDB)**  
  Compare all baselines vs. LoRA+Fusion at equal param budgets.

- **E2: Few-shot (e.g., 128 / 512 examples)**  
  Show LoRA+Fusion stability/benefits when data is scarce.

- **E3: Transfer / Fusion sources**  
  Train adapters on AG News + SST-2, fuse them, fine-tune LoRA on TweetEval (or IMDB).  
  Measure gains vs. LoRA-only and Fusion-only.

- **E4: Budget sensitivity**  
  Sweep LoRA rank (r = 4, 8, 16) while adjusting adapter sizes to keep total params aligned.

- **E5: Efficiency**  
  Report trainable params, throughput (it/s), time-to-X-accuracy, GPU memory, inference latency (≈ unchanged).

---

## 6) Metrics

- **Primary:** Accuracy / F1 (macro for imbalanced).
- **Efficiency:** Trainable parameter count, peak GPU memory, steps/time to reach best dev score.
- **Robustness/Transfer:** Performance degradation under domain shift; few-shot delta vs. full-data.

---

## 7) Ablations (to isolate what matters)

- No-Fusion vs. Fusion (with equal params).
- Fusion sources: which source adapters help most? (topic vs. sentiment)
- Where to place LoRA: attention-only vs. attention+FFN.
- Freeze strategies: fusion-only trainable vs. fusion+LoRA trainable.

---

## 8) Expected results (what “success” looks like)

- LoRA+AdapterFusion beats best single-method PEFT by **+0.5–2.0 pts** on avg across tasks at fixed budget.
- Clear wins in few-shot and cross-domain settings.
- Equal or lower time-to-target accuracy vs. LoRA-only.

---

## 9) Novelty knobs (pick at least one)

- **Fusion-aware LoRA placement:** dynamically enable LoRA only in layers with high fusion entropy.
- **Param-budget allocator:** small search to split budget between LoRA rank vs. adapter size automatically.
- **Task-similarity guided Fusion:** weight source adapters by cosine similarity of CLS embeddings before learning fusion weights.

---

## 10) Feasibility & compute

- **Intensity:** Medium. One mid-range GPU (T4/A10/3060/4060) is enough. DistilBERT variants run on Colab-free with patience.
- **Run time:** Each run on SST-2/AG News typically <1–2 hours on a single modest GPU with low epochs + early stopping.

---

## 12) Report structure (conference-style)

- Title, Abstract
- **1. Introduction** — problem, PEFT limits, our hybrid idea
- **2. Related Work** — LoRA, Adapters, AdapterFusion, PEFT surveys
- **3. Method** — how we integrate LoRA with Fusion; param budget
- **4. Experiments** — datasets, baselines, metrics
- **5. Results** — main tables, few-shot, transfer, efficiency
- **6. Ablations** — placement, sources, budget splits
- **7. Conclusion & Limitations**
- Acknowledgements
- References

**Bonus section:** Target Venue: ECML-PKDD/ICMLA 2025; formatting compliance noted.

---

## 13) Timed milestones (aligned to class timeline)

- **Aug 27 (Topic submission):** Freeze scope + target venue (ECML-PKDD or ICMLA).
- **Sep 15 (Paper collection):** 10 papers (LoRA, Adapters, Fusion, PEFT surveys); build Related Work notes.
- **Sep 22 (Title submission):** Finalize title + abstract draft.
- **Sep 29 (Data acquisition):** Scripts for SST-2, AG News, IMDB; sanity baselines run.
- **Oct 1 (Dataset description):** Add dataset tables & examples to report.
- **Oct 29 (Related Work section):** Complete; include PEFT taxonomy figure.
- **Nov 1–15 (Core experiments):** E1–E3; produce main tables/plots.
- **Nov 16–24 (Ablations & novelty):** E4–E5 + chosen novelty knob.
- **Dec 1 (Final report & slides):** Camera-ready PDF; slides with results & demo.

---

## 14) Deliverables

- Code (with README run steps).
- Report in target conference template (bonus section included).
- Slides: problem, method diagram, baselines, results, ablations, takeaway.
- Repro pack: config files + seeds + exact checkpoints for top runs.

---

## 15) Risk & mitigation

- **Risk:** Fusion training unstable on tiny data.  
  **Mitigation:** early stopping, weight decay, freeze some layers, use DistilBERT.

- **Risk:** Budget fairness hard to match.  
  **Mitigation:** implement param counter; auto-adjust LoRA rank/adapter size to match.

- **Risk:** Transfer benefit small.  
  **Mitigation:** pick closer source tasks (e.g., sentiment ↔ reviews).
