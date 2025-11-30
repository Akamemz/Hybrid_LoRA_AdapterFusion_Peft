## Datasets (selected benchmarks)

---

### **SST-2 (Stanford Sentiment Treebank)**
- **Task:** Binary sentiment classification  
- **Domain:** Movie reviews (Rotten Tomatoes)  
- **Data:** Single sentences from reviews  
- **Labels:** `0 = Negative`, `1 = Positive`  
- **Size:** Small (~67k train examples)  
- **Why this dataset (Role vs. hypotheses):**  
  - Fast, clean baseline for single-task performance (**H1**).  
  - Great for low-data and few-shot stress tests (**H2**, **H4**).  
  - Serves as the initial sanity-check benchmark for the pipeline.

---

### **AG News (AG’s News Corpus)**
- **Task:** 4-class topic classification  
- **Domain:** News articles  
- **Data:** Titles + descriptions  
- **Labels:** `1 = World`, `2 = Sports`, `3 = Business`, `4 = Sci/Tech`  
- **Size:** Medium (~120k train examples)  
- **Why this dataset (Role vs. hypotheses):**  
  - Non-sentiment, medium-scale baseline to test generality (**H1**).  
  - Natural **source task** for transfer experiments into sentiment tasks (**H2**).

---

### **IMDB (Internet Movie Database)**
- **Task:** Binary sentiment classification  
- **Domain:** Movie reviews  
- **Data:** Full-length, multi-sentence reviews (longer sequences vs. SST-2)  
- **Labels:** `0 = Negative`, `1 = Positive`  
- **Size:** Medium (~25k train examples; much longer inputs)  
- **Why this dataset (Role vs. hypotheses):**  
  - Tests robustness to long sequences and narrative complexity (**H4**).  
  - Natural **target task** for domain transfer from SST-2 → IMDB (**H2**).
