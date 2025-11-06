# BA-LoRA Interactive Demo

Interactive Streamlit demonstration of Budget-Aware Adaptive LoRA process.

## 🎯 Purpose

This demo is designed for the **CS 8267 project presentation** to:
- Visually explain the BA-LoRA process step-by-step
- Show why gradient importance ≠ task importance (r=0.42)
- Demonstrate parameter budget matching
- Make negative results understandable and scientifically valuable

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to demo directory
cd ML_Project/src/demo

# Install requirements
pip install -r requirements_demo.txt

# Run the demo
streamlit run streamlit_ba_lora_demo.py
```

The demo will open in your browser at `http://localhost:8501`

---

## 📋 Demo Modes

The application has **5 interactive modes**:

### 1. **Full Pipeline** (Recommended for Presentation)
- Complete step-by-step walkthrough
- Phase 1: Gradient importance estimation
- Phase 2: Budget-aware rank allocation
- Phase 3: Performance analysis
- **Best for**: Main presentation demo

### 2. **Importance Estimation**
- Focus on gradient-based importance calculation
- Interactive importance visualization
- Statistical analysis
- **Best for**: Explaining how importance estimation works

### 3. **Rank Allocation**
- Demonstrates budget-aware allocation algorithm
- Parameter budget matching verification
- Comparison with vanilla LoRA
- **Best for**: Explaining the allocation strategy

### 4. **Performance Analysis** ⭐
- **KEY INSIGHT**: Shows why BA-LoRA underperforms
- Visualizes weak correlation (r=0.42)
- Explains gradient vs. actual importance mismatch
- **Best for**: Defending negative results scientifically

### 5. **Load Real Results**
- Load actual experimental results from JSON files
- Visualize real performance data
- Compare methods on real experiments
- **Best for**: Showing actual experimental outcomes

---

## 🎬 Presentation Flow (15-20 minutes)

### Recommended Demo Sequence:

**1. Introduction (2 min)**
- Open demo, show title and overview
- Explain research question

**2. Full Pipeline Demo (8-10 min)**

**Step 1: Importance Estimation (2-3 min)**
- Click "Run Importance Estimation"
- Show gradient importance bar chart
- Explain: "Middle layers have higher gradients"
- Show importance values table

**Step 2: Rank Allocation (2-3 min)**
- Click "Allocate Ranks"
- Show rank comparison chart
- **Emphasize**: "Parameters match exactly - fair comparison!"
- Show budget verification metrics

**Step 3: Performance Analysis (3-4 min)** ⭐ **MOST IMPORTANT**
- Click "Simulate Performance"
- Show 3-panel visualization:
  1. Gradient vs. Actual Importance (bars)
  2. **Correlation scatter** (r=0.42) - KEY INSIGHT
  3. Performance comparison (-0.31pp gap)
- Read the warning box explaining why weak correlation matters
- Show summary statistics

**3. Switch to Performance Analysis Mode (3-4 min)**
- Go to sidebar → Select "Performance Analysis"
- Show detailed correlation analysis
- Highlight mismatch examples table
- **Key message**: "This explains why BA-LoRA underperforms"

**4. (Optional) Load Real Results (2-3 min)**
- Show actual experimental data
- Confirm simulated patterns match reality

**5. Q&A (remaining time)**
- Use interactive controls to answer questions
- Adjust parameters to show different scenarios

---

## 🎨 Interactive Features

### Sidebar Controls

**Model Parameters:**
- Number of Layers (4-12)
- Hidden Dimension (512-1024)
- Base Rank (2-32)

**BA-LoRA Parameters:**
- Gradient Samples (1000-10000)
- Allocation Strategy (Proportional, Top-K, Threshold)

**Demo Modes:**
- Full Pipeline
- Importance Estimation
- Rank Allocation
- Performance Analysis
- Load Real Results

### Key Visualizations

1. **Gradient Importance Bar Chart**
   - Color-coded by importance level
   - Shows which layers have high/low gradients

2. **Rank Allocation Comparison**
   - Side-by-side bars: Vanilla LoRA vs BA-LoRA
   - Parameter distribution pie chart

3. **Correlation Scatter Plot** ⭐
   - **Most important visualization**
   - Shows weak r=0.42 correlation
   - Explains performance gap

4. **Performance Comparison**
   - Bar chart with delta annotation
   - Color-coded (yellow for negative, green for positive)

---

## 💡 Key Messages to Convey

### During Demo:

1. **Fair Comparison**
   > "Notice the parameters match exactly: 294,912 vs 294,912. This isolates the effect of adaptive allocation."

2. **Sensible Allocation**
   > "BA-LoRA produces a reasonable pattern: higher ranks for middle layers. This aligns with transformer literature."

3. **The Key Insight** ⭐
   > "But here's the problem: gradient importance correlates only weakly (r=0.42) with actual task importance. This explains the performance gap."

4. **Scientific Value**
   > "This isn't a failure—it's a scientific finding. We now understand WHY gradient-based allocation doesn't work for simple classification."

5. **When It Might Work**
   > "Complex reasoning tasks may have stronger correlations, making adaptive allocation more effective."

---

## 🔧 Troubleshooting

### Demo won't start

```bash
# Make sure you're in the right directory
cd ML_Project/src/demo

# Check if streamlit is installed
streamlit --version

# If not installed
pip install streamlit
```

### Port already in use

```bash
# Use a different port
streamlit run streamlit_ba_lora_demo.py --server.port 8502
```

### Plots not showing

```bash
# Ensure plotting libraries installed
pip install matplotlib seaborn plotly
```

### Loading real results fails

```bash
# Check results directory path
ls -la results/results_sst2

# Make sure JSON files exist
ls results/results_sst2/*.json
```

---

## 📊 What to Show Evaluators

### For Technical Depth:
- Show the code in "Full Pipeline" mode expandable sections
- Explain the budget matching algorithm
- Discuss the Frobenius norm calculation

### For Scientific Insight:
- Focus on "Performance Analysis" mode
- Emphasize the correlation analysis
- Explain mismatch examples

### For Experimental Rigor:
- Use "Load Real Results" mode
- Show statistical analysis
- Demonstrate reproducibility

---

## 🎓 Teaching Points

This demo helps explain:

1. **Parameter-Efficient Fine-Tuning**
   - What is LoRA?
   - Why adapt ranks?
   - What is parameter budget?

2. **Importance Estimation**
   - Gradient-based metrics
   - Limitations of early gradients
   - Alternative metrics

3. **Fair Comparison**
   - Why control parameter counts?
   - How to isolate effects?
   - Scientific methodology

4. **Negative Results**
   - When methods don't work
   - Understanding failure modes
   - Scientific value of transparency

---

## 📁 File Structure

```
demo/
├── streamlit_ba_lora_demo.py    # Main demo application
├── requirements_demo.txt         # Demo dependencies
└── README.md                     # This file
```

---

## 🎯 Presentation Checklist

Before your presentation:

- [ ] Install all dependencies
- [ ] Test demo locally
- [ ] Practice navigation between modes
- [ ] Prepare to explain correlation plot
- [ ] Have backup static images (in case demo fails)
- [ ] Test loading real results (if using)
- [ ] Set up projector/screen sharing
- [ ] Close unnecessary applications
- [ ] Disable notifications

During presentation:

- [ ] Start in "Full Pipeline" mode
- [ ] Go through steps sequentially
- [ ] Emphasize budget matching
- [ ] Highlight correlation insight (r=0.42)
- [ ] Switch to "Performance Analysis" for detail
- [ ] Use sidebar to adjust parameters if asked
- [ ] Keep within time limit (15-20 min)

---

## 💬 Example Dialogue

**When showing correlation:**
> "This is the key insight from our research. The scatter plot shows gradient importance versus actual task importance. Notice the correlation is only 0.42—that's weak. This means gradient magnitude doesn't reliably predict which layers matter most for the task. And that explains why our adaptive allocation doesn't improve performance."

**When showing budget match:**
> "This is crucial for scientific validity. Both methods use exactly 294,912 trainable parameters. We're not improving by just using more parameters—we're testing whether adaptive allocation at the same budget helps. And our answer is: not for simple classification tasks."

**When handling negative results:**
> "Some might see this as a failed experiment, but we disagree. We rigorously tested a reasonable hypothesis, found it doesn't work, and explained WHY through systematic analysis. That's valuable science. Now we know gradient-based allocation is insufficient for simple tasks, and we've identified what to try instead."

---

## 🔗 Resources

- **Main Project README**: `../../README.md`
- **Defense Framework**: `../../research_paper/BA_LoRA_Defense_Framework.md`
- **Report Template**: `../../research_paper/BA_LoRA_Report_Template.md`

---

**Built for CS 8267 Advanced Machine Learning**
*Kennesaw State University*
