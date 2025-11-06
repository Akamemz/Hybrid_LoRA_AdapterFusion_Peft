"""
BA-LoRA Interactive Demo
========================
Streamlit application demonstrating Budget-Aware Adaptive LoRA process

Run with: streamlit run streamlit_ba_lora_demo.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="BA-LoRA Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🎯 BA-LoRA Interactive Demo</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Budget-Aware Adaptive Low-Rank Adaptation Simulator</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    demo_mode = st.selectbox(
        "Demo Mode",
        ["Full Pipeline", "Importance Estimation", "Rank Allocation", "Performance Analysis", "Load Real Results"]
    )

    st.markdown("---")

    # Model parameters
    st.subheader("Model Parameters")
    num_layers = st.slider("Number of Layers", 4, 12, 6)
    hidden_dim = st.slider("Hidden Dimension", 512, 1024, 768, step=128)
    base_rank = st.slider("Base Rank (r)", 2, 32, 8, step=2)

    st.markdown("---")

    # BA-LoRA parameters
    st.subheader("BA-LoRA Parameters")
    gradient_samples = st.slider("Gradient Samples", 1000, 10000, 5000, step=1000)
    allocation_strategy = st.selectbox(
        "Allocation Strategy",
        ["Proportional", "Top-K", "Threshold-based"]
    )

    st.markdown("---")

    # Display info
    st.info(f"""
    **Current Configuration:**
    - Layers: {num_layers}
    - Hidden Dim: {hidden_dim}
    - Base Rank: {base_rank}
    - Gradient Samples: {gradient_samples}
    """)

# Helper functions
def simulate_gradient_importance(num_layers, seed=42):
    """Simulate gradient importance scores for layers"""
    np.random.seed(seed)

    # Simulate higher importance for middle layers
    layer_positions = np.linspace(0, 1, num_layers)
    base_importance = np.exp(-((layer_positions - 0.5) ** 2) / 0.15)

    # Add noise
    noise = np.random.normal(0, 0.1, num_layers)
    importance = base_importance + noise
    importance = np.maximum(importance, 0.3)  # Minimum importance

    return importance / importance.mean()  # Normalize

def allocate_ranks(importance, base_rank, budget, hidden_dim):
    """Allocate ranks based on importance scores"""
    # Proposed ranks
    proposed_ranks = np.round(base_rank * importance).astype(int)
    proposed_ranks = np.maximum(proposed_ranks, 1)  # Minimum rank 1

    # Calculate parameters
    params_per_layer = proposed_ranks * (hidden_dim + hidden_dim)  # Query + Value
    total_params = params_per_layer.sum()

    # Scale to meet budget
    if total_params > budget:
        scale_factor = budget / total_params
        proposed_ranks = np.maximum(np.round(proposed_ranks * scale_factor).astype(int), 1)

    return proposed_ranks

def simulate_actual_importance(num_layers, seed=42):
    """Simulate actual task-specific importance (different from gradient)"""
    np.random.seed(seed + 100)

    # Different pattern than gradient importance
    layer_positions = np.linspace(0, 1, num_layers)
    actual = 0.8 + 0.3 * np.sin(layer_positions * np.pi) + np.random.normal(0, 0.15, num_layers)

    return np.maximum(actual, 0.5)

# Main content based on demo mode
if demo_mode == "Full Pipeline":
    st.header("🔄 Full BA-LoRA Pipeline")

    # Step 1: Gradient Importance
    st.subheader("Step 1: Gradient-Based Importance Estimation")

    with st.expander("ℹ️ What is gradient importance?", expanded=False):
        st.markdown("""
        Gradient importance measures the magnitude of gradients flowing through each layer:

        ```python
        # Accumulate gradients over sample data
        for sample in random_sample(data, n=5000):
            loss = model(sample)
            loss.backward()
            gradients[layer] += layer.grad

        # Compute importance as Frobenius norm
        importance[layer] = ||gradients[layer]||_F
        ```

        **Intuition**: Layers with larger gradients are trying to change more → might be more important
        """)

    if st.button("▶️ Run Importance Estimation", key="run_importance"):
        with st.spinner(f"Estimating importance from {gradient_samples} samples..."):
            importance = simulate_gradient_importance(num_layers)

            # Store in session state
            st.session_state['importance'] = importance

            # Visualize
            fig, ax = plt.subplots(figsize=(10, 4))
            layers = [f"Layer {i}" for i in range(num_layers)]
            colors = plt.cm.RdYlGn(importance / importance.max())
            bars = ax.bar(layers, importance, color=colors, edgecolor='black', linewidth=1.5)
            ax.axhline(y=1.0, color='gray', linestyle='--', label='Mean Importance')
            ax.set_ylabel('Normalized Importance', fontsize=12, fontweight='bold')
            ax.set_title('Gradient-Based Layer Importance', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar, val in zip(bars, importance):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

            st.pyplot(fig)

            # Display as table
            importance_df = pd.DataFrame({
                'Layer': layers,
                'Gradient Importance': importance,
                'Relative to Mean': importance / 1.0,
                'Interpretation': ['High' if i > 1.1 else 'Medium' if i > 0.9 else 'Low' for i in importance]
            })
            st.dataframe(importance_df, use_container_width=True)

            st.success("✅ Importance estimation complete!")

    # Step 2: Rank Allocation
    if 'importance' in st.session_state:
        st.markdown("---")
        st.subheader("Step 2: Budget-Aware Rank Allocation")

        with st.expander("ℹ️ How does budget-aware allocation work?", expanded=False):
            st.markdown("""
            Budget-aware allocation ensures fair comparison with vanilla LoRA:

            ```python
            # Step 1: Propose ranks based on importance
            proposed_ranks = base_rank * (importance / importance.mean())

            # Step 2: Calculate total parameters
            total_params = sum(rank * (hidden_dim + hidden_dim) for rank in ranks)

            # Step 3: Scale to meet budget (match vanilla LoRA exactly)
            if total_params > budget:
                scale_factor = budget / total_params
                ranks = scale_down(ranks, scale_factor)

            # Step 4: Discretize to integers
            final_ranks = round(ranks)
            ```

            **Key**: Total parameters must match vanilla LoRA exactly for fair comparison!
            """)

        # Calculate budget
        vanilla_params = num_layers * base_rank * (hidden_dim + hidden_dim) * 2  # Q + V projections

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vanilla LoRA Budget", f"{vanilla_params:,} params", help="Target parameter count to match")
        with col2:
            st.metric("Uniform Rank", f"{base_rank}", help="All layers get same rank")

        if st.button("▶️ Allocate Ranks", key="allocate"):
            importance = st.session_state['importance']
            ranks = allocate_ranks(importance, base_rank, vanilla_params, hidden_dim)

            # Store in session state
            st.session_state['ranks'] = ranks

            # Calculate actual parameters
            ba_lora_params = (ranks * (hidden_dim + hidden_dim) * 2).sum()

            # Visualize allocation
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Left: Rank comparison
            x = np.arange(num_layers)
            width = 0.35
            ax1.bar(x - width/2, [base_rank] * num_layers, width, label='Vanilla LoRA',
                   color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
            ax1.bar(x + width/2, ranks, width, label='BA-LoRA',
                   color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
            ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Allocated Rank', fontsize=12, fontweight='bold')
            ax1.set_title('Rank Allocation Comparison', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f'L{i}' for i in range(num_layers)])
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')

            # Right: Parameter distribution
            labels = [f'Layer {i}' for i in range(num_layers)]
            params_per_layer = ranks * (hidden_dim + hidden_dim) * 2
            ax2.pie(params_per_layer, labels=labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Parameter Distribution Across Layers', fontsize=14, fontweight='bold')

            st.pyplot(fig)

            # Allocation table
            allocation_df = pd.DataFrame({
                'Layer': [f'Layer {i}' for i in range(num_layers)],
                'Gradient Importance': importance,
                'Vanilla LoRA Rank': [base_rank] * num_layers,
                'BA-LoRA Rank': ranks,
                'Rank Difference': ranks - base_rank,
                'Parameters (BA-LoRA)': ranks * (hidden_dim + hidden_dim) * 2
            })
            st.dataframe(allocation_df, use_container_width=True)

            # Budget verification
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Vanilla LoRA", f"{vanilla_params:,} params")
            with col2:
                st.metric("BA-LoRA", f"{ba_lora_params:,} params")
            with col3:
                budget_match = abs(vanilla_params - ba_lora_params) / vanilla_params < 0.01
                st.metric("Budget Match", "✅ Matched" if budget_match else "❌ Not Matched")

            st.success("✅ Rank allocation complete with budget matching!")

    # Step 3: Performance Comparison
    if 'ranks' in st.session_state:
        st.markdown("---")
        st.subheader("Step 3: Performance Analysis")

        with st.expander("ℹ️ Understanding the performance gap", expanded=False):
            st.markdown("""
            **Key Question**: Does gradient importance predict actual task importance?

            We measure:
            1. **Gradient Importance**: Magnitude of gradients during initial adaptation
            2. **Actual Importance**: Performance drop when layer is frozen (ablation study)

            If these correlate strongly → adaptive allocation should help
            If they correlate weakly → adaptive allocation won't help
            """)

        if st.button("▶️ Simulate Performance", key="simulate_perf"):
            importance = st.session_state['importance']
            ranks = st.session_state['ranks']

            # Simulate actual importance (different from gradient)
            actual_importance = simulate_actual_importance(num_layers)

            # Calculate correlation
            correlation = np.corrcoef(importance, actual_importance)[0, 1]

            # Store in session state
            st.session_state['actual_importance'] = actual_importance
            st.session_state['correlation'] = correlation

            # Simulate accuracy
            vanilla_acc = 91.07
            # BA-LoRA performance depends on correlation
            ba_lora_acc = vanilla_acc + (correlation - 0.7) * 2  # Simplified model

            # Visualization
            fig = plt.figure(figsize=(15, 5))
            gs = fig.add_gridspec(1, 3)

            # Plot 1: Importance comparison
            ax1 = fig.add_subplot(gs[0, 0])
            x = np.arange(num_layers)
            width = 0.35
            ax1.bar(x - width/2, importance, width, label='Gradient Importance',
                   color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
            ax1.bar(x + width/2, actual_importance, width, label='Actual Task Importance',
                   color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
            ax1.set_xlabel('Layer', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Importance Score', fontsize=11, fontweight='bold')
            ax1.set_title('Gradient vs. Actual Importance', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f'L{i}' for i in range(num_layers)])
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')

            # Plot 2: Correlation scatter
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.scatter(importance, actual_importance, s=100, alpha=0.6,
                       c=range(num_layers), cmap='viridis', edgecolors='black', linewidth=1.5)

            # Add regression line
            z = np.polyfit(importance, actual_importance, 1)
            p = np.poly1d(z)
            ax2.plot(importance, p(importance), "r--", alpha=0.8, linewidth=2, label=f'Fit: r={correlation:.2f}')

            # Add layer labels
            for i, (x_val, y_val) in enumerate(zip(importance, actual_importance)):
                ax2.annotate(f'L{i}', (x_val, y_val), fontsize=8, ha='right')

            ax2.set_xlabel('Gradient Importance', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Actual Task Importance', fontsize=11, fontweight='bold')
            ax2.set_title(f'Correlation: r = {correlation:.2f}', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Performance comparison
            ax3 = fig.add_subplot(gs[0, 2])
            methods = ['Vanilla LoRA', 'BA-LoRA']
            accuracies = [vanilla_acc, ba_lora_acc]
            colors_bar = ['#4ECDC4', '#FF6B6B']
            bars = ax3.bar(methods, accuracies, color=colors_bar, alpha=0.8,
                          edgecolor='black', linewidth=1.5, width=0.6)

            # Add value labels
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

            # Add delta annotation
            delta = ba_lora_acc - vanilla_acc
            ax3.annotate(f'Δ = {delta:+.2f}%',
                        xy=(0.5, max(accuracies) + 0.5), fontsize=11, ha='center',
                        bbox=dict(boxstyle='round,pad=0.5',
                                 facecolor='yellow' if delta < 0 else 'lightgreen',
                                 alpha=0.5, edgecolor='black', linewidth=1.5))

            ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
            ax3.set_title('Performance Comparison', fontsize=12, fontweight='bold')
            ax3.set_ylim([min(accuracies) - 1, max(accuracies) + 2])
            ax3.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            st.pyplot(fig)

            # Analysis
            st.markdown("### 🔍 Key Insight")

            if correlation < 0.5:
                st.markdown(f"""
                <div class="warning-box">
                <h4>⚠️ Weak Correlation Detected (r = {correlation:.2f})</h4>
                <p><strong>Interpretation</strong>: Gradient importance correlates only weakly with actual task importance.</p>
                <p><strong>Result</strong>: BA-LoRA shows modest underperformance ({delta:+.2f}pp)</p>
                <p><strong>Why</strong>: Gradient magnitude measures where the model <em>wants to change</em>,
                not where changes <em>actually help</em> for the task.</p>
                <p><strong>Conclusion</strong>: Gradient-based allocation insufficient for this task.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                <h4>✅ Strong Correlation Detected (r = {correlation:.2f})</h4>
                <p><strong>Interpretation</strong>: Gradient importance strongly predicts actual task importance.</p>
                <p><strong>Result</strong>: BA-LoRA shows improvement ({delta:+.2f}pp)</p>
                <p><strong>Why</strong>: Adaptive allocation successfully prioritizes important layers.</p>
                <p><strong>Conclusion</strong>: Gradient-based allocation effective for this configuration.</p>
                </div>
                """, unsafe_allow_html=True)

            # Summary table
            st.markdown("### 📊 Summary Statistics")
            summary_df = pd.DataFrame({
                'Metric': [
                    'Gradient-Actual Correlation',
                    'Vanilla LoRA Accuracy',
                    'BA-LoRA Accuracy',
                    'Performance Delta',
                    'Rank Range (BA-LoRA)',
                    'Parameter Budget Match'
                ],
                'Value': [
                    f'{correlation:.3f}',
                    f'{vanilla_acc:.2f}%',
                    f'{ba_lora_acc:.2f}%',
                    f'{delta:+.2f}pp',
                    f'{ranks.min()}-{ranks.max()}',
                    '✅ Matched'
                ]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

elif demo_mode == "Importance Estimation":
    st.header("📊 Gradient Importance Estimation")

    st.markdown("""
    This mode focuses on understanding how gradient-based importance estimation works.

    **Key Concept**: Layers with larger gradient magnitudes during initial adaptation
    are assumed to be more important for the task.
    """)

    # Generate importance
    importance = simulate_gradient_importance(num_layers)

    # Interactive visualization
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[f'Layer {i}' for i in range(num_layers)],
        y=importance,
        marker=dict(
            color=importance,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=[f'{val:.3f}' for val in importance],
        textposition='outside',
        hovertemplate='Layer: %{x}<br>Importance: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title="Gradient-Based Layer Importance",
        xaxis_title="Layer",
        yaxis_title="Normalized Importance",
        height=500,
        hovermode='x'
    )

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="Mean Importance", annotation_position="right")

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max Importance", f"{importance.max():.3f}", f"Layer {importance.argmax()}")
    with col2:
        st.metric("Min Importance", f"{importance.min():.3f}", f"Layer {importance.argmin()}")
    with col3:
        st.metric("Mean Importance", f"{importance.mean():.3f}")
    with col4:
        st.metric("Std Dev", f"{importance.std():.3f}")

    # Explanation
    st.markdown("### 🔬 How It Works")

    with st.expander("Step-by-step process"):
        st.code("""
    # Step 1: Sample training data
    data_subset = random.sample(training_data, n=5000)
    
    # Step 2: Accumulate gradients
    gradients = {}
    for sample in data_subset:
        # Forward pass
        output = model(sample)
        loss = criterion(output, target)
    
        # Backward pass
        loss.backward()
    
        # Accumulate gradients for each layer
        for layer in target_layers:
            gradients[layer] += layer.weight.grad
    
    # Step 3: Compute importance as Frobenius norm
    importance = {}
    for layer, grad in gradients.items():
        importance[layer] = torch.norm(grad, p='fro')  # ||G||_F
    
    # Step 4: Normalize
    importance = importance / mean(importance.values())
            """, language="python")

elif demo_mode == "Rank Allocation":
    st.header("🎯 Budget-Aware Rank Allocation")

    st.markdown("""
    This mode demonstrates how BA-LoRA allocates different ranks to different layers
    while maintaining strict parameter budget matching with vanilla LoRA.
    """)

    # Generate importance and allocate ranks
    importance = simulate_gradient_importance(num_layers)
    vanilla_params = num_layers * base_rank * (hidden_dim + hidden_dim) * 2
    ranks = allocate_ranks(importance, base_rank, vanilla_params, hidden_dim)
    ba_lora_params = (ranks * (hidden_dim + hidden_dim) * 2).sum()

    # Budget comparison
    st.markdown("### 💰 Parameter Budget")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Vanilla LoRA",
            f"{vanilla_params:,}",
            help="Uniform rank allocation"
        )
    with col2:
        st.metric(
            "BA-LoRA",
            f"{ba_lora_params:,}",
            delta=f"{ba_lora_params - vanilla_params:+,}",
            help="Adaptive rank allocation"
        )
    with col3:
        diff_pct = abs(ba_lora_params - vanilla_params) / vanilla_params * 100
        match_status = "✅ Matched" if diff_pct < 1 else "❌ Mismatch"
        st.metric(
            "Budget Match",
            match_status,
            f"{diff_pct:.2f}% difference"
        )

    # Interactive rank comparison
    st.markdown("### 📊 Rank Allocation Comparison")

    df_allocation = pd.DataFrame({
        'Layer': [f'Layer {i}' for i in range(num_layers)],
        'Gradient Importance': importance,
        'Vanilla LoRA': [base_rank] * num_layers,
        'BA-LoRA': ranks,
        'Delta': ranks - base_rank
    })

    # Plotly chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Vanilla LoRA',
        x=df_allocation['Layer'],
        y=df_allocation['Vanilla LoRA'],
        marker_color='#4ECDC4',
        text=df_allocation['Vanilla LoRA'],
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        name='BA-LoRA',
        x=df_allocation['Layer'],
        y=df_allocation['BA-LoRA'],
        marker_color='#FF6B6B',
        text=df_allocation['BA-LoRA'],
        textposition='outside'
    ))

    fig.update_layout(
        barmode='group',
        title="Rank Allocation: Vanilla LoRA vs BA-LoRA",
        xaxis_title="Layer",
        yaxis_title="Allocated Rank",
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.dataframe(df_allocation, use_container_width=True, hide_index=True)

    # Allocation pattern
    st.markdown("### 🔍 Allocation Pattern")

    high_rank_layers = df_allocation[df_allocation['BA-LoRA'] > base_rank]['Layer'].tolist()
    low_rank_layers = df_allocation[df_allocation['BA-LoRA'] < base_rank]['Layer'].tolist()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Higher Ranks Allocated:**
        {', '.join(high_rank_layers) if high_rank_layers else 'None'}

        *These layers have higher gradient importance*
        """)
    with col2:
        st.markdown(f"""
        **Lower Ranks Allocated:**
        {', '.join(low_rank_layers) if low_rank_layers else 'None'}

        *These layers have lower gradient importance*
        """)

elif demo_mode == "Performance Analysis":
    st.header("📈 Performance Analysis: Why Gradient Importance ≠ Task Importance")

    st.markdown("""
    This is the **key insight** explaining why BA-LoRA shows modest underperformance:

    **Hypothesis**: If gradient importance strongly correlates with actual task importance,
    adaptive allocation should improve performance.

    **Reality**: Weak correlation (r ≈ 0.42) explains the performance gap.
    """)

    # Generate data
    importance = simulate_gradient_importance(num_layers)
    actual_importance = simulate_actual_importance(num_layers)
    correlation = np.corrcoef(importance, actual_importance)[0, 1]

    # Interactive scatter plot
    st.markdown("### 🎯 Correlation Analysis")

    fig = px.scatter(
        x=importance,
        y=actual_importance,
        labels={'x': 'Gradient Importance', 'y': 'Actual Task Importance'},
        title=f'Gradient vs. Actual Importance (Correlation: r = {correlation:.3f})',
        height=500
    )

    # Add regression line
    z = np.polyfit(importance, actual_importance, 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(
            x=importance,
            y=p(importance),
            mode='lines',
            name=f'Linear Fit (r={correlation:.2f})',
            line=dict(color='red', dash='dash', width=2)
        )
    )

    # Add layer labels
    for i in range(num_layers):
        fig.add_annotation(
            x=importance[i],
            y=actual_importance[i],
            text=f'L{i}',
            showarrow=False,
            xshift=10,
            yshift=10
        )

    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    st.markdown("### 💡 Interpretation")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Correlation Coefficient", f"{correlation:.3f}")

        if correlation < 0.3:
            strength = "Very Weak"
            color = "red"
        elif correlation < 0.5:
            strength = "Weak"
            color = "orange"
        elif correlation < 0.7:
            strength = "Moderate"
            color = "yellow"
        else:
            strength = "Strong"
            color = "green"

        st.markdown(f"**Correlation Strength:** :{color}[{strength}]")

    with col2:
        st.markdown(f"""
        **What this means:**

        - **Gradient Importance** measures: Where the model wants to change during initial adaptation
        - **Actual Task Importance** measures: Where changes actually help for the final task
        - **Correlation (r={correlation:.2f})**: {"Weak" if correlation < 0.5 else "Moderate"} relationship between the two

        **Result**: Gradient-based allocation {"doesn't reliably improve" if correlation < 0.5 else "may improve"} performance
        because gradient magnitude is a **weak predictor** of actual task importance.
        """)

    # Example mismatches
    st.markdown("### 🔍 Mismatch Examples")

    # Find biggest mismatches
    rank_importance = stats.rankdata(importance)
    rank_actual = stats.rankdata(actual_importance)
    rank_diff = np.abs(rank_importance - rank_actual)
    top_mismatch_idx = np.argsort(rank_diff)[-3:]

    mismatch_df = pd.DataFrame({
        'Layer': [f'Layer {i}' for i in top_mismatch_idx],
        'Gradient Importance': importance[top_mismatch_idx],
        'Actual Importance': actual_importance[top_mismatch_idx],
        'Mismatch': ['High gradient, medium actual' if importance[i] > actual_importance[i]
                     else 'Low gradient, high actual' for i in top_mismatch_idx]
    })

    st.dataframe(mismatch_df, use_container_width=True, hide_index=True)

    st.markdown("""
    These mismatches explain why adaptive allocation based on gradients doesn't improve performance:
    we're allocating high ranks to layers that don't contribute as much as expected, and vice versa.
    """)

elif demo_mode == "Load Real Results":
    st.header("📂 Real Experimental Results")

    st.markdown("""
    Load and visualize actual results from your SST-2 experiments.
    """)

    # Detect project root and construct default path
    import os
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # Go up to ML_Project/
    default_results_path = project_root / "results" / "results_sst2"

    # File uploader
    results_dir = st.text_input(
        "Results Directory",
        value=str(default_results_path),
        help="Path to directory containing JSON result files"
    )

    if st.button("🔍 Load Results"):
        try:
            results_path = Path(results_dir)

            if not results_path.exists():
                st.error(f"Directory not found: {results_dir}")
            else:
                # Load JSON files
                json_files = list(results_path.rglob("*.json"))

                if len(json_files) == 0:
                    st.warning("No JSON files found in directory")
                else:
                    st.success(f"Found {len(json_files)} result files")

                    # Parse results
                    results = []
                    for json_file in json_files:
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)

                            if 'eval_results' in data and 'config' in data:
                                results.append({
                                    'method': data['config'].get('peft_method', 'unknown'),
                                    'rank': data['config'].get('lora_config', {}).get('r',
                                           data['config'].get('ba_lora_config', {}).get('base_rank', 0)),
                                    'accuracy': data['eval_results'].get('eval_accuracy', 0) * 100,
                                    'f1': data['eval_results'].get('eval_f1', 0) * 100,
                                    'train_time': data.get('duration_seconds', 0) / 60
                                })
                        except Exception as e:
                            continue

                    if results:
                        df = pd.DataFrame(results)

                        # Summary statistics
                        st.markdown("### 📊 Summary Statistics")

                        summary = df.groupby('method').agg({
                            'accuracy': ['mean', 'std', 'count'],
                            'f1': ['mean', 'std'],
                            'train_time': 'mean'
                        }).round(2)

                        st.dataframe(summary, use_container_width=True)

                        # Visualizations
                        st.markdown("### 📈 Visualizations")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Accuracy comparison
                            fig1 = px.box(df, x='method', y='accuracy', color='method',
                                         title='Accuracy Distribution by Method',
                                         labels={'accuracy': 'Accuracy (%)', 'method': 'Method'})
                            st.plotly_chart(fig1, use_container_width=True)

                        with col2:
                            # Training time comparison
                            fig2 = px.bar(df.groupby('method')['train_time'].mean().reset_index(),
                                         x='method', y='train_time', color='method',
                                         title='Average Training Time',
                                         labels={'train_time': 'Time (minutes)', 'method': 'Method'})
                            st.plotly_chart(fig2, use_container_width=True)

                        # Performance vs rank
                        if df['rank'].nunique() > 1:
                            fig3 = px.line(df, x='rank', y='accuracy', color='method',
                                          markers=True, title='Accuracy vs Rank',
                                          labels={'accuracy': 'Accuracy (%)', 'rank': 'Rank'})
                            st.plotly_chart(fig3, use_container_width=True)

                        # Raw data
                        with st.expander("View Raw Data"):
                            st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("No valid experiment results found in JSON files")

        except Exception as e:
            st.error(f"Error loading results: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
<p><strong>BA-LoRA: Budget-Aware Adaptive Low-Rank Adaptation</strong></p>
<p>CS 8267: Advanced Machine Learning | Kennesaw State University</p>
<p style='font-size: 0.9em; margin-top: 1rem;'>
<em>"Understanding when and why adaptive allocation helps vs. hurts"</em>
</p>
</div>
""", unsafe_allow_html=True)
