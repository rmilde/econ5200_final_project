import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="401(k) Eligibility & Savings", layout="wide")
st.title("Does 401(k) Eligibility Cause Higher Savings?")
st.markdown("**ECON 5200 Consulting Report | Rebecca Milde | Spring 2026**")
st.markdown("---")

# --- Pre-computed results from DML analysis ---
NAIVE_ESTIMATE = 27372
NAIVE_CI_LOW = 24578
NAIVE_CI_HIGH = 30165

CAUSAL_ESTIMATE = 14441     # DML with Gradient Boosting
CAUSAL_SE = 1567
CAUSAL_CI_LOW = 11369
CAUSAL_CI_HIGH = 17513

ROBUST_ESTIMATE = 13838     # DML with Random Forest
ROBUST_CI_LOW = 10625
ROBUST_CI_HIGH = 17051

# --- Sidebar: What-If Controls ---
st.sidebar.header("What-If Scenarios")
st.sidebar.markdown("Adjust the sliders to explore how the causal estimate changes under different assumptions.")

treatment_multiplier = st.sidebar.slider(
    "Treatment intensity multiplier",
    min_value=0.5, max_value=3.0, value=1.0, step=0.1,
    help="Simulates expanding or contracting the effect of 401(k) eligibility"
)

n_workers = st.sidebar.slider(
    "Number of newly eligible workers",
    min_value=100, max_value=100000, value=10000, step=100,
    help="How many workers a firm or policy would newly enroll"
)

policy_scale = st.sidebar.selectbox(
    "Policy scale",
    ["Single firm (~500 workers)", "Mid-size employer (~5,000 workers)",
     "Large employer (~50,000 workers)", "National policy (~1M workers)"],
    index=1
)

scale_map = {
    "Single firm (~500 workers)": 500,
    "Mid-size employer (~5,000 workers)": 5000,
    "Large employer (~50,000 workers)": 50000,
    "National policy (~1M workers)": 1000000
}
policy_n = scale_map[policy_scale]

# --- Compute What-If Estimates ---
adjusted_ate = CAUSAL_ESTIMATE * treatment_multiplier
adjusted_se = CAUSAL_SE * treatment_multiplier
ci_lower = adjusted_ate - 1.96 * adjusted_se
ci_upper = adjusted_ate + 1.96 * adjusted_se
total_assets = adjusted_ate * policy_n

# --- Section 1: Key Findings ---
st.header("Key Findings")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Naive OLS Estimate", f"${NAIVE_ESTIMATE:,}",
            help="Simple comparison, biased upward by income-driven selection")
col2.metric("Causal Estimate (DML)", f"${CAUSAL_ESTIMATE:,}",
            delta=f"-${NAIVE_ESTIMATE - CAUSAL_ESTIMATE:,} vs naive",
            delta_color="off",
            help="Double Machine Learning with Gradient Boosting nuisance models")
col3.metric("Robustness Check (RF)", f"${ROBUST_ESTIMATE:,}",
            delta=f"${abs(CAUSAL_ESTIMATE - ROBUST_ESTIMATE):,} difference",
            delta_color="off",
            help="DML with Random Forest nuisance models — confirms stability")
col4.metric("Confounding Bias Removed", f"${NAIVE_ESTIMATE - CAUSAL_ESTIMATE:,}",
            help="Portion of naive estimate attributable to income-driven selection bias")

st.markdown(f"""
> **Interpretation:** 401(k) eligibility causally increases net financial assets by an estimated
> **${CAUSAL_ESTIMATE:,}** (95% CI: [${CAUSAL_CI_LOW:,}, ${CAUSAL_CI_HIGH:,}]). The naive OLS estimate
> of ${NAIVE_ESTIMATE:,} overstates the effect by ${NAIVE_ESTIMATE - CAUSAL_ESTIMATE:,} due to
> income-driven selection bias — eligible workers earn ~$16,500 more on average and would save
> more regardless of plan access. DML removes this confounding.
""")

# --- Section 2: Naive vs. Causal Plot ---
st.header("Naive vs. Causal Estimate")

fig_compare = go.Figure()

estimates_labels = ['Naive OLS', 'Causal (DML, GB)', 'Robustness (DML, RF)']
estimates_vals = [NAIVE_ESTIMATE, CAUSAL_ESTIMATE, ROBUST_ESTIMATE]
ci_lows = [NAIVE_CI_LOW, CAUSAL_CI_LOW, ROBUST_CI_LOW]
ci_highs = [NAIVE_CI_HIGH, CAUSAL_CI_HIGH, ROBUST_CI_HIGH]
colors = ['#e53935', '#1a237e', '#1565c0']

for label, val, low, high, color in zip(estimates_labels, estimates_vals, ci_lows, ci_highs, colors):
    fig_compare.add_trace(go.Scatter(
        x=[label], y=[val],
        error_y=dict(type='data', symmetric=False,
                     array=[high - val], arrayminus=[val - low]),
        mode='markers',
        marker=dict(size=14, color=color),
        name=label
    ))
    fig_compare.add_annotation(
        x=label, y=val,
        text=f"${val:,}",
        showarrow=False,
        xshift=55, font=dict(size=12, color=color)
    )

fig_compare.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig_compare.update_layout(
    title="Effect of 401(k) Eligibility on Net Financial Assets",
    yaxis_title="Estimated Effect ($)",
    template="plotly_white",
    showlegend=False,
    height=450
)
st.plotly_chart(fig_compare, use_container_width=True)

# --- Section 3: What-If Scenario ---
st.header("What-If: Adjust Treatment Intensity")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Adjusted Causal Effect", f"${adjusted_ate:,.0f}")
col_b.metric("95% CI Lower", f"${ci_lower:,.0f}")
col_c.metric("95% CI Upper", f"${ci_upper:,.0f}")

st.markdown(f"""
> If treatment intensity is multiplied by **{treatment_multiplier:.1f}x**, the estimated effect
> changes to **${adjusted_ate:,.0f}** (95% CI: [${ci_lower:,.0f}, ${ci_upper:,.0f}]).
""")

# Uncertainty band chart
multipliers = np.arange(0.5, 3.1, 0.1)
ates = CAUSAL_ESTIMATE * multipliers
ses = CAUSAL_SE * multipliers

fig_whatif = go.Figure()
fig_whatif.add_trace(go.Scatter(
    x=multipliers, y=ates + 1.96 * ses,
    mode="lines", line=dict(width=0), showlegend=False
))
fig_whatif.add_trace(go.Scatter(
    x=multipliers, y=ates - 1.96 * ses,
    mode="lines", line=dict(width=0), fill="tonexty",
    fillcolor="rgba(26,35,126,0.15)", name="95% CI"
))
fig_whatif.add_trace(go.Scatter(
    x=multipliers, y=ates,
    mode="lines", line=dict(color="#1a237e", width=2), name="Estimated Effect"
))
fig_whatif.add_vline(
    x=treatment_multiplier, line_dash="dash", line_color="red",
    annotation_text=f"Current: {treatment_multiplier:.1f}x  (${adjusted_ate:,.0f})",
    annotation_position="top right"
)
fig_whatif.update_layout(
    title="What-If: Estimated Effect vs. Treatment Intensity Multiplier",
    xaxis_title="Treatment Intensity Multiplier",
    yaxis_title="Estimated Causal Effect ($)",
    template="plotly_white",
    height=400
)
st.plotly_chart(fig_whatif, use_container_width=True)

# --- Section 4: Policy Impact ---
st.header("Policy Impact Calculator")

st.markdown(f"""
If **{policy_scale}** gained 401(k) eligibility with the estimated causal effect of
**${adjusted_ate:,.0f}** per worker, the aggregate increase in financial assets would be:
""")

col_p1, col_p2 = st.columns(2)
col_p1.metric("Aggregate Asset Increase", f"${total_assets:,.0f}",
              help=f"Causal estimate × {policy_n:,} workers")
col_p2.metric("Per-Worker Effect", f"${adjusted_ate:,.0f}",
              help="Average treatment effect at current multiplier")

# --- Section 5: Counterfactual ---
st.header("Counterfactual: What if Eligibility Doubled?")

counterfactual_ate = CAUSAL_ESTIMATE * 2.0
counterfactual_ci_low = counterfactual_ate - 1.96 * CAUSAL_SE * 2.0
counterfactual_ci_high = counterfactual_ate + 1.96 * CAUSAL_SE * 2.0

st.write(
    f"If the effect of eligibility doubled (e.g., due to auto-enrollment or employer matching), "
    f"the estimated effect would be **${counterfactual_ate:,.0f}** "
    f"(95% CI: [${counterfactual_ci_low:,.0f}, ${counterfactual_ci_high:,.0f}])."
)

# --- Section 6: Identification Summary ---
st.header("Identification Strategy")
st.markdown("""
| Component | Detail |
|-----------|--------|
| **Method** | Double Machine Learning (DML) — Partially Linear Regression |
| **Nuisance models** | Gradient Boosting (primary), Random Forest (robustness) |
| **Cross-fitting** | 5-fold |
| **Key assumption** | Conditional independence: after controlling for income, age, family size, education, and IRA participation, 401(k) eligibility is as good as random |
| **Dataset** | Chernozhukov et al. (2018) fetch_401K — N = 9,915 |
| **Controls** | inc, age, fsize, educ, pira, incsq |
| **Most serious threat** | Unobserved employer characteristics (firm quality) — may bias estimate upward |
""")

st.markdown("---")
st.caption("ECON 5200 Final Project | Spring 2026 | Rebecca Milde | Data: Chernozhukov et al. (2018)")
