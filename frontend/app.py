"""
PolicyPilot – Streamlit Triage Dashboard
------------------------------------------
Multi-page Streamlit application for AFCA complaint triage.

Pages:
  1. 🔍 Triage      – Single complaint triage with full results
  2. 📦 Batch       – CSV upload for bulk triage
  3. 📊 Analytics   – Complaint data analytics and insights
  4. 🧪 Eval        – Prompt version comparison dashboard
  5. 💼 Executive   – ROI calculator and business impact summary
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.chain import TriageChain
from app.core.entities import EntityExtractor
from app.core.risk import RiskScorer
from app.core.config import get_settings
from app.core.prompts.registry import list_versions

settings = get_settings()

# ── Colour palette ────────────────────────────────────────────────────
NAVY   = "#1e3a5f"
TEAL   = "#0c7b93"
SKY    = "#00b4d8"
SLATE  = "#334155"
LIGHT  = "#f0f4f8"
WHITE  = "#ffffff"

RISK_COLOURS = {
    "CRITICAL": "#dc2626",
    "HIGH":     "#ea580c",
    "MEDIUM":   "#d97706",
    "LOW":      "#16a34a",
}

PRODUCT_COLOURS = [
    "#1e3a5f", "#0c7b93", "#00b4d8", "#0284c7",
    "#7c3aed", "#db2777", "#059669", "#d97706",
]

# ── Metric row helper ────────────────────────────────────────────────
def metric_row(items: list[tuple]) -> None:
    """Render a row of st.metric cards. items = [(label, value, delta), ...]"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label = item[0]
        value = item[1]
        delta = item[2] if len(item) > 2 else None
        col.metric(label, value, delta)

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="PolicyPilot – AFCA Complaint Triage",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #0c7b93 60%, #00b4d8 100%);
    padding: 1.6rem 2rem 1.4rem;
    border-radius: 14px;
    margin-bottom: 1.6rem;
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.hero h1 { margin:0; font-size:1.75rem; font-weight:700; letter-spacing:-0.3px; }
.hero p  { margin:0.4rem 0 0; opacity:0.85; font-size:0.9rem; }

/* ── KPI cards ── */
.kpi-row { display:flex; gap:1rem; margin-bottom:1.4rem; }
.kpi {
    flex:1; background:white; border-radius:12px;
    padding:1.1rem 1.3rem; border-top:3px solid #0c7b93;
    box-shadow:0 2px 10px rgba(0,0,0,0.06);
}
.kpi .label { font-size:0.72rem; color:#64748b; text-transform:uppercase;
              letter-spacing:0.6px; font-weight:600; }
.kpi .value { font-size:1.8rem; font-weight:700; color:#1e3a5f; margin-top:2px; }
.kpi .delta { font-size:0.78rem; color:#16a34a; margin-top:2px; font-weight:500; }

/* ── Risk badges ── */
.badge {
    display:inline-block; padding:4px 14px; border-radius:20px;
    font-weight:600; font-size:0.82rem; letter-spacing:0.3px;
}
.badge-CRITICAL { background:#fef2f2; color:#dc2626; border:1px solid #fca5a5; }
.badge-HIGH     { background:#fff7ed; color:#ea580c; border:1px solid #fdba74; }
.badge-MEDIUM   { background:#fffbeb; color:#d97706; border:1px solid #fcd34d; }
.badge-LOW      { background:#f0fdf4; color:#16a34a; border:1px solid #86efac; }

/* ── Result card ── */
.result-card {
    background:white; border:1px solid #e2e8f0;
    border-radius:12px; padding:1.4rem 1.6rem;
    margin-bottom:1rem; box-shadow:0 2px 8px rgba(0,0,0,0.05);
}

/* ── Entity pill ── */
.pill {
    display:inline-block; background:#eff6ff; color:#1d4ed8;
    border:1px solid #bfdbfe; padding:3px 10px;
    border-radius:12px; font-size:0.78rem; margin:2px; font-weight:500;
}
.pill-green  { background:#f0fdf4; color:#166534; border-color:#bbf7d0; }
.pill-purple { background:#faf5ff; color:#6b21a8; border-color:#e9d5ff; }
.pill-orange { background:#fff7ed; color:#9a3412; border-color:#fed7aa; }

/* ── Policy ref tag ── */
.ref-tag {
    display:inline-block; background:#eff6ff; color:#1e40af;
    padding:3px 11px; border-radius:14px; font-size:0.79rem;
    margin:2px; font-weight:600; border:1px solid #bfdbfe;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2440 0%, #1e3a5f 50%, #0c4a6e 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox > div { border-color: rgba(255,255,255,0.2); }

/* ── Section heading ── */
.section-title {
    font-size:1rem; font-weight:700; color:#1e3a5f;
    margin:1.2rem 0 0.6rem; padding-bottom:0.4rem;
    border-bottom:2px solid #e2e8f0;
}

/* Hide branding */
#MainMenu { visibility:hidden; }
footer    { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────

def plotly_defaults(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter", size=12, color="#334155"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0", linewidth=1)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0")
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='padding:0.4rem 0 0.2rem'>"
    "<span style='font-size:1.5rem'>🛡️</span>"
    "<span style='font-size:1.1rem; font-weight:700; margin-left:8px;'>PolicyPilot</span>"
    "</div>"
    "<div style='font-size:0.8rem; opacity:0.65; margin-bottom:1rem'>AFCA Complaint Triage</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🔍 Triage", "📦 Batch Upload", "📊 Analytics", "🧪 Eval Dashboard", "💼 Executive Summary"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Settings**")
prompt_version = st.sidebar.selectbox(
    "Prompt Version",
    ["v3", "v2", "v1"],
    help="v3 = Few-shot + CoT (recommended)",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.72rem; opacity:0.5; line-height:1.6'>"
    "PolicyPilot v1.0 · Melbourne<br>AI-powered compliance copilot<br>"
    "ASIC RG 271 · September 2021"
    "</div>",
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 1: TRIAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if page == "🔍 Triage":
    st.markdown("""
    <div class="hero">
        <h1>🔍 Complaint Triage</h1>
        <p>Paste a complaint to get AI-powered classification, risk scoring, and a draft IDR response</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        complaint_text = st.text_area(
            "Complaint Text",
            placeholder=(
                "Paste the customer complaint here…\n\n"
                "Example: I noticed three transactions on my Visa credit card "
                "totalling $4,280 that I did not authorise."
            ),
            height=190,
        )
    with col2:
        complaint_id = st.text_input("Complaint ID", placeholder="AFCA-2024-XXXXX")
        product = st.selectbox("Product", [
            "(Auto-detect)", "Credit Card", "Home Loan", "General Insurance",
            "Personal Loan", "Superannuation", "Banking", "Life Insurance",
        ])
        source = st.selectbox("Source", ["email", "phone", "web", "letter"])

    run = st.button("🚀 Triage Complaint", type="primary", use_container_width=True)

    if run:
        if not complaint_text or len(complaint_text.strip()) < 10:
            st.error("Please enter a complaint text (minimum 10 characters).")
        else:
            with st.spinner("Analysing complaint…"):
                chain = TriageChain(prompt_version=prompt_version)
                result = chain.run(
                    complaint_text=complaint_text,
                    complaint_id=complaint_id or None,
                    product=None if product == "(Auto-detect)" else product,
                    source=source,
                )

            st.markdown("---")

            # ── Top KPI strip ─────────────────────────────────────
            risk_col = RISK_COLOURS.get(result.risk_assessment.risk_level, "#64748b")
            metric_row([
                ("Category",    result.category),
                ("Risk Score",  f"{result.risk_assessment.overall_score:.2f}"),
                ("Risk Level",  result.risk_assessment.risk_level),
                ("Priority",    f"P{result.risk_assessment.recommended_priority}"),
                ("IDR Deadline",f"{result.risk_assessment.idr_deadline_days} days"),
            ])

            # ── Risk gauge ────────────────────────────────────────
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result.risk_assessment.overall_score,
                domain={"x": [0, 1], "y": [0, 1]},
                number={"font": {"size": 36, "color": NAVY}, "suffix": ""},
                gauge={
                    "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#94a3b8"},
                    "bar": {"color": risk_col, "thickness": 0.25},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 0.3], "color": "#dcfce7"},
                        {"range": [0.3, 0.6], "color": "#fef9c3"},
                        {"range": [0.6, 0.8], "color": "#ffedd5"},
                        {"range": [0.8, 1.0], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": risk_col, "width": 3},
                        "thickness": 0.75,
                        "value": result.risk_assessment.overall_score,
                    },
                },
            ))
            gauge.update_layout(
                height=200, margin=dict(l=20, r=20, t=20, b=0),
                paper_bgcolor="white", font=dict(family="Inter"),
            )

            g_col, t_col = st.columns([1, 2])
            with g_col:
                st.plotly_chart(gauge, use_container_width=True)
            with t_col:
                st.markdown('<div class="section-title">Risk Factors</div>', unsafe_allow_html=True)
                for f in result.risk_assessment.factors:
                    st.markdown(f"⚠️ {f}")

            # ── Detail tabs ───────────────────────────────────────
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📝 Draft Response", "🔎 Reasoning",
                "📋 Entities", "📚 Policy References", "📈 Metadata",
            ])

            with tab1:
                st.markdown('<div class="section-title">Draft IDR Response</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="result-card" style="border-left:4px solid {TEAL};">'
                    f'{result.draft_response}</div>',
                    unsafe_allow_html=True,
                )

            with tab2:
                st.markdown('<div class="section-title">Triage Reasoning</div>', unsafe_allow_html=True)
                st.markdown(result.reasoning)

            with tab3:
                st.markdown('<div class="section-title">Extracted Entities</div>', unsafe_allow_html=True)
                entities = result.entities.to_dict()
                pill_classes = {
                    "amounts": "pill-green",
                    "dates": "pill-purple",
                    "products": "pill-orange",
                }
                for key, values in entities.items():
                    if values:
                        cls = pill_classes.get(key, "pill")
                        pills = " ".join(
                            f'<span class="pill {cls}">{v}</span>' for v in values
                        )
                        st.markdown(
                            f"**{key.replace('_', ' ').title()}** &nbsp; {pills}",
                            unsafe_allow_html=True,
                        )

            with tab4:
                st.markdown('<div class="section-title">Relevant Policy References</div>', unsafe_allow_html=True)
                if result.policy_refs:
                    refs_html = " ".join(
                        f'<span class="ref-tag">{r}</span>' for r in result.policy_refs
                    )
                    st.markdown(refs_html, unsafe_allow_html=True)
                    st.markdown("")
                if result.retrieved_docs:
                    st.markdown('<div class="section-title">Retrieved Regulatory Context</div>', unsafe_allow_html=True)
                    for doc in result.retrieved_docs[:5]:
                        with st.expander(
                            f"📄 {doc.section_id} — {doc.title[:60]} (score: {doc.score:.3f})"
                        ):
                            st.markdown(f"**Source:** {doc.guide} &nbsp;|&nbsp; **Category:** {doc.category}")
                            st.markdown(doc.text)

            with tab5:
                st.markdown('<div class="section-title">Pipeline Metadata</div>', unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f"**Prompt Version:** `{result.prompt_version}`")
                    st.markdown(f"**Model:** `{result.model}`")
                    st.markdown(f"**Latency:** `{result.latency_ms:.1f} ms`")
                with m2:
                    st.markdown(f"**Tokens Used:** `{result.tokens_used}`")
                    st.markdown(f"**Cost:** `${result.cost_usd:.4f}`")
                    st.markdown(f"**Sub-category:** `{result.sub_category}`")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 2: BATCH UPLOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "📦 Batch Upload":
    st.markdown("""
    <div class="hero">
        <h1>📦 Batch Triage</h1>
        <p>Upload a CSV with multiple complaints for bulk triage processing</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV (must have a 'complaint_text' column)",
        type=["csv"],
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded:** {len(df)} complaints")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Run Batch Triage", type="primary"):
            chain = TriageChain(prompt_version=prompt_version)
            results = []
            progress = st.progress(0)
            status = st.empty()

            for i, row in df.iterrows():
                text = str(row.get("complaint_text", row.iloc[0]))
                cid  = str(row.get("complaint_id", f"BATCH-{i+1:03d}"))
                status.text(f"Processing {cid}… ({i+1}/{len(df)})")
                result = chain.run(complaint_text=text, complaint_id=cid)
                results.append({
                    "complaint_id":  cid,
                    "category":      result.category,
                    "sub_category":  result.sub_category,
                    "risk_score":    round(result.risk_assessment.overall_score, 3),
                    "risk_level":    result.risk_assessment.risk_level,
                    "priority":      f"P{result.risk_assessment.recommended_priority}",
                    "idr_deadline":  result.risk_assessment.idr_deadline_days,
                    "policy_refs":   ", ".join(result.policy_refs),
                    "latency_ms":    round(result.latency_ms, 1),
                })
                progress.progress((i + 1) / len(df))

            status.text("✅ Batch triage complete!")
            results_df = pd.DataFrame(results)

            metric_row([
                ("Total Processed", str(len(results_df))),
                ("Critical / High", str(len(results_df[results_df["risk_level"].isin(["CRITICAL", "HIGH"])]))),
                ("Avg Risk Score",  f"{results_df['risk_score'].mean():.2f}"),
                ("Avg Latency",     f"{results_df['latency_ms'].mean():.0f} ms"),
            ])

            # Risk distribution donut
            risk_counts = results_df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            fig_risk = px.pie(
                risk_counts, values="Count", names="Risk Level", hole=0.55,
                color="Risk Level",
                color_discrete_map=RISK_COLOURS,
                title="Risk Level Distribution",
            )
            plotly_defaults(fig_risk, height=280)
            st.plotly_chart(fig_risk, use_container_width=True)

            st.dataframe(results_df, use_container_width=True, height=380)
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV", csv,
                "policypilot_batch_results.csv", "text/csv",
            )
    else:
        st.info("💡 Upload a CSV file with a `complaint_text` column to get started.")
        if st.button("Load Sample AFCA Data"):
            try:
                sample_df = pd.read_csv(settings.data_raw / "afca_complaints_sample.csv")
                st.dataframe(
                    sample_df[["complaint_id", "product", "issue", "complaint_text"]].head(10),
                    use_container_width=True,
                )
            except FileNotFoundError:
                st.error("Sample data not found. Run `python -m app.ingest.load_afca --dry-run` first.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 3: ANALYTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "📊 Analytics":
    st.markdown("""
    <div class="hero">
        <h1>📊 Complaint Analytics</h1>
        <p>AFCA Datacube — 100 complaints across 7 financial product categories</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        df = pd.read_csv(settings.data_raw / "afca_complaints_sample.csv")
    except FileNotFoundError:
        st.error("Data not found. Run `python -m app.ingest.load_afca --dry-run` first.")
        st.stop()

    total_comp = df["compensation_aud"].sum()
    avg_res    = df["resolution_days"].mean()
    high_comp  = df[df["compensation_aud"] > 0]["compensation_aud"].mean()

    metric_row([
        ("Total Complaints",    f"{len(df):,}"),
        ("Product Categories",  str(df["product"].nunique())),
        ("Avg Resolution",      f"{avg_res:.0f} days"),
        ("Total Compensation",  f"${total_comp:,.0f}"),
        ("Avg Payout (cases)",  f"${high_comp:,.0f}"),
    ])

    # ── Row 1: Product donut + State horizontal bar ───────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Complaints by Product</div>', unsafe_allow_html=True)
        prod = df["product"].value_counts().reset_index()
        prod.columns = ["Product", "Complaints"]
        fig_prod = px.pie(
            prod, values="Complaints", names="Product", hole=0.52,
            color_discrete_sequence=PRODUCT_COLOURS,
        )
        fig_prod.update_traces(
            textposition="outside",
            textinfo="percent+label",
            textfont_size=11,
        )
        plotly_defaults(fig_prod, height=320)
        fig_prod.update_layout(showlegend=False)
        st.plotly_chart(fig_prod, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Complaints by State</div>', unsafe_allow_html=True)
        state = df["state"].value_counts().reset_index()
        state.columns = ["State", "Complaints"]
        fig_state = px.bar(
            state, x="Complaints", y="State", orientation="h",
            color="Complaints",
            color_continuous_scale=[[0, SKY], [0.5, TEAL], [1, NAVY]],
            text="Complaints",
        )
        fig_state.update_traces(textposition="outside", textfont_size=11)
        fig_state.update_layout(coloraxis_showscale=False, yaxis_categoryorder="total ascending")
        plotly_defaults(fig_state, height=320)
        st.plotly_chart(fig_state, use_container_width=True)

    # ── Row 2: Resolution outcomes grouped + Resolution days histogram ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Resolution Outcomes</div>', unsafe_allow_html=True)
        res_prod = (
            df.groupby(["product", "resolution"])
            .size().reset_index(name="count")
        )
        fig_res = px.bar(
            res_prod, x="product", y="count", color="resolution",
            barmode="stack",
            color_discrete_sequence=[NAVY, TEAL, SKY, "#0284c7", "#7c3aed", "#db2777"],
        )
        fig_res.update_layout(xaxis_tickangle=-30)
        plotly_defaults(fig_res, height=320)
        st.plotly_chart(fig_res, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title">Resolution Time Distribution</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            df, x="resolution_days", nbins=20,
            color_discrete_sequence=[TEAL],
            labels={"resolution_days": "Days to Resolve"},
        )
        fig_hist.update_traces(marker_line_color=NAVY, marker_line_width=0.6)
        fig_hist.add_vline(
            x=df["resolution_days"].mean(), line_dash="dash",
            line_color=NAVY, annotation_text=f"Mean {avg_res:.0f}d",
            annotation_font_color=NAVY,
        )
        plotly_defaults(fig_hist, height=320)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Row 3: Avg compensation by product (lollipop) ─────────────
    st.markdown('<div class="section-title">Average Compensation by Product (AUD)</div>', unsafe_allow_html=True)
    comp_prod = (
        df[df["compensation_aud"] > 0]
        .groupby("product")["compensation_aud"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )
    comp_prod.columns = ["Product", "Avg Compensation"]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=comp_prod["Avg Compensation"],
        y=comp_prod["Product"],
        orientation="h",
        marker_color=[PRODUCT_COLOURS[i % len(PRODUCT_COLOURS)] for i in range(len(comp_prod))],
        text=[f"${v:,.0f}" for v in comp_prod["Avg Compensation"]],
        textposition="outside",
        textfont=dict(size=11),
    ))
    plotly_defaults(fig_comp, height=280)
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Styled summary table ───────────────────────────────────────
    st.markdown('<div class="section-title">Product-Level Summary</div>', unsafe_allow_html=True)
    summary = df.groupby("product").agg(
        Complaints=("complaint_id", "count"),
        Avg_Compensation=("compensation_aud", "mean"),
        Total_Compensation=("compensation_aud", "sum"),
        Avg_Resolution_Days=("resolution_days", "mean"),
    ).round(0).sort_values("Complaints", ascending=False)
    summary.columns = ["Complaints", "Avg Comp ($)", "Total Comp ($)", "Avg Days"]
    st.dataframe(
        summary.style
            .background_gradient(subset=["Complaints"], cmap="Blues")
            .background_gradient(subset=["Total Comp ($)"], cmap="Greens")
            .format({"Avg Comp ($)": "${:,.0f}", "Total Comp ($)": "${:,.0f}", "Avg Days": "{:.0f}"}),
        use_container_width=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 4: EVAL DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "🧪 Eval Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>🧪 Prompt Evaluation Dashboard</h1>
        <p>Data-driven comparison of v1 (Naive) vs v2 (Chain-of-Thought) vs v3 (Few-shot + CoT)</p>
    </div>
    """, unsafe_allow_html=True)

    eval_data: dict = {}
    for v in ["v1", "v2", "v3"]:
        p = settings.data_processed / f"eval_{v}_results.json"
        if p.exists():
            with open(p) as f:
                eval_data[v] = json.load(f)

    if not eval_data:
        st.warning("No evaluation results found. Run:")
        st.code("python -m app.eval.compare_prompts", language="bash")
        if st.button("🧪 Run Evaluation Now"):
            with st.spinner("Running evaluation…"):
                from app.eval.compare_prompts import compare_prompts
                compare_prompts()
            st.success("✅ Done! Reload to see results.")
            st.rerun()
    else:
        versions = list(eval_data.keys())
        metrics  = {v: eval_data[v]["metrics"] for v in versions}

        # ── KPI strip for best version (v3) ───────────────────────
        best = metrics.get("v3", list(metrics.values())[-1])
        metric_row([
            ("Category Accuracy (v3)", f"{best['category_accuracy']:.0%}"),
            ("Risk Accuracy (v3)",     f"{best['risk_level_accuracy']:.0%}"),
            ("Entity Recall (v3)",     f"{best['mean_entity_recall']:.0%}"),
            ("Avg Latency (v3)",       f"{best['mean_latency_ms']:.0f} ms"),
            ("Cost / Triage (v3)",     f"${best['cost_per_triage_usd']:.4f}"),
        ])

        # ── Grouped bar: accuracy comparison ──────────────────────
        st.markdown('<div class="section-title">Accuracy by Prompt Version</div>', unsafe_allow_html=True)
        acc_df = pd.DataFrame({
            "Version":           versions,
            "Category Accuracy": [metrics[v]["category_accuracy"]      for v in versions],
            "Risk Accuracy":     [metrics[v]["risk_level_accuracy"]     for v in versions],
            "Entity Recall":     [metrics[v]["mean_entity_recall"]      for v in versions],
        })
        fig_acc = px.bar(
            acc_df.melt(id_vars="Version", var_name="Metric", value_name="Score"),
            x="Version", y="Score", color="Metric", barmode="group",
            text_auto=".0%",
            color_discrete_sequence=[NAVY, TEAL, SKY],
        )
        fig_acc.update_traces(textposition="outside", textfont_size=10)
        fig_acc.update_yaxes(tickformat=".0%", range=[0, 1.1])
        plotly_defaults(fig_acc, height=340)
        st.plotly_chart(fig_acc, use_container_width=True)

        # ── Cost vs latency scatter ────────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-title">Cost per Triage</div>', unsafe_allow_html=True)
            cost_df = pd.DataFrame({
                "Version": versions,
                "Cost ($)": [metrics[v]["cost_per_triage_usd"] for v in versions],
            })
            fig_cost = px.bar(
                cost_df, x="Version", y="Cost ($)",
                color="Version", text_auto="$.5f",
                color_discrete_sequence=PRODUCT_COLOURS,
            )
            fig_cost.update_traces(textposition="outside")
            plotly_defaults(fig_cost, height=260)
            st.plotly_chart(fig_cost, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">Avg Latency (ms)</div>', unsafe_allow_html=True)
            lat_df = pd.DataFrame({
                "Version": versions,
                "Latency (ms)": [metrics[v]["mean_latency_ms"] for v in versions],
            })
            fig_lat = px.bar(
                lat_df, x="Version", y="Latency (ms)",
                color="Version", text_auto=".0f",
                color_discrete_sequence=[TEAL, SKY, NAVY],
            )
            fig_lat.update_traces(textposition="outside")
            plotly_defaults(fig_lat, height=260)
            st.plotly_chart(fig_lat, use_container_width=True)

        # ── Comparison table ───────────────────────────────────────
        st.markdown('<div class="section-title">Full Comparison Table</div>', unsafe_allow_html=True)
        table = []
        for v in versions:
            m = metrics[v]
            table.append({
                "Version":           v,
                "Category Accuracy": f"{m['category_accuracy']:.1%}",
                "Risk Accuracy":     f"{m['risk_level_accuracy']:.1%}",
                "Entity Recall":     f"{m['mean_entity_recall']:.1%}",
                "Avg Latency":       f"{m['mean_latency_ms']:.0f} ms",
                "Cost / Request":    f"${m['cost_per_triage_usd']:.4f}",
                "Total Tokens":      m["total_tokens"],
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        # ── Individual results explorer ───────────────────────────
        st.markdown('<div class="section-title">Individual Results Explorer</div>', unsafe_allow_html=True)
        sel = st.selectbox("Select prompt version:", versions)
        if sel in eval_data:
            st.dataframe(
                pd.DataFrame(eval_data[sel]["results"]),
                use_container_width=True, height=360,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 5: EXECUTIVE SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "💼 Executive Summary":
    st.markdown("""
    <div class="hero">
        <h1>💼 Executive Summary</h1>
        <p>Business impact analysis and ROI calculator for AI-powered triage</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Current State (Manual)**")
        daily_complaints   = st.slider("Daily complaints",              50, 500, 200)
        manual_time_min    = st.slider("Manual triage time (min)",       5,  30,  12)
        analyst_hourly     = st.slider("Analyst hourly rate (AUD)",     40, 120,  65)
        working_days       = st.slider("Working days per year",        200, 260, 250)
    with col2:
        st.markdown("**AI-Powered (PolicyPilot)**")
        ai_time_sec        = st.slider("AI triage time (sec)",           5,  60,  18)
        review_time_min    = st.slider("Human review time (min)",        1,  10,   3)
        ai_cost_per_triage = st.slider("AI cost per triage (AUD)",    0.01, 0.20, 0.04, step=0.01)

    # Calculations
    manual_daily_h   = daily_complaints * manual_time_min / 60
    manual_daily_c   = manual_daily_h * analyst_hourly
    manual_yearly    = manual_daily_c * working_days

    ai_daily_h       = daily_complaints * (ai_time_sec / 3600 + review_time_min / 60)
    ai_daily_compute = daily_complaints * ai_cost_per_triage
    ai_daily_labor   = (daily_complaints * review_time_min / 60) * analyst_hourly
    ai_yearly        = (ai_daily_compute + ai_daily_labor) * working_days

    savings_yearly   = manual_yearly - ai_yearly
    savings_pct      = savings_yearly / manual_yearly * 100 if manual_yearly > 0 else 0
    time_saved_h     = (manual_daily_h - ai_daily_h) * working_days
    speedup          = (manual_time_min * 60) / ai_time_sec

    st.markdown("---")

    # ── KPI strip ─────────────────────────────────────────────────
    metric_row([
        ("Annual Manual Cost", f"${manual_yearly:,.0f}"),
        ("Annual AI Cost",     f"${ai_yearly:,.0f}"),
        ("Annual Savings",     f"${savings_yearly:,.0f}", f"↓ {savings_pct:.0f}%"),
        ("Hours Saved / Year", f"{time_saved_h:,.0f}"),
        ("Speed Improvement",  f"{speedup:.0f}× faster"),
    ])

    # ── Before vs After waterfall chart ───────────────────────────
    st.markdown('<div class="section-title">Cost Breakdown — Manual vs AI-Powered</div>', unsafe_allow_html=True)

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Manual Process", "AI Compute Cost", "Human Review Cost", "Total AI Cost"],
        y=[manual_yearly, -(manual_yearly - ai_daily_compute * working_days),
           -(ai_daily_labor * working_days - ai_daily_compute * working_days * 0),
           0],
        text=[f"${manual_yearly:,.0f}", f"−${(manual_yearly - ai_yearly - ai_daily_labor*working_days):,.0f}",
              f"${ai_daily_labor*working_days:,.0f}", f"${ai_yearly:,.0f}"],
        textposition="outside",
        connector={"line": {"color": "#e2e8f0"}},
        decreasing={"marker": {"color": "#16a34a"}},
        increasing={"marker": {"color": "#ea580c"}},
        totals={"marker": {"color": TEAL}},
    ))
    plotly_defaults(fig_wf, height=340)
    fig_wf.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_wf, use_container_width=True)

    # ── Side-by-side annual costs ─────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Annual Cost Comparison</div>', unsafe_allow_html=True)
        fig_bar = go.Figure(go.Bar(
            x=["Manual", "PolicyPilot AI"],
            y=[manual_yearly, ai_yearly],
            marker_color=[NAVY, TEAL],
            text=[f"${manual_yearly:,.0f}", f"${ai_yearly:,.0f}"],
            textposition="outside",
            textfont=dict(size=13, color=NAVY),
        ))
        plotly_defaults(fig_bar, height=280)
        fig_bar.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-title">Time per Complaint</div>', unsafe_allow_html=True)
        fig_time = go.Figure(go.Bar(
            x=["Manual Process", "PolicyPilot AI"],
            y=[manual_time_min * 60, ai_time_sec + review_time_min * 60],
            marker_color=[NAVY, SKY],
            text=[f"{manual_time_min} min", f"{ai_time_sec}s + {review_time_min}min review"],
            textposition="outside",
        ))
        plotly_defaults(fig_time, height=280)
        fig_time.update_yaxes(title_text="Seconds")
        st.plotly_chart(fig_time, use_container_width=True)

    # ── Benefits + Architecture ───────────────────────────────────
    st.markdown('<div class="section-title">Key Benefits</div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f"""
- **⚡ Speed** — {manual_time_min} min → {ai_time_sec}s per complaint ({speedup:.0f}× faster)
- **💰 Cost** — Save **${savings_yearly:,.0f} AUD/year** ({savings_pct:.0f}% reduction)
- **📈 Scale** — {daily_complaints} complaints/day without additional headcount
""")
    with b2:
        st.markdown("""
- **🎯 Consistency** — Same ASIC regulatory framework applied to every complaint
- **📋 Compliance** — Automatic RG 271 timeframe tracking and audit trail
- **🔍 Attribution** — Full reasoning and source citation for every decision
""")

    st.markdown('<div class="section-title">Technical Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
| Component | Technology | Role |
|-----------|-----------|------|
| **AI Engine** | GPT-4o-mini + LangChain | Classification & response generation |
| **Knowledge Base** | Qdrant (vector) + ASIC RG 271 | 196 regulatory paragraphs, cosine search |
| **Database** | PostgreSQL | Complaint history and audit log |
| **API** | FastAPI (async) | Production REST endpoints |
| **Monitoring** | Prometheus | Latency, cost, accuracy tracking |
| **Frontend** | Streamlit | Interactive triage dashboard |
""")
