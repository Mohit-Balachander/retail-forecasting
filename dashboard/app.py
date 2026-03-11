"""
dashboard/app.py
----------------
Streamlit Dashboard — Smart Retail Forecasting Platform
Theme: Deep teal + amber gold — premium data analytics aesthetic
"""

# ── Auto-setup for Streamlit Cloud ─────────────────────────
import subprocess, sys as _sys
from pathlib import Path as _Path
if not _Path("data/validated/validated_train.csv").exists() or not _Path("models/saved/xgboost_model.json").exists():
    with open("setup_log.txt", "w") as _log:
        subprocess.run([_sys.executable, "setup_data.py"], stdout=_log, stderr=_log)



import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Retail Forecast AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS Theme ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-base:      #050c14;
    --bg-surface:   #081420;
    --bg-card:      #0b1d2e;
    --bg-hover:     #0f2540;
    --teal-bright:  #00e5c8;
    --teal-mid:     #00b5a0;
    --teal-dim:     #007a6e;
    --gold:         #f5a623;
    --gold-dim:     #c47e10;
    --coral:        #ff5f6d;
    --text-primary: #e8f4f2;
    --text-secondary:#7fa9a3;
    --border:       #112535;
    --border-glow:  rgba(0,229,200,0.25);
}

/* Base */
html, body, .stApp {
    background-color: var(--bg-base) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060f1a 0%, #040b14 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

/* ── Selectbox & Widgets ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--teal-dim) !important;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, var(--bg-card), #091a2a) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--teal-dim) !important;
    border-radius: 12px !important;
    padding: 20px 18px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4) !important;
}
[data-testid="metric-container"]:hover {
    border-top-color: var(--teal-bright) !important;
    box-shadow: 0 4px 32px rgba(0,229,200,0.12) !important;
    transform: translateY(-1px) !important;
}
[data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 30px !important;
    font-weight: 400 !important;
}
[data-testid="stMetricDelta"] {
    color: var(--teal-bright) !important;
    font-size: 13px !important;
}

/* ── Section Headers ── */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    font-weight: 400;
    color: var(--text-primary);
    margin: 32px 0 4px 0;
    letter-spacing: 0.3px;
}
.section-sub {
    font-size: 12px;
    color: var(--text-secondary);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, var(--teal-dim), transparent);
    margin: 8px 0 20px 0;
    border: none;
}

/* ── Hero Header ── */
.hero-wrap {
    background: linear-gradient(135deg, #071828 0%, #040d18 60%, #07181a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,229,200,0.07), transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 34px;
    color: var(--text-primary);
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 13px;
    color: var(--text-secondary);
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,229,200,0.1);
    border: 1px solid var(--teal-dim);
    color: var(--teal-bright);
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 99px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 12px;
}

/* ── Insight Box ── */
.insight-box {
    background: linear-gradient(135deg, #071e1a, #040f0d);
    border: 1px solid var(--teal-dim);
    border-left: 3px solid var(--teal-bright);
    border-radius: 10px;
    padding: 18px 22px;
    margin: 16px 0;
    color: #a8e6df;
    font-size: 15px;
    line-height: 1.75;
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    font-family: 'DM Serif Display', serif;
    font-size: 20px;
    color: var(--teal-bright);
    margin-bottom: 4px;
}
.sidebar-tagline {
    font-size: 10px;
    color: var(--text-secondary);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

/* ── Sidebar metric ── */
.sb-metric {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.sb-metric-label {
    font-size: 10px;
    color: var(--text-secondary);
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.sb-metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: var(--teal-bright);
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--teal-dim), #005f54) !important;
    color: white !important;
    border: 1px solid var(--teal-dim) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--teal-mid), var(--teal-dim)) !important;
    border-color: var(--teal-bright) !important;
    box-shadow: 0 0 20px rgba(0,229,200,0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--teal-bright) !important; }

/* ── Plotly chart bg fix ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme config ─────────────────────────────────────
PLOT_BG   = "#070f1a"
PAPER_BG  = "#070f1a"
GRID_COL  = "#0e2030"
TEAL      = "#00e5c8"
TEAL_MID  = "#00b5a0"
GOLD      = "#f5a623"
CORAL     = "#ff5f6d"
TEXT_COL  = "#7fa9a3"

def apply_plot_theme(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="DM Sans, sans-serif", color=TEXT_COL, size=12),
        legend=dict(
            orientation="h", y=1.08,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#7fa9a3")
        ),
        xaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, zeroline=False),
        yaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, zeroline=False),
        title=dict(font=dict(size=13, color="#4a8a84"), x=0),
    )
    return fig


# ── Data Loading ────────────────────────────────────────────
@st.cache_data
def load_data():
    for p in ["data/validated/validated_train.csv", "data/processed/cleaned_train.csv"]:
        if Path(p).exists():
            return pd.read_csv(p, parse_dates=["Date"])
    st.error("No data found. Run ETL pipeline first.")
    st.stop()

@st.cache_data
def load_meta():
    meta = {}
    for key, path in [("xgboost","models/saved/xgboost_metadata.json"),
                      ("lstm","models/saved/lstm_metadata.json")]:
        if Path(path).exists():
            meta[key] = json.load(open(path))
    return meta

@st.cache_resource
def load_model():
    try:
        import xgboost as xgb
        m = xgb.XGBRegressor()
        m.load_model("models/saved/xgboost_model.json")
        return m
    except:
        return None

FEATURE_COLS = [
    "Store","DayOfWeek","Promo","SchoolHoliday","StoreType","Assortment",
    "CompetitionDistance","Promo2","Year","Month","Day","WeekOfYear",
    "IsWeekend","Quarter","CompetitionOpenMonths","IsPromoMonth"
]

def predict_store(model, df, store_id, n=60):
    from sklearn.preprocessing import LabelEncoder
    sdf = df[df["Store"]==store_id].copy().sort_values("Date").tail(n)
    le  = LabelEncoder()
    for col in ["StoreType","Assortment","StateHoliday"]:
        if col in sdf.columns:
            sdf[col] = le.fit_transform(sdf[col].astype(str))
    avail = [c for c in FEATURE_COLS if c in sdf.columns]
    sdf["Predicted"] = model.predict(sdf[avail])
    return sdf[["Date","Sales","Predicted","Promo","SchoolHoliday"]]


# ── Load Everything ─────────────────────────────────────────
df    = load_data()
meta  = load_meta()
model = load_model()


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🛒 RetailIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Demand Forecasting Platform</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#4a8a84;margin-bottom:10px">FILTERS</p>', unsafe_allow_html=True)

    store_ids = sorted(df["Store"].unique())
    sel_store = st.selectbox("Store", store_ids)

    types = ["All"] + sorted(df["StoreType"].dropna().unique().tolist())
    sel_type = st.selectbox("Store Type", types)

    date_rng = st.slider(
        "Date Range",
        min_value=df["Date"].min().to_pydatetime(),
        max_value=df["Date"].max().to_pydatetime(),
        value=(pd.Timestamp("2015-01-01").to_pydatetime(),
               df["Date"].max().to_pydatetime())
    )

    st.markdown("---")
    st.markdown('<p style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#4a8a84;margin-bottom:10px">MODEL PERFORMANCE</p>', unsafe_allow_html=True)

    if "xgboost" in meta:
        m = meta["xgboost"]["metrics"]
        st.markdown(f"""
        <div class="sb-metric">
            <div class="sb-metric-label">XGBoost R²</div>
            <div class="sb-metric-value">{m['r2']:.3f}</div>
        </div>
        <div class="sb-metric">
            <div class="sb-metric-label">XGBoost RMSE</div>
            <div class="sb-metric-value" style="font-size:18px;color:#f5a623">{m['rmse']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    if "lstm" in meta:
        m = meta["lstm"]["metrics"]
        st.markdown(f"""
        <div class="sb-metric">
            <div class="sb-metric-label">LSTM R²</div>
            <div class="sb-metric-value" style="font-size:18px;color:#7fa9a3">{m['r2']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:10px;color:#2a4a44;text-align:center">Python · XGBoost · LSTM · SHAP · Streamlit</p>', unsafe_allow_html=True)


# ── Filter ──────────────────────────────────────────────────
filt = df[
    (df["Date"] >= pd.Timestamp(date_rng[0])) &
    (df["Date"] <= pd.Timestamp(date_rng[1]))
]
if sel_type != "All":
    filt = filt[filt["StoreType"] == sel_type]


# ════════════════════════════════════════════════════════════
# HERO HEADER
# ════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-title">🛒 Smart Retail Analytics &amp; Demand Forecasting</div>
    <div class="hero-subtitle">Rossmann Store Sales Intelligence Platform</div>
    <div>
        <span class="hero-badge">✦ {filt['Store'].nunique()} Stores</span>&nbsp;
        <span class="hero-badge">✦ {len(filt):,} Records</span>&nbsp;
        <span class="hero-badge">✦ XGBoost + LSTM</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# KPI CARDS
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

total_sales   = filt["Sales"].sum()
avg_daily     = filt.groupby("Date")["Sales"].sum().mean()
avg_cust      = filt["Customers"].mean() if "Customers" in filt.columns else 0
promo_lift    = ((filt[filt["Promo"]==1]["Sales"].mean() /
                  filt[filt["Promo"]==0]["Sales"].mean()) - 1) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 Total Revenue",      f"€{total_sales/1e6:.1f}M")
c2.metric("📅 Avg Daily Sales",    f"€{avg_daily:,.0f}")
c3.metric("👥 Avg Customers",      f"{avg_cust:,.0f}")
c4.metric("🎯 Promo Sales Lift",   f"+{promo_lift:.1f}%")


# ════════════════════════════════════════════════════════════
# SALES TREND
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Sales Trend Over Time</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Daily revenue with 7-day and 30-day moving averages</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

daily       = filt.groupby("Date")["Sales"].sum().reset_index()
daily["MA7"]  = daily["Sales"].rolling(7).mean()
daily["MA30"] = daily["Sales"].rolling(30).mean()

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=daily["Date"], y=daily["Sales"], name="Daily Sales",
    line=dict(color=TEAL, width=1), opacity=0.3,
    fill="tozeroy", fillcolor="rgba(0,229,200,0.04)"
))
fig_trend.add_trace(go.Scatter(
    x=daily["Date"], y=daily["MA7"], name="7-Day MA",
    line=dict(color=TEAL, width=2.5)
))
fig_trend.add_trace(go.Scatter(
    x=daily["Date"], y=daily["MA30"], name="30-Day MA",
    line=dict(color=GOLD, width=2, dash="dot")
))
apply_plot_theme(fig_trend, 340)
st.plotly_chart(fig_trend, use_container_width=True)


# ════════════════════════════════════════════════════════════
# FORECAST
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Demand Forecast — Actual vs Predicted</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-sub">Store {sel_store} · Last 60 days · XGBoost model</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

if model:
    pred_df = predict_store(model, df, sel_store, 60)
    fig_fc  = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=pred_df["Date"], y=pred_df["Sales"],
        name="Actual", line=dict(color=TEAL, width=2.5)
    ))
    fig_fc.add_trace(go.Scatter(
        x=pred_df["Date"], y=pred_df["Predicted"],
        name="Forecast", line=dict(color=GOLD, width=2, dash="dash")
    ))
    # Confidence band
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([pred_df["Date"], pred_df["Date"][::-1]]),
        y=pd.concat([pred_df["Predicted"]*1.1, pred_df["Predicted"][::-1]*0.9]),
        fill="toself", fillcolor="rgba(245,166,35,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence Band", showlegend=True
    ))
    promo = pred_df[pred_df["Promo"]==1]
    fig_fc.add_trace(go.Scatter(
        x=promo["Date"], y=promo["Sales"], mode="markers",
        name="Promo Day", marker=dict(color=CORAL, size=7, symbol="star",
        line=dict(color="white", width=0.5))
    ))
    apply_plot_theme(fig_fc, 340)
    st.plotly_chart(fig_fc, use_container_width=True)
else:
    st.warning("Train XGBoost model first: `python -m models.train_xgboost`")


# ════════════════════════════════════════════════════════════
# STORE COMPARISON + HEATMAP
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Store Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    top10 = (filt.groupby("Store")["Sales"].sum()
             .sort_values(ascending=False).head(10).reset_index())
    fig_bar = go.Figure(go.Bar(
        x=top10["Store"].astype(str), y=top10["Sales"],
        marker=dict(
            color=top10["Sales"],
            colorscale=[[0,"#003d36"],[0.5,TEAL_MID],[1,TEAL]],
            line=dict(width=0)
        )
    ))
    fig_bar.update_layout(title="Top 10 Stores by Revenue",
                          xaxis_title="Store ID", yaxis_title="Total Sales")
    apply_plot_theme(fig_bar, 300)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_r:
    if "StoreType" in filt.columns:
        pc = filt.groupby(["StoreType","Promo"])["Sales"].mean().reset_index()
        pc["PromoLabel"] = pc["Promo"].map({0:"No Promo",1:"With Promo"})
        fig_promo = go.Figure()
        colors_map = {"No Promo": "#1a4a44", "With Promo": TEAL}
        for label, grp in pc.groupby("PromoLabel"):
            fig_promo.add_trace(go.Bar(
                x=grp["StoreType"], y=grp["Sales"],
                name=label, marker_color=colors_map[label],
                marker_line_width=0
            ))
        fig_promo.update_layout(
            title="Promo vs No Promo by Store Type",
            barmode="group", xaxis_title="Store Type", yaxis_title="Avg Sales"
        )
        apply_plot_theme(fig_promo, 300)
        st.plotly_chart(fig_promo, use_container_width=True)


# ════════════════════════════════════════════════════════════
# HEATMAP
# ════════════════════════════════════════════════════════════
hm = filt.groupby(["Month","DayOfWeek"])["Sales"].mean().unstack()
hm.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(hm.columns)]
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"][:len(hm)]
hm.index = month_labels

fig_heat = go.Figure(go.Heatmap(
    z=hm.values, x=hm.columns, y=hm.index,
    colorscale=[[0,"#030e14"],[0.3,"#003d36"],[0.7,TEAL_MID],[1,TEAL]],
    hoverongaps=False
))
fig_heat.update_layout(title="Avg Sales Heatmap — Month × Day of Week")
apply_plot_theme(fig_heat, 300)
st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════════════════════════════════
# AI EXPLAINER
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🤖 AI Sales Explainer</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Select any store and date to understand what drove sales</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

e1, e2, e3 = st.columns([1,1,1])
with e1:
    exp_store = st.selectbox("Store", store_ids, key="exp_s")
with e2:
    sdates = sorted(df[df["Store"]==exp_store]["Date"].unique())
    exp_date = st.selectbox("Date", [str(d)[:10] for d in sdates[-30:][::-1]], key="exp_d")
with e3:
    st.markdown("<br>", unsafe_allow_html=True)
    go_explain = st.button("🔍 Analyse This Day", use_container_width=True)

if go_explain and model:
    with st.spinner("Running SHAP analysis..."):
        try:
            import shap
            from sklearn.preprocessing import LabelEncoder

            rdf = df[(df["Store"]==exp_store) & (df["Date"]==exp_date)].copy()
            if rdf.empty:
                st.warning("No data found for this date.")
            else:
                le = LabelEncoder()
                for col in ["StoreType","Assortment","StateHoliday"]:
                    if col in df.columns:
                        rdf[col] = le.fit_transform(rdf[col].astype(str))
                avail   = [c for c in FEATURE_COLS if c in rdf.columns]
                X       = rdf[avail]
                actual  = float(rdf["Sales"].values[0])
                pred    = float(model.predict(X)[0])
                pct     = ((actual - pred) / pred) * 100

                m1, m2, m3 = st.columns(3)
                m1.metric("Actual Sales",    f"€{actual:,.0f}")
                m2.metric("Predicted Sales", f"€{pred:,.0f}")
                delta_color = "normal" if pct >= 0 else "inverse"
                m3.metric("vs Forecast", f"{pct:+.1f}%", delta_color=delta_color)

                # SHAP
                explainer  = shap.TreeExplainer(model)
                shap_vals  = explainer.shap_values(X)
                shap_df = pd.DataFrame({
                    "Feature":       avail,
                    "SHAP":          shap_vals[0],
                    "Feature Value": X.iloc[0].values
                }).sort_values("SHAP", key=abs, ascending=True).tail(8)

                bar_colors = [CORAL if v < 0 else TEAL for v in shap_df["SHAP"]]
                fig_shap = go.Figure(go.Bar(
                    x=shap_df["SHAP"],
                    y=[f"{r['Feature']}  =  {r['Feature Value']:.1f}" for _, r in shap_df.iterrows()],
                    orientation="h",
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    text=[f"{v:+.0f}" for v in shap_df["SHAP"]],
                    textposition="outside",
                    textfont=dict(color=TEXT_COL, size=11)
                ))
                fig_shap.update_layout(
                    title=f"Feature Impact on Store {exp_store} Sales — {exp_date}",
                    xaxis_title="Impact on Predicted Sales (€)"
                )
                apply_plot_theme(fig_shap, 340)
                st.plotly_chart(fig_shap, use_container_width=True)

                direction = "increased" if pct >= 0 else "dropped"
                top       = shap_df.iloc[-1]
                promo_txt = "An active promotion contributed positively." if rdf["Promo"].values[0]==1 else "No promotion was active."
                st.markdown(f"""
                <div class="insight-box">
                💡 <strong>AI Insight:</strong>&nbsp; Sales {direction} <strong>{abs(pct):.1f}%</strong>
                for Store {exp_store} on {exp_date}.
                The strongest driver was <strong>{top['Feature']}</strong>
                with an estimated impact of <strong>€{abs(top['SHAP']):,.0f}</strong> on the forecast.
                {promo_txt}
                </div>
                """, unsafe_allow_html=True)

        except ImportError:
            st.error("Run: pip install shap")

elif go_explain and not model:
    st.error("Run `python -m models.train_xgboost` first.")


# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;font-size:12px;color:#1e3a34;letter-spacing:1.5px">RETAILIQ · BUILT WITH PYTHON · XGBOOST · LSTM · SHAP · STREAMLIT</p>',
    unsafe_allow_html=True
)