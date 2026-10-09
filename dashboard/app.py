"""
CardioMine — Streamlit Dashboard
Patient Analysis & Heart Disease Risk Mining System

Pages:
  1. Overview
  2. OLAP Analytics
  3. Patient Risk Prediction
  4. Patient Clusters
  5. Model Evaluation
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow imports from project root
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

# ──────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioMine — Heart Disease Risk Mining",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Base ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: #ffffff;
    color: #334155;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0fdfa 0%, #ccfbf1 100%);
    color: #0f766e;
    border-right: 1px solid rgba(20, 184, 166, 0.3);
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(20, 184, 166, 0.15) 0%, rgba(13, 148, 136, 0.08) 100%);
    border: 1px solid rgba(20, 184, 166, 0.3);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(20, 184, 166, 0.6);
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #14b8a6, #0d9488);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.kpi-label {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Section header */
.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 6px;
    padding-bottom: 10px;
    border-bottom: 2px solid rgba(20, 184, 166, 0.4);
}

/* Risk badge */
.risk-low    { background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #34d399; border-radius: 8px; padding: 4px 12px; }
.risk-medium { background: rgba(245, 158, 11, 0.2); border: 1px solid #f59e0b; color: #fbbf24; border-radius: 8px; padding: 4px 12px; }
.risk-high   { background: rgba(239, 68, 68, 0.2);  border: 1px solid #ef4444; color: #f43f5e; border-radius: 8px; padding: 4px 12px; }

/* Disclaimer */
.disclaimer {
    background: rgba(245, 158, 11, 0.1);
    border-left: 3px solid #f59e0b;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    color: #fbbf24;
    margin-top: 12px;
}

/* Metric override */
[data-testid="metric-container"] {
    background: rgba(20, 184, 166, 0.08);
    border: 1px solid rgba(20, 184, 166, 0.2);
    border-radius: 12px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Plotly default template
# ──────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_white"
COLOR_PALETTE   = px.colors.qualitative.Vivid
ACCENT_COLORS   = ["#14b8a6", "#0d9488", "#0f766e", "#2dd4bf", "#5eead4", "#99f6e4"]


# ──────────────────────────────────────────────────────────────────
# Authentication logic
# ──────────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown('<div style="text-align: center; margin-top: 80px;">', unsafe_allow_html=True)
    st.markdown("## 🔐 CardioMine Login")
    st.write("")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login", type="primary", use_container_width=True)
            
            if submit_button:
                if username == "admin" and password == "12345678":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────────────────────────
# Data loading helpers (cached)
# ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    path = os.path.join(ROOT, "data", "processed", "heart_disease_clean.csv")
    if not os.path.exists(path):
        st.error(
            "❌ Processed data not found. Run the ETL pipeline first:\n\n"
            "`python -m etl.load`"
        )
        st.stop()
    df = pd.read_csv(path)
    return df

def get_main_data() -> pd.DataFrame:
    if "main_df" in st.session_state:
        return st.session_state["main_df"]
    return None


@st.cache_resource
def load_rf_model():
    try:
        from mining.classification import load_model
        return load_model()
    except FileNotFoundError:
        return None


@st.cache_resource
def load_kmeans_model():
    try:
        from mining.clustering import load_cluster_model
        return load_cluster_model()
    except FileNotFoundError:
        return None


@st.cache_resource
def load_regression_pkg():
    try:
        from mining.regression import load_regression_model
        return load_regression_model()
    except FileNotFoundError:
        return None


# ──────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🫀 CardioMine")
    st.markdown("*Patient Analysis & Heart Disease Risk Mining System*")
    st.divider()

    st.markdown("### ⚙️ Data Management")
    uploaded_main = st.file_uploader("Upload Raw UCI Dataset", type=["csv"], help="Upload the raw heart_disease.csv to use across the dashboard.")
    if uploaded_main is not None:
        try:
            raw_df = pd.read_csv(uploaded_main)
            
            # Map alternate target column names if needed
            if 'target' not in raw_df.columns:
                if 'num' in raw_df.columns:
                    raw_df['target'] = raw_df['num']
                elif 'disease' in raw_df.columns:
                    raw_df['target'] = raw_df['disease']
                else:
                    for col in raw_df.columns:
                        if col.lower() == 'target':
                            raw_df.rename(columns={col: 'target'}, inplace=True)
                            break
            
            if 'target' not in raw_df.columns:
                raise KeyError("The dataset is missing a 'target' or 'num' column, which is required for dashboard analytics.")
                
            from etl.transform import transform
            clean_df = transform(raw_df)
            st.session_state["main_df"] = clean_df

            # ── Auto-train mining models ──────────────────────────
            with st.spinner("🔄 Training mining models on uploaded dataset…"):
                try:
                    from mining.preprocess import prepare_classification_data, prepare_clustering_data
                    from mining import classification as clf_mod
                    from mining import clustering as clust_mod
                    from mining import regression as reg_mod

                    # Classification
                    X_train, X_test, y_train, y_test, feat_names = prepare_classification_data(clean_df)
                    rf  = clf_mod.train_random_forest(X_train, y_train)
                    lr  = clf_mod.train_logistic_regression(X_train, y_train)
                    j48 = clf_mod.train_j48(X_train, y_train)
                    nb  = clf_mod.train_naive_bayes(X_train, y_train)
                    all_metrics = {
                        "random_forest":       clf_mod.evaluate_model(rf,  X_test, y_test, "Random Forest"),
                        "logistic_regression": clf_mod.evaluate_model(lr,  X_test, y_test, "Logistic Regression"),
                        "j48":                 clf_mod.evaluate_model(j48, X_test, y_test, "J48 Decision Tree"),
                        "naive_bayes":         clf_mod.evaluate_model(nb,  X_test, y_test, "Naive Bayes"),
                    }
                    fi = clf_mod.get_feature_importance(rf, feat_names)
                    all_metrics["random_forest"]["feature_importance"] = fi.to_dict("records")
                    all_metrics["j48"]["tree_text"] = clf_mod.get_j48_text(j48, feat_names)
                    all_metrics["j48"]["feature_importance"] = clf_mod.get_feature_importance(j48, feat_names).to_dict("records")
                    all_metrics["logistic_regression"]["coefficients"] = clf_mod.get_logistic_coefficients(lr, feat_names).to_dict("records")
                    all_metrics["naive_bayes"]["class_stats"] = clf_mod.get_nb_class_stats(nb, feat_names)
                    clf_mod.save_model(rf, lr, j48, nb, feat_names, all_metrics)

                    # Clustering
                    X_scaled, scaler, clust_feats = prepare_clustering_data(clean_df)
                    elbow = clust_mod.elbow_analysis(X_scaled)
                    km = clust_mod.train_kmeans(X_scaled, n_clusters=3)
                    profiles = clust_mod.get_cluster_profiles(clean_df, km.labels_, clust_feats)
                    clust_mod.save_cluster_model(km, scaler, clust_feats, profiles, elbow)

                    # Regression
                    reg_mod.train_regression(clean_df)

                    # Clear cached resources so new models are picked up
                    load_rf_model.clear()
                    load_kmeans_model.clear()
                    load_regression_pkg.clear()

                    st.success("✅ Dataset loaded & mining models trained successfully!")
                except Exception as train_err:
                    st.warning(f"Dataset loaded, but model training failed: {train_err}")
                    st.success("Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error processing dataset: {e}")
    st.divider()
    
    page = st.radio(
        "Navigate to",
        [
            "📊 Overview",
            "🔬 OLAP Analytics",
            "🩺 Risk Prediction",
            "🔵 Patient Clusters",
            "📈 Model Comparison",
            "📉 Regression Analysis",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Dataset:** UCI Heart Disease (Cleveland)")
    st.markdown("**Subject:** Data Warehousing & Mining")
    st.markdown("**Schema:** Star Schema (MySQL)")

# ──────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────
df = get_main_data()

if df is None:
    st.markdown('<div style="text-align: center; margin-top: 100px;">', unsafe_allow_html=True)
    st.markdown("## 🫀 Welcome to CardioMine")
    st.markdown("#### Please upload a raw UCI Heart Disease dataset using the **Data Management** section in the sidebar to get started.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()
rf_package = load_rf_model()
km_package = load_kmeans_model()
reg_package = load_regression_pkg()


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.markdown('<p class="section-header">📊 CardioMine — Patient Overview</p>', unsafe_allow_html=True)
    st.caption("Population-level statistics from the UCI Heart Disease (Cleveland) dataset.")

    # ── KPI row ─────────────────────────────────────────────────
    total = len(df)
    diseased = int(df["target"].sum())
    prevalence = diseased / total * 100
    avg_age = df["age"].mean()
    avg_chol = df["cholesterol"].mean()

    high_risk = 0
    if rf_package:
        from mining.classification import load_model
        pkg = load_model()
        feat_names = pkg["feature_names"]
        X_all = df[[c for c in feat_names if c in df.columns]].fillna(df.median(numeric_only=True))
        probs = pkg["model"].predict_proba(X_all)[:, 1]
        high_risk = int((probs >= 0.65).sum())
    else:
        high_risk = diseased  # Fallback

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in zip(
        [c1, c2, c3, c4, c5],
        [total, f"{prevalence:.1f}%", f"{avg_age:.1f}", f"{avg_chol:.0f} mg/dL", high_risk],
        ["Total Patients", "Disease Prevalence", "Average Age", "Avg Cholesterol", "High-Risk Patients"],
    ):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Charts row 1 ────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.pie(
            df,
            names="disease_status",
            title="Heart Disease Distribution",
            color_discrete_sequence=["#14b8a6", "#f43f5e"],
            hole=0.5,
            template=PLOTLY_TEMPLATE,
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with col_b:
        age_counts = (
            df.groupby(["age_group", "disease_status"])
            .size()
            .reset_index(name="count")
        )
        order = ["Young", "Adult", "Middle Age", "Senior"]
        age_counts["age_group"] = pd.Categorical(age_counts["age_group"], categories=order, ordered=True)
        age_counts.sort_values("age_group", inplace=True)

        fig2 = px.bar(
            age_counts,
            x="age_group",
            y="count",
            color="disease_status",
            barmode="group",
            title="Disease by Age Group",
            color_discrete_sequence=["#14b8a6", "#f43f5e"],
            template=PLOTLY_TEMPLATE,
            labels={"age_group": "Age Group", "count": "Count", "disease_status": "Status"},
        )
        st.plotly_chart(fig2, use_container_width=True, theme=None)

    # ── Charts row 2 ────────────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        gender_data = (
            df.groupby(["gender", "disease_status"])
            .size()
            .reset_index(name="count")
        )
        fig3 = px.bar(
            gender_data,
            x="gender",
            y="count",
            color="disease_status",
            barmode="group",
            title="Disease by Gender",
            color_discrete_sequence=["#14b8a6", "#f43f5e"],
            template=PLOTLY_TEMPLATE,
            labels={"gender": "Gender", "count": "Count", "disease_status": "Status"},
        )
        st.plotly_chart(fig3, use_container_width=True, theme=None)

    with col_d:
        fig4 = px.histogram(
            df,
            x="cholesterol",
            color="disease_status",
            barmode="overlay",
            nbins=30,
            opacity=0.75,
            title="Cholesterol Distribution",
            color_discrete_sequence=["#14b8a6", "#f43f5e"],
            template=PLOTLY_TEMPLATE,
            labels={"cholesterol": "Cholesterol (mg/dL)", "disease_status": "Status"},
        )
        st.plotly_chart(fig4, use_container_width=True, theme=None)

    # ── Scatter: Age vs Max Heart Rate ──────────────────────────
    st.markdown("### Scatter: Age vs Max Heart Rate")
    fig5 = px.scatter(
        df,
        x="age",
        y="max_heart_rate",
        color="disease_status",
        size="cholesterol",
        hover_data=["gender", "cholesterol_category", "bp_category"],
        color_discrete_sequence=["#14b8a6", "#f43f5e"],
        template=PLOTLY_TEMPLATE,
        title="Age vs Max Heart Rate (bubble size = Cholesterol)",
        labels={"age": "Age", "max_heart_rate": "Max Heart Rate", "disease_status": "Status"},
    )
    st.plotly_chart(fig5, use_container_width=True, theme=None)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — OLAP ANALYTICS
# ══════════════════════════════════════════════════════════════════

elif page == "🔬 OLAP Analytics":
    st.markdown('<p class="section-header">🔬 OLAP Analytics</p>', unsafe_allow_html=True)
    st.caption("Interactive multidimensional analysis. Use filters to Slice & Dice the dataset.")

    # ── Filter sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filters (Slice & Dice)")
        age_filter    = st.multiselect("Age Group",            ["Young", "Adult", "Middle Age", "Senior"], default=["Young", "Adult", "Middle Age", "Senior"])
        gender_filter = st.multiselect("Gender",               df["gender"].unique().tolist(), default=df["gender"].unique().tolist())
        disease_filter= st.multiselect("Disease Status",       df["disease_status"].unique().tolist(), default=df["disease_status"].unique().tolist())
        chol_filter   = st.multiselect("Cholesterol Category", df["cholesterol_category"].unique().tolist(), default=df["cholesterol_category"].unique().tolist())
        bp_filter     = st.multiselect("BP Category",          df["bp_category"].unique().tolist(), default=df["bp_category"].unique().tolist())

    # Apply filters
    mask = (
        df["age_group"].isin(age_filter)
        & df["gender"].isin(gender_filter)
        & df["disease_status"].isin(disease_filter)
        & df["cholesterol_category"].isin(chol_filter)
        & df["bp_category"].isin(bp_filter)
    )
    fdf = df[mask].copy()

    st.markdown(f"**Showing {len(fdf)} of {len(df)} patients** matching current filters.")

    if len(fdf) == 0:
        st.warning("No records match current filters. Adjust the selections.")
        st.stop()

    # ── OLAP: Roll-up table ─────────────────────────────────────
    st.markdown("#### Roll-up: Disease Prevalence by Age Group → Population")
    rollup = (
        fdf.groupby("age_group", observed=True)
        .agg(
            total_patients=("patient_id", "count"),
            disease_count=("target", "sum"),
            avg_cholesterol=("cholesterol", "mean"),
            avg_resting_bp=("resting_bp", "mean"),
        )
        .reset_index()
    )
    rollup["disease_prevalence_pct"] = (rollup["disease_count"] / rollup["total_patients"] * 100).round(1)
    rollup["avg_cholesterol"] = rollup["avg_cholesterol"].round(1)
    rollup["avg_resting_bp"] = rollup["avg_resting_bp"].round(1)
    st.dataframe(rollup, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(
            rollup,
            x="age_group",
            y="disease_prevalence_pct",
            color="age_group",
            color_discrete_sequence=ACCENT_COLORS,
            title="Disease Prevalence % by Age Group (Roll-up)",
            template=PLOTLY_TEMPLATE,
            labels={"age_group": "Age Group", "disease_prevalence_pct": "Prevalence (%)"},
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with col_b:
        fig2 = px.bar(
            fdf.groupby("cholesterol_category", observed=True)["target"].mean().reset_index()
            .rename(columns={"target": "disease_rate"}),
            x="cholesterol_category",
            y="disease_rate",
            color="cholesterol_category",
            color_discrete_sequence=ACCENT_COLORS,
            title="Disease Rate by Cholesterol Category (Slice)",
            template=PLOTLY_TEMPLATE,
            labels={"cholesterol_category": "Cholesterol Category", "disease_rate": "Disease Rate"},
        )
        st.plotly_chart(fig2, use_container_width=True, theme=None)

    # ── OLAP: Heatmap (Dice) ────────────────────────────────────
    st.markdown("#### Dice: Disease Prevalence (Age Group × Cholesterol Category)")
    pivot = (
        fdf.groupby(["age_group", "cholesterol_category"], observed=True)["target"]
        .mean()
        .unstack(fill_value=0)
        * 100
    ).round(1)

    age_order = [a for a in ["Young", "Adult", "Middle Age", "Senior"] if a in pivot.index]
    pivot = pivot.reindex(age_order)

    fig3 = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Purples",
            text=pivot.values.round(1),
            texttemplate="%{text}%",
            textfont={"size": 12},
        )
    )
    fig3.update_layout(
        title="Dice: Disease Prevalence Heatmap (Age × Cholesterol)",
        template=PLOTLY_TEMPLATE,
        height=350,
    )
    st.plotly_chart(fig3, use_container_width=True, theme=None)

    # ── OLAP: Drill-down by quarter ─────────────────────────────
    st.markdown("#### Drill-down: Disease Count by Year → Quarter → Month")
    if "date_id" not in fdf.columns:
        # Simulate date dimension inline (since we might not have MySQL)
        import datetime
        base = datetime.date(2020, 1, 1)
        fdf = fdf.copy()
        fdf["date_id"] = range(1, len(fdf) + 1)
        fdf["year"]    = [(base + datetime.timedelta(days=i - 1)).year    for i in fdf["date_id"]]
        fdf["month"]   = [(base + datetime.timedelta(days=i - 1)).month   for i in fdf["date_id"]]
        fdf["quarter"] = [(base + datetime.timedelta(days=i - 1)).month // 4 + 1 for i in fdf["date_id"]]

    drill_level = st.selectbox("Drill-down level", ["Year", "Quarter", "Month"])
    if drill_level == "Year":
        group_col = "year"
    elif drill_level == "Quarter":
        group_col = "quarter"
    else:
        group_col = "month"

    drill = fdf.groupby(group_col, observed=True).agg(
        total=("patient_id", "count"),
        disease_count=("target", "sum"),
    ).reset_index()
    drill["label"] = drill_level + " " + drill[group_col].astype(str)

    fig4 = px.line(
        drill,
        x="label",
        y=["total", "disease_count"],
        markers=True,
        color_discrete_sequence=["#14b8a6", "#f43f5e"],
        title=f"Drill-down: Patient Count by {drill_level}",
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig4, use_container_width=True, theme=None)


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — RISK PREDICTION
# ══════════════════════════════════════════════════════════════════

elif page == "🩺 Risk Prediction":
    st.markdown('<p class="section-header">🩺 Patient Risk Prediction</p>', unsafe_allow_html=True)
    st.caption("**Algorithm used:** Random Forest Classifier")

    if rf_package is None:
        st.warning(
            "⚠️ Model not found. Run the mining pipeline first:\n\n"
            "`python -m mining.classification`"
        )
        st.stop()

    model        = rf_package["model"]
    feature_names = rf_package["feature_names"]

    st.markdown("Enter patient clinical parameters below or upload a batch CSV to estimate heart disease risk.")

    st.markdown(
        '<div class="disclaimer">⚠️ This tool is for academic demonstration only. '
        'It does <strong>not</strong> constitute a medical diagnosis. '
        'Always consult a qualified physician.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    tab_single, tab_batch = st.tabs(["👤 Single Patient Entry", "📁 Batch Prediction (CSV)"])

    with tab_single:
        col1, col2, col3 = st.columns(3)

        with col1:
            age     = st.number_input("Age (years)", min_value=20, max_value=100, value=55, step=1)
            sex     = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
            cp      = st.selectbox("Chest Pain Type", options=[
                ("Typical Angina", 1),
                ("Atypical Angina", 2),
                ("Non-Anginal Pain", 3),
                ("Asymptomatic", 4),
            ], format_func=lambda x: x[0])

        with col2:
            trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=130, step=1)
            chol     = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=240, step=1)
            fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
            restecg  = st.selectbox("Resting ECG Result", options=[
                ("Normal", 0),
                ("ST-T Abnormality", 1),
                ("Left Ventricular Hypertrophy", 2),
            ], format_func=lambda x: x[0])

        with col3:
            thalach  = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
            exang    = st.selectbox("Exercise-Induced Angina?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
            oldpeak  = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

        patient_data = {
            "age":                age,
            "sex":                sex[1],
            "cp":                 cp[1],
            "resting_bp":         trestbps,
            "trestbps":           trestbps,
            "cholesterol":        chol,
            "chol":               chol,
            "fasting_blood_sugar": fbs[1],
            "fbs":                fbs[1],
            "restecg":            restecg[1],
            "max_heart_rate":     thalach,
            "thalach":            thalach,
            "exang":              exang[1],
            "st_depression":      oldpeak,
            "oldpeak":            oldpeak,
        }

        if st.button("🔮 Predict Risk", use_container_width=True, type="primary"):
            x = np.array([[patient_data.get(f, 0) for f in feature_names]])
            prob = model.predict_proba(x)[0][1]
            pct  = prob * 100

            if prob < 0.35:
                category  = "Low Risk"
                css_class = "risk-low"
                gauge_color = "#10b981"
            elif prob < 0.65:
                category  = "Moderate Risk"
                css_class = "risk-medium"
                gauge_color = "#f59e0b"
            else:
                category  = "High Risk"
                css_class = "risk-high"
                gauge_color = "#ef4444"

            st.markdown("---")
            col_r1, col_r2 = st.columns([1, 2])

            with col_r1:
                st.markdown(f"### Estimated Risk Probability")
                st.caption("*Based on Random Forest model*")
                st.markdown(
                    f'<h2 style="color:{gauge_color}; font-size:3rem; font-weight:700;">{pct:.1f}%</h2>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<span class="{css_class}" style="font-size:1.1rem; font-weight:600;">{category}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="disclaimer">Not a medical diagnosis. For academic use only.</div>',
                    unsafe_allow_html=True,
                )

            with col_r2:
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pct,
                    number={"suffix": "%", "font": {"size": 32, "color": gauge_color}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                        "bar": {"color": gauge_color, "thickness": 0.3},
                        "bgcolor": "rgba(0,0,0,0)",
                        "bordercolor": "#334155",
                        "steps": [
                            {"range": [0, 35],   "color": "rgba(16,185,129,0.15)"},
                            {"range": [35, 65],  "color": "rgba(245,158,11,0.15)"},
                            {"range": [65, 100], "color": "rgba(239,68,68,0.15)"},
                        ],
                        "threshold": {"line": {"color": gauge_color, "width": 3}, "value": pct},
                    },
                ))
                fig_gauge.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=280,
                    margin=dict(t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_gauge, use_container_width=True, theme=None)

            # Feature contribution (rough - use mean shift from training set)
            st.markdown("#### Feature Values vs. Population Averages")
            pop_means = df[[c for c in feature_names if c in df.columns]].mean()
            patient_vals = pd.Series([patient_data.get(f, 0) for f in feature_names], index=feature_names)

            comparison = pd.DataFrame({
                "Feature":       feature_names,
                "Your Value":    patient_vals.values,
                "Population Avg": [pop_means.get(f, 0) for f in feature_names],
            })
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=comparison["Feature"],
                y=comparison["Your Value"],
                name="Your Value",
                marker_color="#14b8a6",
            ))
            fig_comp.add_trace(go.Bar(
                x=comparison["Feature"],
                y=comparison["Population Avg"],
                name="Population Avg",
                marker_color="#64748b",
            ))
            fig_comp.update_layout(
                barmode="group",
                template=PLOTLY_TEMPLATE,
                height=320,
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_comp, use_container_width=True, theme=None)



    with tab_batch:
        st.markdown("### Upload Patient Data")
        st.caption("Upload a CSV file containing patient data. The file must contain columns matching the required features.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded {len(batch_df)} records.**")
            
            # Map alternative names if needed (e.g. from the raw dataset)
            if 'trestbps' in batch_df.columns and 'resting_bp' not in batch_df.columns: batch_df['resting_bp'] = batch_df['trestbps']
            if 'chol' in batch_df.columns and 'cholesterol' not in batch_df.columns: batch_df['cholesterol'] = batch_df['chol']
            if 'fbs' in batch_df.columns and 'fasting_blood_sugar' not in batch_df.columns: batch_df['fasting_blood_sugar'] = batch_df['fbs']
            if 'thalach' in batch_df.columns and 'max_heart_rate' not in batch_df.columns: batch_df['max_heart_rate'] = batch_df['thalach']
            if 'oldpeak' in batch_df.columns and 'st_depression' not in batch_df.columns: batch_df['st_depression'] = batch_df['oldpeak']
            
            missing_cols = [col for col in feature_names if col not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Perform predictions
                X_batch = batch_df[feature_names].fillna(0)
                probs = model.predict_proba(X_batch)[:, 1]
                batch_df["Risk_Probability"] = probs.round(3)
                batch_df["Risk_Category"] = np.where(probs < 0.35, "Low", np.where(probs < 0.65, "Moderate", "High"))
                
                st.markdown("#### Prediction Results")
                st.dataframe(batch_df, use_container_width=True)
                
                # Plot distribution
                risk_counts = batch_df["Risk_Category"].value_counts().reset_index()
                risk_counts.columns = ["Risk_Category", "Count"]
                fig_batch = px.pie(
                    risk_counts,
                    names="Risk_Category",
                    values="Count",
                    title="Batch Risk Distribution",
                    color="Risk_Category",
                    color_discrete_map={"Low": "#10b981", "Moderate": "#f59e0b", "High": "#ef4444"},
                    template=PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig_batch, use_container_width=True, theme=None)

# ══════════════════════════════════════════════════════════════════
# PAGE 4 — PATIENT CLUSTERS
# ══════════════════════════════════════════════════════════════════

elif page == "🔵 Patient Clusters":
    st.markdown('<p class="section-header">🔵 Patient Clusters (K-Means)</p>', unsafe_allow_html=True)
    st.caption("**Algorithm used:** K-Means Clustering")

    if km_package is None:
        st.warning(
            "⚠️ Cluster model not found. Run the mining pipeline first:\n\n"
            "`python -m mining.clustering`"
        )
        st.stop()

    km            = km_package["kmeans"]
    scaler        = km_package["scaler"]
    feature_names = km_package["feature_names"]
    profiles      = pd.DataFrame(km_package["profiles"])
    elbow         = km_package.get("elbow", {})

    # Assign cluster labels to dataframe
    X_clust = df[[c for c in feature_names if c in df.columns]].fillna(df.median(numeric_only=True))
    X_scaled = scaler.transform(X_clust)
    df["cluster"] = km.predict(X_scaled)
    df["cluster_label"] = "Cluster " + (df["cluster"] + 1).astype(str)

    # ── Elbow chart ─────────────────────────────────────────────
    if elbow:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=elbow["k"], y=elbow["inertia"],
                mode="lines+markers",
                marker=dict(color="#14b8a6", size=8),
                line=dict(color="#14b8a6"),
                name="WCSS (Inertia)"
            ))
            fig_elbow.update_layout(
                title="Elbow Method — Optimal k",
                xaxis_title="Number of Clusters (k)",
                yaxis_title="WCSS (Inertia)",
                template=PLOTLY_TEMPLATE,
                height=320,
            )
            st.plotly_chart(fig_elbow, use_container_width=True, theme=None)

        with col_b:
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(
                x=elbow["k"], y=elbow["silhouette"],
                mode="lines+markers",
                marker=dict(color="#0d9488", size=8),
                line=dict(color="#0d9488"),
                name="Silhouette Score"
            ))
            fig_sil.update_layout(
                title="Silhouette Score vs k",
                xaxis_title="Number of Clusters (k)",
                yaxis_title="Silhouette Score",
                template=PLOTLY_TEMPLATE,
                height=320,
            )
            st.plotly_chart(fig_sil, use_container_width=True, theme=None)

    # ── Cluster profile cards ────────────────────────────────────
    st.markdown("### Cluster Profiles")
    st.caption("Clusters were identified from the data. Interpretations are data-driven, not pre-assigned.")

    n_clusters = len(profiles)
    cols = st.columns(n_clusters)

    for i, (col, row) in enumerate(zip(cols, profiles.itertuples())):
        with col:
            cluster_df = df[df["cluster"] == row.cluster]
            n = int(row.count)
            prev = getattr(row, "disease_prevalence_pct", "N/A")

            st.markdown(
                f"""
                <div class="kpi-card">
                <div class="kpi-value">Cluster {i + 1}</div>
                <div class="kpi-label">{n} patients</div>
                <br/>
                """,
                unsafe_allow_html=True,
            )
            for feat in feature_names:
                key = f"avg_{feat}"
                val = getattr(row, key, None)
                if val is not None:
                    st.markdown(f"**{feat}:** {val}")
            if isinstance(prev, (int, float)):
                st.markdown(f"**Disease prevalence:** {prev}%")
            st.markdown("</div>", unsafe_allow_html=True)

    # ── 2D scatter of clusters (Age vs Max HR) ───────────────────
    st.markdown("### Cluster Scatter (Age × Max Heart Rate)")
    fig_scatter = px.scatter(
        df,
        x="age",
        y="max_heart_rate",
        color="cluster_label",
        symbol="disease_status",
        hover_data=["gender", "cholesterol_category", "cluster_label"],
        color_discrete_sequence=ACCENT_COLORS,
        template=PLOTLY_TEMPLATE,
        title="Patient Clusters — Age vs Max Heart Rate",
        labels={"age": "Age", "max_heart_rate": "Max Heart Rate"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True, theme=None)

    # ── Box plots per cluster ────────────────────────────────────
    st.markdown("### Feature Distributions by Cluster")
    feat_choice = st.selectbox("Feature to examine", feature_names)
    fig_box = px.box(
        df,
        x="cluster_label",
        y=feat_choice,
        color="cluster_label",
        color_discrete_sequence=ACCENT_COLORS,
        template=PLOTLY_TEMPLATE,
        title=f"{feat_choice} distribution by cluster",
        points="all",
    )
    st.plotly_chart(fig_box, use_container_width=True, theme=None)



# ══════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON (All 4 Classifiers)
# ══════════════════════════════════════════════════════════════════

elif page == "📈 Model Comparison":
    from sklearn.metrics import roc_curve

    st.markdown('<p class="section-header">📈 Model Comparison</p>', unsafe_allow_html=True)
    st.caption("**Algorithms compared:** Random Forest, Logistic Regression, J48 (Decision Tree), Naive Bayes")
    st.caption("Side-by-side evaluation of all four classifiers on the same test set.")

    if rf_package is None:
        st.warning("⚠️ Models not found. Run the mining pipeline first:\n\n`python run_pipeline.py`")
        st.stop()

    all_metrics = rf_package.get("all_metrics", {})
    if not all_metrics:
        st.error("Old model format detected. Re-run `python run_pipeline.py` to rebuild with all classifiers.")
        st.stop()

    MODEL_DISPLAY = {
        "random_forest":       ("Random Forest",       "#14b8a6"),
        "logistic_regression": ("Logistic Regression", "#38bdf8"),
        "j48":                 ("J48 Decision Tree",   "#34d399"),
        "naive_bayes":         ("Naive Bayes",         "#fb923c"),
    }

    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    def fmt(m, key):
        val = m.get(key, "N/A")
        return f"{val:.4f}" if isinstance(val, float) else "N/A"

    # ── Summary table ────────────────────────────────────────────
    st.markdown("### Performance Summary")
    rows = []
    for key, (display_name, _) in MODEL_DISPLAY.items():
        m = all_metrics.get(key, {})
        row = {"Model": display_name}
        row.update({label: fmt(m, k) for label, k in zip(metric_labels, metric_keys)})
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ── Grouped bar: all metrics × all models ────────────────────
    st.markdown("### Metric Comparison Chart")
    bar_rows = []
    for key, (display_name, color) in MODEL_DISPLAY.items():
        m = all_metrics.get(key, {})
        for label, mk in zip(metric_labels, metric_keys):
            bar_rows.append({"Model": display_name, "Metric": label, "Score": m.get(mk, 0)})
    bar_df = pd.DataFrame(bar_rows)

    fig_bar = px.bar(
        bar_df,
        x="Metric", y="Score", color="Model",
        barmode="group",
        color_discrete_sequence=[c for _, c in MODEL_DISPLAY.values()],
        template=PLOTLY_TEMPLATE,
        range_y=[0, 1.05],
        title="All Classifiers — All Metrics",
    )
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True, theme=None)

    # ── Overlay ROC curves ───────────────────────────────────────
    st.markdown("### ROC Curves (All Models)")
    fig_roc = go.Figure()
    for key, (display_name, color) in MODEL_DISPLAY.items():
        m = all_metrics.get(key, {})
        y_test_arr = np.array(m.get("y_test", []))
        y_prob_arr = np.array(m.get("y_prob", []))
        if len(y_test_arr) > 0:
            fpr, tpr, _ = roc_curve(y_test_arr, y_prob_arr)
            auc = m.get("roc_auc", 0)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"{display_name} (AUC={auc:.3f})",
                line=dict(color=color, width=2),
            ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Classifier",
        line=dict(color="#475569", dash="dash"),
    ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template=PLOTLY_TEMPLATE, height=420,
    )
    st.plotly_chart(fig_roc, use_container_width=True, theme=None)

    # ── Confusion matrices — 2×2 grid ────────────────────────────
    st.markdown("### Confusion Matrices")
    cm_keys = list(MODEL_DISPLAY.items())
    row1, row2 = cm_keys[:2], cm_keys[2:]

    for row_pair in [row1, row2]:
        cols = st.columns(2)
        for col, (key, (display_name, color)) in zip(cols, row_pair):
            m = all_metrics.get(key, {})
            cm_data = m.get("confusion_matrix", [[0, 0], [0, 0]])
            cm_arr = np.array(cm_data)
            with col:
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm_arr,
                    x=["Pred: No Disease", "Pred: Disease"],
                    y=["True: No Disease", "True: Disease"],
                    colorscale=[[0, "#0f172a"], [1, color]],
                    showscale=False,
                    text=cm_arr,
                    texttemplate="%{text}",
                    textfont={"size": 20, "color": "white"},
                ))
                fig_cm.update_layout(
                    title=display_name,
                    template=PLOTLY_TEMPLATE,
                    height=280,
                    margin=dict(t=40, b=10),
                )
                st.plotly_chart(fig_cm, use_container_width=True, theme=None)

    # ── Per-model detail tabs ─────────────────────────────────────
    st.markdown("### Per-Model Deep Dive")
    tabs = st.tabs([dn for _, (dn, _) in MODEL_DISPLAY.items()])

    for tab, (key, (display_name, color)) in zip(tabs, MODEL_DISPLAY.items()):
        m = all_metrics.get(key, {})
        with tab:
            # Feature importance (RF and J48)
            fi_data = m.get("feature_importance", [])
            if fi_data:
                fi_df = pd.DataFrame(fi_data).sort_values("importance", ascending=True)
                fig_fi = px.bar(
                    fi_df, x="importance", y="feature", orientation="h",
                    color="importance", color_continuous_scale="Purples",
                    template=PLOTLY_TEMPLATE,
                    title=f"Feature Importance — {display_name}",
                )
                fig_fi.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_fi, use_container_width=True, theme=None)

            # Logistic Regression coefficients
            coef_data = m.get("coefficients", [])
            if coef_data:
                coef_df = pd.DataFrame(coef_data)
                coef_df["direction"] = coef_df["coefficient"].apply(lambda x: "↑ Higher Risk" if x > 0 else "↓ Lower Risk")
                fig_coef = px.bar(
                    coef_df.sort_values("coefficient"),
                    x="coefficient", y="feature", orientation="h",
                    color="direction",
                    color_discrete_map={"↑ Higher Risk": "#f43f5e", "↓ Lower Risk": "#34d399"},
                    template=PLOTLY_TEMPLATE,
                    title="Log-Odds Coefficients — Logistic Regression",
                )
                fig_coef.update_layout(height=380)
                st.plotly_chart(fig_coef, use_container_width=True, theme=None)

            # J48 tree text
            tree_text = m.get("tree_text", "")
            if tree_text:
                st.markdown("**Decision Tree Rules (J48 / C4.5):**")
                st.code(tree_text, language="text")

            # Naive Bayes class stats
            class_stats = m.get("class_stats", {})
            if class_stats:
                st.markdown("**Naive Bayes — Class Conditional Means:**")
                col_nd, col_hd = st.columns(2)
                for col_widget, cls in zip([col_nd, col_hd], ["No Disease", "Heart Disease"]):
                    with col_widget:
                        st.caption(cls)
                        st.dataframe(pd.DataFrame(class_stats.get(cls, [])), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 6 — REGRESSION ANALYSIS
# ══════════════════════════════════════════════════════════════════

elif page == "📉 Regression Analysis":
    st.markdown('<p class="section-header">📉 Regression Analysis</p>', unsafe_allow_html=True)
    st.caption("**Algorithm used:** Linear / Polynomial Regression")
    st.caption(
        "Linear Regression predicts **max_heart_rate** from clinical features. "
        "Demonstrates the regression mining technique."
    )

    if reg_package is None:
        st.warning("⚠️ Regression model not found. Run the mining pipeline first:\n\n`python run_pipeline.py`")
        st.stop()

    lr_reg   = reg_package["linear_regression"]
    ridge    = reg_package["ridge_regression"]
    feat_reg = reg_package["feature_names"]
    lr_met   = reg_package["linear_metrics"]
    rr_met   = reg_package["ridge_metrics"]
    coefs    = pd.DataFrame(reg_package["coefficients"])

    # ── Summary KPIs ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["Linear R²", "Linear MAE", "Ridge R²", "Ridge MAE"],
        [lr_met["r2"], lr_met["mae"], rr_met["r2"], rr_met["mae"]],
    ):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Model Comparison")

    # ── Metric table ─────────────────────────────────────────────
    reg_comp = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Linear Regression": [lr_met["mae"], lr_met["rmse"], lr_met["r2"]],
        "Ridge Regression":  [rr_met["mae"], rr_met["rmse"], rr_met["r2"]],
    })
    st.dataframe(reg_comp, use_container_width=True, hide_index=True)

    # ── Grouped bar ───────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        bar_reg = reg_comp.melt(id_vars="Metric", var_name="Model", value_name="Value")
        fig_rbar = px.bar(
            bar_reg, x="Metric", y="Value", color="Model", barmode="group",
            color_discrete_sequence=["#14b8a6", "#38bdf8"],
            template=PLOTLY_TEMPLATE,
            title="Regression Metric Comparison",
        )
        st.plotly_chart(fig_rbar, use_container_width=True, theme=None)

    with col_b:
        # Coefficients
        fig_coef = px.bar(
            coefs.sort_values("coefficient"),
            x="coefficient", y="feature", orientation="h",
            color="coefficient",
            color_continuous_scale=["#34d399", "#f43f5e"],
            template=PLOTLY_TEMPLATE,
            title="Linear Regression Coefficients",
        )
        fig_coef.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_coef, use_container_width=True, theme=None)

    # ── Actual vs Predicted scatter ───────────────────────────────
    st.markdown("### Actual vs Predicted — Max Heart Rate")
    y_test  = np.array(lr_met["y_test"])
    y_pred  = np.array(lr_met["y_pred"])
    y_pred_r = np.array(rr_met["y_pred"])

    scatter_df = pd.DataFrame({
        "Actual":         y_test,
        "Linear Pred":    y_pred,
        "Ridge Pred":     y_pred_r,
        "Residual (Lin)": y_test - y_pred,
    })

    col_c, col_d = st.columns(2)
    with col_c:
        fig_ap = go.Figure()
        fig_ap.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            name="Linear Regression",
            marker=dict(color="#14b8a6", size=7, opacity=0.7),
        ))
        fig_ap.add_trace(go.Scatter(
            x=y_test, y=y_pred_r, mode="markers",
            name="Ridge Regression",
            marker=dict(color="#38bdf8", size=7, opacity=0.7),
        ))
        # Perfect prediction line
        mn, mx = int(y_test.min()), int(y_test.max())
        fig_ap.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            name="Perfect Prediction",
            line=dict(color="#64748b", dash="dash"),
        ))
        fig_ap.update_layout(
            xaxis_title="Actual Max Heart Rate",
            yaxis_title="Predicted Max Heart Rate",
            template=PLOTLY_TEMPLATE, height=400,
            title="Actual vs Predicted",
        )
        st.plotly_chart(fig_ap, use_container_width=True, theme=None)

    with col_d:
        # Residual distribution
        fig_res = px.histogram(
            scatter_df, x="Residual (Lin)", nbins=20,
            color_discrete_sequence=["#14b8a6"],
            template=PLOTLY_TEMPLATE,
            title="Residual Distribution (Linear Regression)",
            labels={"Residual (Lin)": "Residual (Actual − Predicted)"},
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="#f43f5e")
        st.plotly_chart(fig_res, use_container_width=True, theme=None)

    # ── Regression formula ────────────────────────────────────────
    st.markdown("### Regression Equation")
    intercept = round(lr_reg.intercept_, 3)
    terms = " + ".join(
        f"({row['coefficient']:.3f} × {row['feature']})"
        for _, row in coefs.iterrows()
    )
    formula = f"max_heart_rate = {intercept} + {terms}"
    st.code(formula, language="text")

    st.markdown(
        '<div class="disclaimer">ℹ️ This regression model is for academic demonstration of the regression '
        'mining technique. Results are not for clinical use.</div>',
        unsafe_allow_html=True,
    )

