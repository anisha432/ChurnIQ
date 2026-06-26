"""
ChurnIQ — AI-Powered Customer Churn Analytics Platform
Section 1: Customer Churn Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib
import shap
import warnings

from utils import (
    load_telco_data, clean_data, engineer_features,
    preprocess_for_training, train_logistic_regression,
    train_random_forest, evaluate_model, get_feature_importance,
    select_best_model, save_artifacts, load_artifacts,
    prepare_single_prediction, get_risk_category, get_confidence_score,
    calculate_kpis, generate_business_insights, profile_data, get_column_details
)

warnings.filterwarnings('ignore')

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="ChurnIQ — AI Churn Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CUSTOM CSS =====================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --primary: #6C5CE7;
    --primary-light: #A29BFE;
    --secondary: #00CEC9;
    --accent: #FD79A8;
    --success: #00B894;
    --warning: #FDCB6E;
    --danger: #E17055;
    --critical: #D63031;
    --bg-dark: #0B0B1A;
    --bg-card: rgba(255, 255, 255, 0.04);
    --bg-card-hover: rgba(255, 255, 255, 0.08);
    --border: rgba(255, 255, 255, 0.08);
    --text: #E8E8F0;
    --text-muted: #7F8C9B;
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #0B0B1A 0%, #1a1a3e 50%, #0B0B1A 100%);
    color: var(--text);
}

/* Hide default Streamlit elements */
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f2e 0%, #1a1a3e 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* Sidebar Nav Items */
.css-1d391kg {
    padding: 4px 12px !important;
}
.css-1d391kg a {
    border-radius: 10px !important;
    padding: 10px 16px !important;
    transition: all 0.3s ease !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}
.css-1d391kg a:hover {
    background: rgba(108, 92, 231, 0.15) !important;
    border-left: 3px solid var(--primary) !important;
}
.css-1d391kg a[aria-current="page"] {
    background: linear-gradient(135deg, rgba(108, 92, 231, 0.25), rgba(0, 206, 201, 0.15)) !important;
    border-left: 3px solid var(--primary) !important;
    font-weight: 600 !important;
}

/* Glass Card */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    transition: all 0.3s ease;
}
.glass-card:hover {
    background: var(--bg-card-hover);
    border-color: rgba(108, 92, 231, 0.3);
    transform: translateY(-2px);
}

/* KPI Card */
.kpi-card {
    background: linear-gradient(135deg, rgba(108, 92, 231, 0.12), rgba(0, 206, 201, 0.08));
    backdrop-filter: blur(20px);
    border: 1px solid rgba(108, 92, 231, 0.2);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: all 0.4s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    border-radius: 16px 16px 0 0;
}
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(108, 92, 231, 0.4);
    box-shadow: 0 12px 40px rgba(108, 92, 231, 0.15);
}

.kpi-value {
    font-size: 28px;
    font-weight: 800;
    background: linear-gradient(135deg, #fff, var(--primary-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
    line-height: 1.2;
}
.kpi-label {
    font-size: 12px;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Risk Badges */
.risk-low {
    background: rgba(0, 184, 148, 0.15);
    border: 1px solid rgba(0, 184, 148, 0.4);
    color: #00B894;
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 20px;
    font-weight: 700;
    text-align: center;
}
.risk-medium {
    background: rgba(253, 203, 110, 0.15);
    border: 1px solid rgba(253, 203, 110, 0.4);
    color: #FDCB6E;
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 20px;
    font-weight: 700;
    text-align: center;
}
.risk-high {
    background: rgba(225, 112, 85, 0.15);
    border: 1px solid rgba(225, 112, 85, 0.4);
    color: #E17055;
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 20px;
    font-weight: 700;
    text-align: center;
}
.risk-critical {
    background: rgba(214, 48, 49, 0.15);
    border: 1px solid rgba(214, 48, 49, 0.4);
    color: #D63031;
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 20px;
    font-weight: 700;
    text-align: center;
    animation: pulse-risk 2s infinite;
}

@keyframes pulse-risk {
    0%, 100% { box-shadow: 0 0 0 0 rgba(214, 48, 49, 0.4); }
    50% { box-shadow: 0 0 20px 4px rgba(214, 48, 49, 0.2); }
}

/* Animated gradient text */
.gradient-text {
    font-size: 48px;
    font-weight: 900;
    background: linear-gradient(135deg, #6C5CE7, #00CEC9, #FD79A8, #6C5CE7);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradient-shift 4s ease infinite;
}

@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Section Title */
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-subtitle {
    font-size: 14px;
    color: var(--text-muted);
    margin-bottom: 24px;
}

/* Custom Button */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), #5A4BD1) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    transition: all 0.3s ease !important;
    font-size: 15px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #7C6CF7, #6C5CE7) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(108, 92, 231, 0.35) !important;
}

/* Progress Bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
    border-radius: 10px !important;
}

/* Dataframe Styling */
.dataframe {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
.dataframe th {
    background: rgba(108, 92, 231, 0.2) !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--border) !important;
}
.dataframe td {
    color: var(--text) !important;
    border-bottom: 1px solid var(--border) !important;
}
.dataframe tr:hover {
    background: rgba(108, 92, 231, 0.08) !important;
}

/* Selectbox & Input */
.stSelectbox, stNumberInput, stTextInput {
    border-radius: 10px !important;
}

/* Metric Card */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

/* Insight Card */
.insight-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 12px;
    transition: all 0.3s ease;
    border-left: 4px solid var(--primary);
}
.insight-card:hover {
    background: var(--bg-card-hover);
    border-left-color: var(--secondary);
}
.insight-critical { border-left-color: var(--critical) !important; }
.insight-high { border-left-color: var(--danger) !important; }
.insight-medium { border-left-color: var(--warning) !important; }
.insight-strategic { border-left-color: var(--secondary) !important; }

/* Confidence Gauge */
.gauge-container {
    text-align: center;
    padding: 20px;
}
.gauge-value {
    font-size: 42px;
    font-weight: 800;
    line-height: 1;
}
.gauge-label {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 4px;
}

/* Divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 24px 0;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(108, 92, 231, 0.3); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(108, 92, 231, 0.5); }

/* Fade-in animation */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in {
    animation: fadeInUp 0.6s ease forwards;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(108, 92, 231, 0.2) !important;
    color: #fff !important;
}

/* Prediction Result Card */
.prediction-result {
    background: linear-gradient(135deg, rgba(108, 92, 231, 0.1), rgba(0, 206, 201, 0.05));
    border: 1px solid rgba(108, 92, 231, 0.3);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
}

/* Logo text in sidebar */
.sidebar-logo {
    font-size: 22px;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary-light), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding: 8px 0;
}

/* Model status badge */
.status-ready {
    background: rgba(0, 184, 148, 0.15);
    border: 1px solid rgba(0, 184, 148, 0.4);
    color: #00B894;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
.status-not-ready {
    background: rgba(225, 112, 85, 0.15);
    border: 1px solid rgba(225, 112, 85, 0.4);
    color: #E17055;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ===================== PLOTLY TEMPLATE =====================
PLOTLY_TEMPLATE = "plotly_dark"
px.defaults.template = PLOTLY_TEMPLATE
px.defaults.color_continuous_scale = ["#6C5CE7", "#00CEC9"]

CUSTOM_PLOTLY_LAYOUT = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': {'color': '#E8E8F0', 'family': 'Inter'},
    'xaxis': {'gridcolor': 'rgba(255,255,255,0.06)', 'zerolinecolor': 'rgba(255,255,255,0.06)'},
    'yaxis': {'gridcolor': 'rgba(255,255,255,0.06)', 'zerolinecolor': 'rgba(255,255,255,0.06)'},
    'colorway': ['#6C5CE7', '#00CEC9', '#FD79A8', '#FDCB6E', '#E17055', '#00B894', '#74B9FF', '#A29BFE'],
}


def apply_plotly_layout(fig):
    """Apply custom dark layout to a Plotly figure."""
    fig.update_layout(**CUSTOM_PLOTLY_LAYOUT)
    fig.update_layout(
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.1)',
            font=dict(size=12, color='#E8E8F0')
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


# ===================== SESSION STATE =====================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'data_loaded': False,
        'data': None,
        'cleaned_data': None,
        'featured_data': None,
        'model_trained': False,
        'model': None,
        'scaler': None,
        'feature_columns': None,
        'numerical_cols': None,
        'training_results': None,
        'best_model_name': None,
        'feature_importance': None,
        'X_test': None,
        'y_test': None,
        'prediction_made': False,
        'last_prediction': None,
        'last_risk': None,
        'last_confidence': None,
        'shap_values': None,
        'shap_calculated': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def load_model_artifacts():
    """Try to load existing model artifacts."""
    artifacts = load_artifacts()
    if artifacts is not None:
        st.session_state.model = artifacts['model']
        st.session_state.scaler = artifacts['scaler']
        st.session_state.feature_columns = artifacts['feature_columns']
        st.session_state.numerical_cols = artifacts['numerical_cols']
        st.session_state.training_results = artifacts['results']
        st.session_state.best_model_name = artifacts['best_model_name']
        st.session_state.model_trained = True
        return True
    return False


# Try to load existing artifacts on startup
load_model_artifacts()


# ===================== SIDEBAR =====================
def render_sidebar():
    """Render the sidebar navigation and status."""
    with st.sidebar:
        # Logo
        st.markdown('<div class="sidebar-logo">🔮 ChurnIQ</div>', unsafe_allow_html=True)
        st.caption("AI-Powered Churn Analytics")
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Navigation
        page = st.radio(
            "Navigation",
            [
                "🏠  Home",
                "📊  Data Overview",
                "📈  Exploratory Analysis",
                "🤖  Model Training",
                "🔮  Customer Prediction",
                "🎯  SHAP Explainability",
                "💡  Business Insights",
            ],
            label_visibility="collapsed",
            index=0
        )

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Model Status
        st.subheader("Model Status")
        if st.session_state.model_trained:
            st.markdown(
                f'<span class="status-ready">✅ Model Ready</span>',
                unsafe_allow_html=True
            )
            st.caption(f"Best: {st.session_state.best_model_name}")
        else:
            st.markdown(
                '<span class="status-not-ready">⏳ Not Trained</span>',
                unsafe_allow_html=True
            )
            st.caption("Go to Model Training to train")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Quick Stats
        if st.session_state.data is not None:
            st.subheader("Dataset Info")
            st.caption(f"Rows: {len(st.session_state.data):,}")
            st.caption(f"Columns: {len(st.session_state.data.columns)}")
            if 'Churn' in st.session_state.data.columns:
                churn_col = st.session_state.data['Churn']
                churn_rate = churn_col.mean() if churn_col.dtype in [int, float] else \
                    (churn_col == 1).mean() if churn_col.dtype == int else \
                    (churn_col == 'Yes').mean()
                st.caption(f"Churn Rate: {churn_rate:.1%}")

        # Footer
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.caption("ChurnIQ v1.0")
        st.caption("Section 1: Churn Prediction")

        # Map page names
        page_map = {
            "🏠  Home": "Home",
            "📊  Data Overview": "Data Overview",
            "📈  Exploratory Analysis": "EDA",
            "🤖  Model Training": "Model Training",
            "🔮  Customer Prediction": "Prediction",
            "🎯  SHAP Explainability": "SHAP",
            "💡  Business Insights": "Insights",
        }
        return page_map.get(page, "Home")


# ===================== LANDING PAGE =====================
def show_landing_page():
    """Show the premium landing page."""
    st.markdown(
        '<div style="text-align:center; padding: 40px 0 20px 0;">'
        '<div class="gradient-text">ChurnIQ</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align:center; font-size:18px; color:#7F8C9B; max-width:600px; margin:0 auto 40px auto;">'
        'AI-Powered Customer Churn Prediction & Explainability Platform. '
        'Predict, understand, and prevent customer churn with machine learning.'
        '</p>',
        unsafe_allow_html=True
    )

    # Feature Cards
    col1, col2, col3 = st.columns(3)
    features = [
        ("🤖", "Smart Prediction", "Logistic Regression & Random Forest models trained on Telco data with automated best model selection."),
        ("🎯", "SHAP Explainability", "Understand WHY each customer is predicted to churn with SHAP force plots and feature contributions."),
        ("💡", "Business Insights", "AI-generated actionable recommendations to reduce churn and increase customer lifetime value."),
    ]

    for col, (icon, title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(
                f'<div class="glass-card">'
                f'<div style="font-size:36px; margin-bottom:12px;">{icon}</div>'
                f'<div style="font-size:16px; font-weight:700; color:#fff; margin-bottom:8px;">{title}</div>'
                f'<div style="font-size:13px; color:#7F8C9B; line-height:1.6;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown('<div style="height:40px"></div>', unsafe_allow_html=True)

    # If model is trained, show quick metrics
    if st.session_state.model_trained and st.session_state.training_results:
        st.markdown('<div class="section-title">🏆 Model Performance Summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Best model results from last training</div>', unsafe_allow_html=True)

        results = st.session_state.training_results
        best = max(results, key=lambda x: x['ROC-AUC'])

        kpi_cols = st.columns(4)
        metrics_display = [
            ("Accuracy", f"{best['Accuracy']:.1%}"),
            ("Precision", f"{best['Precision']:.1%}"),
            ("Recall", f"{best['Recall']:.1%}"),
            ("ROC-AUC", f"{best['ROC-AUC']:.1%}"),
        ]

        for col, (label, value) in zip(kpi_cols, metrics_display):
            with col:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div class="kpi-value">{value}</div>'
                    f'<div class="kpi-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown(f'<p style="text-align:center; color:#7F8C9B; font-size:13px; margin-top:12px;">'
                    f'Best Model: <strong style="color:#A29BFE;">{best["Model"]}</strong> | '
                    f'CV AUC: {best["CV Mean AUC"]:.4f} ± {best["CV Std AUC"]:.4f}</p>',
                    unsafe_allow_html=True)

    # CTA
    st.markdown('<div style="height:40px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;">'
        '<p style="color:#7F8C9B; font-size:14px;">Navigate using the sidebar to explore Data, Train Models, and Make Predictions</p>'
        '</div>',
        unsafe_allow_html=True
    )


# ===================== DATA OVERVIEW =====================
def show_data_overview():
    """Show dataset overview with profiling."""
    # Load data if not loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading Telco Customer Churn dataset..."):
            try:
                df = load_telco_data()
                st.session_state.data = df
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
                return

    df = st.session_state.data

    # Clean if not cleaned
    if st.session_state.cleaned_data is None:
        with st.spinner("Cleaning data..."):
            st.session_state.cleaned_data = clean_data(df)
            st.session_state.featured_data = engineer_features(st.session_state.cleaned_data)

    df_clean = st.session_state.cleaned_data
    df_featured = st.session_state.featured_data

    st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Telco Customer Churn — Data profiling and summary statistics</div>', unsafe_allow_html=True)

    # KPI Cards
    profile = profile_data(df_clean)
    kpi_cols = st.columns(5)
    kpi_items = [
        ("Total Rows", f"{profile['total_rows']:,}"),
        ("Total Columns", str(profile['total_columns'])),
        ("Numeric Cols", str(profile['numeric_cols'])),
        ("Categorical Cols", str(profile['categorical_cols'])),
        ("Missing Values", str(profile['missing_values'])),
    ]
    for col, (label, value) in zip(kpi_cols, kpi_items):
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value">{value}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📋 Cleaned Data", "📋 Column Details"])

    with tab1:
        st.markdown("**Original Dataset (first 100 rows)**")
        st.dataframe(df.head(100), use_container_width=True, height=400)

    with tab2:
        st.markdown("**Cleaned & Feature-Engineered Dataset (first 100 rows)**")
        st.dataframe(df_featured.head(100), use_container_width=True, height=400)

    with tab3:
        st.markdown("**Column-wise Details**")
        col_details = get_column_details(df_clean)
        st.dataframe(col_details, use_container_width=True, height=400)

    # Download button
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    csv = df_featured.to_csv(index=False)
    st.download_button(
        label="📥 Download Cleaned Dataset (CSV)",
        data=csv,
        file_name="telco_churn_cleaned.csv",
        mime="text/csv"
    )


# ===================== EXPLORATORY DATA ANALYSIS =====================
def show_eda():
    """Show comprehensive EDA with interactive charts."""
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please go to Data Overview first to load the dataset.")
        return

    if st.session_state.featured_data is None:
        st.session_state.cleaned_data = clean_data(st.session_state.data)
        st.session_state.featured_data = engineer_features(st.session_state.cleaned_data)

    df = st.session_state.cleaned_data

    st.markdown('<div class="section-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Interactive visualizations uncovering churn patterns</div>', unsafe_allow_html=True)

    # ---- Row 1: Churn Distribution + KPIs ----
    col1, col2 = st.columns([1, 2])

    with col1:
        # Churn Distribution Pie
        churn_counts = df['Churn'].value_counts()
        labels_map = {0: 'Retained', 1: 'Churned'}
        fig_pie = go.Figure(data=[go.Pie(
            labels=[labels_map.get(k, k) for k in churn_counts.index],
            values=churn_counts.values,
            hole=0.65,
            marker_colors=['#00B894', '#E17055'],
            textinfo='label+percent',
            textfont=dict(size=13, color='#E8E8F0'),
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percent: %{percent}<extra></extra>'
        )])
        fig_pie.update_layout(
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.1, font=dict(color='#E8E8F0')),
            margin=dict(t=10, b=40, l=10, r=10),
            annotations=[dict(
                text=f'{churn_counts[1]/len(df)*100:.1f}%',
                x=0.5, y=0.5, font=dict(size=28, color='#E17055', weight='bold'),
                showarrow=False
            )]
        )
        fig_pie = apply_plotly_layout(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Key Metrics
        kpis = calculate_kpis(df)
        metric_cols = st.columns(4)
        metric_items = [
            ("Churn Rate", kpis['Overall Churn Rate'], '#E17055'),
            ("Avg Tenure", kpis['Avg Tenure (months)'], '#6C5CE7'),
            ("Avg Monthly $", kpis['Avg Monthly Charges'], '#00CEC9'),
            ("Revenue at Risk", kpis['Monthly Revenue at Risk'], '#FD79A8'),
        ]
        for col, (label, value, color) in zip(metric_cols, metric_items):
            with col:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div class="kpi-value" style="-webkit-text-fill-color:{color};">{value}</div>'
                    f'<div class="kpi-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # ---- Row 2: Tenure & Charges Distribution ----
    col1, col2 = st.columns(2)

    with col1:
        fig_tenure = px.histogram(
            df, x='tenure', color='Churn',
            color_discrete_map={0: '#6C5CE7', 1: '#E17055'},
            nbins=40,
            title='Tenure Distribution by Churn',
            labels={'tenure': 'Tenure (months)', 'Churn': 'Status'}
        )
        fig_tenure.update_layout(legend_title_text='Status')
        fig_tenure.data[0].name = 'Retained'
        fig_tenure.data[1].name = 'Churned'
        fig_tenure = apply_plotly_layout(fig_tenure)
        st.plotly_chart(fig_tenure, use_container_width=True)

    with col2:
        fig_charges = px.violin(
            df, y='MonthlyCharges', x='Churn',
            color='Churn',
            color_discrete_map={0: '#6C5CE7', 1: '#E17055'},
            box=True,
            title='Monthly Charges Distribution by Churn',
            labels={'MonthlyCharges': 'Monthly Charges ($)', 'Churn': 'Status'}
        )
        fig_charges.data[0].name = 'Retained'
        fig_charges.data[1].name = 'Churned'
        fig_charges = apply_plotly_layout(fig_charges)
        st.plotly_chart(fig_charges, use_container_width=True)

    # ---- Row 3: Contract & Internet Service ----
    col1, col2 = st.columns(2)

    with col1:
        contract_churn = df.groupby('Contract')['Churn'].mean().reset_index()
        contract_churn['Churn'] = contract_churn['Churn'] * 100
        fig_contract = px.bar(
            contract_churn, x='Contract', y='Churn',
            title='Churn Rate by Contract Type',
            labels={'Contract': 'Contract Type', 'Churn': 'Churn Rate (%)'},
            color='Churn',
            color_continuous_scale=['#00B894', '#E17055'],
            text_auto='.1f'
        )
        fig_contract.update_traces(textposition='outside', marker_color='#6C5CE7')
        fig_contract.update_layout(showlegend=False)
        fig_contract = apply_plotly_layout(fig_contract)
        st.plotly_chart(fig_contract, use_container_width=True)

    with col2:
        internet_churn = df.groupby('InternetService')['Churn'].mean().reset_index()
        internet_churn['Churn'] = internet_churn['Churn'] * 100
        fig_internet = px.bar(
            internet_churn, x='InternetService', y='Churn',
            title='Churn Rate by Internet Service',
            labels={'InternetService': 'Internet Service', 'Churn': 'Churn Rate (%)'},
            color='Churn',
            color_continuous_scale=['#00B894', '#E17055'],
            text_auto='.1f'
        )
        fig_internet.update_traces(textposition='outside', marker_color='#00CEC9')
        fig_internet.update_layout(showlegend=False)
        fig_internet = apply_plotly_layout(fig_internet)
        st.plotly_chart(fig_internet, use_container_width=True)

    # ---- Row 4: Payment Method & Gender ----
    col1, col2 = st.columns(2)

    with col1:
        pay_churn = df.groupby('PaymentMethod')['Churn'].mean().reset_index()
        pay_churn['Churn'] = pay_churn['Churn'] * 100
        pay_churn = pay_churn.sort_values('Churn', ascending=True)
        fig_pay = px.bar(
            pay_churn, x='Churn', y='PaymentMethod',
            orientation='h',
            title='Churn Rate by Payment Method',
            labels={'PaymentMethod': 'Payment Method', 'Churn': 'Churn Rate (%)'},
            color='Churn',
            color_continuous_scale=['#00B894', '#E17055'],
            text_auto='.1f'
        )
        fig_pay.update_traces(textposition='outside', marker_color='#FD79A8')
        fig_pay.update_layout(showlegend=False)
        fig_pay = apply_plotly_layout(fig_pay)
        st.plotly_chart(fig_pay, use_container_width=True)

    with col2:
        # Senior Citizen & Partner Analysis
        fig_demo = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "bar"}, {"type": "bar"}]],
            subplot_titles=("Churn by Senior Citizen", "Churn by Partner Status")
        )
        for i, col_name in enumerate(['SeniorCitizen', 'Partner']):
            temp = df.groupby(col_name)['Churn'].mean().reset_index()
            temp['Churn'] = temp['Churn'] * 100
            labels = {0: 'No', 1: 'Yes'} if col_name == 'Partner' else {0: 'No', 1: 'Yes'}
            temp[col_name] = temp[col_name].map(labels)
            fig_demo.add_trace(
                go.Bar(
                    x=temp[col_name], y=temp['Churn'],
                    text=[f'{v:.1f}%' for v in temp['Churn']],
                    textposition='outside',
                    marker_color=['#6C5CE7', '#00CEC9'],
                    showlegend=False
                ),
                row=1, col=i+1
            )
        fig_demo.update_layout(title_text='Demographic Analysis', title_x=0.5)
        fig_demo = apply_plotly_layout(fig_demo)
        st.plotly_chart(fig_demo, use_container_width=True)

    # ---- Row 5: Services Heatmap ----
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔥 Service-Level Churn Analysis</div>', unsafe_allow_html=True)

    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    service_churn = []
    for svc in services:
        for val in df[svc].unique():
            subset = df[df[svc] == val]
            if len(subset) > 10:
                service_churn.append({
                    'Service': svc.replace('Online', '').replace('Streaming', ''),
                    'Value': val,
                    'Churn Rate': subset['Churn'].mean() * 100,
                    'Count': len(subset)
                })
    svc_df = pd.DataFrame(service_churn)

    fig_svc = px.bar(
        svc_df[svc_df['Value'] == 'Yes'].sort_values('Churn Rate', ascending=True),
        x='Churn Rate', y='Service',
        orientation='h',
        title='Churn Rate When Service = Yes',
        labels={'Service': '', 'Churn Rate': 'Churn Rate (%)'},
        color='Churn Rate',
        color_continuous_scale=['#00B894', '#E17055'],
        text_auto='.1f'
    )
    fig_svc.update_traces(textposition='outside')
    fig_svc = apply_plotly_layout(fig_svc)
    st.plotly_chart(fig_svc, use_container_width=True)

    # ---- Row 6: Correlation Heatmap ----
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔗 Correlation Heatmap</div>', unsafe_allow_html=True)

    corr_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    corr_matrix = df[corr_cols].corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale=['#0B0B1A', '#6C5CE7', '#FD79A8'],
        title='Feature Correlation Matrix',
        labels=dict(x='', y=''),
        aspect='auto'
    )
    fig_corr.update_xaxes(side='bottom')
    fig_corr = apply_plotly_layout(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ---- Row 7: Tenure Groups ----
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        df_temp = df.copy()
        df_temp['tenure_group'] = pd.cut(df_temp['tenure'], bins=[-1, 12, 24, 48, 60, 999],
                                         labels=['0-12', '13-24', '25-48', '49-60', '60+'])
        tg_churn = df_temp.groupby('tenure_group')['Churn'].agg(['mean', 'count']).reset_index()
        tg_churn.columns = ['Tenure Group', 'Churn Rate', 'Count']
        tg_churn['Churn Rate'] = tg_churn['Churn Rate'] * 100

        fig_tg = px.bar(
            tg_churn, x='Tenure Group', y='Churn Rate',
            title='Churn Rate by Tenure Group',
            labels={'Tenure Group': '', 'Churn Rate': 'Churn Rate (%)'},
            color='Churn Rate',
            color_continuous_scale=['#00B894', '#E17055'],
            text_auto='.1f'
        )
        fig_tg.update_traces(textposition='outside')
        fig_tg = apply_plotly_layout(fig_tg)
        st.plotly_chart(fig_tg, use_container_width=True)

    with col2:
        # Total Charges vs Tenure scatter
        fig_scatter = px.scatter(
            df.sample(min(1000, len(df)), random_state=42),
            x='tenure', y='TotalCharges', color='Churn',
            color_discrete_map={0: '#6C5CE7', 1: '#E17055'},
            title='Total Charges vs Tenure',
            labels={'tenure': 'Tenure (months)', 'TotalCharges': 'Total Charges ($)'},
            opacity=0.6,
            trendline='ols'
        )
        fig_scatter.data[0].name = 'Retained'
        fig_scatter.data[1].name = 'Churned'
        fig_scatter = apply_plotly_layout(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)


# ===================== MODEL TRAINING =====================
def show_model_training():
    """Show model training interface with progress and results."""
    st.markdown('<div class="section-title">🤖 Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Train and compare Logistic Regression & Random Forest models</div>', unsafe_allow_html=True)

    if st.session_state.data_loaded and st.session_state.featured_data is None:
        st.session_state.cleaned_data = clean_data(st.session_state.data)
        st.session_state.featured_data = engineer_features(st.session_state.cleaned_data)

    # Train Button
    col1, col2 = st.columns([1, 3])
    with col1:
        train_btn = st.button("🚀 Train Models", type="primary", use_container_width=True)

    with col2:
        if st.session_state.model_trained:
            st.markdown(
                f'<div style="padding:8px 0; color:#00B894; font-weight:600;">'
                f'✅ Models already trained. Best: {st.session_state.best_model_name} — '
                f'Click "Train Models" to retrain.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="padding:8px 0; color:#7F8C9B;">'
                'Click "Train Models" to start the training pipeline. This will automatically '
                'load data, clean, engineer features, train, evaluate, and save the best model.</div>',
                unsafe_allow_html=True
            )

    if train_btn:
        # Step-by-step training with progress
        progress_text = st.empty()
        progress_bar = st.progress(0)

        # Step 1: Load Data
        progress_text.markdown("📦 **Step 1/7:** Loading dataset...")
        progress_bar.progress(10)
        if not st.session_state.data_loaded:
            try:
                df = load_telco_data()
                st.session_state.data = df
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Failed to load: {e}")
                return
        df = st.session_state.data
        time.sleep(0.3)

        # Step 2: Clean
        progress_text.markdown("🧹 **Step 2/7:** Cleaning data...")
        progress_bar.progress(25)
        df_clean = clean_data(df)
        st.session_state.cleaned_data = df_clean
        time.sleep(0.3)

        # Step 3: Feature Engineering
        progress_text.markdown("⚙️ **Step 3/7:** Engineering features...")
        progress_bar.progress(40)
        df_featured = engineer_features(df_clean)
        st.session_state.featured_data = df_featured
        time.sleep(0.3)

        # Step 4: Preprocess
        progress_text.markdown("🔄 **Step 4/7:** Encoding, scaling, splitting...")
        progress_bar.progress(55)
        X_train, X_test, y_train, y_test, scaler, feature_cols, num_cols = \
            preprocess_for_training(df_featured)
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        time.sleep(0.3)

        # Step 5: Train Models
        progress_text.markdown("🤖 **Step 5/7:** Training Logistic Regression & Random Forest...")
        progress_bar.progress(70)
        lr_model = train_logistic_regression(X_train, y_train)
        rf_model = train_random_forest(X_train, y_train)
        time.sleep(0.3)

        # Step 6: Evaluate
        progress_text.markdown("📊 **Step 6/7:** Evaluating models...")
        progress_bar.progress(85)
        lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        all_results = [lr_results, rf_results]
        time.sleep(0.3)

        # Step 7: Save
        progress_text.markdown("💾 **Step 7/7:** Selecting best model & saving artifacts...")
        progress_bar.progress(95)

        best = select_best_model(all_results)
        best_name = best['Model']
        if best_name == "Logistic Regression":
            best_model = lr_model
        else:
            best_model = rf_model

        save_artifacts(best_model, scaler, feature_cols, num_cols, all_results, best_name)

        # Update session state
        st.session_state.model = best_model
        st.session_state.scaler = scaler
        st.session_state.feature_columns = feature_cols
        st.session_state.numerical_cols = num_cols
        st.session_state.training_results = all_results
        st.session_state.best_model_name = best_name
        st.session_state.model_trained = True
        st.session_state.shap_calculated = False

        # Feature importance
        lr_fi = get_feature_importance(lr_model, feature_cols, "Logistic Regression")
        rf_fi = get_feature_importance(rf_model, feature_cols, "Random Forest")
        st.session_state.feature_importance = pd.concat([lr_fi, rf_fi], ignore_index=True)

        progress_bar.progress(100)
        progress_text.markdown("✅ **Training Complete!**")
        time.sleep(0.5)
        progress_text.empty()
        progress_bar.empty()

        st.success(f"🎉 Best Model: **{best_name}** | ROC-AUC: {best['ROC-AUC']:.4f} | F1: {best['F1 Score']:.4f}")
        st.rerun()

    # Show results if trained
    if st.session_state.model_trained and st.session_state.training_results:
        results = st.session_state.training_results

        # Comparison Table
        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Model Comparison</div>', unsafe_allow_html=True)

        comparison_data = []
        for r in results:
            comparison_data.append({
                'Model': r['Model'],
                'Accuracy': f"{r['Accuracy']:.4f}",
                'Precision': f"{r['Precision']:.4f}",
                'Recall': f"{r['Recall']:.4f}",
                'F1 Score': f"{r['F1 Score']:.4f}",
                'ROC-AUC': f"{r['ROC-AUC']:.4f}",
                'CV AUC (Mean±Std)': f"{r['CV Mean AUC']:.4f} ± {r['CV Std AUC']:.4f}",
                'Best': '⭐' if r['Model'] == st.session_state.best_model_name else ''
            })

        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

        # KPI Cards for best model
        best = max(results, key=lambda x: x['ROC-AUC'])
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        kpi_cols = st.columns(5)
        kpi_items = [
            ("Accuracy", f"{best['Accuracy']:.1%}", '#6C5CE7'),
            ("Precision", f"{best['Precision']:.1%}", '#00CEC9'),
            ("Recall", f"{best['Recall']:.1%}", '#FD79A8'),
            ("F1 Score", f"{best['F1 Score']:.1%}", '#FDCB6E'),
            ("ROC-AUC", f"{best['ROC-AUC']:.1%}", '#00B894'),
        ]
        for col, (label, value, color) in zip(kpi_cols, kpi_items):
            with col:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div class="kpi-value" style="-webkit-text-fill-color:{color};">{value}</div>'
                    f'<div class="kpi-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # ROC Curves
        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 ROC Curves</div>', unsafe_allow_html=True)

        fig_roc = go.Figure()
        colors = ['#6C5CE7', '#00CEC9']
        for r, color in zip(results, colors):
            fig_roc.add_trace(go.Scatter(
                x=r['FPR'], y=r['TPR'],
                mode='lines',
                name=f"{r['Model']} (AUC={r['ROC-AUC']:.4f})",
                line=dict(color=color, width=2.5)
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='rgba(255,255,255,0.2)', width=1)
        ))
        fig_roc.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05])
        )
        fig_roc = apply_plotly_layout(fig_roc)
        st.plotly_chart(fig_roc, use_container_width=True)

        # Confusion Matrices
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔲 Confusion Matrices</div>', unsafe_allow_html=True)

        cm_cols = st.columns(2)
        for col, r in zip(cm_cols, results):
            with col:
                cm = r['Confusion Matrix']
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale=['#0B0B1A', '#6C5CE7'],
                    labels=dict(x='Predicted', y='Actual'),
                    x=['Retained', 'Churned'],
                    y=['Retained', 'Churned'],
                    title=r['Model']
                )
                fig_cm.update_xaxes(side='bottom')
                fig_cm = apply_plotly_layout(fig_cm)
                st.plotly_chart(fig_cm, use_container_width=True)

        # Feature Importance
        if st.session_state.feature_importance is not None:
            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🏆 Feature Importance (Top 20)</div>', unsafe_allow_html=True)

            fi = st.session_state.feature_importance[
                st.session_state.feature_importance['model'] == st.session_state.best_model_name
            ].head(20)

            fig_fi = px.bar(
                fi.sort_values('importance', ascending=True),
                x='importance', y='feature',
                orientation='h',
                title=f'Top 20 Features — {st.session_state.best_model_name}',
                labels={'feature': '', 'importance': 'Importance'},
                color='importance',
                color_continuous_scale=['#6C5CE7', '#FD79A8']
            )
            fig_fi.update_traces(marker_color='#6C5CE7')
            fig_fi.update_layout(showlegend=False)
            fig_fi = apply_plotly_layout(fig_fi)
            st.plotly_chart(fig_fi, use_container_width=True)


# ===================== CUSTOMER PREDICTION =====================
def show_prediction():
    """Show customer prediction form with risk categorization."""
    st.markdown('<div class="section-title">🔮 Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Enter customer details to predict churn probability and risk level</div>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first by going to **Model Training**.")
        st.info("The model needs to be trained before making predictions. Navigate to '🤖 Model Training' in the sidebar.")
        return

    # Prediction Form
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('**Customer Information**')
    st.caption('Fill in the customer details below to get a churn prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div style="font-size:13px; font-weight:600; color:#A29BFE; margin:12px 0 8px 0;">DEMOGRAPHICS</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"], key="pred_gender")
        senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="pred_senior")
        partner = st.selectbox("Partner", ["No", "Yes"], key="pred_partner")
        dependents = st.selectbox("Dependents", ["No", "Yes"], key="pred_dependents")

    with col2:
        st.markdown('<div style="font-size:13px; font-weight:600; color:#00CEC9; margin:12px 0 8px 0;">SERVICES</div>', unsafe_allow_html=True)
        tenure = st.slider("Tenure (months)", 0, 72, 12, key="pred_tenure")
        phone = st.selectbox("Phone Service", ["Yes", "No"], key="pred_phone")
        multiline = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="pred_multiline")
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="pred_internet")

    with col3:
        st.markdown('<div style="font-size:13px; font-weight:600; color:#FD79A8; margin:12px 0 8px 0;">BILLING</div>', unsafe_allow_html=True)
        online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="pred_onlinesec")
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="pred_onlinebackup")
        device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="pred_deviceprot")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=300.0, value=70.0, step=0.1, key="pred_monthly")

    col1, col2, col3 = st.columns(3)

    with col1:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="pred_techsupport")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="pred_streamingtv")

    with col2:
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="pred_streamingmovies")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="pred_contract")

    with col3:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="pred_paperless")
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key="pred_payment"
        )

    # Calculate TotalCharges
    total_charges = monthly_charges * max(tenure, 1)

    # Predict Button
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Churn", type="primary", use_container_width=True, key="predict_btn")

    st.markdown('</div>', unsafe_allow_html=True)  # Close glass-card

    if predict_btn:
        # Build input dict
        input_dict = {
            'gender': gender,
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiline,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_prot,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
        }

        # Prepare and predict
        X_pred = prepare_single_prediction(
            input_dict,
            st.session_state.feature_columns,
            st.session_state.numerical_cols,
            st.session_state.scaler
        )

        probability = st.session_state.model.predict_proba(X_pred)[0][1]
        prediction = st.session_state.model.predict(X_pred)[0]
        risk_label, risk_color, risk_icon = get_risk_category(probability)
        confidence = get_confidence_score(probability)

        st.session_state.last_prediction = {
            'probability': probability,
            'prediction': prediction,
            'risk': risk_label,
            'risk_color': risk_color,
            'risk_icon': risk_icon,
            'confidence': confidence,
            'input': input_dict
        }
        st.session_state.prediction_made = True
        st.session_state.shap_calculated = False
        st.rerun()

    # Show Prediction Results
    if st.session_state.prediction_made and st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        prob = pred['probability']
        risk = pred['risk']
        risk_color = pred['risk_color']
        risk_icon = pred['risk_icon']
        confidence = pred['confidence']
        prediction = pred['prediction']

        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

        # Main Result Card
        st.markdown(
            f'<div class="prediction-result">'
            f'<div style="font-size:14px; color:#7F8C9B; margin-bottom:8px; text-transform:uppercase; letter-spacing:1px;">Prediction Result</div>'
            f'<div style="font-size:52px; font-weight:900; color:{risk_color}; line-height:1;">{risk_icon} {risk.upper()}</div>'
            f'<div style="font-size:16px; color:#E8E8F0; margin:8px 0;">'
            f'{"This customer is LIKELY to CHURN" if prediction == 1 else "This customer is LIKELY to STAY"}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Metrics Row
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        met_cols = st.columns(3)

        with met_cols[0]:
            # Churn Probability Gauge
            gauge_color = risk_color
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': 'Churn Probability', 'font': {'size': 13, 'color': '#7F8C9B'}},
                number={'suffix': '%', 'font': {'size': 36, 'color': gauge_color, 'weight': 'bold'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#7F8C9B', 'tickfont': {'size': 10}},
                    'bar': {'color': gauge_color, 'thickness': 0.7},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'steps': [
                        {'range': [0, 20], 'color': 'rgba(0,184,148,0.1)'},
                        {'range': [20, 40], 'color': 'rgba(253,203,110,0.1)'},
                        {'range': [40, 65], 'color': 'rgba(225,112,85,0.1)'},
                        {'range': [65, 100], 'color': 'rgba(214,48,49,0.1)'},
                    ],
                    'threshold': {
                        'line': {'color': '#fff', 'width': 2},
                        'thickness': 0.8,
                        'value': prob * 100
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, b=20, l=20, r=20),
                height=200
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with met_cols[1]:
            # Confidence Score
            conf_color = '#00B894' if confidence > 0.6 else '#FDCB6E' if confidence > 0.3 else '#E17055'
            st.markdown(
                f'<div class="kpi-card" style="height:180px; display:flex; flex-direction:column; justify-content:center;">'
                f'<div class="kpi-label">Confidence Score</div>'
                f'<div class="kpi-value" style="-webkit-text-fill-color:{conf_color}; font-size:42px;">{confidence:.1%}</div>'
                f'<div style="font-size:11px; color:#7F8C9B; margin-top:4px;">Distance from 0.5 threshold</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        with met_cols[2]:
            # Risk Category
            st.markdown(
                f'<div class="kpi-card" style="height:180px; display:flex; flex-direction:column; justify-content:center;">'
                f'<div class="kpi-label">Risk Category</div>'
                f'<div style="font-size:42px; margin:8px 0;">{risk_icon}</div>'
                f'<div style="font-size:22px; font-weight:800; color:{risk_color};">{risk}</div>'
                f'<div style="font-size:11px; color:#7F8C9B; margin-top:4px;">'
                f'{"Immediate action needed" if risk in ["Critical", "High"] else "Monitor closely" if risk == "Medium" else "Low priority"}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Probability Bar
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('**Probability Breakdown**')
        stay_pct = (1 - prob) * 100
        churn_pct = prob * 100
        st.markdown(
            f'<div style="display:flex; height:32px; border-radius:8px; overflow:hidden; margin:12px 0;">'
            f'<div style="width:{stay_pct}%; background:linear-gradient(90deg, #00B894, #55EFC4); '
            f'display:flex; align-items:center; justify-content:center; color:#0B0B1A; font-weight:700; font-size:13px;">'
            f'Stay {stay_pct:.1f}%</div>'
            f'<div style="width:{churn_pct}%; background:linear-gradient(90deg, #E17055, #D63031); '
            f'display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; font-size:13px;">'
            f'Churn {churn_pct:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Key Factors
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔑 Key Risk Factors Identified</div>', unsafe_allow_html=True)

        inp = pred['input']
        factors = []

        if inp['Contract'] == 'Month-to-month':
            factors.append(('📝 Month-to-month contract', 'Highest risk contract type — 3x more likely to churn', '#D63031'))
        if inp['InternetService'] == 'Fiber optic':
            factors.append(('🌐 Fiber optic internet', 'Higher churn than DSL — possible service quality issue', '#E17055'))
        if inp['PaymentMethod'] == 'Electronic check':
            factors.append(('💳 Electronic check payment', 'Highest risk payment method', '#E17055'))
        if inp['tenure'] <= 12:
            factors.append(('🆕 Low tenure (≤12 months)', f'New customers have ~50% churn rate in first year', '#D63031'))
        if inp['TechSupport'] == 'No':
            factors.append(('🛡️ No tech support', 'Customers without tech support churn significantly more', '#FDCB6E'))
        if inp['OnlineSecurity'] == 'No':
            factors.append(('🔒 No online security', 'Missing security service increases churn risk', '#FDCB6E'))
        if inp['SeniorCitizen'] == 1:
            factors.append(('👴 Senior citizen', 'Seniors have higher churn rates — need special attention', '#FDCB6E'))
        if inp['PaperlessBilling'] == 'Yes':
            factors.append(('📄 Paperless billing', 'Slight increase in churn — less tangible connection', '#7F8C9B'))

        if not factors:
            factors.append(('✅ No major risk factors', 'This customer profile shows low-risk characteristics', '#00B894'))

        factor_cols = st.columns(min(len(factors), 4))
        for i, (title, desc, color) in enumerate(factors[:8]):
            with factor_cols[i % len(factor_cols)]:
                st.markdown(
                    f'<div class="glass-card" style="border-left:3px solid {color}; padding:16px;">'
                    f'<div style="font-size:13px; font-weight:600; color:#fff; margin-bottom:4px;">{title}</div>'
                    f'<div style="font-size:11px; color:#7F8C9B; line-height:1.5;">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )


# ===================== SHAP EXPLAINABILITY =====================
def show_shap():
    """Show SHAP-based model explainability."""
    st.markdown('<div class="section-title">🎯 SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Understand how each feature contributes to model predictions</div>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first.")
        return

    # Calculate SHAP values
    if not st.session_state.shap_calculated:
        with st.spinner("Computing SHAP values (this may take a moment)..."):
            try:
                model = st.session_state.model
                X_test = st.session_state.X_test

                if X_test is None:
                    st.warning("Test data not available. Please retrain the model.")
                    return

                # Use a sample for SHAP to keep it fast
                sample_size = min(200, len(X_test))
                X_sample = X_test.sample(sample_size, random_state=42)

                # Determine explainer type
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                elif hasattr(model, 'coef_'):
                    explainer = shap.LinearExplainer(model, X_sample)
                else:
                    explainer = shap.KernelExplainer(model.predict_proba, X_sample.iloc[:50])

                shap_values = explainer(X_sample)

                # For models with predict_proba, SHAP returns per-class values
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Get churn class (class 1)

                st.session_state.shap_values = shap_values
                st.session_state.shap_sample = X_sample
                st.session_state.shap_explainer = explainer
                st.session_state.shap_calculated = True
            except Exception as e:
                st.error(f"SHAP computation failed: {e}")
                return

    shap_values = st.session_state.shap_values
    X_sample = st.session_state.shap_sample

    # ---- SHAP Summary Plot ----
    st.markdown('<div class="section-title" style="font-size:18px;">📊 SHAP Summary Plot (Beeswarm)</div>', unsafe_allow_html=True)
    st.caption('Shows feature impact on predictions. Each dot is a customer. Color = feature value (red=high, blue=low). Position = SHAP value (right=increases churn).')

    fig_summary = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    fig_summary.patch.set_facecolor('#0B0B1A')
    ax = fig_summary.axes[0]
    ax.set_facecolor('#0B0B1A')
    ax.tick_params(colors='#E8E8F0')
    ax.xaxis.label.set_color('#E8E8F0')
    ax.yaxis.label.set_color('#E8E8F0')
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.1))
    st.pyplot(fig_summary, bbox_inches='tight')
    plt.close()

    # ---- SHAP Bar Plot ----
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:18px;">📊 Mean |SHAP Value| (Feature Importance)</div>', unsafe_allow_html=True)
    st.caption('Average absolute SHAP value per feature — shows which features matter most overall.')

    fig_bar = plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    fig_bar.patch.set_facecolor('#0B0B1A')
    ax = fig_bar.axes[0]
    ax.set_facecolor('#0B0B1A')
    ax.tick_params(colors='#E8E8F0')
    ax.xaxis.label.set_color('#E8E8F0')
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.1))
    st.pyplot(fig_bar, bbox_inches='tight')
    plt.close()

    # ---- SHAP Waterfall for individual prediction ----
    if st.session_state.prediction_made and st.session_state.last_prediction:
        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size:18px;">💧 SHAP Waterfall Plot (Last Prediction)</div>', unsafe_allow_html=True)
        st.caption('Shows how each feature pushed the prediction from the base value to the final churn probability.')

        try:
            fig_waterfall = plt.figure(figsize=(12, 8))
            shap.plots.waterfall(shap_values[0], max_display=15, show=False)
            fig_waterfall.patch.set_facecolor('#0B0B1A')
            ax = fig_waterfall.axes[0]
            ax.set_facecolor('#0B0B1A')
            ax.tick_params(colors='#E8E8F0')
            for spine in ax.spines.values():
                spine.set_color((1, 1, 1, 0.1))
            st.pyplot(fig_waterfall, bbox_inches='tight')
            plt.close()
        except Exception as e:
            st.info(f"Could not generate waterfall plot: {e}")

    # ---- SHAP Force Plot ----
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:18px;">⚡ SHAP Force Plot (Sample Predictions)</div>', unsafe_allow_html=True)
    st.caption('Visual explanation of individual predictions. Red = increases churn, Blue = decreases churn.')

    try:
        # Use matplotlib-based force plot (more compatible with Streamlit)
        fig_force, ax_force = plt.subplots(figsize=(14, 3))
        shap.force_plot(
            st.session_state.shap_explainer.expected_value if hasattr(st.session_state.shap_explainer, 'expected_value') else 0,
            shap_values.values[:5] if hasattr(shap_values, 'values') else shap_values[:5],
            X_sample.iloc[:5],
            show=False,
            matplotlib=True
        )
        fig_force.patch.set_facecolor('#0B0B1A')
        st.pyplot(fig_force, bbox_inches='tight')
        plt.close()
    except Exception as e:
        st.info(f"Force plot not available for this model type. Summary and bar plots shown above.")
        plt.close('all')

    # ---- Dependence Plots for Top Features ----
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:18px;">📈 SHAP Dependence Plots (Top Features)</div>', unsafe_allow_html=True)
    st.caption('Shows how SHAP values change with feature values — reveals non-linear relationships.')

    # Get top 4 features by mean absolute SHAP
    if hasattr(shap_values, 'values'):
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    top_features_idx = np.argsort(mean_abs_shap)[-4:][::-1]
    top_features = [X_sample.columns[i] for i in top_features_idx]

    dep_cols = st.columns(2)
    for idx, (col_widget, feat_name) in enumerate(zip(dep_cols, top_features)):
        with col_widget:
            try:
                fig_dep = plt.figure(figsize=(8, 5))
                feat_idx = list(X_sample.columns).index(feat_name)
                shap.dependence_plot(
                    feat_idx, shap_values.values if hasattr(shap_values, 'values') else shap_values,
                    X_sample, interaction_index='auto', show=False
                )
                fig_dep.patch.set_facecolor('#0B0B1A')
                ax = fig_dep.axes[0]
                ax.set_facecolor('#0B0B1A')
                ax.tick_params(colors='#E8E8F0')
                ax.xaxis.label.set_color('#E8E8F0')
                ax.yaxis.label.set_color('#E8E8F0')
                ax.title.set_color('#E8E8F0')
                for spine in ax.spines.values():
                    spine.set_color((1, 1, 1, 0.1))
                st.pyplot(fig_dep, bbox_inches='tight')
                plt.close()
            except Exception as e:
                plt.close('all')
                st.caption(f"Could not generate dependence plot for {feat_name}")


# ===================== BUSINESS INSIGHTS =====================
def show_insights():
    """Show AI-generated business insights and recommendations."""
    st.markdown('<div class="section-title">💡 AI-Powered Business Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Automated actionable recommendations based on data analysis and model interpretation</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the data first by going to Data Overview.")
        return

    if st.session_state.cleaned_data is None:
        st.session_state.cleaned_data = clean_data(st.session_state.data)
        st.session_state.featured_data = engineer_features(st.session_state.cleaned_data)

    df = st.session_state.cleaned_data

    # Get feature importance
    fi_df = pd.DataFrame()
    if st.session_state.feature_importance is not None:
        fi_df = st.session_state.feature_importance[
            st.session_state.feature_importance['model'] == st.session_state.best_model_name
        ] if st.session_state.best_model_name else st.session_state.feature_importance

    # Generate insights
    insights = generate_business_insights(df, fi_df)

    # Priority legend
    st.markdown(
        '<div style="display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap;">'
        '<span style="background:rgba(214,48,49,0.15); color:#D63031; padding:4px 12px; border-radius:6px; font-size:12px; font-weight:600;">🔴 Critical</span>'
        '<span style="background:rgba(225,112,85,0.15); color:#E17055; padding:4px 12px; border-radius:6px; font-size:12px; font-weight:600;">🟠 High</span>'
        '<span style="background:rgba(253,203,110,0.15); color:#FDCB6E; padding:4px 12px; border-radius:6px; font-size:12px; font-weight:600;">🟡 Medium</span>'
        '<span style="background:rgba(0,206,201,0.15); color:#00CEC9; padding:4px 12px; border-radius:6px; font-size:12px; font-weight:600;">🔵 Strategic</span>'
        '</div>',
        unsafe_allow_html=True
    )

    # Display insights
    for insight in insights:
        priority_class = f"insight-{insight['priority'].lower()}"
        st.markdown(
            f'<div class="insight-card {priority_class}">'
            f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">'
            f'<span style="font-size:24px;">{insight["icon"]}</span>'
            f'<span style="font-size:16px; font-weight:700; color:#fff;">{insight["title"]}</span>'
            f'<span style="margin-left:auto; font-size:11px; font-weight:600; color:var(--text-muted); '
            f'background:rgba(255,255,255,0.05); padding:2px 10px; border-radius:10px;">{insight["priority"]}</span>'
            f'</div>'
            f'<div style="font-size:13px; color:#B0B0C8; line-height:1.7; padding-left:34px;">{insight["detail"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # KPI Summary Dashboard
    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Executive KPI Summary</div>', unsafe_allow_html=True)

    kpis = calculate_kpis(df)
    kpi_cols = st.columns(4)
    kpi_display = [
        ("Overall Churn Rate", kpis['Overall Churn Rate'], '#E17055'),
        ("Monthly Contract Churn", kpis['Monthly Contract Churn'], '#D63031'),
        ("New Customer Churn", kpis['New Customer Churn (≤12mo)'], '#FD79A8'),
        ("Monthly Revenue at Risk", kpis['Monthly Revenue at Risk'], '#FDCB6E'),
    ]
    for col, (label, value, color) in zip(kpi_cols, kpi_display):
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value" style="-webkit-text-fill-color:{color}; font-size:24px;">{value}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    kpi_cols2 = st.columns(4)
    kpi_display2 = [
        ("Total Customers", kpis['Total Customers'], '#6C5CE7'),
        ("Avg Monthly Charges", kpis['Avg Monthly Charges'], '#00CEC9'),
        ("Senior Churn Rate", kpis['Senior Citizen Churn'], '#A29BFE'),
        ("Annual Revenue at Risk", kpis['Annual Revenue at Risk'], '#D63031'),
    ]
    for col, (label, value, color) in zip(kpi_cols2, kpi_display2):
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value" style="-webkit-text-fill-color:{color}; font-size:24px;">{value}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Churn by Segment Chart
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📉 Churn Rate by Customer Segment</div>', unsafe_allow_html=True)

    segments = {
        'Contract Type': df.groupby('Contract')['Churn'].mean() * 100,
        'Internet Service': df.groupby('InternetService')['Churn'].mean() * 100,
        'Payment Method': df.groupby('PaymentMethod')['Churn'].mean() * 100,
        'Senior Citizen': df.groupby('SeniorCitizen')['Churn'].mean() * 100,
    }

    seg_cols = st.columns(2)
    for col, (seg_name, seg_data) in zip(seg_cols, segments.items()):
        with col:
            seg_df = seg_data.reset_index()
            seg_df.columns = ['Segment', 'Churn Rate']
            seg_df = seg_df.sort_values('Churn Rate', ascending=True)
            fig_seg = px.bar(
                seg_df, x='Churn Rate', y='Segment',
                orientation='h',
                title=seg_name,
                labels={'Segment': '', 'Churn Rate': 'Churn Rate (%)'},
                text_auto='.1f'
            )
            fig_seg.update_traces(
                textposition='outside',
                marker_color='#6C5CE7'
            )
            fig_seg.update_layout(showlegend=False)
            fig_seg = apply_plotly_layout(fig_seg)
            st.plotly_chart(fig_seg, use_container_width=True)

    # Download Insights Report
    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Generate text report
    report_lines = ["ChurnIQ — Business Insights Report\n", "=" * 50 + "\n"]
    for insight in insights:
        report_lines.append(f"\n[{insight['priority']}] {insight['title']}")
        report_lines.append(f"{insight['detail']}\n")
    report_lines.append("\n" + "=" * 50)
    report_lines.append("\nKey KPIs:")
    for k, v in kpis.items():
        report_lines.append(f"  {k}: {v}")

    report_text = "\n".join(report_lines)
    st.download_button(
        label="📥 Download Insights Report (TXT)",
        data=report_text,
        file_name="churniq_insights_report.txt",
        mime="text/plain"
    )


# ===================== MAIN APP =====================
def main():
    """Main application entry point."""
    page = render_sidebar()

    if page == "Home":
        show_landing_page()
    elif page == "Data Overview":
        show_data_overview()
    elif page == "EDA":
        show_eda()
    elif page == "Model Training":
        show_model_training()
    elif page == "Prediction":
        show_prediction()
    elif page == "SHAP":
        show_shap()
    elif page == "Insights":
        show_insights()


if __name__ == "__main__":
    main()