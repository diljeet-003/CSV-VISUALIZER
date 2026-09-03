import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_cleaner import clean_data
from utils.ml_models import apply_kmeans
from utils.insights import generate_insights
from utils.deep_learning import train_lstm
from utils.automl import run_regression, run_classification

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI CSV Visualizer",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# GLOBAL STYLES
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root & Background ── */
.stApp {
    background: #07090f;
    color: #e8eaf0;
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 4rem;
    max-width: 1300px;
}

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    color: #f0f2f8 !important;
    letter-spacing: -0.02em;
}

/* ── Hero title font ── */
.hero-title {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 38px !important;
    font-weight: 900 !important;
    letter-spacing: 0.04em !important;
    background: linear-gradient(135deg, #ffffff 0%, #a8b4ff 50%, #5b6df5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1017 !important;
    border-right: 1px solid #1e2330;
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
}

/* Sidebar radio container */
div[data-testid="stRadio"] > div {
    gap: 6px;
    display: flex;
    flex-direction: column;
}
div[data-testid="stRadio"] label {
    background: #12161f;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 11px 16px !important;
    color: #8892a4 !important;
    font-size: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all 0.2s ease;
    cursor: pointer;
}
div[data-testid="stRadio"] label:hover {
    background: #171d2b;
    border-color: #3d4fe0;
    color: #c8d0e8 !important;
}
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, #1a2060 0%, #111a40 100%);
    border-color: #4a5bf0;
    color: #a8b4ff !important;
}

/* ── Metric Cards ── */
div[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1e2535;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    transition: border-color 0.2s;
}
div[data-testid="metric-container"]:hover {
    border-color: #2e3d6e;
}
div[data-testid="metric-container"] label {
    color: #5a677d !important;
    font-size: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #c8d4f0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3d4fe0, #5b6df5) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 16px rgba(61, 79, 224, 0.25) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4a5bf0, #6b7df8) !important;
    box-shadow: 0 6px 22px rgba(61, 79, 224, 0.4) !important;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0px);
}

/* ── Download Button ── */
.stDownloadButton > button {
    background: transparent !important;
    color: #5b6df5 !important;
    border: 1px solid #2e3d6e !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: #0d1628 !important;
    border-color: #4a5bf0 !important;
    color: #a8b4ff !important;
}

/* ── File Uploader ── */
section[data-testid="stFileUploaderDropzone"] {
    background: #0d1117 !important;
    border: 1.5px dashed #1e2535 !important;
    border-radius: 14px !important;
    padding: 2.5rem !important;
    transition: border-color 0.2s ease;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #3d4fe0 !important;
    background: #0e1220 !important;
}
section[data-testid="stFileUploaderDropzone"] span {
    color: #8892a4 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div {
    background: #0d1117 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 10px !important;
    color: #c8d4f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stSelectbox"] > div > div:hover {
    border-color: #3d4fe0 !important;
}

/* ── Text Input ── */
div[data-testid="stTextInput"] input {
    background: #0d1117 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 10px !important;
    color: #c8d4f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 10px 14px !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #3d4fe0 !important;
    box-shadow: 0 0 0 3px rgba(61, 79, 224, 0.15) !important;
}

/* ── DataFrame ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid #1e2535 !important;
}
.stDataFrame th {
    background: #0d1117 !important;
    color: #5a677d !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.stDataFrame td {
    background: #090c13 !important;
    color: #c8d4f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}

/* ── Alerts ── */
.stSuccess > div {
    background: #071a12 !important;
    border: 1px solid #0d4a28 !important;
    border-radius: 10px !important;
    color: #4cde9a !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stInfo > div {
    background: #071220 !important;
    border: 1px solid #0d2d5a !important;
    border-radius: 10px !important;
    color: #5b8ef0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stWarning > div {
    background: #1a1205 !important;
    border: 1px solid #4a3200 !important;
    border-radius: 10px !important;
    color: #f0a940 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stError > div {
    background: #1a0707 !important;
    border: 1px solid #4a0d0d !important;
    border-radius: 10px !important;
    color: #f05555 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #3d4fe0 !important;
}

/* ── Divider ── */
hr {
    border-color: #1e2535 !important;
    margin: 1.5rem 0;
}

/* ── Section card ── */
.section-card {
    background: #0d1117;
    border: 1px solid #1e2535;
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
}

/* ── Section subheader ── */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3d4fe0;
    margin-bottom: 0.4rem;
}

/* ── Hero banner ── */
.hero-banner {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 2rem;
    padding: 1.75rem 2rem;
    background: linear-gradient(135deg, #0d1117 0%, #0f1428 60%, #101830 100%);
    border: 1px solid #1e2535;
    border-radius: 20px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(61,79,224,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(91,109,245,0.07) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-logo {
    width: 72px;
    height: 72px;
    flex-shrink: 0;
    border-radius: 18px;
    background: linear-gradient(135deg, #1a2060, #0d1535);
    border: 1px solid #2e3d6e;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 34px;
    box-shadow: 0 8px 24px rgba(61,79,224,0.3);
}
.hero-text {
    flex: 1;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #5b6df5;
    background: rgba(61,79,224,0.12);
    border: 1px solid rgba(61,79,224,0.25);
    border-radius: 20px;
    padding: 3px 10px;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 14px;
    color: #5a677d;
    margin-top: 6px;
    font-family: 'DM Sans', sans-serif;
    line-height: 1.5;
}

/* ── Plotly chart background ── */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("""
    <div style='padding: 0 0.5rem 1.5rem;'>
        <div style='font-family: Syne, sans-serif; font-size: 20px; font-weight: 800;
                    color: #f0f2f8; letter-spacing: -0.02em; margin-bottom: 4px;'>
            AI CSV Visualizer
        </div>
        <div style='font-size: 12px; color: #3d4fe0; font-family: DM Sans, sans-serif;
                    text-transform: uppercase; letter-spacing: 0.1em;'>
            Analytics Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:11px; color:#5a677d; text-transform:uppercase; "
                "letter-spacing:0.1em; margin-bottom:8px; font-family: DM Sans, sans-serif;'>Navigation</div>",
                unsafe_allow_html=True)

    page = st.radio(
        "Select Module",
        ["📊 Dashboard", "🤖 AI Chatbot", "🧬 Deep Learning", "🧠 AutoML"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e2535; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0d1117; border:1px solid #1e2535; border-radius:12px;
                padding:14px 16px;'>
        <div style='font-size:11px; color:#5a677d; text-transform:uppercase;
                    letter-spacing:0.1em; margin-bottom:8px; font-family: DM Sans, sans-serif;'>
            How it works
        </div>
        <div style='font-size:13px; color:#8892a4; line-height:1.7; font-family: DM Sans, sans-serif;'>
            1. Upload a CSV file<br>
            2. AI cleans &amp; clusters data<br>
            3. Explore visualizations<br>
            4. Run AutoML or LSTM
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# HEADER  —  hero banner with logo image + Orbitron title
# ==========================================

st.markdown("""
<div class='hero-banner'>
    <div class='hero-logo'>📊</div>
    <div class='hero-text'>
        <div class='hero-badge'>AI · ML · Deep Learning</div>
        <div class='hero-title'>AI CSV Visualizer</div>
        <div class='hero-subtitle'>
            Upload a dataset and instantly get insights, forecasts, and ML models.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Drop your CSV file here or click to browse",
    type=["csv"],
    help="Accepts .csv files up to 200MB"
)

# ==========================================
# AI CHART RECOMMENDATION
# ==========================================

def recommend_chart(df):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    recommendations = []
    if len(numeric_cols) >= 2:
        recommendations.append({"chart": "Scatter Plot", "x": numeric_cols[0], "y": numeric_cols[1]})
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        recommendations.append({"chart": "Bar Chart", "x": categorical_cols[0], "y": numeric_cols[0]})
    if len(numeric_cols) > 0:
        recommendations.append({"chart": "Histogram", "x": numeric_cols[0]})
    return recommendations

# ==========================================
# SIMPLE AI CHATBOT
# ==========================================

def simple_ai_chat(df, query):
    q = query.lower()
    if "columns" in q:
        return df.columns.tolist()
    elif "missing" in q:
        return df.isnull().sum()
    elif "mean" in q:
        return df.mean(numeric_only=True)
    elif "head" in q:
        return df.head()
    elif "describe" in q:
        return df.describe()
    elif "correlation" in q:
        return df.corr(numeric_only=True)
    else:
        return "Try: columns, missing, mean, head, describe, correlation"

# ==========================================
# PLOTLY DARK THEME HELPER
# ==========================================

def dark_chart_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Syne", size=15, color="#c8d4f0")) if title else {},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0a0d14",
        font=dict(family="DM Sans", color="#8892a4", size=12),
        xaxis=dict(
            gridcolor="#1a2030", linecolor="#1e2535",
            tickfont=dict(color="#5a677d"), title_font=dict(color="#8892a4")
        ),
        yaxis=dict(
            gridcolor="#1a2030", linecolor="#1e2535",
            tickfont=dict(color="#5a677d"), title_font=dict(color="#8892a4")
        ),
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)", bordercolor="#1e2535",
            borderwidth=1, font=dict(color="#8892a4")
        ),
        margin=dict(l=16, r=16, t=40 if title else 20, b=16),
    )
    fig.update_traces(marker_line_width=0)
    return fig

# ==========================================
# MAIN APP
# ==========================================

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.success("✓  Dataset loaded successfully")

    df = clean_data(df)
    df = apply_kmeans(df)

    # ─────────────── DASHBOARD ───────────────
    if page == "📊 Dashboard":

        # Dataset Preview
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Dataset Preview</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Syne,sans-serif; font-size:20px; font-weight:700; "
                    "color:#f0f2f8; margin-bottom:1rem;'>Raw Data</div>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.markdown("<br>", unsafe_allow_html=True)

        # Insights
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>AI Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Syne,sans-serif; font-size:20px; font-weight:700; "
                    "color:#f0f2f8; margin-bottom:1rem;'>Automated Insights</div>", unsafe_allow_html=True)
        generate_insights(df)
        st.markdown("</div>", unsafe_allow_html=True)

        # Charts
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Visualization</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Syne,sans-serif; font-size:20px; font-weight:700; "
                    "color:#f0f2f8; margin-bottom:1.5rem;'>AI-Recommended Charts</div>", unsafe_allow_html=True)

        recs = recommend_chart(df)
        if recs:
            cols = st.columns(min(len(recs), 2))
            for i, rec in enumerate(recs):
                with cols[i % 2]:
                    if rec["chart"] == "Scatter Plot":
                        fig = px.scatter(df, x=rec["x"], y=rec["y"],
                                         color_discrete_sequence=["#4a5bf0"])
                        fig = dark_chart_layout(fig, f"{rec['x']} vs {rec['y']}")
                        st.plotly_chart(fig, use_container_width=True)

                    elif rec["chart"] == "Bar Chart":
                        fig = px.bar(df, x=rec["x"], y=rec["y"],
                                     color_discrete_sequence=["#3d4fe0"])
                        fig = dark_chart_layout(fig, f"{rec['y']} by {rec['x']}")
                        st.plotly_chart(fig, use_container_width=True)

                    elif rec["chart"] == "Histogram":
                        fig = px.histogram(df, x=rec["x"],
                                           color_discrete_sequence=["#5b6df5"])
                        fig = dark_chart_layout(fig, f"Distribution of {rec['x']}")
                        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Download
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:DM Sans,sans-serif; font-size:14px; color:#8892a4; "
                    "margin-bottom:0.75rem;'>Export the cleaned and clustered dataset.</div>",
                    unsafe_allow_html=True)
        st.download_button(
            "⬇  Download Processed CSV",
            df.to_csv(index=False).encode(),
            "processed_data.csv",
            "text/csv"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────── AI CHATBOT ───────────────
    elif page == "🤖 AI Chatbot":

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Natural Language Query</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Syne,sans-serif; font-size:20px; font-weight:700; "
                    "color:#f0f2f8; margin-bottom:0.5rem;'>CSV AI Chatbot</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:13px; color:#5a677d; margin-bottom:1.25rem; "
                    "font-family:DM Sans,sans-serif;'>"
                    "Ask questions like: <em>columns</em>, <em>missing</em>, "
                    "<em>mean</em>, <em>describe</em>, <em>correlation</em>"
                    "</div>", unsafe_allow_html=True)

        user_query = st.text_input(
            "Your question",
            placeholder="e.g. show me the correlation matrix",
            label_visibility="collapsed"
        )

        if st.button("Generate AI Response"):
            if user_query:
                with st.spinner("Analyzing your data…"):
                    response = simple_ai_chat(df, user_query)
                st.success("Response ready")
                st.write(response)
            else:
                st.warning("Please enter a question first.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────── DEEP LEARNING ───────────────
    elif page == "🧬 Deep Learning":

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Time-Series Forecasting</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Syne,sans-serif; font-size:20px; font-weight:700; "
                    "color:#f0f2f8; margin-bottom:0.5rem;'>LSTM Neural Network</div>",
                    unsafe_allow_html=True)
        st.markdown("<div style='font-size:13px; color:#5a677d; margin-bottom:1.25rem; "
                    "font-family:DM Sans,sans-serif;'>"
                    "Train a Long Short-Term Memory model to forecast a numeric column."
                    "</div>", unsafe_allow_html=True)

        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        if len(numeric_cols) > 0:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                target_column = st.selectbox("Target column for prediction", numeric_cols)
            with col_b:
                st.markdown("<br>", unsafe_allow_html=True)
                run_model = st.button("Train Model")

            if run_model:
                with st.spinner("Training LSTM model…"):
                    try:
                        actual, predictions = train_lstm(df, target_column)
                        st.success("Model trained successfully")

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=actual.flatten(), name="Actual",
                            line=dict(color="#5b8ef0", width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            y=predictions.flatten(), name="Predicted",
                            line=dict(color="#f05580", width=2, dash="dot")
                        ))
                        fig = dark_chart_layout(fig, f"LSTM Forecast — {target_column}")
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Training error: {e}")
        else:
            st.info("No numeric columns found in the uploaded dataset.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────── AUTOML ───────────────
    elif page == "🧠 AutoML":

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Automated Machine Learning</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Syne,sans-serif; font-size:20px; font-weight:700; "
                    "color:#f0f2f8; margin-bottom:0.5rem;'>AutoML Training</div>",
                    unsafe_allow_html=True)
        st.markdown("<div style='font-size:13px; color:#5a677d; margin-bottom:1.25rem; "
                    "font-family:DM Sans,sans-serif;'>"
                    "Automatically compare multiple ML models and select the best performer."
                    "</div>", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([3, 2, 1])
        with col_a:
            target_column = st.selectbox("Target column", df.columns)
        with col_b:
            problem_type = st.selectbox("Problem type", ["Regression", "Classification"])
        with col_c:
            st.markdown("<br>", unsafe_allow_html=True)
            run_automl = st.button("Train AutoML")

        if run_automl:
            with st.spinner("Running AutoML pipeline…"):
                try:
                    if problem_type == "Regression":
                        best_model, results, predictions = run_regression(df, target_column)
                    else:
                        best_model, results, predictions = run_classification(df, target_column)

                    st.success("AutoML training completed")

                    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
                    st.markdown("<div style='font-size:12px; color:#5a677d; text-transform:uppercase; "
                                "letter-spacing:0.08em; margin-bottom:6px; font-family:DM Sans,sans-serif;'>"
                                "Best Model</div>", unsafe_allow_html=True)
                    st.write(best_model)

                    st.markdown("<div style='font-size:12px; color:#5a677d; text-transform:uppercase; "
                                "letter-spacing:0.08em; margin:1rem 0 6px; font-family:DM Sans,sans-serif;'>"
                                "Model Comparison</div>", unsafe_allow_html=True)
                    st.dataframe(results, use_container_width=True)

                    st.markdown("<div style='font-size:12px; color:#5a677d; text-transform:uppercase; "
                                "letter-spacing:0.08em; margin:1rem 0 6px; font-family:DM Sans,sans-serif;'>"
                                "Predictions (first 5 rows)</div>", unsafe_allow_html=True)
                    st.dataframe(predictions.head(), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"AutoML error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='background:#0d1117; border:1px dashed #1e2535; border-radius:16px;
                padding:3rem; text-align:center; margin-top:1rem;'>
        <div style='font-size:40px; margin-bottom:12px;'>📂</div>
        <div style='font-family:Syne,sans-serif; font-size:20px; font-weight:700;
                    color:#f0f2f8; margin-bottom:8px;'>No dataset loaded</div>
        <div style='font-size:14px; color:#5a677d; font-family:DM Sans,sans-serif;'>
            Upload a CSV file above to unlock all AI features.
        </div>
    </div>
    """, unsafe_allow_html=True)
