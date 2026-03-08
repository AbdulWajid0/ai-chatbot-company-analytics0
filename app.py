"""
AI Business Intelligence Chatbot — Main Application
=====================================================
A Streamlit-based conversational BI assistant powered by
Google Gemini API for natural language company data analysis.

Run: streamlit run app.py
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import APP_TITLE, APP_ICON, PAGE_LAYOUT, GOOGLE_API_KEY
from modules.data_loader import load_all_datasets, get_dataset_summary
from modules.nlp_engine import initialize_gemini, get_chat_session, send_message, clean_response_text
from modules.intent_handler import handle_intent
from modules.insight_generator import generate_executive_summary, generate_quick_insights, get_suggested_questions
from modules.report_generator import generate_report
from modules.forecasting import forecast_revenue, get_seasonal_pattern, get_growth_metrics
from modules.visualizer import create_chart

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI BI Chatbot",
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Premium Modern Theme
# ============================================================
st.markdown("""
<style>
    /* === MAIN BACKGROUND === */
    .stApp {
        background: radial-gradient(circle at top left, #1a1a2e 0%, #16213e 50%, #0f172a 100%);
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }

    /* === SIDEBAR STYLING === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(0, 212, 170, 0.1);
        box-shadow: 2px 0 10px rgba(0,0,0,0.2);
    }
    
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
    }

    /* === HEADER === */
    .main-header {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 30px 40px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    .main-header h1 {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8em;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        margin: 12px 0 0 0;
        font-size: 1.15em;
        font-weight: 400;
    }

    /* === KPI CARDS (Glassmorphism) === */
    .kpi-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 22px 20px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        height: 100%;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 242, 254, 0.4);
        box-shadow: 0 12px 25px rgba(0, 242, 254, 0.15);
    }
    .kpi-value {
        font-size: 2em;
        font-weight: 800;
        background: linear-gradient(135deg, #00D4AA, #4FACFE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.9em;
        font-weight: 500;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* === CHAT MESSAGES === */
    .chat-user {
        background: rgba(79, 172, 254, 0.15);
        border: 1px solid rgba(79, 172, 254, 0.3);
        border-radius: 18px 18px 0 18px;
        padding: 16px 22px;
        margin: 15px 0 15px auto;
        color: #f8fafc;
        max-width: 85%;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        font-size: 1.05em;
        line-height: 1.5;
    }
    .chat-bot {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px 18px 18px 0;
        padding: 16px 22px;
        margin: 15px auto 15px 0;
        color: #e2e8f0;
        max-width: 90%;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        font-size: 1.05em;
        line-height: 1.6;
    }

    /* === INSIGHT BOX === */
    .insight-box {
        background: linear-gradient(135deg, rgba(250, 204, 21, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
        border-left: 4px solid #FACC15;
        border-radius: 0 12px 12px 0;
        padding: 18px 22px;
        margin: 15px 0;
        color: #fef08a;
        font-size: 1em;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* === EXPANDER/ACCORDION === */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        color: #e2e8f0 !important;
    }

    /* === BUTTONS === */
    div.stButton > button {
        background: rgba(0, 242, 254, 0.1) !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
        color: #4FACFE !important;
        border-radius: 100px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    }
    div.stButton > button:hover {
        background: rgba(0, 242, 254, 0.2) !important;
        border-color: #00F2FE !important;
        transform: translateY(-2px) !important;
        color: #fff !important;
        box-shadow: 0 6px 15px rgba(0, 242, 254, 0.2) !important;
    }
    
    /* === TAB STYLING === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(255,255,255,0.05);
        border-bottom: none;
        color: #94a3b8;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(47, 65, 86, 0.8) !important;
        color: #4FACFE !important;
        border-top: 2px solid #00F2FE !important;
    }

    /* === DATA TABLES === */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* === METRICS === */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #f1f5f9 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #94a3b8 !important;
    }

    /* === CHAT INPUT === */
    .stChatInputContainer {
        border-radius: 24px !important;
        padding-bottom: 20px !important;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = None
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = None
    if "page" not in st.session_state:
        st.session_state.page = "chat"

init_session_state()


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    datasets = load_all_datasets()
    return datasets

datasets = load_data()
st.session_state.datasets = datasets


# ============================================================
# GEMINI INITIALIZATION
# ============================================================
def init_gemini():
    if st.session_state.gemini_model is None:
        model = initialize_gemini()
        if model:
            st.session_state.gemini_model = model
            dataset_summary = get_dataset_summary(datasets)
            st.session_state.chat_session = get_chat_session(model, dataset_summary)

init_gemini()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🤖 AI BI Chatbot")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        ["💬 Chat", "📊 Dashboard", "🔮 Forecast", "📄 About"],
        label_visibility="collapsed"
    )
    st.session_state.page = page

    st.markdown("---")

    # API Status
    if st.session_state.gemini_model:
        st.success("✅ Gemini API Connected")
    else:
        st.error("❌ Gemini API Not Connected")
        st.info("Set your API key in the `.env` file")

    # Dataset Status
    st.markdown("### 📁 Loaded Datasets")
    for name, df in datasets.items():
        st.markdown(f"- **{name.title()}**: {len(df):,} records")

    if not datasets:
        st.warning("⚠️ No datasets found! Run `python generate_data.py` first.")

    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.markdown("""
    - Ask about **sales, HR, or finance** data
    - Request **charts, trends, rankings**
    - Ask for **KPIs and forecasts**
    - Download **PDF reports**
    """)


# ============================================================
# MAIN CONTENT AREA
# ============================================================

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Business Intelligence Chatbot</h1>
    <p>Ask me anything about your company data — sales, HR, finance</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE: CHAT
# ============================================================
if st.session_state.page == "💬 Chat":

    # --- Suggested Questions ---
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        suggestions = get_suggested_questions()
        cols = st.columns(2)
        for i, question in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(question, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.rerun()

    # --- Chat History ---
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            st.markdown(f'<div class="chat-user">{content}</div>', unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f'<div class="chat-bot">✨ <strong>BI Assistant</strong><br><br>{content.get("text", "")}</div>', unsafe_allow_html=True)

                # Display chart if present
                if content.get("chart"):
                    st.plotly_chart(content["chart"], use_container_width=True)

                # Display data table if present
                if content.get("result_df") is not None:
                    with st.expander("📋 View Data Table", expanded=False):
                        st.dataframe(content["result_df"], use_container_width=True)

                # Display AI insight if present
                if content.get("insight"):
                    st.markdown(f'<div class="insight-box">💡 <strong>Executive Insight:</strong><br>{content["insight"]}</div>', unsafe_allow_html=True)

                # PDF Report download
                if content.get("report_path"):
                    with open(content["report_path"], "rb") as f:
                        st.download_button(
                            "📥 Download Executive Report (PDF)",
                            data=f,
                            file_name=os.path.basename(content["report_path"]),
                            mime="application/pdf"
                        )

    # --- Chat Input ---
    user_query = st.chat_input("Ask me about your company data...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

        if st.session_state.chat_session is None:
            bot_response = {
                "text": "⚠️ Gemini API is not connected. Please set your API key in the `.env` file and restart the app.",
            }
        else:
            with st.spinner("🔍 Analyzing your data..."):
                # Send to Gemini
                raw_response, parsed_json = send_message(st.session_state.chat_session, user_query)

                # Process intent and get results
                response = handle_intent(
                    datasets, parsed_json, raw_response,
                    st.session_state.gemini_model, user_query
                )

                # Build bot response
                bot_response = {
                    "text": response["response_text"],
                    "chart": response["chart"],
                    "result_df": response["result_df"],
                    "insight": response["insight"],
                    "report_path": None,
                }

                # Generate PDF report if there are results
                if response["result_df"] is not None and len(response["result_df"]) > 0:
                    try:
                        report_path = generate_report(
                            user_query,
                            response["result_df"],
                            response.get("insight", ""),
                        )
                        bot_response["report_path"] = report_path
                    except Exception:
                        pass

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.rerun()


# ============================================================
# PAGE: DASHBOARD
# ============================================================
elif st.session_state.page == "📊 Dashboard":

    if not datasets:
        st.warning("⚠️ No datasets loaded. Run `python generate_data.py` first.")
    else:
        # --- KPI Cards ---
        if "sales" in datasets:
            sales = datasets["sales"]
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">${sales['revenue'].sum()/1_000_000:.1f}M</div>
                    <div class="kpi-label">Total Revenue</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">${sales['profit'].sum()/1_000_000:.1f}M</div>
                    <div class="kpi-label">Total Profit</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{sales['profit_margin'].mean():.1f}%</div>
                    <div class="kpi-label">Avg Profit Margin</div>
                </div>""", unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{sales['quantity'].sum():,}</div>
                    <div class="kpi-label">Units Sold</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

        # --- Executive Summary ---
        st.markdown("### 📋 Executive Summary")
        summary = generate_executive_summary(datasets)
        st.markdown(summary)

        st.markdown("---")

        # --- Quick Charts (Now with Tabs) ---
        st.markdown("### 📊 Key Visualizations")

        if "sales" in datasets:
            sales = datasets["sales"]

            tab1, tab2, tab3 = st.tabs(["💰 Revenue Trends", "🛒 Product & Category", "🌍 Regional Performance"])
            
            with tab1:
                monthly = sales.groupby("year_month", as_index=False)["revenue"].sum()
                monthly = monthly.rename(columns={"year_month": "Period"})
                fig1 = create_chart(monthly, "line", "Monthly Revenue Trend")
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)

            with tab2:
                col_a, col_b = st.columns(2)
                with col_a:
                    top_products = sales.groupby("product", as_index=False)["revenue"].sum()
                    top_products = top_products.sort_values("revenue", ascending=False).head(5)
                    fig2 = create_chart(top_products, "bar", "Top 5 Products")
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True)
                with col_b:
                    cat_revenue = sales.groupby("category", as_index=False)["revenue"].sum()
                    fig3 = create_chart(cat_revenue, "pie", "Revenue by Category")
                    if fig3:
                        st.plotly_chart(fig3, use_container_width=True)

            with tab3:
                region_rev = sales.groupby("region", as_index=False)["revenue"].sum()
                fig4 = create_chart(region_rev, "bar", "Revenue by Region")
                if fig4:
                    st.plotly_chart(fig4, use_container_width=True)

        # --- Quick Insights ---
        st.markdown("---")
        st.markdown("### 💡 Quick Insights")
        insights = generate_quick_insights(datasets)
        for insight in insights:
            st.markdown(f"- {insight}")


# ============================================================
# PAGE: FORECAST
# ============================================================
elif st.session_state.page == "🔮 Forecast":

    st.markdown("### 🔮 Revenue Forecast")

    if "sales" not in datasets:
        st.warning("⚠️ Sales data not loaded.")
    else:
        sales = datasets["sales"]

        # Forecast
        forecast_periods = st.slider("Forecast periods (months ahead):", 1, 6, 3)
        forecast_df = forecast_revenue(sales, periods=forecast_periods)

        if forecast_df is not None:
            # Forecast Chart
            fig = create_chart(forecast_df[["Date", "Revenue", "Type"]], "line", "Revenue Forecast")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Growth Metrics
            metrics = get_growth_metrics(sales)
            if metrics:
                st.markdown("### 📈 Growth Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    mom = metrics.get("mom_growth", 0)
                    st.metric("Month-over-Month", f"{mom:+.1f}%",
                              delta=f"{mom:.1f}%")
                with col2:
                    qoq = metrics.get("qoq_growth", 0)
                    st.metric("Quarter-over-Quarter", f"{qoq:+.1f}%",
                              delta=f"{qoq:.1f}%")
                with col3:
                    yoy = metrics.get("yoy_growth", 0)
                    st.metric("Year-over-Year", f"{yoy:+.1f}%",
                              delta=f"{yoy:.1f}%")

            # Seasonal Pattern
            st.markdown("---")
            st.markdown("### 🌊 Seasonal Pattern")
            seasonal = get_seasonal_pattern(sales)
            if seasonal is not None:
                fig = create_chart(
                    seasonal[["month_name", "avg_revenue"]].rename(columns={"month_name": "Month", "avg_revenue": "Avg Revenue"}),
                    "bar", "Average Revenue by Month"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                st.dataframe(seasonal, use_container_width=True)


# ============================================================
# PAGE: ABOUT
# ============================================================
elif st.session_state.page == "📄 About":
    st.markdown("""
    ### About This Project

    **AI Chatbot with Company Data Analysis Capability** is a Conversational Business Intelligence Assistant
    built as an internship project.

    #### 🏗 Architecture
    | Layer | Purpose | Technology |
    |-------|---------|-----------|
    | **Layer 1** | Chat Interface | Streamlit |
    | **Layer 2** | NLP Processing | Google Gemini API |
    | **Layer 3** | Analytics Engine | Pandas, NumPy |
    | **Layer 4** | Visualization | Plotly, Matplotlib |
    | **Layer 5** | Report Generation | FPDF2 |

    #### 🤖 Capabilities
    - Natural language query processing
    - Dynamic data analysis (filter, aggregate, rank, trend)
    - Auto-generated interactive charts
    - Revenue forecasting with seasonal analysis
    - PDF report generation
    - Executive KPI dashboards

    #### 👥 Team
    Built by a team of 9 interns.

    #### 📊 Datasets
    - **Sales Data**: Product sales, revenue, profit, regions, channels
    - **HR Data**: Employee records, performance, attrition, departments
    - **Finance Data**: Budget vs actual, department expenses, variance
    """)
