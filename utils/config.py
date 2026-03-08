"""
Configuration module for the AI Chatbot application.
Loads environment variables and defines app-wide settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Configuration (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Fast & capable on Groq

# --- App Configuration ---
APP_TITLE = "🤖 AI Business Intelligence Chatbot"
APP_ICON = "🤖"
PAGE_LAYOUT = "wide"

# --- Data Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# --- Dataset Files ---
SALES_DATA = "sales_data.csv"
HR_DATA = "hr_data.csv"
FINANCE_DATA = "finance_data.csv"

# --- Chart Settings ---
CHART_THEME = "plotly_dark"
CHART_HEIGHT = 450
CHART_COLORS = [
    "#00D4AA", "#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF",
    "#FF8B94", "#B8D4E3", "#F7DC6F", "#BB8FCE", "#85C1E9"
]

# --- PDF Report Settings ---
REPORT_TITLE = "AI Business Intelligence Report"
REPORT_FONT = "Helvetica"
