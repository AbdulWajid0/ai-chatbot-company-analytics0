"""
Data Loader Module (Team 1 — Data Engineering)
================================================
Handles loading, cleaning, and validating company datasets.
Provides a unified interface for all data access across the app.
"""

import pandas as pd
import numpy as np
import os
import streamlit as st
from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SALES_DATA, HR_DATA, FINANCE_DATA


@st.cache_data(ttl=3600)
def load_sales_data():
    """Load and preprocess sales dataset."""
    filepath = os.path.join(RAW_DATA_DIR, SALES_DATA)
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath, parse_dates=["date"])

    # --- Cleaning ---
    df.dropna(subset=["date", "product", "revenue"], inplace=True)
    df["sales_channel"] = df["sales_channel"].fillna("Unknown")

    # --- Feature Engineering ---
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["quarter"] = df["date"].dt.quarter
    df["quarter_label"] = "Q" + df["quarter"].astype(str)
    df["weekday"] = df["date"].dt.day_name()
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # Profit margin
    df["profit_margin"] = np.where(
        df["revenue"] > 0,
        (df["profit"] / df["revenue"]) * 100,
        0
    )

    return df


@st.cache_data(ttl=3600)
def load_hr_data():
    """Load and preprocess HR dataset."""
    filepath = os.path.join(RAW_DATA_DIR, HR_DATA)
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath, parse_dates=["join_date"])

    # --- Feature Engineering ---
    df["tenure_years"] = ((pd.Timestamp.now() - df["join_date"]).dt.days / 365.25).round(1)
    df["attrition_label"] = df["attrition"].map({1: "Left", 0: "Active"})
    df["performance_label"] = df["performance_score"].map({
        1: "Poor", 2: "Below Average", 3: "Average", 4: "Good", 5: "Excellent"
    })
    df["salary_band"] = pd.cut(
        df["salary"],
        bins=[0, 40000, 70000, 100000, 150000, float("inf")],
        labels=["Entry", "Mid", "Senior", "Lead", "Executive"]
    )

    return df


@st.cache_data(ttl=3600)
def load_finance_data():
    """Load and preprocess finance dataset."""
    filepath = os.path.join(RAW_DATA_DIR, FINANCE_DATA)
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath, parse_dates=["date"])

    # --- Feature Engineering ---
    df["over_budget"] = df["variance"] > 0
    df["quarter"] = ((df["month"] - 1) // 3) + 1
    df["quarter_label"] = "Q" + df["quarter"].astype(str)
    df["year_quarter"] = df["year"].astype(str) + " " + df["quarter_label"]

    return df


def load_all_datasets():
    """Load all datasets and return as a dictionary."""
    datasets = {}

    sales = load_sales_data()
    if sales is not None:
        datasets["sales"] = sales

    hr = load_hr_data()
    if hr is not None:
        datasets["hr"] = hr

    finance = load_finance_data()
    if finance is not None:
        datasets["finance"] = finance

    return datasets


def get_dataset_summary(datasets):
    """Generate a summary of all loaded datasets for the LLM context."""
    summary_parts = []

    if "sales" in datasets:
        df = datasets["sales"]
        summary_parts.append(f"""📊 SALES DATA:
- Records: {len(df):,}
- Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
- Products: {df['product'].nunique()} ({', '.join(df['product'].unique()[:5])}...)
- Categories: {', '.join(df['category'].unique())}
- Regions: {', '.join(df['region'].unique())}
- Channels: {', '.join(df['sales_channel'].dropna().unique())}
- Columns: {', '.join(df.columns.tolist())}
- Total Revenue: ${df['revenue'].sum():,.2f}
- Total Profit: ${df['profit'].sum():,.2f}""")

    if "hr" in datasets:
        df = datasets["hr"]
        summary_parts.append(f"""👥 HR DATA:
- Employees: {len(df):,}
- Departments: {', '.join(df['department'].unique())}
- Attrition Rate: {df['attrition'].mean()*100:.1f}%
- Avg Salary: ${df['salary'].mean():,.0f}
- Columns: {', '.join(df.columns.tolist())}""")

    if "finance" in datasets:
        df = datasets["finance"]
        summary_parts.append(f"""💰 FINANCE DATA:
- Records: {len(df):,}
- Period: {df['year'].min()} to {df['year'].max()}
- Expense Types: {', '.join(df['expense_type'].unique())}
- Departments: {', '.join(df['department'].unique())}
- Columns: {', '.join(df.columns.tolist())}
- Total Budget: ${df['budget'].sum():,.2f}
- Total Actual: ${df['actual_amount'].sum():,.2f}""")

    return "\n\n".join(summary_parts)
