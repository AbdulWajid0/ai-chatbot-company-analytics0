"""
Insight Generator Module (Team 4 — Visualization & Insights)
=============================================================
Generates text-based executive insights and recommendations
from analysis results, independent of the LLM.
"""

import pandas as pd
import numpy as np


def generate_executive_summary(datasets):
    """Generate a quick executive summary of all data."""
    parts = []

    if "sales" in datasets:
        df = datasets["sales"]
        total_revenue = df["revenue"].sum()
        total_profit = df["profit"].sum()
        avg_margin = df["profit_margin"].mean()
        top_product = df.groupby("product")["revenue"].sum().idxmax()
        top_region = df.groupby("region")["revenue"].sum().idxmax()

        parts.append(f"""### 📊 Sales Overview
- **Total Revenue:** ${total_revenue:,.2f}
- **Total Profit:** ${total_profit:,.2f}
- **Avg Profit Margin:** {avg_margin:.1f}%
- **Top Product:** {top_product}
- **Strongest Region:** {top_region}
- **Transactions:** {len(df):,}""")

    if "hr" in datasets:
        df = datasets["hr"]
        attrition_rate = df["attrition"].mean() * 100
        avg_performance = df["performance_score"].mean()
        avg_salary = df["salary"].mean()

        parts.append(f"""### 👥 HR Overview
- **Employees:** {len(df):,}
- **Attrition Rate:** {attrition_rate:.1f}%
- **Avg Performance:** {avg_performance:.1f}/5.0
- **Avg Salary:** ${avg_salary:,.0f}
- **Departments:** {df['department'].nunique()}""")

    if "finance" in datasets:
        df = datasets["finance"]
        total_budget = df["budget"].sum()
        total_actual = df["actual_amount"].sum()
        variance = total_actual - total_budget

        parts.append(f"""### 💰 Finance Overview
- **Total Budget:** ${total_budget:,.2f}
- **Total Spending:** ${total_actual:,.2f}
- **Budget Variance:** ${variance:,.2f} ({'Over' if variance > 0 else 'Under'} Budget)
- **Over-Budget Incidents:** {df['over_budget'].sum():,}""")

    return "\n\n".join(parts)


def generate_quick_insights(datasets):
    """Generate actionable quick insights from the data."""
    insights = []

    if "sales" in datasets:
        df = datasets["sales"]

        # Best and worst performing months
        monthly_rev = df.groupby("year_month")["revenue"].sum()
        best_month = monthly_rev.idxmax()
        worst_month = monthly_rev.idxmin()
        insights.append(f"📈 **Best sales month:** {best_month} (${monthly_rev.max():,.0f})")
        insights.append(f"📉 **Weakest sales month:** {worst_month} (${monthly_rev.min():,.0f})")

        # Category performance
        cat_rev = df.groupby("category")["revenue"].sum()
        insights.append(f"🏆 **Top category:** {cat_rev.idxmax()} (${cat_rev.max():,.0f})")

        # Channel insights
        channel_rev = df.groupby("sales_channel")["revenue"].sum().dropna()
        if len(channel_rev) > 0:
            insights.append(f"🛒 **Best channel:** {channel_rev.idxmax()} (${channel_rev.max():,.0f})")

    if "hr" in datasets:
        df = datasets["hr"]

        # Department with highest attrition
        dept_attrition = df.groupby("department")["attrition"].mean() * 100
        highest_attrition_dept = dept_attrition.idxmax()
        insights.append(f"⚠️ **Highest attrition:** {highest_attrition_dept} ({dept_attrition.max():.1f}%)")

        # Best performing department
        dept_perf = df.groupby("department")["performance_score"].mean()
        best_dept = dept_perf.idxmax()
        insights.append(f"⭐ **Best performing dept:** {best_dept} ({dept_perf.max():.2f}/5)")

    if "finance" in datasets:
        df = datasets["finance"]

        # Most over-budget expense
        exp_variance = df.groupby("expense_type")["variance"].sum()
        most_over = exp_variance.idxmax()
        insights.append(f"💸 **Most over-budget:** {most_over} (${exp_variance.max():,.0f} over)")

    return insights


def get_suggested_questions():
    """Return a list of suggested questions for the user."""
    return [
        "📊 What were last quarter's total sales?",
        "🏆 Show me top 5 products by revenue",
        "📈 Show the monthly revenue trend",
        "🌍 Which region has the highest profit margin?",
        "👥 What is the employee attrition rate by department?",
        "💰 Which departments are over budget?",
        "🔮 Predict next month's revenue",
        "📊 Compare sales performance across regions",
        "👥 Show salary distribution by department",
        "💰 What are the KPIs for this quarter?",
    ]
