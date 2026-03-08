"""
Visualization Module (Team 4 — Visualization & Insights)
==========================================================
Generates interactive Plotly charts from analysis results.
Supports bar, line, pie, heatmap, scatter, histogram, and box plots.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.config import CHART_THEME, CHART_HEIGHT, CHART_COLORS


def create_chart(df, chart_type, title="", params=None):
    """
    Create a Plotly chart based on the chart type and data.

    Args:
        df: DataFrame with analysis results
        chart_type: Type of chart (bar, line, pie, heatmap, scatter, histogram, box)
        title: Chart title
        params: Additional parameters from intent

    Returns:
        Plotly figure object or None
    """
    if df is None or len(df) == 0:
        return None

    params = params or {}

    try:
        if chart_type == "bar":
            fig = create_bar_chart(df, title, params)
        elif chart_type == "line":
            fig = create_line_chart(df, title, params)
        elif chart_type == "pie":
            fig = create_pie_chart(df, title, params)
        elif chart_type == "heatmap":
            fig = create_heatmap(df, title, params)
        elif chart_type == "scatter":
            fig = create_scatter_chart(df, title, params)
        elif chart_type == "histogram":
            fig = create_histogram(df, title, params)
        elif chart_type == "box":
            fig = create_box_chart(df, title, params)
        else:
            fig = create_bar_chart(df, title, params)  # Default to bar

        # Apply consistent styling
        fig = apply_chart_styling(fig, title)
        return fig

    except Exception as e:
        print(f"Chart creation error: {e}")
        return None


def create_bar_chart(df, title, params):
    """Create a bar chart."""
    cols = df.columns.tolist()
    # Remove non-useful columns
    skip_cols = ["rank", "Type"]

    # Find category (x) and value (y) columns
    category_col = None
    value_col = None

    for col in cols:
        if col in skip_cols:
            continue
        if df[col].dtype == "object" or col in ["Period"]:
            if category_col is None:
                category_col = col
        elif np.issubdtype(df[col].dtype, np.number):
            if value_col is None:
                value_col = col

    if category_col is None:
        category_col = cols[0]
    if value_col is None:
        value_col = cols[-1]

    fig = px.bar(
        df,
        x=category_col,
        y=value_col,
        title=title,
        color_discrete_sequence=CHART_COLORS,
        text=value_col,
    )
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    return fig


def create_line_chart(df, title, params):
    """Create a line chart."""
    cols = df.columns.tolist()

    x_col = None
    y_cols = []

    for col in cols:
        if col in ["Period", "year_month", "date", "month", "year"] or "period" in col.lower():
            x_col = col
        elif np.issubdtype(df[col].dtype, np.number) and col not in ["growth_rate"]:
            y_cols.append(col)

    if x_col is None:
        x_col = cols[0]
    if not y_cols:
        y_cols = [cols[-1]]

    fig = go.Figure()

    colors = CHART_COLORS[:len(y_cols)]
    for i, y_col in enumerate(y_cols[:3]):  # Max 3 lines
        # Check if this is a forecast
        if "Type" in df.columns:
            historical = df[df["Type"] == "Historical"]
            forecast = df[df["Type"] == "Forecast"]

            fig.add_trace(go.Scatter(
                x=historical[x_col], y=historical[y_col],
                mode='lines+markers', name=f"{y_col} (Historical)",
                line=dict(color=colors[i], width=2),
            ))
            if len(forecast) > 0:
                fig.add_trace(go.Scatter(
                    x=forecast[x_col], y=forecast[y_col],
                    mode='markers', name=f"{y_col} (Forecast)",
                    marker=dict(size=12, color="#FF6B6B", symbol="star"),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=df[x_col], y=df[y_col],
                mode='lines+markers', name=y_col,
                line=dict(color=colors[i], width=2),
            ))

    fig.update_layout(title=title)
    return fig


def create_pie_chart(df, title, params):
    """Create a pie chart."""
    cols = df.columns.tolist()

    label_col = None
    value_col = None

    for col in cols:
        if df[col].dtype == "object" and label_col is None:
            label_col = col
        elif np.issubdtype(df[col].dtype, np.number) and value_col is None:
            value_col = col

    if label_col is None:
        label_col = cols[0]
    if value_col is None:
        value_col = cols[-1]

    fig = px.pie(
        df,
        names=label_col,
        values=value_col,
        title=title,
        color_discrete_sequence=CHART_COLORS,
        hole=0.4,  # Donut style
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def create_heatmap(df, title, params):
    """Create a heatmap."""
    # Try to create a pivot table for the heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if len(cat_cols) >= 2 and len(numeric_cols) >= 1:
        pivot = df.pivot_table(
            values=numeric_cols[0],
            index=cat_cols[0],
            columns=cat_cols[1] if len(cat_cols) > 1 else cat_cols[0],
            aggfunc="sum",
            fill_value=0
        )
        fig = px.imshow(
            pivot, title=title,
            color_continuous_scale="YlOrRd",
            text_auto=True,
        )
    else:
        # Fallback: correlation heatmap of numeric columns
        corr = df[numeric_cols].corr()
        fig = px.imshow(
            corr, title=title,
            color_continuous_scale="RdBu_r",
            text_auto=".2f",
        )

    return fig


def create_scatter_chart(df, title, params):
    """Create a scatter plot."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) >= 2:
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
    else:
        x_col = df.columns[0]
        y_col = df.columns[-1]

    color_col = None
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        color_col = cat_cols[0]

    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col,
        title=title,
        color_discrete_sequence=CHART_COLORS,
    )
    return fig


def create_histogram(df, title, params):
    """Create a histogram."""
    metric = params.get("metric", None)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if metric and metric in df.columns:
        col = metric
    elif numeric_cols:
        col = numeric_cols[0]
    else:
        col = df.columns[0]

    fig = px.histogram(
        df, x=col, title=title,
        color_discrete_sequence=CHART_COLORS,
        nbins=20,
    )
    return fig


def create_box_chart(df, title, params):
    """Create a box plot."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    y_col = numeric_cols[0] if numeric_cols else df.columns[-1]
    x_col = cat_cols[0] if cat_cols else None

    fig = px.box(
        df, x=x_col, y=y_col, title=title,
        color=x_col,
        color_discrete_sequence=CHART_COLORS,
    )
    return fig


def apply_chart_styling(fig, title):
    """Apply consistent dark theme styling to charts."""
    fig.update_layout(
        template=CHART_THEME,
        height=CHART_HEIGHT,
        title=dict(
            text=title,
            font=dict(size=18, color="#FFFFFF"),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        font=dict(color="#CCCCCC", size=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            font=dict(size=11),
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
