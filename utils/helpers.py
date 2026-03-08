"""
Helper utility functions used across the project.
"""

import os
import pandas as pd
from datetime import datetime


def format_currency(value):
    """Format a number as currency string."""
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:,.2f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:,.1f}K"
    else:
        return f"${value:,.2f}"


def format_percentage(value):
    """Format a number as percentage string."""
    return f"{value:+.1f}%"


def get_quarter(month):
    """Return quarter string from month number."""
    return f"Q{(month - 1) // 3 + 1}"


def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def get_date_range_label(df, date_col="date"):
    """Get a human-readable date range label from a DataFrame."""
    if date_col not in df.columns:
        return "Unknown date range"
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    return f"{min_date.strftime('%b %Y')} – {max_date.strftime('%b %Y')}"


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def get_timestamp():
    """Get current timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def truncate_text(text, max_length=100):
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
