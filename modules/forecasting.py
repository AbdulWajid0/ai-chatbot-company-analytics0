"""
Forecasting Module (Team 3 — Analytics Engine)
================================================
Provides time-series forecasting capabilities using
statistical methods (Moving Average, Linear Regression).
"""

import pandas as pd
import numpy as np


def forecast_revenue(df, periods=3):
    """
    Forecast future revenue using moving average and linear trend.

    Args:
        df: Sales DataFrame with 'date' and 'revenue' columns
        periods: Number of future periods (months) to forecast

    Returns:
        DataFrame with historical and forecasted values
    """
    if "date" not in df.columns or "revenue" not in df.columns:
        return None

    # Aggregate monthly revenue
    monthly = df.set_index("date").resample("ME")["revenue"].sum().reset_index()
    monthly.columns = ["Date", "Revenue"]
    monthly = monthly.sort_values("Date")

    # Calculate moving averages
    monthly["MA_3"] = monthly["Revenue"].rolling(window=3, min_periods=1).mean()
    monthly["MA_6"] = monthly["Revenue"].rolling(window=6, min_periods=1).mean()

    # Linear regression for trend
    x = np.arange(len(monthly))
    y = monthly["Revenue"].values

    # Fit linear trend
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs[0], coeffs[1]
    else:
        slope, intercept = 0, y[-1] if len(y) > 0 else 0

    # Generate forecast
    monthly["Type"] = "Historical"
    forecast_rows = []

    last_date = monthly["Date"].max()
    for i in range(1, periods + 1):
        future_date = last_date + pd.DateOffset(months=i)
        future_x = len(monthly) + i - 1

        # Trend-based forecast
        trend_value = slope * future_x + intercept

        # Moving average based forecast
        recent_values = monthly["Revenue"].tail(3).values
        ma_value = np.mean(recent_values)

        # Weighted combination: 60% trend + 40% MA
        forecast_value = 0.6 * trend_value + 0.4 * ma_value

        # Add some realistic variance
        forecast_value = max(0, forecast_value)

        forecast_rows.append({
            "Date": future_date,
            "Revenue": round(forecast_value, 2),
            "MA_3": None,
            "MA_6": None,
            "Type": "Forecast",
        })

    forecast_df = pd.concat([monthly, pd.DataFrame(forecast_rows)], ignore_index=True)

    # Add confidence bands (simple: ±15% for forecast periods)
    forecast_df["Lower_Bound"] = forecast_df.apply(
        lambda row: round(row["Revenue"] * 0.85, 2) if row["Type"] == "Forecast" else None,
        axis=1
    )
    forecast_df["Upper_Bound"] = forecast_df.apply(
        lambda row: round(row["Revenue"] * 1.15, 2) if row["Type"] == "Forecast" else None,
        axis=1
    )

    # Store trend info
    forecast_df.attrs["slope"] = slope
    forecast_df.attrs["trend"] = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"

    return forecast_df


def get_seasonal_pattern(df):
    """
    Detect seasonal patterns in the sales data.

    Returns:
        DataFrame showing average revenue by month
    """
    if "date" not in df.columns or "revenue" not in df.columns:
        return None

    df_copy = df.copy()
    df_copy["month"] = df_copy["date"].dt.month
    df_copy["month_name"] = df_copy["date"].dt.strftime("%B")

    seasonal = df_copy.groupby(["month", "month_name"], as_index=False)["revenue"].agg(
        avg_revenue="mean",
        total_revenue="sum",
        transaction_count="count"
    )
    seasonal = seasonal.sort_values("month")

    # Mark peak/low seasons
    avg = seasonal["avg_revenue"].mean()
    seasonal["season_type"] = seasonal["avg_revenue"].apply(
        lambda x: "🔥 Peak" if x > avg * 1.1 else ("❄️ Low" if x < avg * 0.9 else "➡️ Normal")
    )

    return seasonal


def get_growth_metrics(df):
    """
    Calculate period-over-period growth metrics.

    Returns:
        Dictionary with growth metrics
    """
    if "date" not in df.columns or "revenue" not in df.columns:
        return {}

    monthly = df.set_index("date").resample("ME")["revenue"].sum()

    if len(monthly) < 2:
        return {}

    metrics = {
        "last_month_revenue": monthly.iloc[-1],
        "prev_month_revenue": monthly.iloc[-2],
        "mom_growth": ((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100) if monthly.iloc[-2] != 0 else 0,
    }

    # Quarter over quarter
    quarterly = df.set_index("date").resample("QE")["revenue"].sum()
    if len(quarterly) >= 2:
        metrics["last_quarter_revenue"] = quarterly.iloc[-1]
        metrics["prev_quarter_revenue"] = quarterly.iloc[-2]
        metrics["qoq_growth"] = ((quarterly.iloc[-1] - quarterly.iloc[-2]) / quarterly.iloc[-2] * 100) if quarterly.iloc[-2] != 0 else 0

    # Year over year
    yearly = df.set_index("date").resample("YE")["revenue"].sum()
    if len(yearly) >= 2:
        metrics["last_year_revenue"] = yearly.iloc[-1]
        metrics["prev_year_revenue"] = yearly.iloc[-2]
        metrics["yoy_growth"] = ((yearly.iloc[-1] - yearly.iloc[-2]) / yearly.iloc[-2] * 100) if yearly.iloc[-2] != 0 else 0

    return metrics
