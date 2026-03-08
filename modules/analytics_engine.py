"""
Analytics Engine Module (Team 3 — Analytics)
=============================================
Core data analysis engine that executes analytical operations
on company datasets based on parsed intents from the NLP layer.
"""

import pandas as pd
import numpy as np


def execute_analysis(datasets, parsed_intent):
    """
    Main entry point: Execute analysis based on parsed intent from Gemini.
    Returns: (result_df, summary_text)
    """
    if parsed_intent is None:
        return None, None

    dataset_name = parsed_intent.get("dataset", "sales")
    operation = parsed_intent.get("operation", "summary")
    params = parsed_intent.get("parameters", {})

    if dataset_name not in datasets:
        return None, f"Dataset '{dataset_name}' not found."

    df = datasets[dataset_name].copy()

    # Apply filters first
    df = apply_filters(df, params.get("filters", {}))

    # Execute operation
    if operation == "filter":
        result = df
    elif operation == "aggregate":
        result = aggregate_data(df, params)
    elif operation == "group_by":
        result = group_by_data(df, params)
    elif operation == "rank":
        result = rank_data(df, params)
    elif operation == "trend":
        result = trend_analysis(df, params)
    elif operation == "kpi":
        result = calculate_kpis(df, dataset_name, params)
    elif operation == "forecast":
        result = simple_forecast(df, params)
    elif operation == "summary":
        result = dataset_summary(df, params)
    else:
        result = df

    # Generate summary text
    summary = generate_result_summary(result, parsed_intent)

    return result, summary


def apply_filters(df, filters):
    """Apply column-value filters to a DataFrame."""
    if not filters:
        return df

    for col, value in filters.items():
        if col not in df.columns:
            continue

        if isinstance(value, list):
            df = df[df[col].isin(value)]
        elif isinstance(value, dict):
            # Range filter: {"min": 100, "max": 500}
            if "min" in value:
                df = df[df[col] >= value["min"]]
            if "max" in value:
                df = df[df[col] <= value["max"]]
        elif isinstance(value, str):
            # Case-insensitive string match
            df = df[df[col].astype(str).str.lower() == value.lower()]
        else:
            df = df[df[col] == value]

    return df


def aggregate_data(df, params):
    """Perform aggregation on specified columns."""
    metric = params.get("metric", "revenue")
    agg_func = params.get("agg_function", "sum")

    if metric not in df.columns:
        # Find a numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        metric = numeric_cols[0] if numeric_cols else None

    if metric is None:
        return df.describe()

    agg_map = {
        "sum": df[metric].sum(),
        "mean": df[metric].mean(),
        "count": len(df),
        "min": df[metric].min(),
        "max": df[metric].max(),
        "median": df[metric].median(),
    }

    value = agg_map.get(agg_func, df[metric].sum())
    result = pd.DataFrame({
        "Metric": [metric],
        "Operation": [agg_func],
        "Value": [round(value, 2)]
    })
    return result


def group_by_data(df, params):
    """Group data and aggregate."""
    group_cols = params.get("group_by", [])
    metric = params.get("metric", "revenue")
    agg_func = params.get("agg_function", "sum")
    sort_order = params.get("sort_order", "desc")
    top_n = params.get("top_n")

    if not group_cols:
        return df

    # Validate columns exist
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return df

    if metric not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        metric = numeric_cols[0] if numeric_cols else None

    if metric is None:
        return df

    # Perform groupby
    result = df.groupby(group_cols, as_index=False).agg({metric: agg_func})
    result.columns = group_cols + [f"{metric}_{agg_func}"]

    # Sort
    sort_col = result.columns[-1]
    ascending = sort_order == "asc"
    result = result.sort_values(sort_col, ascending=ascending)

    # Top N
    if top_n and isinstance(top_n, int):
        result = result.head(top_n)

    result = result.reset_index(drop=True)
    return result


def rank_data(df, params):
    """Rank data by a metric."""
    metric = params.get("metric", "revenue")
    sort_order = params.get("sort_order", "desc")
    top_n = params.get("top_n", 10)
    group_cols = params.get("group_by", params.get("columns", []))

    if not group_cols:
        # Default: rank by product for sales, department for HR
        if "product" in df.columns:
            group_cols = ["product"]
        elif "department" in df.columns:
            group_cols = ["department"]
        elif "expense_type" in df.columns:
            group_cols = ["expense_type"]
        else:
            group_cols = [df.columns[0]]

    group_cols = [c for c in group_cols if c in df.columns]

    if metric not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        metric = numeric_cols[0] if numeric_cols else None

    if metric is None or not group_cols:
        return df.head(top_n)

    result = df.groupby(group_cols, as_index=False)[metric].sum()
    ascending = sort_order == "asc"
    result = result.sort_values(metric, ascending=ascending).head(top_n)
    result["rank"] = range(1, len(result) + 1)
    result = result.reset_index(drop=True)
    return result


def trend_analysis(df, params):
    """Analyze trends over time."""
    metric = params.get("metric", "revenue")
    time_col = None

    # Find appropriate time column
    for col in ["year_month", "date", "year_quarter", "quarter_label"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is None or metric not in df.columns:
        return df

    if time_col == "date":
        # Group by month
        df["_period"] = df["date"].dt.to_period("M").astype(str)
        result = df.groupby("_period", as_index=False)[metric].sum()
        result = result.rename(columns={"_period": "Period"})
    else:
        result = df.groupby(time_col, as_index=False)[metric].sum()
        result = result.rename(columns={time_col: "Period"})

    result = result.sort_values("Period")

    # Calculate growth rate
    result["growth_rate"] = result[metric].pct_change() * 100
    result["growth_rate"] = result["growth_rate"].round(2)

    result = result.reset_index(drop=True)
    return result


def calculate_kpis(df, dataset_name, params):
    """Calculate key performance indicators based on dataset type."""
    kpis = {}

    if dataset_name == "sales":
        kpis = {
            "Total Revenue": f"${df['revenue'].sum():,.2f}",
            "Total Profit": f"${df['profit'].sum():,.2f}",
            "Avg Profit Margin": f"{df['profit_margin'].mean():.1f}%",
            "Total Units Sold": f"{df['quantity'].sum():,}",
            "Avg Order Value": f"${df['revenue'].mean():,.2f}",
            "Unique Products": str(df['product'].nunique()),
            "Top Product": df.groupby('product')['revenue'].sum().idxmax(),
            "Top Region": df.groupby('region')['revenue'].sum().idxmax(),
            "Total Transactions": f"{len(df):,}",
        }
    elif dataset_name == "hr":
        kpis = {
            "Total Employees": str(len(df)),
            "Attrition Rate": f"{df['attrition'].mean() * 100:.1f}%",
            "Avg Salary": f"${df['salary'].mean():,.0f}",
            "Median Salary": f"${df['salary'].median():,.0f}",
            "Avg Performance": f"{df['performance_score'].mean():.2f}/5",
            "Avg Tenure": f"{df['tenure_years'].mean():.1f} years",
            "Gender Ratio (M:F)": f"{(df['gender']=='Male').sum()}:{(df['gender']=='Female').sum()}",
            "Departments": str(df['department'].nunique()),
            "Top Department": df.groupby('department').size().idxmax(),
        }
    elif dataset_name == "finance":
        kpis = {
            "Total Budget": f"${df['budget'].sum():,.2f}",
            "Total Spending": f"${df['actual_amount'].sum():,.2f}",
            "Budget Variance": f"${df['variance'].sum():,.2f}",
            "Over-Budget %": f"{df['over_budget'].mean() * 100:.1f}%",
            "Biggest Expense": df.groupby('expense_type')['actual_amount'].sum().idxmax(),
            "Top Spending Dept": df.groupby('department')['actual_amount'].sum().idxmax(),
        }

    result = pd.DataFrame({
        "KPI": list(kpis.keys()),
        "Value": list(kpis.values())
    })
    return result


def simple_forecast(df, params):
    """Simple moving average forecast."""
    metric = params.get("metric", "revenue")

    if metric not in df.columns:
        return pd.DataFrame({"message": ["Unable to forecast: metric not found"]})

    # Group by month for time series
    if "date" in df.columns:
        monthly = df.set_index("date").resample("ME")[metric].sum().reset_index()
        monthly.columns = ["Period", metric]
    elif "year_month" in df.columns:
        monthly = df.groupby("year_month", as_index=False)[metric].sum()
        monthly.columns = ["Period", metric]
    else:
        return pd.DataFrame({"message": ["Unable to forecast: no time column found"]})

    # Calculate moving averages
    monthly["MA_3"] = monthly[metric].rolling(window=3).mean()
    monthly["MA_6"] = monthly[metric].rolling(window=6).mean()

    # Simple forecast: use the trend of the last 3 months
    if len(monthly) >= 3:
        last_3 = monthly[metric].tail(3).values
        trend = (last_3[-1] - last_3[0]) / 2
        forecast_value = last_3[-1] + trend
        monthly["Type"] = "Historical"

        # Add forecast row
        forecast_row = pd.DataFrame({
            "Period": ["Forecast (Next Month)"],
            metric: [round(forecast_value, 2)],
            "MA_3": [None],
            "MA_6": [None],
            "Type": ["Forecast"]
        })
        monthly = pd.concat([monthly, forecast_row], ignore_index=True)

    return monthly


def dataset_summary(df, params):
    """Generate a summary of the dataset."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df[numeric_cols].describe().T
    summary = summary[["count", "mean", "min", "max", "std"]]
    summary = summary.round(2)
    summary = summary.reset_index()
    summary.columns = ["Column", "Count", "Mean", "Min", "Max", "Std Dev"]
    return summary


def generate_result_summary(result, parsed_intent):
    """Generate a text summary of the analysis results."""
    if result is None:
        return "No results found."

    if isinstance(result, pd.DataFrame):
        if len(result) == 0:
            return "No matching data found for the specified criteria."

        title = parsed_intent.get("title", "Analysis Results")
        num_rows = len(result)

        # Build summary based on result contents
        summary_parts = [f"**{title}**", f"Found {num_rows} result(s)."]

        # Add top-level numeric insights
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols[:3]:
            if col not in ["rank", "growth_rate"]:
                summary_parts.append(f"- {col}: Total = {result[col].sum():,.2f}, Avg = {result[col].mean():,.2f}")

        return "\n".join(summary_parts)

    return str(result)
