"""
Intent Handler Module (Team 2 — NLP & Intent Recognition)
==========================================================
Maps intents from the NLP engine to analytics operations.
Acts as the bridge between Gemini's output and the analytics engine.
"""

from modules.analytics_engine import execute_analysis
from modules.visualizer import create_chart
from modules.nlp_engine import generate_insight, clean_response_text


def handle_intent(datasets, parsed_json, raw_response, model, user_query):
    """
    Process a parsed intent and return analysis results, chart, and insights.

    Returns:
        dict with keys: 'result_df', 'chart', 'summary', 'insight', 'response_text', 'chart_type'
    """
    response = {
        "result_df": None,
        "chart": None,
        "summary": None,
        "insight": None,
        "response_text": clean_response_text(raw_response),
        "chart_type": None,
    }

    if parsed_json is None:
        # No structured data — just a general conversation response
        return response

    # Execute the analytics operation
    result_df, summary = execute_analysis(datasets, parsed_json)
    response["result_df"] = result_df
    response["summary"] = summary

    # Generate chart if applicable
    chart_type = parsed_json.get("chart_type", "none")
    title = parsed_json.get("title", "Analysis Result")

    if chart_type and chart_type != "none" and result_df is not None and len(result_df) > 0:
        chart = create_chart(result_df, chart_type, title, parsed_json.get("parameters", {}))
        response["chart"] = chart
        response["chart_type"] = chart_type

    # Generate AI insight from results
    if result_df is not None and len(result_df) > 0 and model is not None:
        results_summary = ""
        if summary:
            results_summary += summary + "\n"
        results_summary += result_df.head(10).to_string()

        insight = generate_insight(model, user_query, title, results_summary)
        response["insight"] = insight

    return response
