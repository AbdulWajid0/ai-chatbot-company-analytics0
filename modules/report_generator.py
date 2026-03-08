"""
Report Generator Module (Team 5 — Chatbot Integration)
========================================================
Generates downloadable PDF analytics reports from
conversation results and visualizations.
"""

import os
from datetime import datetime
from fpdf import FPDF
from utils.config import REPORTS_DIR, REPORT_TITLE
from utils.helpers import ensure_directory, get_timestamp


class AnalyticsReport(FPDF):
    """Custom PDF report generator for analytics results."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(0, 100, 180)
        self.cell(0, 10, REPORT_TITLE, ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", ln=True, align="C")
        self.ln(5)
        # Line separator
        self.set_draw_color(0, 100, 180)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"AI Business Intelligence Report | Page {self.page_no()}/{{nb}}", align="C")

    def add_title_section(self, title):
        """Add a section title."""
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 70, 140)
        self.cell(0, 10, title, ln=True)
        self.ln(3)

    def add_subtitle(self, subtitle):
        """Add a subsection title."""
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, subtitle, ln=True)
        self.ln(2)

    def add_body_text(self, text):
        """Add body text."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        # Handle multi-line text and special characters
        clean_text = text.encode('latin-1', errors='replace').decode('latin-1')
        self.multi_cell(0, 6, clean_text)
        self.ln(3)

    def add_key_value(self, key, value):
        """Add a key-value pair line."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(50, 50, 50)
        self.cell(60, 7, f"{key}:", align="L")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(80, 80, 80)
        clean_value = str(value).encode('latin-1', errors='replace').decode('latin-1')
        self.cell(0, 7, clean_value, ln=True)

    def add_table(self, df, max_rows=20):
        """Add a data table from a DataFrame."""
        if df is None or len(df) == 0:
            return

        df_display = df.head(max_rows)
        columns = df_display.columns.tolist()

        # Calculate column widths
        page_width = 190  # usable width
        col_width = page_width / len(columns)

        # Header
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(0, 100, 180)
        self.set_text_color(255, 255, 255)
        for col in columns:
            col_name = str(col)[:20]  # Truncate long names
            self.cell(col_width, 7, col_name, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 8)
        self.set_text_color(60, 60, 60)
        fill = False
        for _, row in df_display.iterrows():
            if fill:
                self.set_fill_color(240, 245, 255)
            else:
                self.set_fill_color(255, 255, 255)

            for col in columns:
                val = str(row[col])[:25]  # Truncate long values
                clean_val = val.encode('latin-1', errors='replace').decode('latin-1')
                self.cell(col_width, 6, clean_val, border=1, fill=True, align="C")
            self.ln()
            fill = not fill

        if len(df) > max_rows:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 6, f"... and {len(df) - max_rows} more rows", ln=True, align="C")

        self.ln(5)

    def add_divider(self):
        """Add a horizontal divider."""
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)


def generate_report(query, result_df, insight_text, chart_path=None):
    """
    Generate a PDF report from analysis results.

    Args:
        query: User's original query
        result_df: Analysis results DataFrame
        insight_text: AI-generated insight text
        chart_path: Path to saved chart image (optional)

    Returns:
        Path to the generated PDF file
    """
    ensure_directory(REPORTS_DIR)

    pdf = AnalyticsReport()
    pdf.alias_nb_pages()
    pdf.add_page()

    # 1. Query Section
    pdf.add_title_section("Query Details")
    pdf.add_key_value("User Query", query)
    pdf.add_key_value("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pdf.ln(3)

    # 2. Results Section
    pdf.add_divider()
    pdf.add_title_section("Analysis Results")
    if result_df is not None and len(result_df) > 0:
        pdf.add_table(result_df)
    else:
        pdf.add_body_text("No tabular results for this query.")

    # 3. Chart Section (if chart image available)
    if chart_path and os.path.exists(chart_path):
        pdf.add_divider()
        pdf.add_title_section("Visualization")
        try:
            pdf.image(chart_path, x=15, w=180)
            pdf.ln(5)
        except Exception:
            pdf.add_body_text("(Chart could not be embedded)")

    # 4. Insights Section
    if insight_text:
        pdf.add_divider()
        pdf.add_title_section("AI Insights & Recommendations")
        pdf.add_body_text(insight_text)

    # Save PDF
    timestamp = get_timestamp()
    filename = f"report_{timestamp}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)
    pdf.output(filepath)

    return filepath
