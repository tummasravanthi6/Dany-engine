from dataclasses import dataclass
from typing import Any, Dict


# =========================================================
# Report Schema (THE PRODUCT)
# =========================================================

@dataclass
class Report:
    title: str
    executive_summary: Dict[str, Any]
    data_overview: str
    cleaning_actions: str
    key_insights: str
    modeling_results: str
    predictions_confidence: str
    trust_warnings: str
    limitations_assumptions: str


# =========================================================
# HTML Renderer (PLAIN, READABLE)
# =========================================================

def render_report_to_html(report: Report) -> str:
    """
    Converts a Report object into a readable HTML document.
    No CSS frameworks. No styling tricks.
    """

    def section(title: str, content: str) -> str:
        return f"<h2>{title}</h2><p>{content}</p>"

    html = f"""
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>{report.title}</title>
    </head>
    <body>
        <h1>{report.title}</h1>

        {section("Executive Summary", str(report.executive_summary))}
        {section("Data Overview", report.data_overview)}
        {section("Cleaning Actions", report.cleaning_actions)}
        {section("Key Insights", report.key_insights)}
        {section("Modeling Results", report.modeling_results)}
        {section("Predictions & Confidence", report.predictions_confidence)}
        {section("Trust Warnings", report.trust_warnings)}
        {section("Limitations & Assumptions", report.limitations_assumptions)}

    </body>
    </html>
    """

    return html
