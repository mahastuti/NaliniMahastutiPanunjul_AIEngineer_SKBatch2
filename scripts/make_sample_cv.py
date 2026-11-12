"""generate the synthetic sample cv pdf. fictional person, fictional employers."""

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

LINES: list[tuple[str, str]] = [
    ("title", "Dewi Contoh"),
    ("note", "Synthetic sample CV - fictional person, for demonstration only"),
    ("gap", ""),
    ("h", "SUMMARY"),
    ("p", "Data analyst focused on reporting, dashboards and demand forecasting."),
    ("gap", ""),
    ("h", "EXPERIENCE"),
    ("p", "Data Analyst - PT Contoh Analitika | Jan 2021 - Mar 2024"),
    ("p", "- Built SQL-based dashboards in Tableau for weekly sales reporting."),
    ("p", "- Automated weekly reports with Python (pandas), saving about 6 hours per week."),
    ("p", "- Forecasted product demand with ARIMA models."),
    ("p", "Research Assistant - Universitas Contoh | 2019 - 2020"),
    ("p", "- Cleaned survey datasets in R and prepared descriptive statistics."),
    ("gap", ""),
    ("h", "EDUCATION"),
    ("p", "Bachelor of Statistics - Universitas Contoh | 2015 - 2019"),
    ("gap", ""),
    ("h", "SKILLS"),
    ("p", "Python, SQL, R, Tableau, Microsoft Excel"),
    ("gap", ""),
    ("h", "PROJECTS"),
    ("p", "Retail Demand Forecasting - Time series forecasting project comparing ARIMA and Prophet on public retail data."),
    ("gap", ""),
    ("h", "CERTIFICATIONS"),
    ("p", "Google Data Analytics Certificate (2022)"),
]


def build(path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    _, height = A4
    y = height - 60
    for style, text in LINES:
        if style == "title":
            pdf.setFont("Helvetica-Bold", 20)
            step = 30
        elif style == "h":
            pdf.setFont("Helvetica-Bold", 12)
            step = 20
        elif style == "note":
            pdf.setFont("Helvetica-Oblique", 9)
            step = 16
        elif style == "gap":
            y -= 8
            continue
        else:
            pdf.setFont("Helvetica", 10)
            step = 15
        pdf.drawString(50, y, text)
        y -= step
    pdf.save()


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/synthetic_cv.pdf")
    build(target)
    print(f"wrote {target}")
