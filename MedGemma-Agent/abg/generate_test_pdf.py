"""
Generate a realistic test ABG lab report PDF using reportlab.
Run: python abg/generate_test_pdf.py
Output: test_abg_report.pdf in the project root.
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

OUTPUT = Path(__file__).parent.parent / "test_abg_report.pdf"

styles = getSampleStyleSheet()

HOSPITAL_BLUE = colors.HexColor("#1d4ed8")
HEADER_BG     = colors.HexColor("#eff6ff")
ALT_ROW       = colors.HexColor("#f8fafc")
NORMAL_GREEN  = colors.HexColor("#16a34a")
ABNORMAL_RED  = colors.HexColor("#dc2626")
BORDER_GREY   = colors.HexColor("#cbd5e1")

title_style = ParagraphStyle(
    "HospTitle", parent=styles["Title"],
    fontSize=16, textColor=HOSPITAL_BLUE, spaceAfter=2, alignment=TA_CENTER,
)
subtitle_style = ParagraphStyle(
    "HospSubtitle", parent=styles["Normal"],
    fontSize=10, textColor=colors.HexColor("#64748b"), alignment=TA_CENTER,
)
section_style = ParagraphStyle(
    "SectionHead", parent=styles["Heading2"],
    fontSize=10, textColor=HOSPITAL_BLUE, spaceAfter=4, spaceBefore=8,
    borderPad=4,
)
label_style = ParagraphStyle(
    "Label", parent=styles["Normal"],
    fontSize=9, textColor=colors.HexColor("#374151"),
)
normal_val = ParagraphStyle(
    "NormalVal", parent=styles["Normal"],
    fontSize=9, textColor=NORMAL_GREEN, fontName="Helvetica-Bold",
)
abnormal_val = ParagraphStyle(
    "AbnormalVal", parent=styles["Normal"],
    fontSize=9, textColor=ABNORMAL_RED, fontName="Helvetica-Bold",
)
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=8.5, textColor=colors.HexColor("#1e293b"), leading=13,
)
disclaimer_style = ParagraphStyle(
    "Disclaimer", parent=styles["Normal"],
    fontSize=7.5, textColor=colors.HexColor("#94a3b8"), alignment=TA_CENTER,
)

def flag(val, lo, hi, invert=False):
    """Return 'H', 'L', or '' based on range."""
    if val < lo:
        return "L"
    if val > hi:
        return "H"
    return ""

def make_result_row(param, value, unit, ref_range, flag_str, is_abnormal):
    val_style = abnormal_val if is_abnormal else normal_val
    return [
        Paragraph(param, label_style),
        Paragraph(f"<b>{value}</b>", val_style),
        Paragraph(unit, label_style),
        Paragraph(ref_range, label_style),
        Paragraph(f"<b>{flag_str}</b>", abnormal_val if flag_str else label_style),
    ]


def build():
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("CITY GENERAL HOSPITAL", title_style))
    story.append(Paragraph("Department of Pathology & Laboratory Medicine", subtitle_style))
    story.append(Paragraph("Arterial Blood Gas Analysis Report", subtitle_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=HOSPITAL_BLUE))
    story.append(Spacer(1, 0.3*cm))

    # ── Patient & Specimen Info ───────────────────────────────────────────────
    info_data = [
        [Paragraph("<b>Patient ID:</b> ABG-TEST-001", label_style),
         Paragraph("<b>Ward / Unit:</b> Post-Op Recovery (ICU-4)", label_style)],
        [Paragraph("<b>Patient Name:</b> Test Patient", label_style),
         Paragraph("<b>Requesting Clinician:</b> Dr. J. Smith", label_style)],
        [Paragraph("<b>DOB:</b> 1975-03-14  (Age: 50)", label_style),
         Paragraph("<b>Collection Time:</b> 2026-04-15  09:42", label_style)],
        [Paragraph("<b>MRN:</b> MRN-2026-00541", label_style),
         Paragraph("<b>Report Time:</b> 2026-04-15  09:55", label_style)],
    ]
    info_table = Table(info_data, colWidths=[9*cm, 9*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HEADER_BG),
        ("BOX",        (0, 0), (-1, -1), 0.5, BORDER_GREY),
        ("INNERGRID",  (0, 0), (-1, -1), 0.3, BORDER_GREY),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Clinical Context ──────────────────────────────────────────────────────
    story.append(Paragraph("CLINICAL CONTEXT", section_style))
    story.append(Paragraph(
        "Post-operative day 2 following emergency laparotomy for bowel perforation. "
        "Patient currently on 40% O2 via Venturi mask (FiO2: 0.40). "
        "No known COPD. No ketosis. Haemodynamically borderline — MAP 62 mmHg.",
        body_style
    ))
    story.append(Spacer(1, 0.4*cm))

    # ── Primary ABG Results ───────────────────────────────────────────────────
    story.append(Paragraph("ARTERIAL BLOOD GAS RESULTS", section_style))

    col_hdrs = [
        Paragraph("<b>Parameter</b>", label_style),
        Paragraph("<b>Result</b>",    label_style),
        Paragraph("<b>Unit</b>",      label_style),
        Paragraph("<b>Reference</b>", label_style),
        Paragraph("<b>Flag</b>",      label_style),
    ]

    abg_rows = [
        col_hdrs,
        make_result_row("pH",      "7.28", "",      "7.35 – 7.45", "L", True),
        make_result_row("PaO2",    "68",   "mmHg",  "80 – 100",    "L", True),
        make_result_row("PaCO2",   "52",   "mmHg",  "35 – 45",     "H", True),
        make_result_row("HCO3-",   "22",   "mmol/L","22 – 26",     "",  False),
        make_result_row("Base Excess (BE)", "-4.2", "mmol/L", "-2 – +2", "L", True),
        make_result_row("Lactate", "3.1",  "mmol/L","0.5 – 2.0",   "H", True),
        make_result_row("SpO2 (calc.)", "91", "%",  "95 – 100",    "L", True),
        make_result_row("FiO2",    "0.40", "(fraction)", "0.21 – 1.00", "", False),
    ]

    abg_table = Table(abg_rows, colWidths=[5.5*cm, 3*cm, 2.5*cm, 4*cm, 1.5*cm])
    abg_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), HOSPITAL_BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, ALT_ROW]),
        ("BOX",          (0, 0), (-1, -1), 0.5, BORDER_GREY),
        ("INNERGRID",    (0, 0), (-1, -1), 0.3, BORDER_GREY),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 9),
    ]))
    story.append(abg_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Electrolytes ──────────────────────────────────────────────────────────
    story.append(Paragraph("ELECTROLYTES & METABOLIC PANEL", section_style))

    elec_rows = [
        col_hdrs,
        make_result_row("Na+",     "138",  "mmol/L", "135 – 145",  "",  False),
        make_result_row("Cl-",     "102",  "mmol/L", "98 – 106",   "",  False),
        make_result_row("Albumin", "3.5",  "g/dL",   "3.5 – 5.0",  "",  False),
        make_result_row("Glucose", "6.2",  "mmol/L", "3.9 – 6.1",  "H", True),
    ]

    elec_table = Table(elec_rows, colWidths=[5.5*cm, 3*cm, 2.5*cm, 4*cm, 1.5*cm])
    elec_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), HOSPITAL_BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, ALT_ROW]),
        ("BOX",          (0, 0), (-1, -1), 0.5, BORDER_GREY),
        ("INNERGRID",    (0, 0), (-1, -1), 0.3, BORDER_GREY),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 9),
    ]))
    story.append(elec_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Interpretation Box ────────────────────────────────────────────────────
    story.append(Paragraph("PRELIMINARY INTERPRETATION", section_style))
    interp_text = (
        "<b>Acid-Base:</b> Respiratory acidosis (pH 7.28, PaCO2 52 mmHg) with mild metabolic compensation "
        "(HCO3- 22 mmol/L). Base deficit -4.2 mmol/L consistent with mild metabolic acidosis component.<br/><br/>"
        "<b>Oxygenation:</b> Type II Respiratory Failure — hypoxia (PaO2 68 mmHg on FiO2 0.40, P/F ratio = 170) "
        "combined with hypercapnia (PaCO2 52 mmHg). SpO2 91% subtherapeutic.<br/><br/>"
        "<b>Perfusion:</b> Elevated lactate 3.1 mmol/L — consistent with tissue hypoperfusion. "
        "Clinical picture concerning for early sepsis in post-operative setting.<br/><br/>"
        "<b>Electrolytes:</b> Na 138, Cl 102 — anion gap borderline elevated. Albumin 3.5 g/dL. "
        "Mildly elevated glucose 6.2 mmol/L, no ketosis identified."
    )
    interp_table = Table(
        [[Paragraph(interp_text, body_style)]],
        colWidths=[16.5*cm]
    )
    interp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef9c3")),
        ("BOX",        (0, 0), (-1, -1), 1.0, colors.HexColor("#eab308")),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(interp_table)
    story.append(Spacer(1, 0.6*cm))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_GREY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "LABORATORY USE ONLY — For clinical decision support. Verified by: Auto-analyser ABL90 FLEX PLUS. "
        "This report is confidential and intended solely for the requesting clinician.",
        disclaimer_style
    ))

    doc.build(story)
    print(f"PDF written to: {OUTPUT}")


if __name__ == "__main__":
    build()
