"""
ABG-specific MedGemma prompt templates.

Kept completely separate from the vitals pipeline prompts.
"""

ABG_SYSTEM_PROMPT = """\
You are MedGemma, a clinical decision support AI specialising in arterial blood gas (ABG) interpretation.
Your role is to provide clinical reasoning about a patient's ABG results and their significance.

STRICT RULES:
1. You MUST respond ONLY with a valid JSON object matching the schema below.
2. You MUST NOT recommend specific drugs, doses, or treatments.
3. You MUST reference the actual ABG values in your reasoning.
4. Your confidence score must honestly reflect uncertainty.
5. You are DECISION SUPPORT ONLY — never a replacement for clinical judgment.

OUTPUT SCHEMA (respond with ONLY this JSON, no other text):
{
  "clinical_significance": "<plain English explanation of the ABG pattern and its clinical meaning>",
  "primary_concern": "<the single most urgent clinical concern from this ABG>",
  "contributing_factors": ["<factor from patient history or ABG values>", ...],
  "confidence": <float 0.0-1.0>,
  "limitations": "<note missing context: temperature, respiratory rate, clinical history, medications>",
  "disclaimer": "CLINICAL DECISION SUPPORT ONLY. This output must not replace clinical judgment."
}
"""

ABG_USER_PROMPT_TEMPLATE = """\
PATIENT ID: {patient_id}

ABG VALUES:
- pH:           {pH:.2f}
- PaO2:         {pao2_mmhg:.0f} mmHg  (on FiO2 {fio2:.2f})
- PaCO2:        {paco2_mmhg:.0f} mmHg
- HCO3:         {hco3:.1f} mmol/L
- Base Excess:  {base_excess:+.1f} mmol/L
- Lactate:      {lactate:.1f} mmol/L{electrolytes_line}{copd_line}

RULE-BASED INTERPRETATION:
- Alert Level:       {alert_level}
- Primary Disorder:  {primary_disorder}
- Respiratory:       {respiratory_summary}
- Compensation:      {compensation_summary}
- Anion Gap:         {ag_summary}
- Clinical Alerts:   {clinical_alerts}

SUMMARY: {summary}

PATIENT EHR CONTEXT:
{ehr_context}

Based on the ABG values, rule-based interpretation, and patient history above, provide your
structured clinical reasoning as JSON.
"""


def build_abg_user_prompt(
    patient_id: str,
    pH: float,
    pao2_mmhg: float,
    paco2_mmhg: float,
    hco3: float,
    base_excess: float,
    lactate: float,
    fio2: float,
    na: float | None,
    cl: float | None,
    chronic_copd: bool,
    interp: dict,
    ehr_context: str,
) -> str:
    elec = ""
    if na is not None and cl is not None:
        elec = f"\n- Electrolytes:  Na+ {na:.0f}, Cl- {cl:.0f} mmol/L"

    copd = "\n- Chronic COPD:  Yes" if chronic_copd else ""

    ab   = interp["acid_base"]
    resp = interp["respiratory"]
    comp = interp["compensation"]
    met  = interp["metabolic"]

    resp_summary = resp["type"].value
    if resp.get("pf_ratio"):
        resp_summary += f", P/F {resp['pf_ratio']:.0f}"

    comp_summary = comp.get("compensated") or "N/A"
    if comp.get("expected_value") and comp.get("measured_value"):
        comp_summary += f" (expected {comp['expected_value']}, measured {comp['measured_value']})"

    if met.get("anion_gap_corrected") is not None:
        ag_summary = f"{met['anion_gap_corrected']} mmol/L ({met['gap_type'] or 'N/A'})"
    else:
        ag_summary = "insufficient data"

    alerts_str = "; ".join(interp["clinical_alerts"]) if interp["clinical_alerts"] else "None"

    return ABG_USER_PROMPT_TEMPLATE.format(
        patient_id=patient_id,
        pH=pH,
        pao2_mmhg=pao2_mmhg,
        paco2_mmhg=paco2_mmhg,
        hco3=hco3,
        base_excess=base_excess,
        lactate=lactate,
        fio2=fio2,
        electrolytes_line=elec,
        copd_line=copd,
        alert_level=interp["alert_level"].value,
        primary_disorder=ab["primary_disorder"].value.replace("_", " ").title(),
        respiratory_summary=resp_summary,
        compensation_summary=comp_summary,
        ag_summary=ag_summary,
        clinical_alerts=alerts_str,
        summary=interp["summary"],
        ehr_context=ehr_context or "No EHR data available.",
    )
