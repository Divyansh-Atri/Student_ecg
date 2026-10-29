"""
Tests for guardrails and clinical scoring engine.
Run with: pytest tests/test_guardrails.py -v
"""

import pytest
from datetime import datetime, timezone
from vitals.schemas import VitalsSnapshotInput, AlertLevel
from guardrails.clinical_rules import (
    calculate_news2, calculate_qsofa, determine_alert_level,
)
from guardrails.input_guardrails import validate_input
from guardrails.output_guardrails import validate_llm_output


def make_snapshot(**kwargs):
    defaults = dict(
        patient_id="test-patient",
        systolic_bp=120, diastolic_bp=80,
        heart_rate=72, spo2=98,
        snapshot_timestamp=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return VitalsSnapshotInput(**defaults)


# ── NEWS2 scoring ─────────────────────────────────────────────────────────────

def test_news2_all_normal():
    s = calculate_news2(120, 80, 72, 98)
    assert s.total_score == 0
    assert not s.any_single_critical

def test_news2_critical_sbp():
    s = calculate_news2(88, 60, 72, 98)
    assert s.sbp_score == 3
    assert s.any_single_critical

def test_news2_critical_hr():
    s = calculate_news2(120, 80, 140, 98)
    assert s.heart_rate_score == 3
    assert s.any_single_critical

def test_news2_critical_spo2():
    s = calculate_news2(120, 80, 72, 90)
    assert s.spo2_score == 3
    assert s.any_single_critical

def test_news2_high_total():
    # TC-05: multi-vital deterioration
    s = calculate_news2(85, 52, 136, 90)
    assert s.total_score >= 7
    assert s.any_single_critical


# ── qSOFA ─────────────────────────────────────────────────────────────────────

def test_qsofa_normal():
    q = calculate_qsofa(120, 72, 98)
    assert q.score == 0
    assert not q.high_risk

def test_qsofa_high_risk():
    # TC-04: SBP=88, HR=96, SpO2=91%
    q = calculate_qsofa(88, 96, 91)
    assert q.sbp_flag
    assert q.hr_flag
    assert q.spo2_flag
    assert q.score == 3
    assert q.high_risk


# ── Alert level determination ─────────────────────────────────────────────────

def test_alert_level_normal():
    news2 = calculate_news2(120, 80, 72, 98)
    qsofa = calculate_qsofa(120, 72, 98)
    assert determine_alert_level(news2, qsofa) == AlertLevel.NORMAL

def test_alert_level_critical_from_single_vital():
    news2 = calculate_news2(88, 60, 72, 98)  # SBP score 3
    qsofa = calculate_qsofa(88, 72, 98)
    assert determine_alert_level(news2, qsofa) == AlertLevel.CRITICAL

def test_alert_level_critical_from_qsofa():
    news2 = calculate_news2(102, 65, 95, 93)  # total ~4 = HIGH, but qSOFA≥2
    qsofa = calculate_qsofa(102, 95, 93)
    assert qsofa.high_risk
    assert determine_alert_level(news2, qsofa) == AlertLevel.CRITICAL


# ── Input guardrails ──────────────────────────────────────────────────────────

def test_input_valid_normal():
    result = validate_input(make_snapshot())
    assert result.valid
    assert not result.immediate_critical

def test_input_rejects_impossible_hr():
    with pytest.raises(Exception):
        make_snapshot(heart_rate=500)

def test_input_rejects_sbp_less_than_dbp():
    with pytest.raises(Exception):
        make_snapshot(systolic_bp=70, diastolic_bp=80)

def test_input_immediate_critical_for_critical_vitals():
    result = validate_input(make_snapshot(systolic_bp=85, diastolic_bp=55, heart_rate=140, spo2=89))
    assert result.valid
    assert result.immediate_critical


# ── Output guardrails ─────────────────────────────────────────────────────────

def test_output_valid():
    snapshot = make_snapshot(systolic_bp=88, heart_rate=95, spo2=91)
    llm_out = {
        "risk_level": "CRITICAL",
        "confidence": 0.85,
        "reasoning": "Patient has systolic BP of 88 mmHg which is critically low, heart rate of 95 bpm, and SpO2 of 91%.",
        "sepsis_risk_flag": True,
        "contributing_factors": ["hypotension", "tachycardia"],
        "recommended_action": "immediate_response",
        "limitations": "Temperature and respiratory rate not available.",
        "disclaimer": "CLINICAL DECISION SUPPORT ONLY.",
    }
    result = validate_llm_output(llm_out, AlertLevel.CRITICAL, snapshot)
    assert result.valid

def test_output_rejects_prohibited_content():
    snapshot = make_snapshot()
    llm_out = {
        "risk_level": "CRITICAL",
        "confidence": 0.9,
        "reasoning": "SpO2 98, administer norepinephrine 0.1 mcg/kg/min",
        "sepsis_risk_flag": False,
    }
    result = validate_llm_output(llm_out, AlertLevel.NORMAL, snapshot)
    assert not result.valid

def test_output_rejects_ungrounded_reasoning():
    snapshot = make_snapshot(systolic_bp=88)
    llm_out = {
        "risk_level": "HIGH",
        "confidence": 0.7,
        "reasoning": "The patient appears to be at risk based on general clinical assessment.",
        "sepsis_risk_flag": False,
    }
    result = validate_llm_output(llm_out, AlertLevel.HIGH, snapshot)
    assert not result.valid

def test_output_rejects_large_contradiction():
    snapshot = make_snapshot(systolic_bp=88, heart_rate=140, spo2=88)
    llm_out = {
        "risk_level": "NORMAL",
        "confidence": 0.8,
        "reasoning": "Systolic BP 88, heart rate 140, SpO2 88 — patient seems fine.",
        "sepsis_risk_flag": False,
    }
    result = validate_llm_output(llm_out, AlertLevel.CRITICAL, snapshot)
    assert not result.valid  # NORMAL vs CRITICAL = 3-tier contradiction
