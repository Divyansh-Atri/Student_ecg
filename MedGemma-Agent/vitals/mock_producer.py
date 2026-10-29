"""
Mock vitals producer — loads pre-defined test cases from test_cases.yaml
and packages them as VitalsSnapshotInput objects.
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import yaml

from vitals.schemas import VitalsSnapshotInput

SYNTHETIC_EHR_DIR = Path(__file__).parent.parent / "synthetic-EHRs"


def _load_test_cases() -> dict:
    path = Path(__file__).parent / "test_cases.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["test_cases"]


def list_test_cases() -> list[dict]:
    cases = _load_test_cases()
    return [
        {
            "id": case_id,
            "name": data["name"],
            "clinical_narrative": data.get("clinical_narrative", "").strip(),
            "patient_id": data["patient_id"],
            "vitals": {
                "systolic_bp": data["systolic_bp"],
                "diastolic_bp": data["diastolic_bp"],
                "heart_rate": data["heart_rate"],
                "spo2": data["spo2"],
            },
            "ehr_file": data.get("ehr_file"),
            "expected_alert": data["expected_alert"],
            "notes": data["notes"],
        }
        for case_id, data in cases.items()
    ]


def get_test_case_snapshot(case_id: str) -> VitalsSnapshotInput:
    cases = _load_test_cases()
    if case_id not in cases:
        raise KeyError(f"Test case '{case_id}' not found. Available: {list(cases.keys())}")
    data = cases[case_id]
    return VitalsSnapshotInput(
        patient_id=data["patient_id"],
        systolic_bp=data["systolic_bp"],
        diastolic_bp=data["diastolic_bp"],
        heart_rate=data["heart_rate"],
        spo2=data["spo2"],
        snapshot_timestamp=datetime.now(timezone.utc),
    )


def get_test_case_ehr_path(case_id: str) -> Path | None:
    """Returns the absolute path to the EHR PDF for this test case, or None."""
    cases = _load_test_cases()
    if case_id not in cases:
        return None
    ehr_file = cases[case_id].get("ehr_file")
    if not ehr_file:
        return None
    path = SYNTHETIC_EHR_DIR / ehr_file
    return path if path.exists() else None
