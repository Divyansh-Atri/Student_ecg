"""Load ABG test cases from YAML."""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import yaml

from abg.schemas import ABGInput

_YAML_PATH = Path(__file__).parent / "test_cases.yaml"


def list_abg_test_cases() -> List[Dict]:
    """Return metadata for all ABG test cases (for UI display)."""
    raw = _load_raw()
    out = []
    for tc in raw["test_cases"]:
        out.append({
            "id": tc["id"],
            "name": tc["name"],
            "patient_id": tc["patient_id"],
            "description": tc.get("description", ""),
            "expected_alert": tc["expected_alert"],
            "expected_disorder": tc.get("expected_disorder", ""),
            "notes": tc.get("notes", ""),
            "values": tc["values"],
        })
    return out


def get_abg_test_case_input(case_id: str) -> ABGInput:
    """Build an ABGInput for a given test case ID."""
    raw = _load_raw()
    for tc in raw["test_cases"]:
        if tc["id"] == case_id:
            v = tc["values"]
            return ABGInput(
                patient_id=tc["patient_id"],
                pH=v["pH"],
                pao2_mmhg=v["pao2_mmhg"],
                paco2_mmhg=v["paco2_mmhg"],
                hco3=v["hco3"],
                base_excess=v["base_excess"],
                lactate=v["lactate"],
                fio2=v.get("fio2", 0.21),
                na=v.get("na"),
                cl=v.get("cl"),
                albumin=v.get("albumin"),
                glucose=v.get("glucose"),
                ketones=v.get("ketones"),
                chronic_copd=v.get("chronic_copd", False),
            )
    raise KeyError(f"ABG test case '{case_id}' not found")


def _load_raw() -> Dict:
    with open(_YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)
