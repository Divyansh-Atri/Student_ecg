"""
ABG Interpreter — rule-based arterial blood gas analysis.

Ported from abg_interpreter_v1 notebook.
Accepts values in mmHg (clinical standard); converts to kPa internally.

Analyses:
  - Respiratory failure type (Type 1 / Type 2)
  - Acid-base primary disorder (respiratory/metabolic acidosis/alkalosis)
  - Compensation adequacy (Winters' formula, acute/chronic rules)
  - Anion gap with albumin correction and delta-delta ratio
  - Clinical alert severity level
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

_KPA_PER_MMHG = 1 / 7.5006


# ── Enumerations ──────────────────────────────────────────────────────────────

class RespiratoryFailureType(str, Enum):
    NONE   = "none"
    TYPE_1 = "type1"   # Hypoxemic (low PaO2, normal/low PaCO2)
    TYPE_2 = "type2"   # Hypercapnic (low PaO2, high PaCO2)


class AcidBaseStatus(str, Enum):
    NORMAL    = "normal"
    ACIDOSIS  = "acidosis"
    ALKALOSIS = "alkalosis"


class PrimaryDisorder(str, Enum):
    NONE                  = "none"
    RESPIRATORY_ACIDOSIS  = "respiratory_acidosis"
    RESPIRATORY_ALKALOSIS = "respiratory_alkalosis"
    METABOLIC_ACIDOSIS    = "metabolic_acidosis"
    METABOLIC_ALKALOSIS   = "metabolic_alkalosis"
    MIXED                 = "mixed"


class ABGAlertLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    NORMAL   = "NORMAL"


# ── Internal value holder (kPa) ───────────────────────────────────────────────

@dataclass
class _ABGValues:
    """Internal representation in kPa. Do not use directly — use ABGInterpreter."""
    pH:          float
    PaO2_kPa:   float
    PaCO2_kPa:  float
    HCO3:        float   # mmol/L
    base_excess: float   # mmol/L
    lactate:     float   # mmol/L
    FiO2:        float = 0.21
    chronic_copd: bool = False
    Na:          Optional[float] = None
    Cl:          Optional[float] = None
    albumin:     Optional[float] = None
    glucose:     Optional[float] = None
    ketones:     Optional[bool]  = None

    def PaO2_mmHg(self)  -> float: return self.PaO2_kPa  * 7.5006
    def PaCO2_mmHg(self) -> float: return self.PaCO2_kPa * 7.5006


@dataclass
class _Ref:
    pH_min:          float = 7.35
    pH_max:          float = 7.45
    PaO2_min_kPa:   float = 10.7   # 80 mmHg
    PaCO2_min_kPa:  float = 4.7    # 35 mmHg
    PaCO2_max_kPa:  float = 6.0    # 45 mmHg
    HCO3_min:        float = 22.0
    HCO3_max:        float = 26.0
    AG_normal:       float = 12.0
    AG_min:          float = 8.0


# ── Main interpreter ──────────────────────────────────────────────────────────

class ABGInterpreter:
    """
    Rule-based ABG interpreter.

    Accepts values in mmHg for PaO2 and PaCO2 (standard clinical practice).
    All other values in SI units (pH unitless, HCO3/lactate/electrolytes in mmol/L,
    albumin in g/dL, FiO2 as fraction 0.21–1.0).
    """

    def __init__(
        self,
        pH: float,
        pao2_mmhg: float,
        paco2_mmhg: float,
        hco3: float,
        base_excess: float,
        lactate: float,
        fio2: float = 0.21,
        chronic_copd: bool = False,
        na: Optional[float] = None,
        cl: Optional[float] = None,
        albumin: Optional[float] = None,
        glucose: Optional[float] = None,
        ketones: Optional[bool] = None,
    ):
        self._v = _ABGValues(
            pH=pH,
            PaO2_kPa=pao2_mmhg  * _KPA_PER_MMHG,
            PaCO2_kPa=paco2_mmhg * _KPA_PER_MMHG,
            HCO3=hco3,
            base_excess=base_excess,
            lactate=lactate,
            FiO2=fio2,
            chronic_copd=chronic_copd,
            Na=na, Cl=cl, albumin=albumin,
            glucose=glucose, ketones=ketones,
        )
        self._ref = _Ref()

    # ── Public entry point ─────────────────────────────────────────────────────

    def interpret(self) -> Dict:
        """
        Run full ABG interpretation.

        Returns a dict with keys:
          respiratory, acid_base, compensation, metabolic,
          summary, clinical_alerts, alert_level
        """
        results: Dict = {
            "respiratory":     self._assess_respiratory(),
            "acid_base":       self._classify_acid_base(),
            "compensation":    self._assess_compensation(),
            "metabolic":       self._assess_metabolic(),
            "summary":         None,
            "clinical_alerts": [],
            "alert_level":     None,
        }
        results["summary"]         = self._generate_summary(results)
        results["clinical_alerts"] = self._generate_alerts(results)
        results["alert_level"]     = self._determine_alert_level(results)
        return results

    # ── Respiratory ────────────────────────────────────────────────────────────

    def _assess_respiratory(self) -> Dict:
        v, ref = self._v, self._ref
        out: Dict = {
            "hypoxia": False,
            "type": RespiratoryFailureType.NONE,
            "pf_ratio": None,
            "explanation": [],
        }

        if v.PaO2_kPa < ref.PaO2_min_kPa:
            out["hypoxia"] = True
            out["explanation"].append(
                f"Hypoxia present (PaO2 {v.PaO2_mmHg():.0f} mmHg on FiO2 {v.FiO2:.2f})"
            )
            pf = v.PaO2_mmHg() / v.FiO2
            out["pf_ratio"] = round(pf, 1)

            if v.PaCO2_kPa <= ref.PaCO2_max_kPa:
                out["type"] = RespiratoryFailureType.TYPE_1
                out["explanation"].append(
                    f"Type 1 respiratory failure — hypoxaemia with normal/low PaCO2 ({v.PaCO2_mmHg():.0f} mmHg)"
                )
            else:
                out["type"] = RespiratoryFailureType.TYPE_2
                out["explanation"].append(
                    f"Type 2 respiratory failure — hypoxaemia with elevated PaCO2 ({v.PaCO2_mmHg():.0f} mmHg)"
                )

            if pf < 100:
                out["explanation"].append(f"P/F ratio {pf:.0f} — severe hypoxaemia (ARDS criteria)")
            elif pf < 200:
                out["explanation"].append(f"P/F ratio {pf:.0f} — moderate hypoxaemia")
            elif pf < 300:
                out["explanation"].append(f"P/F ratio {pf:.0f} — mild hypoxaemia")
            else:
                out["explanation"].append(f"P/F ratio {pf:.0f} — borderline")
        else:
            out["explanation"].append(f"No hypoxia (PaO2 {v.PaO2_mmHg():.0f} mmHg)")

        return out

    # ── Acid-base ──────────────────────────────────────────────────────────────

    def _classify_acid_base(self) -> Dict:
        v, ref = self._v, self._ref
        out: Dict = {
            "status": AcidBaseStatus.NORMAL,
            "primary_disorder": PrimaryDisorder.NONE,
            "acute_vs_chronic": None,
            "explanation": [],
        }

        if v.pH < ref.pH_min:
            out["status"] = AcidBaseStatus.ACIDOSIS
            out["explanation"].append(f"Acidaemia (pH {v.pH:.2f})")
        elif v.pH > ref.pH_max:
            out["status"] = AcidBaseStatus.ALKALOSIS
            out["explanation"].append(f"Alkalaemia (pH {v.pH:.2f})")
        else:
            out["explanation"].append(f"pH within normal range ({v.pH:.2f})")

        co2_mmhg = v.PaCO2_mmHg()

        # Respiratory acidosis
        if v.pH < ref.pH_min and v.PaCO2_kPa > ref.PaCO2_max_kPa:
            out["primary_disorder"] = PrimaryDisorder.RESPIRATORY_ACIDOSIS
            out["explanation"].append(
                f"Primary respiratory acidosis (PaCO2 {co2_mmhg:.0f} mmHg)"
            )
            if v.HCO3 >= 28:
                out["acute_vs_chronic"] = "chronic"
                out["explanation"].append(f"Likely chronic — compensated HCO3 {v.HCO3:.1f} mmol/L")
            else:
                out["acute_vs_chronic"] = "acute"
                out["explanation"].append(f"Likely acute — HCO3 {v.HCO3:.1f} mmol/L not significantly elevated")

        # Respiratory alkalosis
        elif v.pH > ref.pH_max and v.PaCO2_kPa < ref.PaCO2_min_kPa:
            out["primary_disorder"] = PrimaryDisorder.RESPIRATORY_ALKALOSIS
            out["explanation"].append(
                f"Primary respiratory alkalosis (PaCO2 {co2_mmhg:.0f} mmHg)"
            )
            if v.HCO3 < ref.HCO3_min:
                out["acute_vs_chronic"] = "chronic"
                out["explanation"].append(f"Likely chronic — compensated HCO3 {v.HCO3:.1f} mmol/L")
            else:
                out["acute_vs_chronic"] = "acute"
                out["explanation"].append(f"Likely acute — HCO3 {v.HCO3:.1f} mmol/L near normal")

        # Metabolic acidosis
        elif v.pH < ref.pH_min and v.HCO3 < ref.HCO3_min:
            out["primary_disorder"] = PrimaryDisorder.METABOLIC_ACIDOSIS
            out["explanation"].append(
                f"Primary metabolic acidosis (HCO3 {v.HCO3:.1f} mmol/L)"
            )

        # Metabolic alkalosis
        elif v.pH > ref.pH_max and v.HCO3 > ref.HCO3_max:
            out["primary_disorder"] = PrimaryDisorder.METABOLIC_ALKALOSIS
            out["explanation"].append(
                f"Primary metabolic alkalosis (HCO3 {v.HCO3:.1f} mmol/L)"
            )

        # Mixed
        elif (
            (v.PaCO2_kPa > ref.PaCO2_max_kPa or v.PaCO2_kPa < ref.PaCO2_min_kPa)
            and (v.HCO3 > ref.HCO3_max or v.HCO3 < ref.HCO3_min)
        ):
            out["primary_disorder"] = PrimaryDisorder.MIXED
            out["explanation"].append("Possible mixed acid-base disorder")

        return out

    # ── Compensation ───────────────────────────────────────────────────────────

    def _assess_compensation(self) -> Dict:
        v = self._v
        out: Dict = {
            "compensated": None,
            "expected_value": None,
            "measured_value": None,
            "explanation": [],
        }

        ab = self._classify_acid_base()
        disorder = ab["primary_disorder"]
        co2_mmhg = v.PaCO2_mmHg()

        if disorder == PrimaryDisorder.METABOLIC_ACIDOSIS:
            # Winters' formula: expected PaCO2 = 1.5 * HCO3 + 8 (±2)
            expected = 1.5 * v.HCO3 + 8
            out["expected_value"] = round(expected, 1)
            out["measured_value"] = round(co2_mmhg, 1)
            diff = co2_mmhg - expected
            if abs(diff) <= 2:
                out["compensated"] = "appropriate"
                out["explanation"].append(
                    f"Appropriate respiratory compensation (Winters': expected PaCO2 {expected:.1f} mmHg, measured {co2_mmhg:.1f} mmHg)"
                )
            elif diff > 2:
                out["compensated"] = "additional_resp_acidosis"
                out["explanation"].append(
                    f"Superimposed respiratory acidosis (PaCO2 {co2_mmhg:.1f} mmHg > expected {expected:.1f} mmHg)"
                )
            else:
                out["compensated"] = "additional_resp_alkalosis"
                out["explanation"].append(
                    f"Superimposed respiratory alkalosis (PaCO2 {co2_mmhg:.1f} mmHg < expected {expected:.1f} mmHg)"
                )

        elif disorder == PrimaryDisorder.METABOLIC_ALKALOSIS:
            expected = 0.7 * (v.HCO3 - 24) + 40
            out["expected_value"] = round(expected, 1)
            out["measured_value"] = round(co2_mmhg, 1)
            diff = co2_mmhg - expected
            if abs(diff) <= 5:
                out["compensated"] = "appropriate"
                out["explanation"].append(
                    f"Appropriate respiratory compensation (expected PaCO2 {expected:.1f} mmHg, measured {co2_mmhg:.1f} mmHg)"
                )
            elif diff > 5:
                out["compensated"] = "additional_resp_acidosis"
                out["explanation"].append(
                    f"Additional respiratory acidosis (PaCO2 {co2_mmhg:.1f} > expected {expected:.1f} mmHg)"
                )
            else:
                out["compensated"] = "additional_resp_alkalosis"
                out["explanation"].append(
                    f"Additional respiratory alkalosis (PaCO2 {co2_mmhg:.1f} < expected {expected:.1f} mmHg)"
                )

        elif disorder in (PrimaryDisorder.RESPIRATORY_ACIDOSIS, PrimaryDisorder.RESPIRATORY_ALKALOSIS):
            delta_co2 = co2_mmhg - 40

            if disorder == PrimaryDisorder.RESPIRATORY_ACIDOSIS:
                exp_acute   = 24 + (delta_co2 / 10) * 1.0
                exp_chronic = 24 + (delta_co2 / 10) * 3.5
            else:
                exp_acute   = 24 - (abs(delta_co2) / 10) * 2.0
                exp_chronic = 24 - (abs(delta_co2) / 10) * 4.5

            if abs(v.HCO3 - exp_acute) <= abs(v.HCO3 - exp_chronic):
                out["compensated"] = "acute"
                out["explanation"].append(
                    f"Consistent with acute pattern (HCO3 {v.HCO3:.1f} mmol/L, expected acute {exp_acute:.1f})"
                )
            else:
                out["compensated"] = "chronic"
                out["explanation"].append(
                    f"Consistent with chronic pattern (HCO3 {v.HCO3:.1f} mmol/L, expected chronic {exp_chronic:.1f})"
                )

        return out

    # ── Metabolic / Anion gap ──────────────────────────────────────────────────

    def _assess_metabolic(self) -> Dict:
        v, ref = self._v, self._ref
        out: Dict = {
            "anion_gap": None,
            "anion_gap_corrected": None,
            "gap_type": None,
            "delta_delta": None,
            "likely_causes": [],
            "explanation": [],
        }

        if v.Na is None or v.Cl is None:
            out["explanation"].append("Na+ and Cl- required for anion gap calculation")
            return out

        ag = v.Na - (v.Cl + v.HCO3)
        out["anion_gap"] = round(ag, 1)

        if v.albumin is not None:
            ag_corr = ag + 2.5 * (4.0 - v.albumin)
            out["anion_gap_corrected"] = round(ag_corr, 1)
            out["explanation"].append(
                f"Anion gap: {ag:.1f} mmol/L (albumin-corrected: {ag_corr:.1f} mmol/L)"
            )
        else:
            ag_corr = ag
            out["anion_gap_corrected"] = round(ag, 1)
            out["explanation"].append(f"Anion gap: {ag:.1f} mmol/L (no albumin correction)")

        if ag_corr > 14:
            out["gap_type"] = "high"
            out["explanation"].append("High anion gap metabolic acidosis (HAGMA)")

            if v.lactate > 2.0:
                out["likely_causes"].append(
                    f"Lactic acidosis (lactate {v.lactate:.1f} mmol/L) — consider sepsis, shock, tissue hypoperfusion"
                )
            if v.ketones:
                out["likely_causes"].append("Ketoacidosis — consider DKA, starvation, alcoholic ketoacidosis")
            if v.glucose and v.glucose > 15:
                out["likely_causes"].append(
                    f"Hyperglycaemia ({v.glucose:.1f} mmol/L) — consider DKA"
                )
            if not out["likely_causes"]:
                out["likely_causes"].append("Consider: uraemia, toxins (MUDPILES), or unmeasured anions")

            # Delta-delta ratio
            delta_ag  = ag_corr - ref.AG_normal
            delta_hco = 24 - v.HCO3
            if delta_hco != 0:
                dd = delta_ag / delta_hco
                out["delta_delta"] = round(dd, 2)
                if 0.8 <= dd <= 1.2:
                    out["explanation"].append(f"Delta-delta {dd:.2f} — pure HAGMA")
                elif dd > 1.2:
                    out["explanation"].append(f"Delta-delta {dd:.2f} — concurrent metabolic alkalosis")
                else:
                    out["explanation"].append(f"Delta-delta {dd:.2f} — additional normal-gap acidosis")

        elif ag_corr < ref.AG_min:
            out["gap_type"] = "low"
            out["explanation"].append("Low anion gap — consider lab error or paraproteinaemia")
        else:
            out["gap_type"] = "normal"
            out["explanation"].append("Normal anion gap metabolic acidosis (NAGMA / hyperchloraemic)")
            out["likely_causes"].append(
                "Consider: diarrhoea, renal tubular acidosis, early renal failure, excess Cl- resuscitation"
            )

        return out

    # ── Summary ────────────────────────────────────────────────────────────────

    def _generate_summary(self, results: Dict) -> str:
        parts: List[str] = []
        resp = results["respiratory"]
        ab   = results["acid_base"]
        comp = results["compensation"]
        met  = results["metabolic"]

        if resp["type"] != RespiratoryFailureType.NONE:
            parts.append(f"{resp['type'].value.replace('type', 'Type ')} respiratory failure")
        elif resp["hypoxia"]:
            parts.append("Hypoxaemia without hypercapnia")
        else:
            parts.append("Normal oxygenation")

        if ab["primary_disorder"] != PrimaryDisorder.NONE:
            name = ab["primary_disorder"].value.replace("_", " ").title()
            prefix = ab.get("acute_vs_chronic") or ""
            parts.append((f"{prefix.title()} {name}" if prefix else name).strip())
        else:
            parts.append("Normal acid-base status")

        if comp.get("compensated") == "appropriate":
            parts.append("with appropriate compensation")
        elif comp.get("compensated") in ("additional_resp_acidosis", "additional_resp_alkalosis"):
            parts.append(f"with {comp['compensated'].replace('_', ' ')}")

        if met.get("gap_type") == "high" and met.get("likely_causes"):
            parts.append(f"({met['likely_causes'][0]})")

        return ". ".join(parts) + "."

    # ── Clinical alerts ────────────────────────────────────────────────────────

    def _generate_alerts(self, results: Dict) -> List[str]:
        v = self._v
        alerts: List[str] = []

        if results["respiratory"].get("pf_ratio") and results["respiratory"]["pf_ratio"] < 200:
            alerts.append(f"Severe hypoxaemia — P/F ratio {results['respiratory']['pf_ratio']:.0f}")

        if v.lactate >= 4.0:
            alerts.append(f"Markedly elevated lactate {v.lactate:.1f} mmol/L — consider sepsis bundle")
        elif v.lactate >= 2.0:
            alerts.append(f"Elevated lactate {v.lactate:.1f} mmol/L")

        if v.pH < 7.20:
            alerts.append(f"Severe acidaemia pH {v.pH:.2f} — urgent review required")
        elif v.pH > 7.55:
            alerts.append(f"Severe alkalaemia pH {v.pH:.2f}")

        if v.chronic_copd and results["compensation"].get("compensated") == "acute":
            alerts.append("Possible COPD exacerbation — acute on chronic respiratory acidosis")

        return alerts

    # ── Alert level ────────────────────────────────────────────────────────────

    def _determine_alert_level(self, results: Dict) -> ABGAlertLevel:
        v = self._v
        pf = results["respiratory"].get("pf_ratio")

        # CRITICAL
        if v.pH < 7.20 or v.pH > 7.55:
            return ABGAlertLevel.CRITICAL
        if pf is not None and pf < 100:
            return ABGAlertLevel.CRITICAL
        if v.lactate >= 4.0:
            return ABGAlertLevel.CRITICAL

        # HIGH
        if v.pH < 7.25 or v.pH > 7.50:
            return ABGAlertLevel.HIGH
        if pf is not None and pf < 200:
            return ABGAlertLevel.HIGH
        if v.lactate >= 2.0:
            return ABGAlertLevel.HIGH
        if results["respiratory"]["type"] == RespiratoryFailureType.TYPE_2:
            return ABGAlertLevel.HIGH

        # MEDIUM
        ab = results["acid_base"]
        if ab["primary_disorder"] != PrimaryDisorder.NONE:
            return ABGAlertLevel.MEDIUM
        if results["respiratory"]["hypoxia"]:
            return ABGAlertLevel.MEDIUM

        return ABGAlertLevel.NORMAL
