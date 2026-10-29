# Software FMEA — Failure Mode and Effects Analysis
**Document ID:** FMEA-001  
**Version:** 1.0  
**Standard:** ISO 14971:2019 + IEC 62304  
**Device:** MedGemma Post-Operative Monitoring Agent  
**Date:** 2026-04-09  

---

## Purpose

Identify software failure modes, their effects on patient safety, and the risk controls that mitigate them. Each hazard is scored for probability (P1–P5) and severity (S1–S5) before and after controls.

---

## Failure Modes

### H-001 — Missed Critical Alert (False Negative)

| Field | Detail |
|---|---|
| **Failure mode** | Agent assigns NORMAL or MEDIUM when vitals indicate CRITICAL |
| **Cause** | LLM hallucinates a safe assessment; output validator fails to catch contradiction |
| **Effect** | Clinician not alerted; patient deteriorates without intervention |
| **Initial risk** | P3 × S5 = **High** |
| **Risk controls** | (1) Rule-based NEWS2/qSOFA runs independently of LLM. (2) CRITICAL from rules is never downgraded by LLM. (3) Contradiction check rejects LLM output if it differs from rule score by >1 tier. (4) Single vital scoring 3 triggers CRITICAL bypass — LLM not consulted. |
| **Residual risk** | P1 × S5 = **Medium** |
| **ALARP rationale** | Full elimination not feasible (rule-based scoring limited to 4 vitals). Residual risk accepted given mandatory clinical review of all outputs. |

---

### H-002 — Spurious Critical Alert (False Positive)

| Field | Detail |
|---|---|
| **Failure mode** | Agent assigns CRITICAL when patient is clinically stable |
| **Cause** | Sensor error produces implausible vitals; LLM overinterprets borderline values |
| **Effect** | Unnecessary clinical intervention; alert fatigue reducing trust in system |
| **Initial risk** | P4 × S2 = **Medium** |
| **Risk controls** | (1) Physiological plausibility check rejects impossible readings before scoring. (2) SBP > DBP cross-field validation. (3) Confidence threshold: LLM confidence <0.6 flags for human review rather than auto-escalating. |
| **Residual risk** | P2 × S2 = **Low** |

---

### H-003 — LLM Recommends Specific Treatment

| Field | Detail |
|---|---|
| **Failure mode** | MedGemma output includes drug names, doses, or direct treatment orders |
| **Cause** | Model generates treatment recommendations outside its sanctioned scope |
| **Effect** | Clinician acts on unsanctioned drug order; patient harm from incorrect treatment |
| **Initial risk** | P3 × S4 = **High** |
| **Risk controls** | (1) Output guardrail regex filter blocks responses containing drug names, dosing language, or administration instructions. (2) Rejected outputs fall back to rule-based score with no LLM reasoning surfaced. (3) Mandatory disclaimer on every output. |
| **Residual risk** | P1 × S4 = **Medium** |
| **ALARP rationale** | Regex filtering cannot catch all possible drug references. Residual risk accepted given mandatory clinical review requirement. |

---

### H-004 — PHI Exposure to LLM

| Field | Detail |
|---|---|
| **Failure mode** | Patient identifiable information sent to MedGemma in prompt |
| **Cause** | De-identification layer fails to strip a HIPAA identifier from EHR chunk |
| **Effect** | PHI included in LLM inference; logged in Ollama logs; potential data breach |
| **Initial risk** | P3 × S3 = **Medium** |
| **Risk controls** | (1) `phi_handler.deidentify()` applies 18-pattern regex strip to all EHR chunks before embedding or LLM use. (2) Patient ID in prompt is pseudonymized (opaque ID, not name). (3) Ollama runs locally — no data leaves the device. |
| **Residual risk** | P2 × S3 = **Medium** |
| **ALARP rationale** | Regex de-identification is not 100% reliable for unstructured clinical text. Documented limitation. Production deployment should use a validated NLP de-identification tool (e.g., Microsoft Presidio). |

---

### H-005 — Audit Log Tampering

| Field | Detail |
|---|---|
| **Failure mode** | Audit trail entries modified or deleted after the fact |
| **Cause** | Direct database access by unauthorized party |
| **Effect** | Loss of compliance evidence; inability to reconstruct clinical decisions |
| **Initial risk** | P2 × S3 = **Medium** |
| **Risk controls** | (1) SHA-256 hash chain: each entry includes hash of previous entry. Tampering with any entry invalidates all subsequent hashes. (2) RBAC: only admin role can read audit log via API. (3) `verify_chain()` function for periodic integrity checks. |
| **Residual risk** | P1 × S3 = **Low** |

---

### H-006 — Ollama Service Unavailability

| Field | Detail |
|---|---|
| **Failure mode** | Ollama process crashes or becomes unresponsive during monitoring |
| **Cause** | Model crash, OOM, or system resource exhaustion |
| **Effect** | No LLM reasoning available; agent cannot complete pipeline |
| **Initial risk** | P3 × S4 = **High** |
| **Risk controls** | (1) Ollama runs as a separate process — its crash does not crash the FastAPI server. (2) `LLM_FALLBACK_ON_TIMEOUT=true` — system continues on rule-based scoring alone. (3) `rule_based_only=true` flag in response tells clinician that LLM was unavailable. (4) Health endpoint continuously monitors Ollama availability. |
| **Residual risk** | P2 × S2 = **Low** |

---

### H-007 — Incomplete qSOFA Assessment

| Field | Detail |
|---|---|
| **Failure mode** | qSOFA score underestimates sepsis risk due to missing signals |
| **Cause** | Full qSOFA requires respiratory rate and altered mentation — not available from the 4 monitored vitals |
| **Effect** | Sepsis risk underscored; delayed alert |
| **Initial risk** | P3 × S4 = **High** |
| **Risk controls** | (1) SpO2 <94% used as proxy for respiratory dysfunction (documented limitation). (2) Limitation explicitly stated in every LLM output and alert response. (3) LLM prompted to note unavailable signals and adjust confidence accordingly. |
| **Residual risk** | P2 × S4 = **Medium** |
| **ALARP rationale** | Adding respiratory rate and mentation assessment requires hardware/clinical workflow changes outside scope of this software. Documented as known limitation. |

---

## Residual Risk Summary

| Hazard | Residual Risk | Accepted |
|---|---|---|
| H-001 Missed critical alert | Medium | Yes — mandatory clinical review |
| H-002 Spurious critical alert | Low | Yes |
| H-003 LLM recommends treatment | Medium | Yes — mandatory clinical review |
| H-004 PHI exposure | Medium | Yes — local inference only; production requires stronger de-id |
| H-005 Audit log tampering | Low | Yes |
| H-006 Ollama unavailability | Low | Yes — rule-based fallback |
| H-007 Incomplete qSOFA | Medium | Yes — documented limitation in all outputs |

**Overall residual risk conclusion:** Acceptable, provided the system is used as clinical decision support only and all alerts are reviewed by a qualified clinician before action is taken.
