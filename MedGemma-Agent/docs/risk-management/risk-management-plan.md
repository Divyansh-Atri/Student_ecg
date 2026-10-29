# Risk Management Plan
**Document ID:** RMP-001  
**Version:** 1.0  
**Standard:** ISO 14971:2019  
**Device:** MedGemma Post-Operative Monitoring Agent (SaMD)  
**Date:** 2026-04-09  

---

## 1. Scope

This Risk Management Plan covers the MedGemma Post-Operative Monitoring Agent — a Software as a Medical Device (SaMD) that monitors post-operative patients for sepsis and cardiovascular deterioration by analyzing vital sign snapshots (SBP, DBP, HR, SpO2) in conjunction with patient EHR context.

**Lifecycle stages covered:** Design, development, testing, deployment, post-production monitoring.

**Intended use:** Clinical decision support for ward-based post-operative patient monitoring. Outputs must be reviewed by a qualified clinician before any action is taken.

**Intended users:** Nurses and clinicians in hospital ward settings.

**Patient population:** Adult post-operative patients, cardiovascular and sepsis risk focus.

---

## 2. Risk Acceptability Criteria

Per ISO 14971:2019, Section 4.

### Probability Categories

| Level | Name | Description |
|---|---|---|
| P5 | Frequent | Expected to occur regularly in use |
| P4 | Probable | Likely to occur |
| P3 | Occasional | May occur |
| P2 | Remote | Unlikely but possible |
| P1 | Improbable | Very unlikely |

### Severity Categories

| Level | Name | Clinical Impact |
|---|---|---|
| S5 | Catastrophic | Patient death |
| S4 | Critical | Permanent or severe patient harm |
| S3 | Serious | Reversible injury requiring intervention |
| S2 | Minor | Temporary discomfort, no treatment needed |
| S1 | Negligible | No patient impact |

### Risk Acceptability Matrix

|  | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| **P5** | Medium | High | High | Unacceptable | Unacceptable |
| **P4** | Medium | Medium | High | High | Unacceptable |
| **P3** | Low | Medium | Medium | High | High |
| **P2** | Low | Low | Medium | Medium | High |
| **P1** | Low | Low | Low | Medium | Medium |

### Required Actions

| Level | Action |
|---|---|
| Low | Accept and document |
| Medium | Reduce if practicable; document ALARP rationale if not |
| High | Risk reduction required; demonstrate ALARP |
| Unacceptable | Design change mandatory; cannot proceed |

---

## 3. Responsibilities

| Role | Responsibility |
|---|---|
| Risk Management Lead | Owns risk file, approves risk controls |
| Software Engineer | Implements risk controls in code |
| Clinical Advisor | Reviews clinical hazard scenarios |
| QA Reviewer | Verifies residual risk acceptability |

---

## 4. Verification Activities

All risk controls are verified through:
- Unit and integration tests (`pytest tests/`)
- Manual test case execution via mock producer (TC-01 through TC-08)
- Code review of guardrail implementations
- Audit log review for compliance controls

---

## 5. Post-Production Monitoring

- Alert audit logs reviewed monthly for unexpected patterns
- False negative / false positive rates tracked per test case
- Any patient safety incident triggers immediate risk file review
- Model or threshold updates require a new risk assessment prior to deployment
