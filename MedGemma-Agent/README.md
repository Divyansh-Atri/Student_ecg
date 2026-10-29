# MedGemma Post-Operative Monitoring Agent

An intelligent post-operative patient monitoring agent powered by MedGemma 4-bit, designed for edge deployment in hospital ward settings.

## What it does

Post-operative patients shifted from ICU to general wards lose access to continuous monitoring. This agent provides two independent clinical tools:

1. **Vitals Monitor** — monitors four key vitals every 5 minutes (Systolic BP, Diastolic BP, Heart Rate, SpO2) to detect early signs of **sepsis and cardiovascular deterioration**, using NEWS2 + qSOFA scoring with MedGemma contextual reasoning.

2. **ABG Interpreter** — on-demand arterial blood gas analysis with rule-based acid-base interpretation (respiratory failure typing, Winters' formula compensation, anion gap with delta-delta ratio) and MedGemma clinical reasoning. Accepts manual entry or PDF lab report upload.

The MOAT is the agent harness, not the model. MedGemma provides contextual reasoning; clinical safety guarantees come from the rule-based guardrail layer built around it.

---

## Architecture

### Vitals Monitoring Pipeline

```
Vitals Snapshot (POST /api/v1/vitals/snapshot)
        │
        ▼
┌─────────────────────┐
│  Input Validator    │  Schema + plausibility checks, PHI check
└────────┬────────────┘
         │  CRITICAL vitals → bypass LLM, immediate alert
         ▼
┌─────────────────────┐
│  EHR Retriever      │  RAG: top-5 relevant chunks from patient EHR
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Clinical Scorer    │  Deterministic NEWS2 + qSOFA scoring
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  LLM Analyzer       │  MedGemma contextual risk reasoning
│  (MedGemma 4-bit)   │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Output Validator   │  Schema, grounding, prohibited content checks
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Alert Router       │  Merges rule score + LLM → final alert level
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Audit Logger       │  HIPAA-compliant SHA-256 hash-chained log
└─────────────────────┘
```

### ABG Interpreter Pipeline (standalone module)

```
PDF Upload or Manual Entry
        │
        ▼
┌─────────────────────┐
│  PDF Parser         │  pdfplumber regex extractor — handles ABL90,
│  (abg/pdf_parser)   │  i-STAT, Radiometer, Cobas printout formats
└────────┬────────────┘
         │  Pre-fills form; uncertain fields highlighted for review
         ▼
┌─────────────────────┐
│  Rule-based Interp  │  Acid-base disorder classification, respiratory
│  (abg/interpreter)  │  failure typing (Type I/II), Winters' formula,
└────────┬────────────┘  anion gap, albumin correction, delta-delta
         ▼
┌─────────────────────┐
│  EHR Context RAG    │  Retrieves patient history for LLM context
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  MedGemma Reasoning │  Standalone httpx call — independent of vitals
│  (structured JSON)  │  LangGraph pipeline
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  ABG Result Store   │  Persisted to abg_results table (SQLite)
└─────────────────────┘
```

---

## Alert Levels

### Vitals Monitor

| Level | Trigger | Action |
|---|---|---|
| CRITICAL | Any single NEWS2 score = 3, total ≥ 7, or qSOFA ≥ 2 | Immediate response |
| HIGH | NEWS2 total 5–6 | Urgent clinical review |
| MEDIUM | NEWS2 total 3–4 | Clinician review |
| NORMAL | NEWS2 total 0–2 | Observation only |

### ABG Interpreter

| Level | Example condition |
|---|---|
| CRITICAL | Lactic acidosis, DKA, mixed acid-base failure, severe Type II RF |
| HIGH | Acute COPD exacerbation, significant metabolic acidosis |
| MEDIUM | Compensated respiratory acidosis, metabolic alkalosis |
| NORMAL | Normal or near-normal ABG |

---

## Clinical Scoring

- **NEWS2** (National Early Warning Score 2) — Royal College of Physicians, 2017
- **qSOFA** (Sepsis-3) — Singer et al., JAMA 2016
- **Winters' Formula** — expected PaCO2 = 1.5×HCO3 + 8 ± 2
- **Anion Gap** = Na − (Cl + HCO3), corrected for albumin (+ 2.5 per 1 g/dL below 4.0)
- **Delta-delta ratio** — distinguishes pure vs. mixed high-AG metabolic acidosis
- **P/F ratio** — PaO2/FiO2 for respiratory failure severity (ARDS threshold: < 300)

---

## Guardrails

**Vitals Input:** Schema validation → physiological plausibility → PHI check → critical bypass

**Vitals Output:** JSON schema enforcement → grounding check → prohibited content filter (no drug names/doses) → confidence threshold → contradiction check (LLM cannot disagree with rule score by >1 tier)

**ABG Input:** Pydantic schema with physiological range validators on all 13 parameters

**ABG PDF Parser:** Post-extraction plausibility validation — out-of-range values are zeroed and flagged rather than silently passed through

---

## Compliance

- **HIPAA:** AES-256 encryption at rest, SHA-256 hash-chained audit trail, 18-identifier PHI de-identification, RBAC
- **ISO 14971:** Risk management file in `docs/risk-management/`
- **Fallback safety:** If Ollama is unavailable, both vitals monitor and ABG interpreter continue on rule-based scoring alone (`rule_based_only: true` in ABG responses)

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | LangGraph (vitals pipeline) |
| LLM | MedGemma 1.5B 4-bit (Q4_K_M GGUF via Ollama) |
| RAG | ChromaDB + all-MiniLM-L6-v2 |
| PDF parsing | pdfplumber (EHR ingestion + ABG lab reports) |
| PDF generation | reportlab (test report generation) |
| API | FastAPI |
| Database | SQLite (SQLAlchemy) — `vitals_alerts` + `abg_results` tables |
| Encryption | Python cryptography (Fernet/AES-256) |

---

## Quick Start

```bash
# Clone and set up
cp .env.example .env
# Edit .env — set ENCRYPTION_KEY and API_KEY (or set DEV_MODE=true for local testing)

# Start
./start.sh

# Stop
./stop.sh
```

See [SETUP.md](SETUP.md) for full installation instructions including Ollama model setup.

---

## API Endpoints

### Vitals Monitoring

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/vitals/snapshot` | Submit vitals for analysis |
| GET | `/api/v1/alerts` | List alerts (filterable by patient, level) |
| GET | `/api/v1/alerts/{id}` | Full alert detail + LLM reasoning |
| GET | `/api/v1/patients/{id}` | Patient summary |
| POST | `/api/v1/ehr/ingest` | Ingest EHR PDF into RAG store |
| GET | `/api/v1/mock/test-cases` | List predefined vitals test cases |
| POST | `/api/v1/mock/run/{id}` | Run a vitals test case end-to-end |
| GET | `/api/v1/audit` | Audit trail (admin only) |
| GET | `/api/v1/health` | System health (Ollama, ChromaDB, DB) |

### ABG Interpreter

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/abg/upload` | Extract ABG values from a PDF lab report (pre-fill) |
| POST | `/api/v1/abg/analyze` | Run full ABG analysis (rule-based + MedGemma) |
| GET | `/api/v1/abg/test-cases` | List predefined ABG test cases |
| POST | `/api/v1/abg/run/{case_id}` | Run a named ABG test case end-to-end |
| GET | `/api/v1/abg/history` | List past ABG interpretations |

Full interactive docs at `http://localhost:8000/docs`.

---

## ABG PDF Upload

The ABG tab accepts real-world lab report PDFs. Supported machine formats:
- **ABL90 FLEX PLUS** (Radiometer)
- **i-STAT** (Abbott)
- **Cobas b 123** (Roche)
- Table-style and inline label:value reports

A sample test PDF is included at `test_abg_report.pdf`. Upload it from the ABG tab to verify the pipeline end-to-end.

Fields extracted: pH, PaO2, PaCO2, HCO3, Base Excess, Lactate, FiO2, Na, Cl, Albumin, Glucose, Ketones flag, COPD flag, Patient ID.

Values are returned for **clinician review before submission** — uncertain fields are highlighted in amber. The system does not auto-submit extracted values.

---

## Project Structure

```
MedGemma-Agent/
├── api/
│   ├── main.py                  # FastAPI app, router registration
│   ├── middleware/auth.py        # RBAC, API key validation
│   └── routes/
│       ├── vitals.py            # Vitals snapshot endpoint
│       ├── alerts.py            # Alert retrieval
│       ├── patients.py          # Patient summaries
│       ├── ehr.py               # EHR PDF ingestion
│       ├── mock.py              # Test case runner
│       ├── audit.py             # Audit trail
│       └── abg.py               # ABG interpreter endpoints
├── abg/
│   ├── interpreter.py           # Rule-based ABG analysis engine
│   ├── pdf_parser.py            # Lab report PDF → structured values
│   ├── prompts.py               # MedGemma ABG system + user prompts
│   ├── schemas.py               # Pydantic models for ABG I/O
│   ├── test_cases.yaml          # 8 clinical ABG test scenarios
│   ├── test_case_loader.py      # YAML loader + ABGInput builder
│   └── generate_test_pdf.py     # reportlab test PDF generator
├── vitals/                      # Vitals monitoring schemas + scoring
├── agent/                       # LangGraph agent nodes
├── rag/                         # ChromaDB retriever + ingestion
├── db/database.py               # SQLAlchemy models (alerts + ABG)
├── config/settings.py           # Pydantic settings (.env)
├── ui/
│   ├── index.html               # Single-page UI
│   └── app.js                   # All frontend logic
├── test_abg_report.pdf          # Sample ABG lab report for testing
├── start.sh / stop.sh           # Server lifecycle scripts
├── SETUP.md                     # Full installation guide
└── .env.example                 # Environment template
```

---

## Disclaimer

This system is **clinical decision support only**. All outputs must be reviewed by a qualified healthcare professional before any clinical action is taken. This system does not diagnose, prescribe, or replace clinical judgment.
