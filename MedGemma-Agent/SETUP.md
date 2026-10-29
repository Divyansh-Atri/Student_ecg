# Setup Guide

Step-by-step instructions to run the MedGemma Monitoring Agent on a new machine.

## Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com) installed
- The MedGemma GGUF model file (`medgemma-1.5-4b-it-Q4_K_M.gguf`) placed in `medgemma-4-bit-model/`
- 4 GB RAM minimum (8 GB recommended for comfortable inference)

---

## Step 1 — Install Ollama

**macOS (Apple Silicon or Intel):**
```bash
brew install ollama
```

**Windows (run in WSL2):**
```bash
sudo apt-get install -y zstd
curl -fsSL https://ollama.com/install.sh | sh
```

Verify Ollama is running:
```bash
curl http://localhost:11434
# Expected: Ollama is running
```

---

## Step 2 — Import the MedGemma model

From the project root:

```bash
cat > Modelfile <<'EOF'
FROM ./medgemma-4-bit-model/medgemma-1.5-4b-it-Q4_K_M.gguf
PARAMETER num_ctx 4096
PARAMETER temperature 0.1
TEMPLATE """<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
"""
SYSTEM "You are MedGemma, a medical AI assistant for post-operative patient monitoring. You provide structured clinical decision support based on vital signs and patient history. You never diagnose, prescribe, or replace clinical judgment."
EOF

ollama create medgemma -f Modelfile
```

Verify:
```bash
ollama list
# Should show: medgemma
```

---

## Step 3 — Python environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows WSL2: same command
pip install -r requirements.txt
```

This installs all dependencies locally. The `all-MiniLM-L6-v2` embedding model (~90 MB) downloads from HuggingFace on first run and is cached in `data/models/` for offline use after that.

---

## Step 4 — Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in:

**Required:**
```
# Generate your encryption key:
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

ENCRYPTION_KEY=<paste output here>
API_KEYS_ADMIN=your-admin-key-here
API_KEYS_CLINICIAN=your-clinician-key-here
```

**Optional (defaults are fine for local use):**
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=medgemma
DATABASE_URL=sqlite:///./data/medgemma_agent.db
```

---

## Step 5 — Start the server

```bash
./start.sh
```

The server starts in the background. You'll see:
```
Server started (PID 12345)

  UI:       http://localhost:8000/ui
  API docs: http://localhost:8000/docs
  Health:   http://localhost:8000/api/v1/health
  Logs:     tail -f server.log
```

To stop:
```bash
./stop.sh
```

---

## Step 6 — Ingest EHRs

Before running test cases, ingest patient EHRs so the RAG pipeline has context.

Via the UI (`http://localhost:8000/ui` → EHR Ingest tab) or via curl:

```bash
curl -X POST http://localhost:8000/api/v1/ehr/ingest \
  -H "X-API-Key: your-clinician-key-here" \
  -F "patient_id=patient_277403" \
  -F "age=67" \
  -F "gender=M" \
  -F "file=@synthetic-EHRs/patient_277403_discharge_summary.pdf"
```

Repeat for each patient PDF in `synthetic-EHRs/`.

---

## Step 7 — Run test cases

Via the UI → Test Cases tab, or:

```bash
curl -X POST http://localhost:8000/api/v1/mock/run/TC-04 \
  -H "X-API-Key: your-clinician-key-here"
```

TC-04 is the qSOFA sepsis test case (expected: CRITICAL alert).

---

## GPU Acceleration (optional but recommended)

**Apple Silicon (Mac Mini):**
Ollama uses Metal automatically — no extra setup needed.

**NVIDIA GPU (Linux/WSL2):**
Ensure CUDA drivers are up to date, then Ollama detects the GPU automatically.
To check: `ollama run medgemma "hello"` — if it's fast (<2s), GPU is active.

**CPU-only:**
Works fine. MedGemma 1.5B 4-bit is small enough to run on CPU for 5-minute interval analysis.

---

## Running tests

```bash
source venv/bin/activate
pytest tests/ -v
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ENCRYPTION_KEY not set` | Generate and add key to `.env` (Step 4) |
| `Ollama unreachable` | Run `curl http://localhost:11434` — if nothing, restart Ollama |
| `Model not found` | Run `ollama list` — if medgemma missing, redo Step 2 |
| `ModuleNotFoundError` | Make sure venv is activated: `source venv/bin/activate` |
| `403 Forbidden` | Check API key in request header matches `.env` |
| Server slow on first request | Embedding model loading on first call — subsequent calls are fast |
