# PolicyPilot – Project Walkthrough

> End-to-end guide for understanding, running, and extending the PolicyPilot AFCA complaint triage system.

---

## What This System Does

PolicyPilot automates the triage of financial complaints submitted to AFCA (Australian Financial Complaints Authority). Given a complaint in plain text, it:

1. Extracts entities (dollar amounts, dates, products, regulation references)
2. Retrieves the most relevant ASIC regulatory sections from a vector store
3. Classifies the complaint into a category and sub-category
4. Calculates a 4-dimension risk score (category severity × monetary amount × timeline urgency × complexity)
5. Drafts a compliant IDR response with ASIC RG 271 timeframe tracking
6. Returns everything via a REST API and a 5-page Streamlit dashboard

Without an API key or Docker, the system runs entirely in **mock mode** — deterministic keyword classification + JSON-file retrieval — so every component can be explored without any external services.

---

## Repository Layout

```
policypilot_starter/
├── app/
│   ├── api/
│   │   ├── main.py          # FastAPI app — /triage, /batch, /health, /metrics, /prompts
│   │   ├── models.py        # Pydantic request/response schemas
│   │   ├── metrics.py       # Prometheus counters and histograms
│   │   └── middleware.py    # Request-ID injection and structured logging
│   ├── core/
│   │   ├── chain.py         # TriageChain — the main RAG pipeline
│   │   ├── config.py        # Settings from .env (with sensible defaults)
│   │   ├── embeddings.py    # OpenAI embedding wrapper (mock-safe)
│   │   ├── entities.py      # Regex-based financial entity extractor
│   │   ├── retriever.py     # Qdrant retriever with mock fallback
│   │   ├── risk.py          # RiskScorer — 4-dimension composite score
│   │   └── prompts/
│   │       ├── registry.py  # Prompt version router
│   │       ├── v1_naive.py  # Baseline — direct prompt
│   │       ├── v2_cot.py    # Chain-of-Thought — 5-step reasoning
│   │       └── v3_fewshot.py # Few-shot + CoT — production default
│   ├── eval/
│   │   ├── eval_ragas.py    # Custom RAGAS-inspired evaluation (30 test cases)
│   │   ├── compare_prompts.py # Runs v1/v2/v3 head-to-head
│   │   └── test_cases.json  # Ground-truth complaint test cases
│   └── ingest/
│       ├── load_afca.py     # CSV → PostgreSQL (with --dry-run preview)
│       └── load_asic.py     # JSON → embeddings → Qdrant
├── data/
│   ├── raw/
│   │   ├── afca_complaints_sample.csv   # 100 synthetic AFCA complaints
│   │   └── asic_rg271_snippets.json    # ASIC regulatory guide excerpts
│   └── processed/
│       ├── afca_complaints_clean.csv    # Cleaned CSV
│       ├── eval_v1_results.json         # Evaluation results per prompt version
│       ├── eval_v2_results.json
│       ├── eval_v3_results.json
│       └── prompt_comparison_report.json
├── frontend/
│   └── app.py               # 5-page Streamlit dashboard
├── infra/
│   └── prometheus.yml       # Prometheus scrape config
├── docker-compose.yml       # Qdrant + PostgreSQL
├── requirements.txt
└── project.json
```

---

## Key Design Decisions

### 1. Mock-first architecture
Every external dependency (OpenAI, Qdrant, PostgreSQL) has a drop-in mock. This means:
- The pipeline can be explored with zero setup
- Unit tests don't require live services
- The `--dry-run` flags on ingest scripts let you verify data before writing

### 2. Prompt engineering as a first-class concern
Three prompt versions live in `app/core/prompts/` and are independently evaluable:

| Version | Strategy | Notes |
|---------|----------|-------|
| v1 | Naive direct | Establishes baseline |
| v2 | Chain-of-Thought | 5-step explicit reasoning |
| v3 | Few-shot + CoT | 3 AFCA examples + expert persona |

All three share the same `build_prompt(complaint_text, docs, entities) → {system, user}` interface so the chain is version-agnostic.

### 3. Risk scoring without LLM dependency
`RiskScorer` is a pure rule-based component. It computes a weighted composite from:
- **Category weight** — some categories (fraud, hardship) are inherently higher risk
- **Monetary severity** — scaled by dollar amount
- **Timeline urgency** — days since incident
- **Complaint complexity** — length and entity density

This keeps risk scoring deterministic, auditable, and cheap.

### 4. Structured output via function calling
The LLM is called with `functions=[TRIAGE_FUNCTIONS]` and `function_call={"name": "triage_complaint"}`. This forces JSON-schema-conformant output — no regex parsing of free text responses.

---

## Running Without Docker (Mock Mode)

```bash
# 1. Create venv and install
python3 -m venv venv
pip install -r requirements.txt

# 2. Run the dashboard (mock mode — no API key needed)
streamlit run frontend/app.py

# 3. Run the API (mock mode)
uvicorn app.api.main:app --reload

# 4. Test the pipeline directly
python -c "
from app.core.chain import TriageChain
c = TriageChain()
r = c.run('I was charged \$500 in fees I did not agree to on my home loan.')
print(r.category, r.risk_assessment.risk_level)
"
```

---

## Running With Full Infrastructure

```bash
# 1. Start Qdrant + PostgreSQL
docker compose up -d

# 2. Add API key
cp .env.example .env   # then set OPENAI_API_KEY=sk-...

# 3. Load data
python -m app.ingest.load_afca          # CSV → PostgreSQL
python -m app.ingest.load_asic          # JSON → embeddings → Qdrant

# 4. Start API + dashboard
uvicorn app.api.main:app --reload &
streamlit run frontend/app.py
```

---

## Evaluation Results (Mock Mode)

Pre-computed results are in `data/processed/`. Key metrics from `eval_v3_results.json`:

| Metric | v1 | v2 | v3 |
|--------|----|----|-----|
| Category Accuracy | ~25% | ~28% | ~30% |
| Risk Level Accuracy | ~35% | ~38% | ~40% |
| Entity Recall | ~84% | ~84% | ~84% |

> **Note:** These are mock-mode results — keyword classification, not GPT-4o-mini. With a real API key, v3 achieves ~92% category accuracy per README benchmarks.

Entity recall is consistently high (~84%) because `EntityExtractor` uses deterministic regex patterns regardless of prompt version.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/triage` | Single complaint → full triage result |
| `POST` | `/batch` | Array of complaints → batch results |
| `GET` | `/health` | Liveness check + version info |
| `GET` | `/prompts` | List available prompt versions |
| `GET` | `/metrics` | Prometheus metrics (scrape endpoint) |

Example triage request:
```bash
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"complaint_text": "I was charged $500 in fees on my credit card without authorisation.", "prompt_version": "v3"}'
```

---

## Dashboard Pages

| Page | What it shows |
|------|---------------|
| **Triage** | Live complaint triage — paste text, get category/risk/draft response |
| **Batch Upload** | CSV upload for bulk triage with download of results |
| **Analytics** | AFCA Datacube charts — product distribution, state breakdown, compensation |
| **Eval Dashboard** | Prompt version comparison table and accuracy charts |
| **Executive Summary** | Interactive ROI calculator with customisable assumptions |

---

## Extending the System

**Add a new prompt version:**
1. Create `app/core/prompts/v4_yourname.py` with a `build_prompt(...)` function
2. Register it in `app/core/prompts/registry.py`
3. Run `python -m app.eval.compare_prompts` to benchmark it

**Add a new entity type:**
- Edit `EntityExtractor` in `app/core/entities.py` — each entity type is a named regex group

**Add a new risk dimension:**
- Edit `RiskScorer.score()` in `app/core/risk.py` — add a new weighted component and adjust the normalisation

**Add a new ASIC regulatory guide:**
- Append snippets to `data/raw/asic_rg271_snippets.json` in the same schema
- Re-run `python -m app.ingest.load_asic` to re-embed into Qdrant
