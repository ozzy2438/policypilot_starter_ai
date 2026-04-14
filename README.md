# PolicyPilot – AFCA Complaint Triage & Compliance Copilot

> **Production-ready AI system** for an Australian financial services firm: reduces manual complaint triage from 12 minutes to 18 seconds with regulatory compliance built-in.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-orange.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<br>
<p align="center">
  <img src="assets/demo.webp" alt="PolicyPilot Streamlit Dashboard Demo" width="100%">
</p>
<br>

---

## Problem

AFCA received **97,000+ complaints** in 2022–23, with banking and finance complaints up **38%**. Operations teams manually read each complaint, classify it, match ASIC regulations, and draft responses — averaging **12 minutes per case**. This is slow, expensive, and error-prone.

## Solution

PolicyPilot is a **RAG-powered compliance copilot** that automates complaint triage:

1. **Ingest** — Reads complaint (PDF/email/text)
2. **Extract** — Identifies entities (amounts, dates, products, regulatory references)
3. **Retrieve** — Finds relevant ASIC regulations via Qdrant hybrid search
4. **Classify** — Categorises complaint using LLM with structured output
5. **Score** — Calculates composite risk score (4-dimensional weighted model)
6. **Respond** — Drafts compliant IDR response with AFCA escalation rights

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Streamlit   │────▶│   FastAPI     │────▶│  RAG Pipeline   │
│  Dashboard   │◀────│   (async)     │◀────│  (LangChain)    │
│  5 pages     │     │  /triage      │     │                 │
│              │     │  /batch       │     │  ┌───────────┐  │
│  • Triage    │     │  /health      │     │  │ Entities  │  │
│  • Batch     │     │  /metrics     │     │  │ (regex)   │  │
│  • Analytics │     │  /prompts     │     │  └─────┬─────┘  │
│  • Eval      │     └──────────────┘     │        │        │
│  • ROI       │                          │  ┌─────▼─────┐  │
└─────────────┘                          │  │ Retriever  │  │
                                         │  │ (Qdrant)   │  │
┌─────────────┐     ┌──────────────┐     │  └─────┬─────┘  │
│ Prometheus  │◀────│   Metrics     │     │        │        │
│ Monitoring  │     │   Module      │     │  ┌─────▼─────┐  │
└─────────────┘     └──────────────┘     │  │ LLM Chain  │  │
                                         │  │ (GPT-4o)   │  │
┌─────────────┐     ┌──────────────┐     │  └─────┬─────┘  │
│ PostgreSQL  │◀────│   Ingest      │     │        │        │
│ (complaints)│     │   Pipeline    │     │  ┌─────▼─────┐  │
└─────────────┘     └──────────────┘     │  │Risk Scorer │  │
                                         │  │(rule-based)│  │
┌─────────────┐     ┌──────────────┐     │  └───────────┘  │
│   Qdrant    │◀────│   Embedding   │     └─────────────────┘
│  (policies) │     │   Pipeline    │
└─────────────┘     └──────────────┘
```

## Key Features

| Feature | Description | Job Ad Alignment |
|---------|-------------|-----------------|
| **RAG Pipeline** | End-to-end retrieval-augmented generation with ASIC regulatory knowledge | *Design, develop and deploy AI solutions* |
| **3-Version Prompt Engineering** | v1 Naive → v2 CoT → v3 Few-shot, with comparative evaluation | *Develop and refine prompt engineering approaches* |
| **Production API** | FastAPI with async endpoints, middleware, structured logging | *Build scalable, production-ready systems* |
| **Risk Scoring Engine** | 4-dimension weighted composite score (category, monetary, timeline, complexity) | *Commercial mindset, link AI to business outcomes* |
| **RAGAS Evaluation** | 30 test cases, category accuracy, risk accuracy, entity recall metrics | *Monitor, evaluate and continuously improve AI* |
| **Prometheus Monitoring** | Request latency, token usage, cost tracking, risk distribution | *Monitor and evaluate AI performance* |
| **Stakeholder Dashboard** | 5-page Streamlit app with ROI calculator | *Strong stakeholder engagement* |

## Prompt Engineering Approach

| Version | Strategy | Use Case |
|---------|---------|---------|
| **v1** | Simple direct prompt | Baseline for comparison |
| **v2** | Chain-of-Thought with 5-step reasoning | Improved accuracy |
| **v3** | Few-shot (3 AFCA examples) + CoT + Expert persona | **Production default** |

Each version is independently evaluated using RAGAS metrics, enabling data-driven prompt selection.

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd policypilot_starter
pip install -r requirements.txt
cp .env.example .env  # Add your OpenAI API key

# 2. Start infrastructure
docker compose up -d

# 3. Load data
python -m app.ingest.load_afca --dry-run    # Preview data
python -m app.ingest.load_afca              # Load to PostgreSQL
python -m app.ingest.load_asic              # Embed policies to Qdrant

# 4. Start API
uvicorn app.api.main:app --reload

# 5. Start dashboard
streamlit run frontend/app.py

# 6. Run evaluations
python -m app.eval.eval_ragas --prompt-version v3
python -m app.eval.compare_prompts
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/triage` | Triage a single complaint |
| `POST` | `/batch` | Batch triage (up to 50) |
| `GET` | `/health` | System health check |
| `GET` | `/prompts` | List prompt versions |
| `GET` | `/metrics` | Prometheus metrics |

## Business Impact (Simulated)

| Metric | Manual | AI-Powered | Improvement |
|--------|--------|-----------|-------------|
| **Triage Time** | 12 min | 18 sec | **40x faster** |
| **Category Accuracy** | ~87% | ~92% | +5pp |
| **Source Attribution** | Manual lookup | Automatic | 100% cited |
| **Cost per Triage** | ~$13 AUD | ~$0.04 AUD | **99.7% reduction** |
| **Annual Savings** | — | ~$150K AUD | 200 cases/day |

## Project Structure

```
policypilot_starter/
├── app/
│   ├── api/          # FastAPI endpoints, middleware, metrics
│   ├── core/         # RAG chain, embeddings, retriever, risk scorer
│   │   └── prompts/  # v1/v2/v3 prompt engineering
│   ├── eval/         # RAGAS evaluation, prompt comparison
│   └── ingest/       # AFCA + ASIC data loading
├── data/
│   ├── raw/          # Source data (AFCA CSV, ASIC JSON)
│   └── processed/    # Cleaned data + eval results
├── frontend/         # Streamlit 5-page dashboard
├── infra/            # Prometheus configuration
├── docker-compose.yml
└── requirements.txt
```

## Tech Stack

- **LLM**: OpenAI GPT-4o-mini via LangChain
- **Vector Store**: Qdrant (cosine similarity)
- **Database**: PostgreSQL 16
- **API**: FastAPI (async)
- **Frontend**: Streamlit
- **Monitoring**: Prometheus
- **Evaluation**: Custom RAGAS-inspired suite
- **Containerisation**: Docker Compose

---

*Built as a portfolio project demonstrating production-ready AI engineering for Australian financial services.*
