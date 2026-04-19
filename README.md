# Geo-Insight: Which Crises Are Most Overlooked?

A humanitarian prioritization tool that surfaces crises where **documented need outpaces funding coverage** — built for the UN OCHA Datathon 2026.

The system combines FTS/HNO funding gap data with the INFORM Global Crisis Severity Index to answer: *which crises are severe, chronically underfunded, and flying under the radar?*

---

## What It Does

| Tab | Description |
|---|---|
| **Gap Scoring** | Ranks countries by an auditable funding-gap formula across 4 dimensions. Animate across 2019–2025. |
| **Prioritize** | Natural-language mandate → LLM extracts importance weights → MCDA ranks countries across 14 dimensions. |
| **SQL Query** | Ask questions in plain English → auto-generated Databricks SQL → live results table. |

### The Mismatch Score

The core insight: `mismatch_score = inform_severity × (1 − coverage_ratio)`

High where a crisis is simultaneously **severe** (INFORM) and **underfunded** (FTS). The scatter plot in the Prioritize tab shows this as a quadrant — top-left is the focus.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  React + Vite frontend  (http://localhost:5173 in dev)  │
└────────────────────────┬────────────────────────────────┘
                         │  HTTP (proxied in dev)
┌────────────────────────▼────────────────────────────────┐
│  FastAPI backend  (http://localhost:8000)                │
│                                                         │
│  POST /score    → gap_scorer.py   (funding gap ranking) │
│  POST /analyze  → mcda_analyzer.py (NL → MCDA)         │
│  POST /query    → sql_generator.py (NL → SQL)           │
│  POST /refresh  → reload tables from Databricks         │
└──────────┬────────────────────────┬─────────────────────┘
           │                        │
    ┌──────▼──────┐          ┌──────▼──────┐
    │  Databricks │          │ Local parquet│
    │  warehouse  │  ──or──  │  fallback   │
    │ (live data) │          │  data/      │
    └─────────────┘          └─────────────┘
```

**LLM calls** go to Databricks Foundation Model APIs (Llama 4 Maverick → Llama 3.3 70B fallback) via an OpenAI-compatible client — no Anthropic key needed.

**INFORM severity data** is downloaded automatically from HDX on first run and cached to `data/processed/dim_inform_severity.parquet`.

---

## Quickstart

### Option A — Docker (recommended)

```bash
git clone <repo-url>
cd datathon2026-UN-Challenge

cp .env.example .env
# Fill in DATABRICKS_HOST and DATABRICKS_TOKEN (required for AI features)

docker build -t geo-insight .
docker run -p 8000:8000 --env-file .env geo-insight
```

Open `http://localhost:8000`.

---

### Option B — Local dev (conda)

**1. Create environment**

```bash
conda env create -f environment.yml
conda activate geo-insight
```

> After a `git pull`: `conda env update -f environment.yml --prune`

**2. Configure credentials**

```bash
cp .env.example .env
# Set the following in .env:
#   DATABRICKS_HOST        e.g. https://dbc-xxxx.cloud.databricks.com
#   DATABRICKS_TOKEN       personal access token
#   DATABRICKS_HTTP_PATH   /sql/1.0/warehouses/xxxx
#   DATABRICKS_CATALOG     workspace  (default)
#   DATABRICKS_SCHEMA      geo_insight  (default)
```

The backend falls back to local parquet snapshots in `data/processed/` if Databricks is unavailable.

> **SQL Query tab — one-time Databricks setup required**
>
> The SQL Query tab runs live queries against Delta tables in your Databricks workspace.
> Those tables don't exist by default. Run this once to upload them:
>
> ```bash
> python scripts/upload_to_databricks.py
> ```
>
> This requires `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, and `DATABRICKS_HTTP_PATH` in your `.env`.
> It creates the `workspace.geo_insight` schema and uploads all 4 tables as Delta tables.
> After that, the SQL Query tab and `POST /refresh` will work against your live warehouse.

**3. Start the backend**

```bash
uvicorn backend.main:app --reload
# API docs at http://localhost:8000/docs
# Data source shown at http://localhost:8000/health
```

**4. Start the frontend**

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## Scoring Methodology

### Gap Scorer (`/score`)

Four auditable dimensions, each normalised to [0, 1]:

| Dimension | Formula | Meaning |
|---|---|---|
| `need_scale` | `log(1+pin) / max(log(1+pin))` | Absolute scale of humanitarian need |
| `funding_gap` | `1 − coverage_ratio` | Share of requirements unfunded |
| `structural` | Mean historical funding gap | Chronic underfunding across years |
| `trend` | Decline in coverage slope | Worsening trajectory |

```
overlooked_score = weighted_sum(dimensions) × confidence_weight
```

Validated by a Borda ensemble of 4 independent rankings. Countries in the top quartile of all four are flagged as **robustly overlooked**.

### MCDA Analyzer (`/analyze`)

14 dimensions across four groups:

- **Humanitarian Need**: people in need, targeting gap
- **Funding**: funding gap, structural neglect, worsening trend
- **Vulnerability**: water stress, food insecurity, displacement, health fragility, climate vulnerability, governance fragility, disaster risk
- **Severity**: INFORM crisis severity index, mismatch score

User writes a natural-language mandate → LLM assigns importance scores 1–10 → squared and normalised into weights → MCDA score computed per country.

### INFORM Severity

Downloaded from the [INFORM Global Crisis Severity Index](https://data.humdata.org/dataset/inform-global-crisis-severity-index) (HDX). All monthly snapshots for 2019–2025 (~85 files) are downloaded, parsed, and averaged per country-year for robustness. Normalised from the 1–5 scale to [0, 1].

---

## Data Sources

| Source | What | Where |
|---|---|---|
| OCHA FTS | Funding requirements & received (2019–2025) | Databricks warehouse |
| OCHA HNO | People in need by country & sector | Databricks warehouse |
| INFORM Severity | Monthly crisis severity index | HDX (auto-downloaded) |
| dim_country | Country metadata (region, continent) | Databricks warehouse |
| dim_crisis_plan | Crisis confidence & type | Databricks warehouse |

Data is pulled live from Databricks on startup and cached in memory. Call `POST /refresh` to force a reload.

---

## Credentials

### What you need

| Variable | When required | Description |
|---|---|---|
| `DATABRICKS_HOST` | **AI features** (Prioritize tab) | e.g. `https://dbc-xxxx.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | **AI features** (Prioritize tab) | Databricks personal access token |
| `DATABRICKS_HTTP_PATH` | SQL Query tab only | e.g. `/sql/1.0/warehouses/xxxx` |
| `DATABRICKS_CATALOG` | No | Default: `workspace` |
| `DATABRICKS_SCHEMA` | No | Default: `geo_insight` |

**Minimum setup** (Gap Scoring tab works fully offline — data is bundled as parquets):
```
DATABRICKS_HOST=...
DATABRICKS_TOKEN=...
```

The **SQL Query tab** additionally needs `DATABRICKS_HTTP_PATH`. Without it, the tab shows a clear error message.

**No credentials at all?** The Gap Scoring tab still works — it uses the bundled 2019–2025 parquet snapshots. The Prioritize and SQL Query tabs will show a configuration error.

---

## Project Structure

```
├── backend/
│   └── main.py               FastAPI app — all endpoints
├── src/
│   ├── analysis/
│   │   ├── gap_scorer.py     4-dimension funding gap scorer + Borda ensemble
│   │   └── mcda_analyzer.py  14-dimension MCDA + LLM weight extraction
│   ├── data/
│   │   ├── loader.py         Databricks-first data loader (parquet fallback)
│   │   └── inform_loader.py  INFORM severity downloader + monthly averager
│   └── query/
│       ├── sql_generator.py  NL → Databricks SQL via LLM
│       └── databricks_client.py  SQL execution with self-healing retry
├── frontend/
│   └── src/App.tsx           React single-page app (Vite + Tailwind + Recharts)
├── data/
│   └── processed/            Parquet snapshots (fallback + INFORM cache)
├── Dockerfile                Multi-stage: builds frontend, runs backend
├── requirements.txt          Pinned Python deps
└── environment.yml           Conda environment (Python 3.9)
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Backend status + data source |
| `POST` | `/score` | Gap scoring for a given year |
| `POST` | `/analyze` | NL mandate → MCDA country ranking |
| `POST` | `/query` | NL → SQL → results |
| `POST` | `/refresh` | Force reload data from Databricks |

Full interactive docs at `http://localhost:8000/docs`.

---

## Limitations

- Gap scores are **relative** — a low score doesn't mean a crisis is well-funded.
- INFORM severity data covers crises tracked by ACAPS/INFORM; some smaller crises may be missing.
- Vulnerability index scores are static (not time-varying) for countries not in the Databricks dimension tables.
- The SQL query tab requires live Databricks access and will fail gracefully if unavailable.
- This tool is designed for **decision support**, not automated decision-making.

---

*Built at UN OCHA Datathon 2026.*
