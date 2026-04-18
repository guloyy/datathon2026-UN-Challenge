# Geo-Insight: Which Crises Are Most Overlooked?

A system that surfaces humanitarian crises where **documented need outpaces funding coverage**, built for the UN OCHA Datathon 2026.

## Architecture

```
natural-language query
        │
        ▼
┌─────────────────┐    Claude (LLM)
│  Query Parser   │◄──────────────────
└────────┬────────┘
         │  QueryFilter
         ▼
┌─────────────────┐
│  Master Dataset │◄── HNO + HRP + FTS + CBPF
└────────┬────────┘
         │  filtered rows
         ▼
┌─────────────────┐
│  Gap Scorer     │  coverage_gap × need_scale → gap_score
└────────┬────────┘  (+ temporal penalty for chronic underfunding)
         │  ranked DataFrame
         ▼
┌─────────────────┐
│  API / Dashboard│
└─────────────────┘
```

## Gap Score Formula

```
coverage_gap  = 1 − (funding_usd / requirements_usd)   ∈ [0, 1]
need_scale    = log1p(pin) / max(log1p(pin))            ∈ [0, 1]
gap_score     = coverage_gap × need_scale

# Bonus: temporal multiplier for crises underfunded N consecutive years
gap_score    *= 1 + 0.15 × N
```

## Quickstart

Run these steps in order — every command is copy-pasteable.

---

### Step 1 — Clone the repo

```bash
git clone <repo-url>
cd datathon2026-UN-Challenge
```

---

### Step 2 — Create the conda environment

All dependencies (including kaleido for map exports and ipykernel for notebooks) are pinned in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate geo-insight
```

Register the env as a Jupyter kernel so the notebooks use the right Python:

```bash
python -m ipykernel install --user --name geo-insight --display-name "geo-insight"
```

> **After a `git pull`** (syncs new packages without recreating):
> ```bash
> conda env update -f environment.yml --prune
> ```
>
> **If the environment is broken** — delete and start fresh:
> ```bash
> conda env remove -n geo-insight && conda env create -f environment.yml
> python -m ipykernel install --user --name geo-insight --display-name "geo-insight"
> ```

---

### Step 3 — Configure the API key

```bash
cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY=sk-ant-...
```

---

### Step 4 — Download all datasets

Downloads HNO (2024–2026), HRP, FTS funding, population, CBPF, and the Natural Earth shapefile used by the maps. All files go to `data/raw/` (gitignored):

```bash
python scripts/download_data.py
```

```bash
python scripts/download_data.py --list          # preview HDX resources without downloading
python scripts/download_data.py --only hno cbpf # download specific sources only
python scripts/download_data.py --force         # re-download even if files already exist
```

---

### Step 5 — Build the master dataset

Merges HNO + FTS into `data/processed/master.parquet`:

```bash
python -c "from src.ingestion.loaders import build_master_dataset; build_master_dataset()"
```

---

### Step 6 — Verify everything works

```bash
pytest tests/
# Expected: 3 passed
```

---

### Step 7 — Open the notebooks

```bash
jupyter lab
```

Open `notebooks/01_data_exploration.ipynb` to explore the raw data and maps, then `notebooks/02_pipeline_demo.ipynb` to run the full gap-scoring pipeline.
Set the kernel to **geo-insight** (top-right dropdown in JupyterLab).

---

### Step 8 — Run the app (scaffold, not yet tested)

```bash
streamlit run dashboard/app.py                 # Streamlit dashboard
uvicorn src.api.main:app --reload              # FastAPI — docs at http://localhost:8000/docs
```

## Project Structure

```
├── src/
│   ├── ingestion/   loaders.py          — data download + normalisation
│   ├── scoring/     gap_score.py        — coverage gap + need scale + temporal
│   ├── query/       llm_parser.py       — NL → QueryFilter via Claude
│   ├── ranking/     ranker.py           — filter → score → rank pipeline
│   └── api/         main.py             — FastAPI service
├── dashboard/       app.py              — Streamlit UI + map
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_pipeline_demo.ipynb
├── tests/           test_scoring.py
└── data/
    ├── raw/         (gitignored — download from HDX)
    └── processed/   master.parquet
```

## Data Sources & Columns

### 1. HNO — Humanitarian Needs Overview (`hno_2024.csv`, `hno_2025.csv`, `hno_2026.csv`)

**What it is:** The annual global assessment of how many people need humanitarian assistance, broken down by country and sector. Produced by OCHA as part of the Humanitarian Programme Cycle.

**What it tells us:** The *scale of need* — how many people in each country/sector require aid. This is the core "need" signal in the gap score.

**Downloaded from:** https://data.humdata.org/dataset/global-hpc-hno

| Column | Type | Description |
|---|---|---|
| `Country ISO3` | str | ISO 3-letter country code (e.g. `AFG`, `YEM`) |
| `Description` | str | Crisis or plan name (e.g. "Plan caseload", "Food Security and Agriculture") |
| `Cluster` | str | Humanitarian sector code (`ALL`, `FSC`, `HEA`, `EDU`, `NUT`, `PRO`, `SHL`, `WSH`, …) |
| `Category` | str | Sub-category within the cluster |
| `Population` | float | Total population of the geographic area |
| `In Need` | float | **People in need (PIN)** — the primary need signal |
| `Targeted` | float | Number of people the response plan aims to reach |
| `Affected` | float | People affected (broader than "in need") |
| `Reached` | float | People actually reached by response (where reported) |
| `Info` | str | Metadata or notes |

> 2025/2026 files also include `Admin 1–3 PCode/Name` for sub-national breakdown.

---

### 2. HRP — Humanitarian Response Plans (`hrp.csv`)

**What it is:** A registry of all formal humanitarian response plans, flash appeals, and regional response plans registered in the OCHA HPC Tools system.

**What it tells us:** Which crises have a formal plan, over what time period, and what their original and revised financial requirements were. Useful for filtering to "officially recognised" crises and for joining plan codes to FTS funding data.

**Downloaded from:** https://data.humdata.org/dataset/humanitarian-response-plans

> Note: row 0 of this file contains HXL hashtags (e.g. `#response+code`) — skipped automatically by the loader.

| Column | Type | Description |
|---|---|---|
| `code` | str | Unique plan code (e.g. `HAFG26`, `HHTI26`) |
| `internalId` | int | Internal HPC Tools plan ID |
| `startDate` | date | Plan start date |
| `endDate` | date | Plan end date |
| `planVersion` | str | Full plan name / title |
| `categories` | str | Plan type(s) (e.g. "Humanitarian needs and response plan", "Flash appeal") |
| `locations` | str | Country ISO3 codes covered by the plan (pipe-separated for regional plans) |
| `years` | int | Year(s) covered |
| `origRequirements` | float | Original financial requirements in USD |
| `revisedRequirements` | float | Revised financial requirements in USD |

---

### 3. FTS — Global Requirements and Funding (`funding.csv`)

**What it is:** Financial Tracking Service (FTS) data — the most authoritative public record of humanitarian funding requests and receipts, maintained by UNOCHA. One row per plan per year.

**What it tells us:** How much money was requested (`requirements`) and how much was actually received (`funding`) for each crisis plan. This is the core *funding* signal used to compute the coverage ratio.

**Downloaded from:** https://data.humdata.org/dataset/global-requirements-and-funding-data

| Column | Type | Description |
|---|---|---|
| `countryCode` | str | ISO 3-letter country code |
| `id` | float | FTS plan ID |
| `name` | str | Plan name (e.g. "Afghanistan Humanitarian Needs and Response Plan 2026") |
| `code` | str | Plan code (matches HRP `code`) |
| `typeId` | float | Numeric plan type ID |
| `typeName` | str | Plan type label (e.g. "Humanitarian response plan", "Flash appeal") |
| `startDate` | date | Plan start date |
| `endDate` | date | Plan end date |
| `year` | int | Year of the plan |
| `requirements` | float | **Total funding requested (USD)** |
| `funding` | float | **Total funding received (USD)** |
| `percentFunded` | float | Funding as a percentage of requirements (0–100) |

---

### 4. Population — COD Global Population (`population.csv`)

**What it is:** OCHA's Common Operational Dataset (COD) for population — official baseline population figures by country and admin level, disaggregated by age and gender.

**What it tells us:** Total population denominators for computing need as a share of population (e.g. "X% of the country is in need"). Useful as a normalisation factor and for per-capita analysis.

**Downloaded from:** https://data.humdata.org/dataset/cod-ps-global

| Column | Type | Description |
|---|---|---|
| `ISO3` | str | Country ISO3 code |
| `Country` | str | Country name |
| `ADM1_PCODE` / `ADM1_NAME` | str | Admin level 1 code / name (province/state) |
| `ADM2_PCODE` / `ADM2_NAME` | str | Admin level 2 code / name (district) |
| `ADM3_PCODE` / `ADM3_NAME` | str | Admin level 3 code / name |
| `ADM4_PCODE` / `ADM4_NAME` | str | Admin level 4 code / name |
| `Population_group` | str | Group code (e.g. `T_TL` = total, `F_TL` = female, `M_TL` = male) |
| `Gender` | str | `all`, `f`, or `m` |
| `Age_range` | str | Age bracket (e.g. `0-4`, `all`) |
| `Age_min` / `Age_max` | float | Age range bounds |
| `Population` | float | **Population count** |
| `Reference_year` | int | Year of the population estimate |
| `Source` | str | Data source (e.g. national statistics authority) |
| `Contributor` | str | OCHA country office that contributed the data |

---

### 5. CBPF — Country-Based Pooled Fund Allocations (`cbpf.json`)

**What it is:** Allocations from OCHA-managed in-country pooled funds (CBPFs, e.g. Somalia Humanitarian Fund, Yemen Humanitarian Fund). These are flexible, fast-disbursing funds that complement donor-directed funding.

**What it tells us:** How much pooled fund money was approved for each country/year, broken down by recipient organisation type. Useful as a signal of whether a crisis has access to flexible local funding beyond what FTS tracks.

**Downloaded from:** https://cbpfapi.unocha.org/vo2/odata/AllocationBudgetTotalsByYearAndFund

> Note: CBPF uses country *names* (e.g. "Somalia"), not ISO3 codes — a name→ISO3 mapping is needed to join to the master dataset (currently a known gap).

| Column | Type | Description |
|---|---|---|
| `AllocationYear` | int | Year of the allocation |
| `OrganizationType` | str | Recipient type (`International NGO`, `National NGO`, `UN Agency`) |
| `PooledFundName` | str | Country/fund name (e.g. "Somalia", "Yemen") |
| `ApprovedBudget` | float | **Total approved allocation (USD)** |
| `ApprovedReserveBudget` | float | Reserve allocation component |
| `ApprovedStandardBudget` | float | Standard allocation component |
| `PipelineBudget` | float | Pipeline (not yet approved) allocations |
| `FundingType` | int | 1 = standard, 2 = reserve |

---

## Master Dataset (`data/processed/master.parquet`)

**What it is:** A merged, analysis-ready table joining HNO people-in-need figures with FTS funding data on `country_iso3 + year`. One row per country × year × sector combination.

**How it's built:** `src/ingestion/loaders.py → build_master_dataset()`

**Shape:** ~4,000 rows × 9 columns (varies by year coverage)

| Column | Type | Source | Description |
|---|---|---|---|
| `country_iso3` | str | HNO / FTS | ISO3 country code |
| `year` | int | HNO / FTS | Year (1999–2031 in FTS; 2024–2026 in HNO) |
| `pin` | float | HNO | People in need for this country/sector/year — **null for FTS-only rows** |
| `sector` | str | HNO | Cluster/sector code (`ALL`, `FSC`, `HEA`, …) — **null for FTS-only rows** |
| `country_name` | str | FTS | Full crisis plan name or "Not specified" |
| `requirements_usd` | float | FTS | Total funding requested (USD) — **null for HNO-only rows** |
| `funding_usd` | float | FTS | Total funding received (USD) |
| `coverage_ratio` | float | FTS | `funding / requirements` ∈ [0, 1] — **null where requirements is null** |
| `cbpf_allocations_usd` | float | CBPF | Pooled fund allocations (USD) — **currently null pending name→ISO3 fix** |

**Key data quality notes:**
- `pin` and `sector` are ~94% null — most FTS rows cover plan types (flash appeals, regional plans) that have no HNO counterpart. Filter to rows where `pin` is not null for need-based analysis.
- `coverage_ratio` is ~68% null — rows where no financial requirements were set (e.g. purely donor-directed funding without a formal plan).
- `cbpf_allocations_usd` is currently 100% null because CBPF uses country names while the master uses ISO3 codes. A name→ISO3 mapping lookup is needed to fix this join.
- Year range spans 1999–2031 because FTS tracks multi-year plans with open end dates.

---

## Team

Built at UN OCHA Datathon 2026.

## Limitations & Caveats

- Gap scores are **relative**, not absolute — a low score doesn't mean a crisis is adequately funded.
- HNO figures vary in recency; stale data reduces reliability for any single year.
- CBPF allocations are a subset of total funding; absence of CBPF ≠ absence of funding.
- `cbpf_allocations_usd` is currently unjoined — CBPF uses country names, not ISO3 codes.
- This tool is designed for **decision support**, not automated decision-making.
