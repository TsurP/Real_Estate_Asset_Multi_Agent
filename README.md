# Real Estate Asset Multi-Agent Portfolio Assistant

A stateful **LangGraph multi-agent system** that answers natural-language questions about a real-estate portfolio P&L ledger — accurately, without LLM hallucinations on the numbers.

---

## Table of Contents

1. [Solution Overview](#solution-overview)
2. [Setup](#setup)
3. [Architecture](#architecture)
4. [Multi-Agent Workflow](#multi-agent-workflow)
5. [Supported Intents](#supported-intents)
6. [Dataset](#dataset)
7. [Challenges & Solutions](#challenges--solutions)
8. [Project Structure](#project-structure)
9. [Running Tests](#running-tests)

---

## Solution Overview

The system accepts a free-form natural-language question (e.g. *"Which property had the lowest expense ratio among those with above-average revenue in 2024?"*) and produces a precise, cited numeric answer by running it through a pipeline of specialised agents.

**Key design principles:**

- **LLMs classify and extract; pandas computes.** No arithmetic is ever delegated to the LLM — all aggregations, ratios, rankings, and comparisons are executed deterministically with pandas. This eliminates hallucinated numbers.
- **Stateful interrupts for missing information.** When a required field (timeframe, comparison target) is absent, the graph pauses with a `LangGraph interrupt`, surfaces a focused clarifying question to the user, then resumes exactly where it stopped.
- **Deterministic guardrails wrap every LLM call.** Post-processing rules in the extraction agent override noisy LLM output (hallucinated filters, wrong grouping dimensions, etc.) before it reaches computation.
- **Hybrid routing.** The router combines LLM structured output with regex-based deterministic overrides for edge-case intent patterns that LLMs consistently misclassify.

---

## Setup

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) (recommended) **or** pip
- An Anthropic or OpenAI API key

### 1. Clone and install dependencies

**With uv (recommended):**
```bash
git clone <repo-url>
cd Real_Estate_Asset_Multi_Agent
uv sync
```

**With pip:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```bash
# Anthropic (default)
ANTHROPIC_API_KEY=sk-ant-...

# --- OR --- OpenAI
# OPENAI_API_KEY=sk-...
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o-mini

# Optional: override the model (default: claude-haiku-4-5-20251001)
# LLM_MODEL=claude-sonnet-4-6
```

Or export directly in your shell:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the Streamlit app

```bash
# With uv
uv run streamlit run src/app.py

# With pip / activated venv
streamlit run src/app.py
```

The UI opens at `http://localhost:8501` by default. It includes:
- A chat interface with multi-turn memory
- A collapsible debug panel showing intent, extracted fields, computation steps, and node trace
- Sample prompts to get started quickly

### 4. Run tests

```bash
uv run pytest tests/ -v
# or
pytest tests/ -v
```

All 122 tests run without an API key — the test suite mocks LLM calls and exercises routing logic, extraction heuristics, computation functions, and response formatting.

---

## Architecture

```
User Query
    │
    ▼
┌──────────┐   unsupported / low-conf    ┌──────────┐
│  Router  │────────────────────────────►│ Fallback │──► END
│  Agent   │   general_question          └──────────┘
└────┬─────┘──────────────────────────────────────────► Respond ──► END
     │ data intent
     ▼
┌──────────┐   missing required field
│ Extract  │──[interrupt]──► ClarificationInterrupt
│  Agent   │◄──[resume]──────────────────────────────
└────┬─────┘
     │
     ▼
┌──────────┐   no match / ambiguous
│ Retrieve │──[interrupt]──► DisambiguationInterrupt
│  Agent   │◄──[resume]──────────────────────────────
└────┬─────┘
     │
     ▼
┌──────────┐  error      ┌──────────┐    ┌─────────┐
│ Compute  │────────────►│ Fallback │    │ Respond │──► END
│  Agent   │             │  Agent   │    │  Agent  │
└──────────┘             └──────────┘    └─────────┘
```

### Agents at a glance

| Agent | Type | Responsibility |
|---|---|---|
| **RouterAgent** | LLM + deterministic overrides | Classify query into one of 13 supported intents |
| **ExtractionAgent** | LLM + extensive heuristic post-processing | Extract targets, timeframe, ledger filters, grouping dimension, ranking direction, ratio type; interrupt for missing fields |
| **RetrievalAgent** | Deterministic | Fuzzy name resolution (RapidFuzz); filter DataFrame; interrupt for disambiguation |
| **ComputationAgent** | Deterministic (pandas only) | Aggregation, ranking, margin analysis, period comparison, share-of-total — zero LLM involvement |
| **ResponseAgent** | Deterministic templates + LLM | Format computation results into concise prose; LLM only for general questions |
| **FallbackAgent** | LLM | Handle unsupported / no-match / errors; offer alternatives and one clarifying question |

### Shared state

All agents communicate through a single `AgentState` TypedDict passed between LangGraph nodes. Key fields:

```python
class AgentState(TypedDict, total=False):
    user_query: str
    intent: str                  # e.g. "ranking", "margin_analysis"
    targets: list[str]           # resolved entity names
    timeframe: dict              # {"year": "2024", "label": "2024"}
    group_by: str                # "property_name", "ledger_type", …
    query_condition: str         # "above_average_profit", "expenses_without_revenue", …
    ratio_type: str              # "profit_margin" | "expense_ratio"
    margin_pre_filter: str       # "above_average_revenue" | "below_average_revenue"
    ledger_filters: dict         # {ledger_type, ledger_group, …}
    k: int                       # top/bottom K
    rank_direction: str          # "top" | "bottom"
    retrieved_records: list[dict]
    computation_result: dict
    final_answer: str
```

---

## Multi-Agent Workflow

### LangGraph graph definition

```mermaid
flowchart TD
    START([START]) --> router

    router -->|general_question| respond
    router -->|unsupported / low confidence| fallback
    router -->|data intent| extract

    extract -->|missing fields| CLARIFY([ClarificationInterrupt])
    CLARIFY -->|user answers| extract

    extract -->|ready| retrieve

    retrieve -->|no match| fallback
    retrieve -->|ambiguous| DISAMBIG([DisambiguationInterrupt])
    DISAMBIG -->|user picks| retrieve
    retrieve -->|matched| compute

    compute -->|error| fallback
    compute -->|success| respond

    respond --> END([END])
    fallback --> END
```

### Step-by-step for a typical query

> *"Among properties with above-average revenue in 2024, which one had the lowest expense ratio?"*

**1. Router** — The LLM classifies the intent. The deterministic override `_is_margin_data_query` fires (detects "expense ratio" keyword), ensuring `margin_analysis` intent regardless of LLM output.

**2. Extraction** — The LLM extracts `timeframe=2024`, `group_by=property_name`. Deterministic post-processing in `_normalise_margin_analysis_extraction` then:
- detects "expense ratio" → sets `ratio_type = "expense_ratio"`
- detects "above-average revenue" → sets `margin_pre_filter = "above_average_revenue"`
- detects "which one" + "lowest" → sets `k=1`, `rank_direction="bottom"`

**3. Retrieval** — Queries the DataFrame for all 2024 rows (3,181 rows).

**4. Computation** — `compute_margin_analysis` runs:
1. Groups revenue by property; computes mean revenue (avg ≈ €459k)
2. Keeps only properties above the average (Building 120, 140, 160)
3. Computes expense ratio = `|expenses| / revenue` for each
4. Sorts ascending (lowest first), returns rank 1

**5. Response** — Deterministic template formats the single winner with full working shown.

### Interrupt / resume flow

LangGraph's `interrupt()` function is used at two points:

```
extract_node → interrupt("Which time period?") → [graph paused]
                                                       ↑
                                              user types "2024"
                                                       ↓
extract_node ← resume(answer="2024") ← [graph resumed]
```

This is implemented as a true graph pause — the full state is serialised by LangGraph, the user's answer arrives as a separate event, and the graph picks up from exactly the same node without restarting.

---

## Supported Intents

| Intent | Example Query |
|---|---|
| `pnl_total` | "What is the total profit for 2024?" |
| `pnl_by_property` | "Show profit by property for Q1 2025" |
| `pnl_by_tenant` | "Revenue per tenant in 2024" |
| `breakdown` | "Break down expenses by ledger group for 2024" |
| `comparison` | "Compare Building 140 vs Building 160 for 2024" |
| `period_comparison` | "Which property grew the most from Q1 to Q4 2024?" |
| `ranking` | "Top 5 properties by profit in 2024" |
| `margin_analysis` | "Which property had the worst net profit margin in 2024?" |
| `share_of_total` | "What % of total revenue came from Building 120 in 2024?" |
| `full_review` | "Give me a full review of Building 160" |
| `count` | "How many tenants are in Building 140?" |
| `general_question` | "What is NOI in real estate?" |
| `unsupported` | "What is Building 140 worth?" |

---

## Dataset

File: `cortex (2) (1) (2) (1) (1) (3).parquet`

| Column | Description |
|---|---|
| `entity_name` | Legal entity (PropCo only) |
| `property_name` | Building name (Building 120, 140, 160, 17, 180) |
| `tenant_name` | Tenant name (Tenant 1–18); NaN for entity-level rows |
| `ledger_type` | `revenue` or `expenses` |
| `ledger_group` | `rental_income`, `general_expenses`, `management_fees`, … |
| `ledger_category` | `revenue_rent_taxed`, `bank_charges`, … |
| `ledger_code` | Numeric code (e.g. 4800) |
| `ledger_description` | Bilingual free-text description |
| `month` | `YYYY-MXX` (e.g. `2024-M06`) |
| `quarter` | `YYYY-QX` (e.g. `2024-Q3`) |
| `year` | `YYYY` |
| `profit` | Signed float — positive = income, negative = expense |

**Coverage:** 3,924 rows · 2024 (full year) + 2025 Q1 · Entity: PropCo only.

> **This dataset does NOT contain property valuations, market prices, appraisals, or addresses.**

---

## Challenges & Solutions

### 1. LLMs hallucinate numbers — so we banned them from arithmetic

**Problem:** Early prototypes had the LLM compute totals, averages, and ratios directly. Results were plausible-sounding but frequently wrong — off by rounding, sign errors, or silent aggregation mistakes.

**Solution:** Hard architectural split. The LLM is only allowed to *classify* and *extract*; all arithmetic runs through pandas in the `ComputationAgent` which is a pure-python function with zero LLM involvement. Every number in the final answer is deterministically computed and can be reproduced exactly.

---

### 2. LLM intent classification has systematic blind spots

**Problem:** Certain query patterns reliably trip up even capable LLMs. For example:
- "Which month had the highest profit in 2024?" → classified as `period_comparison` instead of `ranking`
- "Among properties with above-average revenue, which had the lowest expense ratio?" → classified as `margin_analysis` but the "expense ratio" detail and the "above-average revenue" pre-filter were silently ignored
- "How many tenants are in Building 140?" → classified as `pnl_by_property`

**Solution:** A library of deterministic override functions in `router.py` that run *after* the LLM and take priority. Each function uses targeted regex patterns to detect a specific structural pattern in the query:
- `_is_time_bucket_extreme_query` → forces `ranking` intent for "which [time bucket] had the [superlative]?" queries
- `_is_margin_data_query` → extended to detect "expense ratio" in addition to "margin"
- `_is_tenant_listing_query`, `_is_count_query`, `_is_ledger_reconciliation_query` → further overrides

This hybrid approach keeps the LLM as the default classifier for the long tail while locking down high-frequency patterns where it consistently fails.

---

### 3. Extraction noise — the LLM over-specifies filters and invents targets

**Problem:** When asked "Top 5 properties by profit in 2024?", the LLM would sometimes set `ledger_type=revenue` (because "profit" rhymes with revenue in its training data), or inject a `targets=["Building 120"]` from the dataset description in its context window. Either mistake silently narrows the query scope and produces wrong answers.

**Solution:** A multi-pass defensive post-processing layer in `extract.py` that runs *after* the LLM returns:

- **Ledger filter guard:** If the user's query contains no explicit ledger keywords (`expense`, `rental income`, `management fee`, …), clear any `ledger_filters` the LLM set.
- **Ledger group specificity guard:** If `ledger_group` is set, verify the query contains a group-specific term; otherwise clear it (to avoid "revenue" → `rental_income` over-inference).
- **Ranking target guard:** For ranking intent, any target that was not explicitly named verbatim in the query string is cleared — ranking means comparing everyone, not filtering to one.
- **Net profit guard:** If the query defines net profit as "revenue minus expenses", clear a mistakenly set `ledger_type=revenue`.

---

### 4. Compound conditional queries span multiple reasoning steps

**Problem:** Queries like *"Among properties with above-average revenue in 2024, which one had the lowest expense ratio?"* require a two-step computation: first filter the candidate set, then rank within it. The system initially either ignored the filter entirely or routed to `compute_presence_condition` which could list the filtered set but couldn't then rank by a ratio metric.

**Solution:** Introduced two new state fields (`ratio_type`, `margin_pre_filter`) that flow from extraction into computation. `compute_margin_analysis` now accepts an optional `pre_filter` argument. When set to `"above_average_revenue"`, it computes per-group revenue, finds the mean, filters the candidate set, and *then* computes the requested ratio on the filtered subset. The result carries the average threshold so the response can cite it.

---

### 5. "Which one?" singular queries returning ranked lists

**Problem:** Queries phrased as "which one had the lowest X?" clearly expect a single winner, but the system defaulted to `k=5` because `_is_single_winner_ranking_query` only matched "which `<entity-type>`" patterns (e.g., "which property", "which tenant"), not the pronoun form "which one".

**Solution:** Extended the singular-winner detection regex to include `\b(which|what)\s+one\b`. This ensures `k=1` is set whenever the query uses the pronoun form, and the response formatter produces a clean single-winner sentence rather than a ranked table.

---

### 6. Multi-turn pronoun and reference resolution

**Problem:** In multi-turn conversations, users naturally refer back to prior results: *"What about that tenant?"*, *"How did it perform in Q1?"*, *"Show me the same breakdown for that building."* Without context, each query is unresolvable.

**Solution:** A four-pass reference resolution chain in `extract.py` that runs when no explicit target was found in the current query:
1. **Temporal refs** — "then", "that year", "same period" → resolve to the last mentioned timeframe in conversation context
2. **Locative refs** — "there", "that building", "that property" → scan context for a property name
3. **Demonstrative entity refs** — "that tenant", "that company", "that one" → scan context for a tenant name
4. **Pronoun refs** — "he", "she", "they", "it" → try tenant first (more specific), then property

The conversation context (last 1-2 Q&A exchanges) is injected by the Streamlit UI into every new `AgentState`.

---

### 7. LangGraph interrupts for graceful clarification

**Problem:** When the user omits a required field (e.g., no timeframe for a P&L query), a naive implementation would either return a wrong answer (using all data) or crash. Forcing the user to re-type the full query after being told what's missing is a poor UX.

**Solution:** LangGraph's `interrupt()` / `resume()` mechanism. When extraction detects a missing required field, it calls `interrupt(clarifying_question)` which pauses graph execution mid-node, serialises the full state, and surfaces the question to the user. When the user answers, the graph resumes from the same node with the answer injected as `_clarification_answer`. The extraction node re-runs with the additional context and fills the missing field, then the graph continues normally — without restarting from the beginning.

---

### 8. Fuzzy entity name resolution

**Problem:** Users spell property and tenant names inconsistently: "building120", "bldg 140", "tenant5", "the small building". Exact string matching fails silently (0 rows returned) or noisily (wrong entity matched).

**Solution:** `RetrievalAgent` runs every target name through RapidFuzz token-sort ratio matching against the known entity list. Matches above a high threshold are accepted automatically. Matches in a middle band trigger a `DisambiguationInterrupt` showing the top candidates so the user can confirm which one they meant. Matches below the low threshold route to `FallbackAgent` with a "did you mean?" suggestion.

---

## Project Structure

```
Real_Estate_Asset_Multi_Agent/
├── src/
│   ├── app.py              ← Streamlit UI (chat + debug panel)
│   ├── graph.py            ← LangGraph StateGraph + LLM factory
│   ├── agents/
│   │   ├── router.py       ← Intent classification + deterministic overrides
│   │   ├── extract.py      ← Entity extraction + defensive post-processing
│   │   ├── retrieve.py     ← Fuzzy name resolution + DataFrame query
│   │   ├── compute.py      ← Pure pandas aggregation (no LLM)
│   │   ├── respond.py      ← Deterministic formatters + LLM for general Q&A
│   │   └── fallback.py     ← Unsupported / no-match / error handler
│   ├── tools/
│   │   └── df_tools.py     ← Sole interface to the DataFrame
│   └── core/
│       ├── state.py        ← AgentState TypedDict + Intent constants
│       └── schemas.py      ← Pydantic models + system prompts
├── tests/
│   ├── test_tools.py
│   ├── test_compute.py
│   ├── test_respond.py
│   └── test_routing.py
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

The test suite covers:
- **122 tests**, all passing without an API key
- Routing heuristics and deterministic overrides
- Computation functions (all intents, edge cases, ties, empty data)
- Response formatters (all result types, expense ratio, pre-filter context)
- DataFrame tool functions (filtering, fuzzy match, timeframe parsing)
