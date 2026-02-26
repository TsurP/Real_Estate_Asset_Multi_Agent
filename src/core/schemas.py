"""
Prompt templates and JSON schemas for LLM-driven agents.
All structured outputs use Pydantic models validated before use.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .state import Intent

# ---------------------------------------------------------------------------
# Pydantic models for structured LLM outputs
# ---------------------------------------------------------------------------


class RouterOutput(BaseModel):
    """Output from the RouterAgent."""
    model_config = ConfigDict(extra="forbid")
    intent: str = Field(
        description=f"Intent classification. One of: {Intent.ALL}"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0 and 1"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Brief reasoning for the classification"
    )

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        if v not in Intent.ALL:
            # Try fuzzy recovery
            v_lower = v.lower().replace("-", "_").replace(" ", "_")
            for intent in Intent.ALL:
                if intent in v_lower or v_lower in intent:
                    return intent
            return Intent.UNSUPPORTED
        return v


class TimeframeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    year: Optional[str] = None
    quarter: Optional[str] = None
    month: Optional[str] = None
    ytd: bool = False
    label: str = ""


class LedgerFilters(BaseModel):
    """Structured ledger filters — typed to satisfy OpenAI strict schema (additionalProperties=false)."""
    model_config = ConfigDict(extra="forbid")
    ledger_type: Optional[str] = None
    ledger_group: Optional[str] = None
    ledger_category: Optional[str] = None
    ledger_code: Optional[int] = None
    description_keyword: Optional[str] = None


class ExtractionOutput(BaseModel):
    """Output from the ExtractionAgent."""
    model_config = ConfigDict(extra="forbid")
    targets: list[str] = Field(
        default_factory=list,
        description="Raw target names mentioned (property, tenant, entity). 1-2 items."
    )
    target_fields: list[str] = Field(
        default_factory=list,
        description="Field keys for targets: property_name, tenant_name, entity_name"
    )
    timeframe: Optional[TimeframeOutput] = Field(
        default=None,
        description="Parsed time reference"
    )
    group_by: Optional[Literal[
        "property_name", "tenant_name", "entity_name",
        "ledger_type", "ledger_group", "ledger_category", "ledger_code",
        "year", "quarter", "month"
    ]] = Field(
        default=None,
        description=(
            "DataFrame column to aggregate by. Must be exactly one of the listed values. "
            "Use 'year', 'quarter', or 'month' when the user asks to see data "
            "'by year', 'per year', 'by quarter', 'by month', etc."
        )
    )
    ledger_filters: LedgerFilters = Field(
        default_factory=LedgerFilters,
        description="Optional filters: ledger_type, ledger_group, ledger_category, ledger_code, description_keyword"
    )
    k: int = Field(default=5, description="K for ranking intent")
    rank_direction: Literal["top", "bottom"] = Field(default="top", description="'top' or 'bottom'")
    ratio_type: Optional[Literal["profit_margin", "expense_ratio"]] = Field(
        default=None,
        description=(
            "For margin_analysis intent: 'expense_ratio' when the user asks for expense ratio "
            "(|expenses| ÷ revenue), 'profit_margin' for net profit margin (profit ÷ revenue). "
            "Leave null to default to profit_margin."
        )
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Fields still needed: timeframe, target, comparison_target"
    )

    @field_validator("target_fields")
    @classmethod
    def validate_target_fields(cls, v: list[str]) -> list[str]:
        valid = {"property_name", "tenant_name", "entity_name"}
        return [f for f in v if f in valid]

    @field_validator("rank_direction", mode="before")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        return "bottom" if str(v).lower() in ("bottom", "worst", "lowest") else "top"


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

DATASET_CONTEXT = """\
You are analyzing a real-estate portfolio P&L ledger with these columns:
- entity_name: legal entity (PropCo)
- property_name: building name (e.g. "Building 140")
- tenant_name: tenant (may be null for entity-level rows)
- ledger_type: "revenue" or "expenses"
- ledger_group: e.g. "rental_income", "general_expenses", "management_fees"
- ledger_category: e.g. "revenue_rent_taxed", "bank_charges"
- ledger_code: numeric code (e.g. 4800)
- ledger_description: free-text description
- month: "YYYY-MXX" (e.g. "2024-M06")
- quarter: "YYYY-QX" (e.g. "2024-Q3")
- year: "YYYY"
- profit: signed float (positive=income, negative=expense)

Available data: years 2024 and 2025 (through Q1 2025).
Entity: PropCo only. Properties: Building 120, 140, 160, 17, 180. Tenants: Tenant 1–18.

IMPORTANT: This dataset does NOT contain property valuations, market prices, or addresses.\
"""

ROUTER_SYSTEM_PROMPT = f"""\
{DATASET_CONTEXT}

You are a routing agent. Classify the user's query into exactly one intent:

- pnl_total: Total portfolio profit for a timeframe (or for a specific tenant/property)
- pnl_by_property: Profit broken down by property; also use this when asking which
  building(s) a specific tenant is in / associated with — the answer comes from grouping
  that tenant's rows by property_name
- pnl_by_tenant: Profit broken down by tenant
- breakdown: Profit broken down by a ledger dimension (type/group/category/code)
- comparison: Compare profit between exactly two NAMED entities (two properties, two tenants,
  or two entities). Both names must be explicitly stated.
- share_of_total: Calculate what percentage/share a specific property, tenant, or entity
  contributes to a total (e.g., "What % of total revenue came from Building 120?",
  "What share of expenses is Tenant 5?", "How much of total profit is Building 140?").
  Use whenever the user asks for a proportion, percentage, share, or fraction of a total.
- period_comparison: Compare profit/revenue/expenses across two DIFFERENT TIME PERIODS
  (e.g., "which property had the largest increase from Q1 to Q4?", "how did profit change
  from 2024 to 2025?", "which tenant grew the most between Q1 and Q2?"). Use this whenever
  the question asks about change/growth/increase/decrease between two time periods —
  NOT about two named entities.
- margin_analysis: Analyze/rank net profit margin (profit ÷ revenue), especially for queries like
  "worst net profit margin", "best margin by property", "highest profit margin in 2024".
- count: Count distinct entities — use when the user asks "how many tenants", "how many
  properties", or "how many buildings"; AND when asking which/who tenants are in a building
  (e.g. "Who lives in Building 120?", "Which tenants are in Building 140?",
  "Who are the tenants in Building 17?"). Do NOT use for profit or financial queries.
- ranking: Top/bottom K properties or tenants by profit
- full_review: Comprehensive multi-section report on an entity/property/tenant/portfolio.
  Use ONLY for broad, unspecific requests: "review", "full review", "overview",
  "give me everything about", "tell me about", "summarise", "report on".
  This returns total P&L + breakdown by property + breakdown by tenant + breakdown by year.
  IMPORTANT: Do NOT use full_review when the user specifies a particular metric (expenses,
  revenue) AND a grouping dimension (by quarter, by month, per property). In that case use
  breakdown instead. E.g. "review of expenses by quarter" → breakdown, NOT full_review.
- general_question: Accounting or real-estate concepts (no data needed)
- unsupported: ONLY for market valuations, sale prices, physical addresses, occupancy rates,
  or appraisals. Do NOT use for review/summary/overview queries.

IMPORTANT — do NOT classify as unsupported:
- "Which building is [tenant] in?" → pnl_by_property (shows tenant revenue by property)
- "Where does [tenant] pay rent?" → pnl_by_property
- "Give me a full review of X" → full_review
- "Overview / summary of X" → full_review
- "Give me a review of expenses by quarter in 2024" → breakdown (metric + dimension = NOT full_review)
- "Show me revenue for each property" → breakdown
- "Which property had the largest increase from Q1 to Q4?" → period_comparison
- "How did profit change from 2024 to 2025?" → period_comparison
- "What % of total revenue came from Building 120?" → share_of_total
- "What share of expenses is Tenant 5?" → share_of_total
- "Which property had the worst net profit margin in 2024?" → margin_analysis
- "How many tenants are in Building 120 in 2024?" → count
- "How many properties does PropCo have?" → count
- "Who lives in Building 120?" → count (lists the tenants in that building)
- "Which tenants are in Building 140?" → count
- "Tell me about X" → full_review
- Any question answerable by filtering/grouping the ledger columns

Return JSON with: intent (string), confidence (0-1 float), notes (string).
"""

EXTRACTION_SYSTEM_PROMPT = f"""\
{DATASET_CONTEXT}

You are an entity extraction agent. Given a user query and its intent, extract:
- targets: list of 1-2 raw target names (property, tenant, entity names as mentioned)
- target_fields: list of field keys (property_name / tenant_name / entity_name)
- timeframe: time reference with fields year, quarter, month, ytd (bool), label (string)
- group_by: dimension to aggregate by (property_name, tenant_name, ledger_type, ledger_group, ledger_category, ledger_code, year, quarter, month)
- ledger_filters: object with optional fields: ledger_type (str), ledger_group (str), ledger_category (str), ledger_code (int), description_keyword (str). Leave null if not applicable.
- k: integer K for ranking (default 5)
- rank_direction: "top" or "bottom"
- ratio_type: for margin_analysis only — "expense_ratio" when user asks for expense ratio (|expenses| ÷ revenue); omit or use "profit_margin" for net profit margin
- missing_fields: list of what's still needed (timeframe / target / comparison_target)

Current date context: use system date to interpret relative expressions.
- "this year" → current year (YTD)
- "last year" → previous year (full)
- "this quarter" → current quarter
- "YTD" → current year through latest available month

Format timeframe as: year="YYYY", quarter="YYYY-QX", month="YYYY-MXX".
Only set the most specific applicable field (month > quarter > year).

IMPORTANT RULES — read carefully before extracting:
1. TIMEFRAME: Only extract if the query explicitly mentions a time period or relative expression
   ("this year", "last year", "Q3 2024", "in 2025", etc.). If no time expression is present,
   leave timeframe null. Do NOT default to the current date or year.
2. TARGETS (ranking intent): For ranking queries ("which tenant paid the most?", "top 5
   properties"), leave `targets` empty — you are ranking ALL entities, not filtering to one.
   Only set targets if a specific named entity is mentioned (e.g. "which building does Tenant 7
   appear in?").
3. LEDGER FILTERS: Only set ledger_filters if the query explicitly names a specific ledger
   dimension (e.g. "rental income", "management fees", "expenses"). Generic words like "paid",
   "earned", or "spent" do NOT justify setting ledger filters — they refer to total P&L.

Return valid JSON only.
"""

RESPONSE_SYSTEM_PROMPT = """\
You are a concise real-estate portfolio analyst. Given the computation results and original query,
write a response:
1. Start with the direct answer (numbers formatted with commas, 2 decimal places for currency).
2. Follow with 2-4 bullet points titled "How I computed this:" explaining steps.
3. Note any assumptions (e.g. timeframe interpretation).
4. Keep total response under 200 words.

Use € symbol for amounts (European portfolio).
"""

FALLBACK_SYSTEM_PROMPT = """\
You are a helpful fallback assistant for a real-estate portfolio P&L system.
When the user asks something unsupported or unclear:
1. Briefly explain why it's not available (1 sentence).
2. Suggest 1-2 relevant things the system CAN answer.
3. Ask ONE specific clarifying question if the query could be salvaged.

IMPORTANT: This system's dataset covers 2024 (full year) and 2025 (through Q1 only).
Entity: PropCo. Properties: Building 120, 140, 160, 17, 180. Tenants: Tenant 1–18.
NEVER claim the data ends in 2023 or any earlier date — that is wrong.

Supported capabilities: total portfolio profit, profit by property/tenant,
breakdowns by ledger type/group/category, comparisons between two named targets,
period-over-period profit change (e.g., Q1 to Q4), top/bottom rankings,
and general real-estate accounting definitions.

NOT supported: property valuations, market prices, appraisals, addresses,
occupancy rates, lease terms (not in this dataset).

Keep response under 100 words. Be friendly and helpful.
"""
