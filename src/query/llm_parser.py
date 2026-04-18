# [SCAFFOLD] Claude-generated — not yet run or tested by the team.
# Entry point: parse_query(natural_language_query) → QueryFilter
# Requires ANTHROPIC_API_KEY in .env

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import anthropic

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


@dataclass
class QueryFilter:
    """Structured filter criteria extracted from a natural-language query."""
    regions: list[str] = field(default_factory=list)
    country_iso3s: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    max_coverage_ratio: Optional[float] = None
    min_pin: Optional[int] = None
    year_range: tuple[int, int] | None = None
    require_hrp: bool = False
    chronic_only: bool = False
    raw_query: str = ""


_SYSTEM_PROMPT = """
You are a humanitarian data analyst assistant.
Given a natural-language query from a decision-maker, extract structured filter criteria
and return them as a JSON object with these optional keys:

  regions        : list of UN region codes or names (e.g. "SSA", "MENA", "LAC")
  country_iso3s  : list of ISO 3-letter country codes
  sectors        : list of humanitarian sectors (food, health, shelter, wash, education, protection)
  max_coverage_ratio : float between 0 and 1 (e.g. 0.10 means "funded less than 10%")
  min_pin        : integer minimum people-in-need threshold
  year_range     : [start_year, end_year] (both inclusive)
  require_hrp    : boolean — true if query specifically asks about HRP countries
  chronic_only   : boolean — true if query asks about multi-year or structural underfunding

Return ONLY valid JSON. Omit keys that are not mentioned or clearly implied.
""".strip()


def parse_query(natural_language_query: str) -> QueryFilter:
    """Parse a free-text query into a QueryFilter using Claude."""
    client = _get_client()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": natural_language_query}],
    )
    raw = message.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    parsed = json.loads(raw)

    year_range = parsed.get("year_range")
    if year_range and len(year_range) == 2:
        year_range = (int(year_range[0]), int(year_range[1]))
    else:
        year_range = None

    return QueryFilter(
        regions=parsed.get("regions", []),
        country_iso3s=parsed.get("country_iso3s", []),
        sectors=parsed.get("sectors", []),
        max_coverage_ratio=parsed.get("max_coverage_ratio"),
        min_pin=parsed.get("min_pin"),
        year_range=year_range,
        require_hrp=parsed.get("require_hrp", False),
        chronic_only=parsed.get("chronic_only", False),
        raw_query=natural_language_query,
    )


# [SCAFFOLD] generate_explanation — not yet tested.
# Produces a plain-English briefing paragraph for a single ranked crisis row.
#
# def generate_explanation(row: dict, query: str) -> str:
#     client = _get_client()
#     prompt = (
#         f"Query: {query}\n\n"
#         f"Crisis data: {json.dumps(row, default=str)}\n\n"
#         "Write one short paragraph (3-4 sentences) explaining to a humanitarian coordinator "
#         "why this crisis appears most overlooked. Be specific about the numbers. "
#         "Acknowledge any data limitations. Do not make up figures not present in the data."
#     )
#     message = client.messages.create(
#         model="claude-sonnet-4-6",
#         max_tokens=300,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return message.content[0].text.strip()
