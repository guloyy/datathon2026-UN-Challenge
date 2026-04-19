"""LLM intent classifier + parser for the /prioritize flow.

Classifies NL queries into one of three modes:

    target_sector   — "prioritize Brazil for water supply"
    target_only     — "prioritize Colombia"
    filter_only     — "how should we fund the Middle East", "how should we allocate water funding"

Reuses the Anthropic client pattern from src/query/llm_parser.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

Mode = Literal["target_sector", "target_only", "filter_only"]

_client = None


def _get_client():
    """Lazy-import anthropic to keep Intent importable in test/offline contexts."""
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


@dataclass
class Intent:
    mode: Mode
    country_iso3: Optional[str] = None
    sector: Optional[str] = None
    region: Optional[str] = None
    emergency_group: Optional[str] = None
    year: Optional[int] = None
    hints: list[str] = field(default_factory=list)
    raw_query: str = ""


SECTOR_CODES = {
    "FSC": "Food Security",
    "HEA": "Health",
    "WSH": "WASH / water sanitation hygiene",
    "SHL": "Shelter",
    "PRO": "Protection",
    "EDU": "Education",
    "NUT": "Nutrition",
    "MPC": "Multipurpose Cash",
    "LOG": "Logistics",
    "ETC": "Emergency Telecoms",
    "CCCM": "Camp Management",
    "ERY": "Early Recovery",
    "COOR": "Coordination",
    "ALL": "All Sectors (pre-aggregated total)",
}

REGION_CODES = {
    "SSA": "Sub-Saharan Africa",
    "MENA": "Middle East & North Africa",
    "APAC": "Asia-Pacific",
    "LAC": "Latin America & Caribbean",
    "EECA": "Eastern Europe & Central Asia",
}

EMERGENCY_GROUPS = ("Conflict-related", "Natural Disaster", "Other")

_SYSTEM_PROMPT = f"""
You are a humanitarian data assistant. Classify the user's natural-language query into
one of three modes and extract structured fields. Return ONLY valid JSON with these keys:

  mode            : "target_sector" | "target_only" | "filter_only"
  country_iso3    : ISO-3 code, only when mode starts with "target"
  sector          : sector code, only when mode == "target_sector"
  region          : region code, only when mode == "filter_only"
  emergency_group : one of {EMERGENCY_GROUPS}, only when filter_only and user mentions an emergency type
  year            : integer year if the user specifies one
  hints           : list of short free-form keywords (e.g. ["water", "largest need"])

MODE RULES:
  - "target_sector": user names a specific country AND a specific sector/cluster
      ("prioritize Brazil for water supply", "I want to advocate for Colombia's food response")
  - "target_only": user names a specific country with no sector
      ("prioritize Yemen", "advocate for Colombia")
  - "filter_only": user does NOT name a target country — either a region question, a sector
    question, or a general funding question
      ("how should we fund the Middle East", "how should we allocate water funding",
       "which conflict crises are most underfunded")

SECTOR CODES (map user phrases to these codes):
{json.dumps(SECTOR_CODES, indent=2)}

REGION CODES:
{json.dumps(REGION_CODES, indent=2)}

Return only the JSON object. No commentary, no markdown fences.
""".strip()


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            body = parts[1]
            if body.lower().startswith("json"):
                body = body[4:]
            return body.strip()
    return raw


def parse_intent(natural_language_query: str) -> Intent:
    """Parse a free-text query into an Intent using Claude."""
    client = _get_client()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": natural_language_query}],
    )
    raw = _strip_fences(message.content[0].text)
    parsed = json.loads(raw)

    mode = parsed.get("mode")
    if mode not in ("target_sector", "target_only", "filter_only"):
        raise ValueError(f"LLM returned invalid mode: {mode!r}")

    return Intent(
        mode=mode,
        country_iso3=(parsed.get("country_iso3") or None),
        sector=(parsed.get("sector") or None),
        region=(parsed.get("region") or None),
        emergency_group=(parsed.get("emergency_group") or None),
        year=int(parsed["year"]) if parsed.get("year") is not None else None,
        hints=list(parsed.get("hints", []) or []),
        raw_query=natural_language_query,
    )
