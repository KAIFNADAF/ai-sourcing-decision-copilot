import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


DEFAULT_SCENARIO = {
    "total_demand": 120000,
    "max_supplier_share": 0.40,
    "min_avg_esg": 70,
    "max_avg_risk": 45,
    "min_suppliers": 3,
    "blocked_regions": [],
    "w_cost": 0.65,
    "w_risk": 0.20,
    "w_esg": 0.10,
    "supplier_selection_penalty": 0.02,
}


ALLOWED_REGIONS = [
    "Ireland",
    "UK",
    "Eastern Europe",
    "Asia Pacific",
    "North America",
    "Western Europe",
]


NUMERIC_FIELDS_INT = {
    "total_demand",
    "min_avg_esg",
    "max_avg_risk",
    "min_suppliers",
}

NUMERIC_FIELDS_FLOAT = {
    "max_supplier_share",
    "w_cost",
    "w_risk",
    "w_esg",
    "supplier_selection_penalty",
}


def _normalize_region_name(region: str) -> str:
    region = region.strip().lower()

    region_map = {
        "ireland": "Ireland",
        "uk": "UK",
        "united kingdom": "UK",
        "great britain": "UK",
        "britain": "UK",
        "asia pacific": "Asia Pacific",
        "apac": "Asia Pacific",
        "eastern europe": "Eastern Europe",
        "east europe": "Eastern Europe",
        "europe east": "Eastern Europe",
        "north america": "North America",
        "na": "North America",
        "western europe": "Western Europe",
        "west europe": "Western Europe",
    }

    return region_map.get(region, region.title())


def _clean_region_list(regions: List[str]) -> List[str]:
    cleaned = []
    for region in regions:
        if not isinstance(region, str):
            continue
        normalized = _normalize_region_name(region)
        if normalized in ALLOWED_REGIONS and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def _extract_json_block(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return match.group(0)


def _parse_numeric_value(value: Any) -> Optional[float]:
    """
    Safely parses numbers from model output.

    Supports:
    - 130000
    - "130000"
    - "130k"
    - "40%"
    - "0.4"
    - "at least 75"

    Returns None for vague text like:
    - "strong"
    - "pretty low"
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return None

    text = value.strip().lower()
    if not text:
        return None

    vague_terms = {
        "strong",
        "very strong",
        "high",
        "low",
        "pretty low",
        "very low",
        "reasonable",
        "a bit more",
        "slightly more",
        "moderate",
        "balanced",
    }
    if text in vague_terms:
        return None

    shorthand_match = re.fullmatch(r"(\d+(\.\d+)?)\s*([km])", text)
    if shorthand_match:
        number = float(shorthand_match.group(1))
        suffix = shorthand_match.group(3)
        if suffix == "k":
            return number * 1_000
        if suffix == "m":
            return number * 1_000_000

    match = re.search(r"-?\d+(\.\d+)?", text)
    if not match:
        return None

    number = float(match.group(0))

    if "%" in text:
        return number / 100.0

    return number


def _coerce_types(parsed: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], Dict[str, str]]:
    """
    Converts only safely-parsable fields.

    Returns:
    - coerced scenario fields
    - assumptions generated during coercion
    - field_sources mapping:
        field -> explicit | default | heuristic
    """
    result: Dict[str, Any] = {}
    assumptions: List[str] = []
    field_sources: Dict[str, str] = {}

    for field, value in parsed.items():
        if field == "blocked_regions":
            if value is None:
                result[field] = []
                field_sources[field] = "default"
            elif isinstance(value, list):
                result[field] = _clean_region_list(value)
                field_sources[field] = "explicit"
            elif isinstance(value, str):
                result[field] = _clean_region_list([value])
                field_sources[field] = "explicit"
            else:
                result[field] = []
                field_sources[field] = "default"
                assumptions.append(
                    f"{field} could not be parsed cleanly, so no blocked regions were applied"
                )
            continue

        if field in NUMERIC_FIELDS_INT:
            parsed_number = _parse_numeric_value(value)
            if parsed_number is not None:
                result[field] = int(round(parsed_number))
                field_sources[field] = "explicit"
            continue

        if field in NUMERIC_FIELDS_FLOAT:
            parsed_number = _parse_numeric_value(value)
            if parsed_number is not None:
                result[field] = float(parsed_number)
                field_sources[field] = "explicit"
            continue

        result[field] = value

    if "blocked_regions" not in result:
        result["blocked_regions"] = []
        field_sources["blocked_regions"] = "default"

    return result, assumptions, field_sources


def _contains_any(text: str, phrases: List[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _apply_heuristic_mappings(
    user_request: str,
    scenario_fields: Dict[str, Any],
    field_sources: Dict[str, str],
) -> Tuple[Dict[str, Any], List[str], Dict[str, str]]:
    """
    Applies narrow, explicit business-language mappings only for fields
    that were not already explicitly parsed.
    """
    text = user_request.lower()
    assumptions: List[str] = []

    updated = dict(scenario_fields)
    sources = dict(field_sources)

    def set_if_missing(field: str, value: Any, assumption: str) -> None:
        if field not in updated:
            updated[field] = value
            sources[field] = "heuristic"
            assumptions.append(assumption)

    # -------------------------
    # ESG language
    # -------------------------
    if "esg" in text:
        if any(word in text for word in ["very strong", "very high", "excellent"]):
            set_if_missing(
                "min_avg_esg",
                85,
                "min_avg_esg was heuristically set to 85 from very strong ESG language",
            )
        elif any(word in text for word in ["strong", "high"]):
            set_if_missing(
                "min_avg_esg",
                80,
                "min_avg_esg was heuristically set to 80 from strong ESG language",
            )
        elif any(word in text for word in ["good", "decent", "solid"]):
            set_if_missing(
                "min_avg_esg",
                75,
                "min_avg_esg was heuristically set to 75 from moderate ESG language",
            )

    # -------------------------
    # Risk language
    # -------------------------
    if "risk" in text:
        if any(word in text for word in ["very low", "extremely low"]):
            set_if_missing(
                "max_avg_risk",
                25,
                "max_avg_risk was heuristically set to 25 from very low risk language",
            )
        elif any(word in text for word in ["pretty low", "low"]):
            set_if_missing(
                "max_avg_risk",
                35,
                "max_avg_risk was heuristically set to 35 from low risk language",
            )
        elif any(word in text for word in ["moderate", "reasonable"]):
            set_if_missing(
                "max_avg_risk",
                45,
                "max_avg_risk was heuristically set to 45 from moderate risk language",
            )

    # -------------------------
    # Cost / trade-off language
    # -------------------------
    if _contains_any(
        text,
        [
            "okay paying a bit more",
            "ok paying a bit more",
            "willing to pay more",
            "can pay more",
            "slightly higher cost is okay",
            "accept a bit more cost",
            "pay a bit more",
        ],
    ):
        set_if_missing(
            "w_cost",
            0.50,
            "w_cost was heuristically reduced to reflect willingness to pay somewhat more",
        )
        set_if_missing(
            "w_esg",
            0.30,
            "w_esg was heuristically increased to reflect willingness to trade some cost for ESG",
        )
        set_if_missing(
            "w_risk",
            0.30,
            "w_risk was heuristically increased to reflect stated concern for low risk",
        )

    # -------------------------
    # Concentration / diversification language
    # -------------------------
    concentration_phrases = [
        "not too concentrated",
        "avoid concentration",
        "too concentrated",
        "2-3 suppliers",
        "across just 2-3 suppliers",
        "spread it out",
        "more balanced supplier base",
    ]
    if _contains_any(text, concentration_phrases):
        set_if_missing(
            "max_supplier_share",
            0.30,
            "max_supplier_share was heuristically tightened to 0.30 from concentration-avoidance language",
        )

        current_min_suppliers = updated.get("min_suppliers")
        if current_min_suppliers is None or current_min_suppliers < 4:
            updated["min_suppliers"] = 4
            if "min_suppliers" not in sources:
                sources["min_suppliers"] = "heuristic"
            assumptions.append(
                "min_suppliers was heuristically set to at least 4 to support broader supplier spread"
            )

    if _contains_any(text, ["diverse supplier base", "diversified supplier base", "broader supplier base"]):
        current_min_suppliers = updated.get("min_suppliers")
        if current_min_suppliers is None or current_min_suppliers < 4:
            updated["min_suppliers"] = 4
            if "min_suppliers" not in sources:
                sources["min_suppliers"] = "heuristic"
            assumptions.append(
                "min_suppliers was heuristically set to at least 4 from diversification language"
            )

    # -------------------------
    # Region blocking language
    # -------------------------
    region_phrase_map = {
        "ireland": "Ireland",
        "uk": "UK",
        "united kingdom": "UK",
        "great britain": "UK",
        "britain": "UK",
        "eastern europe": "Eastern Europe",
        "asia pacific": "Asia Pacific",
        "apac": "Asia Pacific",
        "north america": "North America",
        "western europe": "Western Europe",
    }

    for phrase, canonical_region in region_phrase_map.items():
        if (
            f"don't use {phrase}" in text
            or f"dont use {phrase}" in text
            or f"avoid {phrase}" in text
            or f"block {phrase}" in text
            or f"exclude {phrase}" in text
        ):
            existing = updated.get("blocked_regions", [])
            if canonical_region not in existing:
                updated["blocked_regions"] = _clean_region_list(existing + [canonical_region])
                if "blocked_regions" not in sources:
                    sources["blocked_regions"] = "heuristic"
                assumptions.append(
                    f"blocked_regions was heuristically mapped from natural-language exclusion of {canonical_region}"
                )

    return updated, assumptions, sources


def _apply_defaults(parsed: Dict[str, Any], field_sources: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    final = {}
    updated_sources = dict(field_sources)

    for key, default_value in DEFAULT_SCENARIO.items():
        if key in parsed:
            final[key] = parsed[key]
        else:
            final[key] = default_value
            updated_sources[key] = "default"

    return final, updated_sources


def _normalize_weights(parsed: Dict[str, Any], field_sources: Dict[str, str]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Normalizes the three objective weights to sum to 1 when possible.
    """
    validated = dict(parsed)
    assumptions: List[str] = []

    w_cost = float(validated.get("w_cost", 0.0))
    w_risk = float(validated.get("w_risk", 0.0))
    w_esg = float(validated.get("w_esg", 0.0))

    total = w_cost + w_risk + w_esg
    if total > 0:
        normalized_cost = round(w_cost / total, 4)
        normalized_risk = round(w_risk / total, 4)
        normalized_esg = round(w_esg / total, 4)

        if (
            abs(normalized_cost - w_cost) > 1e-6
            or abs(normalized_risk - w_risk) > 1e-6
            or abs(normalized_esg - w_esg) > 1e-6
        ):
            assumptions.append(
                "Objective weights were normalized to keep cost, risk, and ESG trade-offs on a consistent scale"
            )

        validated["w_cost"] = normalized_cost
        validated["w_risk"] = normalized_risk
        validated["w_esg"] = normalized_esg

    return validated, assumptions


def _validate_ranges(parsed: Dict[str, Any]) -> Dict[str, Any]:
    validated = dict(parsed)

    validated["total_demand"] = max(1, validated["total_demand"])
    validated["max_supplier_share"] = min(max(validated["max_supplier_share"], 0.05), 1.0)
    validated["min_avg_esg"] = min(max(validated["min_avg_esg"], 0), 100)
    validated["max_avg_risk"] = min(max(validated["max_avg_risk"], 0), 100)
    validated["min_suppliers"] = max(1, validated["min_suppliers"])

    validated["w_cost"] = min(max(validated["w_cost"], 0.0), 1.0)
    validated["w_risk"] = min(max(validated["w_risk"], 0.0), 1.0)
    validated["w_esg"] = min(max(validated["w_esg"], 0.0), 1.0)
    validated["supplier_selection_penalty"] = min(max(validated["supplier_selection_penalty"], 0.0), 1.0)

    return validated


def _derive_missing_fields(field_sources: Dict[str, str]) -> List[str]:
    return [key for key in DEFAULT_SCENARIO if field_sources.get(key) == "default"]


def _derive_heuristic_fields(field_sources: Dict[str, str]) -> List[str]:
    return [key for key in DEFAULT_SCENARIO if field_sources.get(key) == "heuristic"]


def _derive_explicit_fields(field_sources: Dict[str, str]) -> List[str]:
    return [key for key in DEFAULT_SCENARIO if field_sources.get(key) == "explicit"]


def _detect_unmapped_user_intent(user_request: str) -> List[str]:
    request = user_request.lower()
    limitations = []

    if "concentrat" in request or "too concentrated" in request or "2-3 suppliers" in request:
        limitations.append(
            "Concentration preference was approximated through max supplier share and supplier count because the schema does not model top-2 concentration directly."
        )

    if "bit more" in request or "slightly more" in request or "okay paying more" in request or "pay a bit more" in request:
        limitations.append(
            "Soft willingness-to-pay language was interpreted heuristically rather than from an explicit numeric trade-off."
        )

    if "pretty low" in request or "low risk" in request or "strong esg" in request or ("esg" in request and "strong" in request):
        limitations.append(
            "Some qualitative preferences were mapped heuristically and should be reviewed before analysis."
        )

    return limitations


def _build_parser_prompt(user_request: str) -> str:
    return f"""
You are a sourcing scenario parser.

Your task is to convert a natural language sourcing request into a structured JSON object.

Rules:
1. Extract only scenario parameters, not sourcing decisions.
2. Do not invent supplier names or allocation outcomes.
3. If a value is not explicitly numeric or clearly inferable, omit it rather than guessing.
4. Only use these keys:
   - total_demand
   - max_supplier_share
   - min_avg_esg
   - max_avg_risk
   - min_suppliers
   - blocked_regions
   - w_cost
   - w_risk
   - w_esg
   - supplier_selection_penalty
   - interpretation
   - assumptions
   - missing_fields
5. blocked_regions must be a list.
6. interpretation must summarize what the user asked for in plain business language.
7. assumptions must be a list of short statements.
8. missing_fields must be a list of keys that were not explicitly provided numerically.
9. Return valid JSON only. No markdown, no explanation outside JSON.
10. Do not put words like "strong", "low", or "reasonable" into numeric fields. Omit those fields instead.

Allowed region values:
- Ireland
- UK
- Eastern Europe
- Asia Pacific
- North America
- Western Europe

User request:
{user_request}
""".strip()


def parse_sourcing_request(user_request: str, model: str = "llama-3.1-8b-instant") -> Dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your environment or .env file.")

    client = Groq(api_key=api_key)
    prompt = _build_parser_prompt(user_request)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You convert sourcing requests into structured JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    raw_text = response.choices[0].message.content.strip()
    json_text = _extract_json_block(raw_text)
    parsed = json.loads(json_text)

    scenario_fields_raw = {
        key: value
        for key, value in parsed.items()
        if key in DEFAULT_SCENARIO
    }

    scenario_fields, coercion_assumptions, field_sources = _coerce_types(scenario_fields_raw)

    scenario_fields, heuristic_assumptions, field_sources = _apply_heuristic_mappings(
        user_request=user_request,
        scenario_fields=scenario_fields,
        field_sources=field_sources,
    )

    scenario_fields, field_sources = _apply_defaults(
        parsed=scenario_fields,
        field_sources=field_sources,
    )

    scenario_fields, weight_assumptions = _normalize_weights(
        parsed=scenario_fields,
        field_sources=field_sources,
    )

    scenario_fields = _validate_ranges(scenario_fields)

    model_assumptions = parsed.get("assumptions", [])
    if not isinstance(model_assumptions, list):
        model_assumptions = []

    limitation_assumptions = _detect_unmapped_user_intent(user_request)

    assumptions: List[str] = []
    assumptions.extend(model_assumptions)
    assumptions.extend(coercion_assumptions)
    assumptions.extend(heuristic_assumptions)
    assumptions.extend(weight_assumptions)
    assumptions.extend(limitation_assumptions)

    missing_fields = _derive_missing_fields(field_sources)
    heuristic_fields = _derive_heuristic_fields(field_sources)
    explicit_fields = _derive_explicit_fields(field_sources)

    if not assumptions and missing_fields:
        assumptions = [
            f"{field} was not explicitly provided, so the default value was used"
            for field in missing_fields
        ]

    return {
        "raw_request": user_request,
        "parsed_scenario": scenario_fields,
        "interpretation": parsed.get(
            "interpretation",
            "Parsed sourcing scenario from user request."
        ),
        "assumptions": assumptions,
        "missing_fields": missing_fields,
        "heuristic_fields": heuristic_fields,
        "explicit_fields": explicit_fields,
        "field_sources": field_sources,
        "raw_model_output": raw_text,
    }


if __name__ == "__main__":
    sample_request = (
        "Need to cover 130k units. Avoid UK and Ireland. "
        "I’m okay paying a bit more if ESG is strong. "
        "Risk should stay pretty low. "
        "Also don’t make it too concentrated across just 2-3 suppliers."
    )

    result = parse_sourcing_request(sample_request)

    print("\nPARSED SCENARIO:")
    for key, value in result["parsed_scenario"].items():
        print(f"{key}: {value}")

    print("\nINTERPRETATION:")
    print(result["interpretation"])

    print("\nASSUMPTIONS:")
    for item in result["assumptions"]:
        print("-", item)

    print("\nEXPLICIT FIELDS:")
    for item in result["explicit_fields"]:
        print("-", item)

    print("\nHEURISTIC FIELDS:")
    for item in result["heuristic_fields"]:
        print("-", item)

    print("\nDEFAULTED FIELDS:")
    for item in result["missing_fields"]:
        print("-", item)

    print("\nFIELD SOURCES:")
    for key, value in result["field_sources"].items():
        print(f"- {key}: {value}")