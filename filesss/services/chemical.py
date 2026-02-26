import json
import os


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chemical_db.json")


def _load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_name(name):
    return " ".join(name.strip().lower().split())


def analyze_chemicals(chemicals_text):
    db = _load_db()
    chemicals = [
        _normalize_name(item)
        for item in chemicals_text.replace(";", ",").split(",")
        if item.strip()
    ]

    warnings = []
    avoid = []
    alternatives = []
    details = []

    for chemical in chemicals:
        info = db.get(chemical)
        if not info:
            details.append(
                {
                    "chemical": chemical,
                    "known": False,
                    "message": "No database entry found. Use with expert guidance.",
                }
            )
            continue

        known_warnings = []
        if info.get("bee_toxicity", "").lower() == "high":
            known_warnings.append("May harm pollinators")
        if info.get("soil_microbe_impact", "").lower() in {"high", "severe"}:
            known_warnings.append("Reduces soil microbial diversity")
        if info.get("human_health_risk", "").lower() in {"high", "moderate"}:
            known_warnings.append("Potential human health risk if misused")

        if known_warnings:
            warnings.extend(known_warnings)
            avoid.append(chemical)

        alternative = info.get("recommended_alternative")
        if alternative:
            alternatives.append(alternative)

        details.append(
            {
                "chemical": chemical,
                "known": True,
                "bee_toxicity": info.get("bee_toxicity", "Unknown"),
                "soil_microbe_impact": info.get("soil_microbe_impact", "Unknown"),
                "human_health_risk": info.get("human_health_risk", "Unknown"),
                "recommended_alternative": alternative or "N/A",
                "warnings": known_warnings,
            }
        )

    return {
        "input_chemicals": chemicals,
        "warnings": sorted(set(warnings)),
        "chemicals_to_avoid": sorted(set(avoid)),
        "alternatives": sorted(set(alternatives)),
        "details": details,
    }
