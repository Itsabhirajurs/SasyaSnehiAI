def generate_advisory(
    disease,
    severity,
    risk_score,
    risk_level,
    chemicals_info,
    soil_type,
    watering_frequency,
):
    urgency = "Immediate action required" if severity == "Severe" else "Monitor and manage"
    spread_warning = (
        "High outbreak potential" if risk_score > 0.75 else "Spread risk currently manageable"
    )

    actions = []
    possible_causes = []

    if severity == "Severe":
        actions.append("Isolate affected plant parts and start treatment immediately")
    elif severity == "Moderate":
        actions.append("Start treatment within 24-48 hours and monitor spread")
    else:
        actions.append("Continue observation and preventive care")

    if "healthy" not in disease.lower():
        possible_causes.extend(
            [
                "High moisture around foliage",
                "Insufficient air circulation",
                "Pathogen carryover from previous crop cycle",
            ]
        )

    if soil_type.strip().lower() == "clay" and watering_frequency >= 5:
        possible_causes.append("Possible root-zone stress due to overwatering in clay soil")
        actions.append("Reduce watering frequency to avoid root rot conditions")

    if chemicals_info.get("warnings"):
        actions.append("Avoid repeated use of high-impact chemicals")
        actions.append("Switch to safer bio-based alternatives where possible")

    consultation_suggestion = None
    if severity == "Severe":
        consultation_suggestion = {
            "message": "Consult nearby agricultural expert.",
            "maps_link": "https://www.google.com/maps/search/agricultural+nursery+near+me",
        }

    recommended_actions = sorted(set(actions))

    return {
        "urgency": urgency,
        "spread_warning": spread_warning,
        "possible_causes": possible_causes,
        "recommended_actions": recommended_actions,
        "chemicals_to_avoid": chemicals_info.get("chemicals_to_avoid", []),
        "safer_alternatives": chemicals_info.get("alternatives", []),
        "consultation_suggestion": consultation_suggestion,
        "summary_text": (
            f"Disease: {disease}. Severity: {severity}. Risk: {risk_level} ({risk_score:.2f}). "
            f"Urgency: {urgency}. {spread_warning}."
        ),
    }
