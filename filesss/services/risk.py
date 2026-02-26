def compute_environmental_risk(base_confidence, weather):
    risk_score = float(base_confidence)

    humidity = weather.get("humidity")
    rain_probability = weather.get("rain_probability")
    temperature = weather.get("temperature")

    if humidity is not None and humidity > 75:
        risk_score += 0.15

    if rain_probability is not None and rain_probability > 60:
        risk_score += 0.10

    if temperature is not None and 18 <= temperature <= 28:
        risk_score += 0.10

    risk_score = min(risk_score, 1.0)

    if risk_score < 0.5:
        risk_level = "Low Risk"
    elif risk_score < 0.75:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    return {
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
    }
