import os

import requests


def fetch_weather(city=None, lat=None, lon=None):
    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        return {
            "available": False,
            "reason": "OPENWEATHER_API_KEY not configured",
            "humidity": None,
            "temperature": None,
            "rain_probability": None,
            "wind_speed": None,
        }

    try:
        params = {"appid": api_key, "units": "metric"}
        if lat and lon:
            params.update({"lat": lat, "lon": lon})
        elif city:
            params.update({"q": city})
        else:
            params.update({"q": "Bengaluru"})

        current_res = requests.get(
            "https://api.openweathermap.org/data/2.5/weather", params=params, timeout=10
        )
        current_res.raise_for_status()
        current_data = current_res.json()

        forecast_params = {
            "appid": api_key,
            "units": "metric",
            "lat": current_data["coord"]["lat"],
            "lon": current_data["coord"]["lon"],
        }
        forecast_res = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params=forecast_params,
            timeout=10,
        )
        forecast_res.raise_for_status()
        forecast_data = forecast_res.json()

        rain_probabilities = [
            int(item.get("pop", 0) * 100) for item in forecast_data.get("list", [])[:8]
        ]
        rain_probability = max(rain_probabilities) if rain_probabilities else 0

        return {
            "available": True,
            "city": current_data.get("name"),
            "humidity": current_data.get("main", {}).get("humidity"),
            "temperature": current_data.get("main", {}).get("temp"),
            "rain_probability": rain_probability,
            "wind_speed": current_data.get("wind", {}).get("speed"),
            "air_quality": "N/A",
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
            "humidity": None,
            "temperature": None,
            "rain_probability": None,
            "wind_speed": None,
        }
