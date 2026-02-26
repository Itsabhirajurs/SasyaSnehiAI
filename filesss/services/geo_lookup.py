import requests


def _normalize_shop(item):
    name = item.get("display_name", "Unknown shop").split(",")[0]
    lat = item.get("lat")
    lon = item.get("lon")
    address = item.get("display_name", "Address unavailable")

    maps_link = None
    if lat and lon:
        maps_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"

    return {
        "name": name,
        "address": address,
        "lat": lat,
        "lon": lon,
        "maps_link": maps_link,
    }


def find_nearby_agri_shops(city=None, lat=None, lon=None, limit=5):
    headers = {
        "User-Agent": "SashyasnehiAI/1.0 (agri-advisory prototype)"
    }

    try:
        params = {"format": "json", "limit": limit}

        if lat is not None and lon is not None:
            query = (
                f"agriculture shop near {lat},{lon} OR fertilizer shop near {lat},{lon} "
                f"OR pesticide shop near {lat},{lon}"
            )
            params["q"] = query
        elif city:
            params["q"] = f"agriculture fertilizer pesticide shop in {city}"
        else:
            return {
                "available": False,
                "reason": "No location provided",
                "shops": [],
            }

        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=headers,
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()

        shops = [_normalize_shop(item) for item in data[:limit]]

        if not shops:
            return {
                "available": False,
                "reason": "No nearby shops found",
                "shops": [],
            }

        return {
            "available": True,
            "reason": None,
            "shops": shops,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
            "shops": [],
        }
