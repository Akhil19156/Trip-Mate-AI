import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Any

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

mcp = FastMCP("Weather MCP Server")

OPENWEATHER_API_KEY = os.getenv(
    "OPENWEATHER_API_KEY"
)

REQUEST_TIMEOUT_SECONDS = 20


def _get_api_key() -> str:
    if not OPENWEATHER_API_KEY:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is missing "
            "from the project .env file."
        )

    return OPENWEATHER_API_KEY


def _request_json(
    url: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as exc:
        details = ""

        failed_response = getattr(
            exc,
            "response",
            None,
        )

        if failed_response is not None:
            details = (
                f" Response: "
                f"{failed_response.text[:500]}"
            )

        raise RuntimeError(
            f"OpenWeather request failed: "
            f"{exc}.{details}"
        ) from exc


@mcp.tool()
def get_current_weather(
    city: str,
) -> dict[str, Any]:
    """Return the current weather for a city."""

    city = city.strip()

    if not city:
        raise ValueError(
            "city cannot be empty"
        )
    data = _request_json(
        "https://api.openweathermap.org/data/2.5/weather",
        {
            "q": city,
            "appid": _get_api_key(),
            "units": "metric",
        },
    )

    return {
        "city": data["name"],
        "temperature_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
    }


@mcp.tool()
def get_forecast(
    city: str,
    days: Annotated[int, Field(ge=1, le=5)] = 5,
) -> dict[str, Any]:
    """
    Return one summary for each requested forecast date (up to five).
    """

    city = city.strip()

    if not city:
        raise ValueError(
            "city cannot be empty"
        )
    if days < 1 or days > 5:
        raise ValueError("days must be between 1 and 5")

    data = _request_json(
        "https://api.openweathermap.org/data/2.5/forecast",
        {
            "q": city,
            "appid": _get_api_key(),
            "units": "metric",
        },
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in data.get("list", []):
        forecast_date = item.get("dt_txt", "")[:10]
        if forecast_date:
            grouped[forecast_date].append(item)

    forecast = []
    for forecast_date in sorted(grouped)[:days]:
        entries = grouped[forecast_date]
        conditions = Counter(
            item.get("weather", [{}])[0].get("description", "unknown")
            for item in entries
        )
        temperatures = [
            item.get("main", {}).get("temp")
            for item in entries
            if isinstance(item.get("main", {}).get("temp"), int | float)
        ]
        if not temperatures:
            continue
        forecast.append(
            {
                "date": forecast_date,
                "min_temp_c": min(temperatures),
                "max_temp_c": max(temperatures),
                "dominant_condition": conditions.most_common(1)[0][0],
            }
        )

    return {
        "city": data.get(
            "city",
            {},
        ).get(
            "name",
            city,
        ),
        "forecast": forecast,
    }


if __name__ == "__main__":
    # mcp_client.py launches this as a stdio subprocess.
    mcp.run(
        transport="stdio",
    )