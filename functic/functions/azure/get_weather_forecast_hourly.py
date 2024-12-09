import os
from datetime import datetime
from textwrap import dedent
from typing import Any, ClassVar, Dict, Text

import requests
from pydantic import Field, SecretStr

from functic import FuncticBaseModel, FuncticConfig
from functic.config import console

AZURE_MAPS_KEY: SecretStr = SecretStr(os.environ["AZURE_MAPS_KEY"])
AZURE_WEATHER_FORECAST_HOURLY_URL = (
    "https://atlas.microsoft.com/weather/forecast/hourly/json"
)
AZURE_WEATHER_FORECAST_HOURLY_API_VERSION = "1.1"


get_weather_forecast_hourly_config = FuncticConfig.model_validate(
    {
        "name": "get_weather_forecast_hourly",
        "description": dedent(
            """
            This function retrieves an hourly weather forecast from the Azure Maps Weather API.
            It accepts parameters such as location query, duration, and language to customize the forecast data.
            The function returns detailed weather information including temperature, humidity, wind, precipitation, and UV index for the specified duration.
            """  # noqa: E501
        ).strip(),
        "function": "functic.functions.azure.get_weather_forecast_hourly:get_weather_forecast_hourly",  # noqa: E501
    }
)


class GetWeatherForecastHourly(FuncticBaseModel):
    # Function config
    functic_config: ClassVar[FuncticConfig] = get_weather_forecast_hourly_config

    # Function arguments
    query: Text = Field(
        ...,
        description=dedent(
            """
            The location query specified as a comma-separated string composed of latitude followed by longitude, e.g., "47.641268,-122.125679".
            This parameter is required to identify the geographical location for which the weather forecast is requested.
            """  # noqa: E501
        ).strip(),
    )
    duration: int = Field(
        default=1,
        description=dedent(
            """
            The time frame for the returned weather forecast. Available values are:
            1 - Return forecast data for the next hour (default).
            12 - Return hourly forecast for the next 12 hours.
            24 - Return hourly forecast for the next 24 hours.
            72 - Return hourly forecast for the next 72 hours (3 days).
            """  # noqa: E501
        ).strip(),
    )
    language: Text = Field(
        default="en",
        description=dedent(
            """
            The language in which the search results should be returned. Should be one of the supported IETF language tags, case insensitive.
            When data in the specified language is not available for a specific field, the default language is used.
            """  # noqa: E501
        ).strip(),
    )

    @classmethod
    def parse_content(cls, response: Dict[Text, Any]) -> Text:
        if not response:
            console.print("Got empty function response content")
            return cls.functic_config.error_content
        if not isinstance(response, Dict):
            console.print(f"Invalid response type: {type(response)}")
            return cls.functic_config.error_content
        try:
            return format_weather_article(response)
        except Exception:
            console.print_exception()
            return cls.functic_config.error_content


def get_weather_forecast_hourly(request: "GetWeatherForecastHourly"):
    url = AZURE_WEATHER_FORECAST_HOURLY_URL
    params = request.model_dump(exclude_none=True)
    params["api-version"] = AZURE_WEATHER_FORECAST_HOURLY_API_VERSION
    params["subscription-key"] = AZURE_MAPS_KEY.get_secret_value()
    response = requests.get(url, params=params)
    return response.json()


def format_weather_article(weather_data: Dict[Text, Any]) -> Text:
    article_parts = []

    # Process each forecast
    for forecast in weather_data.get("forecasts", []):
        try:
            # Parse date
            date_str = forecast.get("date")
            if date_str:
                date = datetime.fromisoformat(date_str)
                date_formatted = date.strftime("%A, %B %d, %Y %I:%M %p")
                article_parts.append(f"\n=== {date_formatted} ===")

            # Weather conditions
            icon_phrase = forecast.get("iconPhrase", "Unknown")
            article_parts.append(f"Conditions: {icon_phrase}")

            # Temperature and Real Feel
            temp = forecast.get("temperature", {}).get("value")
            real_feel = forecast.get("realFeelTemperature", {}).get("value")
            if temp is not None:
                article_parts.append(f"Temperature: {temp}°C")
            if real_feel is not None:
                article_parts.append(f"Feels like: {real_feel}°C")

            # Wind details
            wind = forecast.get("wind", {})
            direction = wind.get("direction", {}).get("localizedDescription", "Unknown")
            speed = wind.get("speed", {}).get("value")
            if speed is not None:
                article_parts.append(f"Wind: {direction} at {speed} km/h")

            # Precipitation details
            has_precipitation = forecast.get("hasPrecipitation", False)
            if has_precipitation:
                precip_type = forecast.get("precipitationType", "Unknown")
                precip_intensity = forecast.get("precipitationIntensity", "Unknown")
                article_parts.append(
                    f"Precipitation: {precip_type} ({precip_intensity})"
                )

            # Additional details
            humidity = forecast.get("relativeHumidity")
            if humidity is not None:
                article_parts.append(f"Humidity: {humidity}%")

            visibility = forecast.get("visibility", {}).get("value")
            if visibility is not None:
                article_parts.append(f"Visibility: {visibility} km")

            cloud_cover = forecast.get("cloudCover")
            if cloud_cover is not None:
                article_parts.append(f"Cloud Cover: {cloud_cover}%")

        except Exception:
            console.print_exception()
            continue  # Skip problematic forecasts

    return "\n".join(article_parts).strip()
