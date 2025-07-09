import pydantic
import requests


def get_weather(
    latitude: float,
    longitude: float,
    timezone: str = "auto",
    include_current: bool = True,
    include_hourly: bool = False,
    include_daily: bool = True,
) -> "GetWeatherResponse":
    """
    Get weather data using the Open-Meteo API (free, no API key required).

    Args:
        latitude: Latitude coordinate (-90 to 90)
        longitude: Longitude coordinate (-180 to 180)
        timezone: Timezone (default: "auto" for automatic detection)
        include_current: Include current weather data
        include_hourly: Include hourly forecast
        include_daily: Include daily forecast

    Returns:
        Dictionary containing weather data from Open-Meteo API

    Raises:
        requests.exceptions.RequestException: If API request fails
        ValueError: If coordinates are invalid
    """
    # Validate coordinates
    if not (-90 <= latitude <= 90):
        raise ValueError(f"Latitude must be between -90 and 90, got {latitude}")
    if not (-180 <= longitude <= 180):
        raise ValueError(f"Longitude must be between -180 and 180, got {longitude}")

    # Build API URL
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": latitude, "longitude": longitude, "timezone": timezone}

    # Add weather parameters based on options
    if include_current:
        params["current"] = (
            "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m"
        )

    if include_hourly:
        params["hourly"] = "temperature_2m,precipitation_probability,wind_speed_10m"

    if include_daily:
        params["daily"] = (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max"
        )

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return GetWeatherResponse.model_validate(response.json())
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Failed to fetch weather data: {e}")


class CurrentWeatherUnits(pydantic.BaseModel):
    """Units for current weather data"""

    apparent_temperature: str
    interval: str
    relative_humidity_2m: str
    temperature_2m: str
    time: str
    wind_speed_10m: str


class CurrentWeather(pydantic.BaseModel):
    """Current weather data"""

    apparent_temperature: float
    interval: int
    relative_humidity_2m: int
    temperature_2m: float
    time: str
    wind_speed_10m: float


class DailyWeatherUnits(pydantic.BaseModel):
    """Units for daily weather data"""

    precipitation_probability_max: str
    temperature_2m_max: str
    temperature_2m_min: str
    time: str
    weather_code: str


class DailyWeather(pydantic.BaseModel):
    """Daily weather forecast data"""

    precipitation_probability_max: list[int]
    temperature_2m_max: list[float]
    temperature_2m_min: list[float]
    time: list[str]
    weather_code: list[int]


class GetWeatherResponse(pydantic.BaseModel):
    """Complete weather response from Open-Meteo API"""

    current: CurrentWeather
    current_units: CurrentWeatherUnits
    daily: DailyWeather
    daily_units: DailyWeatherUnits
    elevation: float
    generationtime_ms: float
    latitude: float
    longitude: float
    timezone: str
    timezone_abbreviation: str
    utc_offset_seconds: int

    def to_description(self) -> str:
        """Generate a human-readable weather description in English"""
        lines = []

        # Location and timezone info
        lines.append(
            f"📍 Weather for coordinates ({self.latitude}°, {self.longitude}°)"
        )
        lines.append(f"🕐 Timezone: {self.timezone} ({self.timezone_abbreviation})")
        lines.append(f"⛰️ Elevation: {self.elevation}m")
        lines.append("")

        # Current weather
        if hasattr(self, "current") and self.current:
            lines.append("🌤️ CURRENT WEATHER")
            lines.append("-" * 30)
            lines.append(f"🌡️ Temperature: {self.current.temperature_2m}°C")
            lines.append(f"🌡️ Feels like: {self.current.apparent_temperature}°C")
            lines.append(f"💧 Humidity: {self.current.relative_humidity_2m}%")
            lines.append(f"💨 Wind Speed: {self.current.wind_speed_10m} km/h")
            lines.append(f"⏰ Last Updated: {self.current.time}")
            lines.append("")

        # Daily forecast
        if hasattr(self, "daily") and self.daily:
            lines.append("📅 7-DAY FORECAST")
            lines.append("-" * 30)

            # Weather code meanings (WMO codes)
            weather_codes = {
                0: "☀️ Clear sky",
                1: "🌤️ Mainly clear",
                2: "⛅ Partly cloudy",
                3: "☁️ Overcast",
                45: "🌫️ Fog",
                48: "🌫️ Depositing rime fog",
                51: "🌦️ Light drizzle",
                53: "🌦️ Moderate drizzle",
                55: "🌦️ Dense drizzle",
                56: "🌧️❄️ Light freezing drizzle",
                57: "🌧️❄️ Dense freezing drizzle",
                61: "🌧️ Slight rain",
                63: "🌧️ Moderate rain",
                65: "🌧️ Heavy rain",
                66: "🌧️❄️ Light freezing rain",
                67: "🌧️❄️ Heavy freezing rain",
                71: "🌨️ Slight snow",
                73: "🌨️ Moderate snow",
                75: "🌨️ Heavy snow",
                77: "❄️ Snow grains",
                80: "🌦️ Slight rain showers",
                81: "🌦️ Moderate rain showers",
                82: "🌦️ Violent rain showers",
                85: "🌨️ Slight snow showers",
                86: "🌨️ Heavy snow showers",
                95: "⛈️ Thunderstorm",
                96: "⛈️🧊 Thunderstorm with hail",
                99: "⛈️🧊 Severe thunderstorm with hail",
            }

            for i, date in enumerate(self.daily.time):
                weather_code = self.daily.weather_code[i]
                weather_desc = weather_codes.get(
                    weather_code, f"🌡️ Weather code {weather_code}"
                )

                lines.append(f"{date}:")
                lines.append(f"  {weather_desc}")
                lines.append(f"  🔼 High: {self.daily.temperature_2m_max[i]}°C")
                lines.append(f"  🔽 Low: {self.daily.temperature_2m_min[i]}°C")
                lines.append(
                    f"  🌧️ Rain chance: {self.daily.precipitation_probability_max[i]}%"
                )
                if i < len(self.daily.time) - 1:  # Don't add blank line after last day
                    lines.append("")

        return "\n".join(lines)


# Example usage:
if __name__ == "__main__":
    # Get weather for Taipei
    try:
        weather_data = get_weather(
            latitude=25.03, longitude=121.57, timezone="Asia/Taipei"
        )

        print("✅ Successfully fetched weather data from Open-Meteo!")
        print("=" * 50)

        # Display human-readable weather description
        print(weather_data.to_description())

        print("\n" + "=" * 50)
        print("📊 Raw API Response (for debugging):")
        print("-" * 30)
        from pprint import pprint

        pprint(weather_data.model_dump())

    except Exception as e:
        print(f"❌ Error: {e}")
