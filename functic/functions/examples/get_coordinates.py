import typing

import agents
import pydantic
import requests

import functic


def get_coordinates(
    request: "GetCoordinates", context: agents.TContext | None = None, *args, **kwargs
) -> "GetCoordinatesResponse":
    """
    Get coordinates for a location using the Open-Meteo Geocoding API.
    Converts place names to latitude/longitude coordinates.
    Free API with no authentication required.

    Args:
        request: GetCoordinates request with location name and search options
        context: Optional agent context for execution

    Returns:
        GetCoordinatesResponse: Location data including coordinates and metadata

    Raises:
        ValueError: If no results found for the location
        requests.exceptions.RequestException: If API request fails
    """
    # Build API URL
    base_url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": request.name,
        "count": request.count,
        "language": request.language,
        "format": "json",
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "results" not in data or not data["results"]:
            raise ValueError(f"No results found for location: {request.name}")

        return GetCoordinatesResponse.model_validate(data)
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Failed to fetch coordinates: {e}")


class GetCoordinatesConfig(functic.FuncticConfig):
    name: typing.Text = pydantic.Field(
        "get_coordinates",
        description="The name of the function.",
        pattern=r"^[a-zA-Z0-9_-]*$",
    )
    description: typing.Text = pydantic.Field(
        "Get coordinates for a location using the Open-Meteo Geocoding API "
        "(free, no API key required).",
        description="A description of the function.",
    )
    function: typing.Text = pydantic.Field(
        "functic.functions.examples.get_coordinates:get_coordinates",
        description="The path of the callable function.",
    )


class GetCoordinates(functic.FuncticBaseModel):
    functic_config: typing.ClassVar[typing.Type[GetCoordinatesConfig]] = (
        GetCoordinatesConfig
    )

    # Args
    name: str
    count: int = 5
    language: str = "en"


class LocationResult(pydantic.BaseModel):
    """Individual location result"""

    id: int
    name: str
    latitude: float
    longitude: float
    elevation: typing.Optional[float] = None
    feature_code: typing.Optional[str] = None
    country_code: typing.Optional[str] = None
    country: typing.Optional[str] = None
    country_id: typing.Optional[int] = None
    timezone: typing.Optional[str] = None
    population: typing.Optional[int] = None
    postcodes: typing.Optional[list[str]] = None
    admin1: typing.Optional[str] = None
    admin2: typing.Optional[str] = None
    admin3: typing.Optional[str] = None
    admin4: typing.Optional[str] = None
    admin1_id: typing.Optional[int] = None
    admin2_id: typing.Optional[int] = None
    admin3_id: typing.Optional[int] = None
    admin4_id: typing.Optional[int] = None


class GetCoordinatesResponse(pydantic.BaseModel):
    """Complete geocoding response from Open-Meteo Geocoding API"""

    results: list[LocationResult]
    generationtime_ms: typing.Optional[float] = None

    def to_description(self) -> str:
        """Generate a human-readable description of the locations found"""
        if not self.results:
            return "❌ No locations found."

        lines = []
        lines.append(f"🗺️ Found {len(self.results)} location(s):")
        lines.append("")

        for i, location in enumerate(self.results, 1):
            lines.append(f"{i}. {location.name}")

            # Add country/region info
            if location.country:
                location_parts = [location.country]
                if location.admin1:
                    location_parts.insert(0, location.admin1)
                lines.append(f"   📍 {', '.join(location_parts)}")

            lines.append(
                f"   🌐 Coordinates: {location.latitude}°, {location.longitude}°"
            )

            if location.timezone:
                lines.append(f"   🕐 Timezone: {location.timezone}")

            if location.population:
                lines.append(f"   👥 Population: {location.population:,}")

            if location.elevation is not None:
                lines.append(f"   ⛰️ Elevation: {location.elevation}m")

            if i < len(self.results):  # Don't add blank line after last result
                lines.append("")

        return "\n".join(lines)


# Example usage:
if __name__ == "__main__":
    # Get coordinates for Hsinchu
    try:
        coordinates_data = get_coordinates(
            request=GetCoordinates(name="Hsinchu", count=3, language="en")
        )

        print("✅ Successfully fetched coordinates from Open-Meteo Geocoding!")
        print("=" * 60)

        # Display human-readable description
        print(coordinates_data.to_description())

        print("\n" + "=" * 60)
        print("📊 Raw API Response (for debugging):")
        print("-" * 30)
        from pprint import pprint

        pprint(coordinates_data.model_dump())

    except Exception as e:
        print(f"❌ Error: {e}")
