import typing

import agents
import pydantic
import requests

import functic


def get_my_ip_info(
    request: "GetMyIpInfo", context: agents.TContext | None = None, *args, **kwargs
) -> "GetMyIpInfoResponse":
    """
    Get your public IP address and location information using ip-api.com.
    Provides detailed information about the client's IP address including
    geographical location, ISP, and network details.
    Free API with no authentication required.

    Args:
        request: GetMyIpInfo request with response format options
        context: Optional agent context for execution

    Returns:
        GetMyIpInfoResponse: IP address and associated metadata

    Raises:
        requests.exceptions.RequestException: If API request fails
        ValueError: If API returns an error status
    """
    # Build API URL
    base_url = "http://ip-api.com/json"
    params = {}

    if request.fields:
        params["fields"] = request.fields
    if request.lang:
        params["lang"] = request.lang

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Check if API returned an error
        if data.get("status") == "fail":
            raise ValueError(f"API error: {data.get('message', 'Unknown error')}")

        return GetMyIpInfoResponse.model_validate(data)
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Failed to fetch IP info: {e}")


class GetMyIpInfoConfig(functic.FuncticConfig):
    name: typing.Text = pydantic.Field(
        "get_my_ip_info",
        description="The name of the function.",
        pattern=r"^[a-zA-Z0-9_-]*$",
    )
    description: typing.Text = pydantic.Field(
        "Get your public IP address and location information using ip-api.com "
        "(free, no API key required).",
        description="A description of the function.",
    )
    function: typing.Text = pydantic.Field(
        "functic.functions.examples.get_my_ip_info:get_my_ip_info",
        description="The path of the callable function.",
    )


class GetMyIpInfo(functic.FuncticBaseModel):
    functic_config: typing.ClassVar[typing.Type[GetMyIpInfoConfig]] = GetMyIpInfoConfig

    # Args
    fields: typing.Optional[str] = None  # Comma-separated list of fields to return
    lang: typing.Optional[str] = None  # Language for country/region names


class GetMyIpInfoResponse(pydantic.BaseModel):
    """Complete IP information response from ip-api.com"""

    status: str
    country: typing.Optional[str] = None
    countryCode: typing.Optional[str] = None
    region: typing.Optional[str] = None
    regionName: typing.Optional[str] = None
    city: typing.Optional[str] = None
    zip: typing.Optional[str] = None
    lat: typing.Optional[float] = None
    lon: typing.Optional[float] = None
    timezone: typing.Optional[str] = None
    isp: typing.Optional[str] = None
    org: typing.Optional[str] = None
    as_: typing.Optional[str] = pydantic.Field(default=None, alias="as")
    query: typing.Optional[str] = None  # This is the IP address
    mobile: typing.Optional[bool] = None
    proxy: typing.Optional[bool] = None
    hosting: typing.Optional[bool] = None
    message: typing.Optional[str] = None  # Error message if status is "fail"

    def to_description(self) -> str:
        """Generate a human-readable description of the IP information"""
        if self.status == "fail":
            return f"❌ Failed to get IP information: {self.message}"

        lines = []
        lines.append("🌐 Your Public IP Information")
        lines.append("=" * 35)

        # Basic IP info
        if self.query:
            lines.append(f"📡 IP Address: {self.query}")

        # Location information
        if any([self.city, self.regionName, self.country]):
            location_parts = []
            if self.city:
                location_parts.append(self.city)
            if self.regionName:
                location_parts.append(self.regionName)
            if self.country:
                location_parts.append(self.country)
            lines.append(f"📍 Location: {', '.join(location_parts)}")

        if self.countryCode:
            lines.append(f"🏳️ Country Code: {self.countryCode}")

        if self.zip:
            lines.append(f"📮 Postal Code: {self.zip}")

        # Coordinates
        if self.lat is not None and self.lon is not None:
            lines.append(f"🗺️ Coordinates: {self.lat}°, {self.lon}°")

        if self.timezone:
            lines.append(f"🕐 Timezone: {self.timezone}")

        lines.append("")

        # Network information
        lines.append("🔌 Network Information")
        lines.append("-" * 25)

        if self.isp:
            lines.append(f"🌐 ISP: {self.isp}")

        if self.org:
            lines.append(f"🏢 Organization: {self.org}")

        if self.as_:
            lines.append(f"🔗 AS Number: {self.as_}")

        # Connection type indicators
        indicators = []
        if self.mobile:
            indicators.append("📱 Mobile")
        if self.proxy:
            indicators.append("🛡️ Proxy")
        if self.hosting:
            indicators.append("☁️ Hosting/VPS")

        if indicators:
            lines.append(f"🏷️ Connection Type: {', '.join(indicators)}")

        return "\n".join(lines)


# Example usage:
if __name__ == "__main__":
    # Get your IP information
    try:
        ip_data = get_my_ip_info(request=GetMyIpInfo())

        print("✅ Successfully fetched IP information from ip-api.com!")
        print("=" * 55)

        # Display human-readable description
        print(ip_data.to_description())

        print("\n" + "=" * 55)
        print("📊 Raw API Response (for debugging):")
        print("-" * 30)
        from pprint import pprint

        pprint(ip_data.model_dump())

    except Exception as e:
        print(f"❌ Error: {e}")
