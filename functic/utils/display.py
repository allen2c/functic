import zoneinfo
from datetime import datetime
from typing import Text


def display_datetime_now(tz: zoneinfo.ZoneInfo = zoneinfo.ZoneInfo("UTC")) -> Text:
    now = datetime.now(tz)
    return now.strftime("%A, %B %d, %Y %I:%M %p (%Z, UTC%z)")
