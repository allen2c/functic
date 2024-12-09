import os
import typing
import zoneinfo
from datetime import datetime

import rich.style
import rich.text

from functic.config import console

if typing.TYPE_CHECKING:
    from openai.types.beta.threads import Message


def display_datetime_now(
    tz: zoneinfo.ZoneInfo = zoneinfo.ZoneInfo("UTC"),
) -> typing.Text:
    now = datetime.now(tz)
    return now.strftime("%A, %B %d, %Y %I:%M %p (%Z, UTC%z)")


def display_thread_message(
    message: "Message",
    *,
    tz: zoneinfo.ZoneInfo = zoneinfo.ZoneInfo(os.getenv("TZ", "UTC")),
    is_print: bool = True,
) -> rich.text.Text:
    output = rich.text.Text("")
    output += rich.text.Text(
        datetime.fromtimestamp(
            message.completed_at or message.created_at, tz
        ).isoformat(),
        style="bright_green",
    )
    output += rich.text.Text(" ", style="default")
    output += rich.text.Text(message.id, style="bright_blue")
    output += rich.text.Text(" ", style="default")
    output += rich.text.Text(f"{message.role:9}:", style="bright_magenta")
    output += rich.text.Text(" ", style="default")
    output += rich.text.Text(
        "".join(
            (
                content_block.text.value
                if content_block.type == "text"
                else str(content_block)
            )
            for content_block in message.content
        ).strip(),
        style="default",
    )

    if is_print:
        console.print(output)

    return output
