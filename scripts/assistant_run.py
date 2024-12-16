import time
import typing
from textwrap import dedent

import openai
import rich.box
import rich.json
import rich.panel
from openai.types.beta.threads import Message

import functic.utils.openai_utils.ensure as ENSURE
from functic.config import console, settings
from functic.functions.assorted.currencies import GetCurrencies
from functic.functions.azure.get_weather_forecast_daily import GetWeatherForecastDaily
from functic.functions.azure.get_weather_forecast_hourly import GetWeatherForecastHourly
from functic.functions.google.get_maps_geocode import GetMapsGeocode
from functic.types.assistant_create import AssistantCreate
from functic.utils.display import display_thread_message
from functic.utils.openai_utils.event_handlers import FuncticEventHandler

ASSISTANT_NAME = "asst_functic"
ASSISTANT_INSTRUCTIONS = dedent(
    """
    You are a taciturn and reticent AI assistant capable of using tools and responding in short, concise plain text. Your responses should be brief and to the point.

    Before responding, analyze the user's input and context in <analysis> tags. Plan your response, considering the most concise way to address the user's request or question.

    After your analysis, provide your final answer in plain text without any formatting, code blocks, lists, or markdown. Keep your response as short as possible while still addressing the user's input.

    Additional guidelines:
    - Do not use any formatting or special characters in your response.
    - If asked for code, lists, or other non-plain text content, politely decline and offer a brief plain text alternative if possible.
    - Use tools only if absolutely necessary to answer the user's query.
    - Always maintain a reserved and succinct tone in your responses.
    """  # noqa: E501
).strip()
ASSISTANT_MODEL = "gpt-4o-mini"
FUNCTION_TOOLS = (
    GetWeatherForecastDaily,
    GetWeatherForecastHourly,
    GetMapsGeocode,
    GetCurrencies,
)
FORCE = False
DEBUG = True


def main():
    client = openai.OpenAI()

    assistant = ENSURE.ensure_assistant(
        ASSISTANT_NAME,
        client,
        cache=settings.local_cache,
        assistant_create=AssistantCreate(
            name=ASSISTANT_NAME,
            instructions=ASSISTANT_INSTRUCTIONS,
            model=ASSISTANT_MODEL,
        ),
        force=FORCE,
    )
    if DEBUG:
        console.print(
            rich.panel.Panel(
                rich.json.JSON(assistant.model_dump_json()),
                box=rich.box.SQUARE,
                title=f"Assistant: {ASSISTANT_NAME} ({assistant.id})",
            )
        )

    # Create a thread
    ts = time.perf_counter()
    thread = client.beta.threads.create()
    thread_messages: typing.List[Message] = []
    thread_messages.append(
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=(
                "How is the weather in Tokyo day after tomorrow? "
                + "And how is a USD to EUR conversion?"
            ),
        )
    )

    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        tools=[tool.function_tool_param for tool in FUNCTION_TOOLS],
        event_handler=FuncticEventHandler(
            client, tools_set=FUNCTION_TOOLS, messages=thread_messages, debug=DEBUG
        ),
    ) as stream:
        stream.until_done()
        for message in stream.messages:
            display_thread_message(message)

    console.print(f"Time taken: {time.perf_counter() - ts:.2f} seconds")


if __name__ == "__main__":
    main()
