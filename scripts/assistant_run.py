import typing
from textwrap import dedent

import openai
import rich.box
import rich.json
import rich.panel
from openai import AssistantEventHandler
from openai.types.beta.assistant_stream_event import AssistantStreamEvent
from openai.types.beta.threads import Message
from openai.types.beta.threads.run import Run
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from typing_extensions import override

import functic.utils.openai_utils.ensure as ENSURE
from functic.config import console, settings
from functic.functions.azure.get_weather_forecast_daily import GetWeatherForecastDaily
from functic.functions.azure.get_weather_forecast_hourly import GetWeatherForecastHourly
from functic.functions.google.get_maps_geocode import GetMapsGeocode
from functic.types.assistant_create import AssistantCreate
from functic.utils.display import display_thread_message

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
FUNCTION_TOOLS = {
    tool.config.name: tool
    for tool in (GetWeatherForecastDaily, GetWeatherForecastHourly, GetMapsGeocode)
}
ASSISTANT_TOOLS = [tool.function_tool for tool in FUNCTION_TOOLS.values()]
FORCE = False
DEBUG = True


class EventHandler(AssistantEventHandler):

    def __init__(
        self,
        client: openai.OpenAI,
        *args,
        messages: typing.Optional[typing.List[Message]] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.client = client
        self.messages = messages or []
        self.debug = debug

    @override
    def on_event(self, event: "AssistantStreamEvent"):
        # Retrieve events that are denoted with 'requires_action'
        # since these will have our tool_calls
        if event.event == "thread.run.requires_action":
            run_id = event.data.id  # Retrieve the run ID from the event data
            self.handle_requires_action(event.data, run_id)

    @override
    def on_message_done(self, message: Message) -> None:
        self.messages.append(message)

    def handle_requires_action(self, data: "Run", run_id: typing.Text) -> None:
        if data.required_action is None:
            return

        tool_outputs: typing.List[ToolOutput] = []

        for tool in data.required_action.submit_tool_outputs.tool_calls:
            if tool.function.name not in FUNCTION_TOOLS:
                raise ValueError(
                    f"Function name '{tool.function.name}' not found, "
                    + f"available functions: {list(FUNCTION_TOOLS.keys())}"
                )

            if self.debug:
                console.print(
                    f"Calling function: '{tool.function.name}' "
                    + f"with args: '{tool.function.arguments}'"
                )
            functic_base_model = FUNCTION_TOOLS[tool.function.name]
            functic_model = functic_base_model.from_args_str(tool.function.arguments)
            functic_model.set_tool_call_id(tool.id)

            # Execute the function
            functic_model.sync_execute()

            if self.debug:
                console.print(f"Tool output: {functic_model.tool_output_param}")
            tool_outputs.append(functic_model.tool_output_param)

        # Submit all tool_outputs at the same time
        self.submit_tool_outputs(tool_outputs, run_id)

    def submit_tool_outputs(
        self,
        tool_outputs: typing.Iterable[ToolOutput],
        run_id: typing.Text,
    ) -> None:
        if self.current_run is None:
            return

        # Use the submit_tool_outputs_stream helper
        with self.client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(
                self.client, messages=self.messages, debug=self.debug
            ),
        ) as stream:
            stream.until_done()


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
            tools=ASSISTANT_TOOLS,
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

    thread = client.beta.threads.create()
    thread_messages: typing.List[Message] = []
    thread_messages.append(
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="How is the weather in Tokyo day after tomorrow?",
        )
    )

    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        event_handler=EventHandler(client, messages=thread_messages, debug=DEBUG),
    ) as stream:
        stream.until_done()

    for message in client.beta.threads.messages.list(thread_id=thread.id, order="asc"):
        display_thread_message(message)


if __name__ == "__main__":
    main()
