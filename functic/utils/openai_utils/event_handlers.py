import typing

import openai
from openai import AssistantEventHandler
from openai.types.beta.assistant_stream_event import AssistantStreamEvent
from openai.types.beta.threads import Message
from openai.types.beta.threads.run import Run
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from typing_extensions import override

from functic import FuncticBaseModel
from functic.config import console


class FuncticEventHandler(AssistantEventHandler):

    def __init__(
        self,
        client: openai.OpenAI,
        *args,
        tools_set: typing.Optional[
            typing.Iterable[typing.Type[FuncticBaseModel]]
        ] = None,
        messages: typing.Optional[typing.List[Message]] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.client = client
        self.tools_set = list(tools_set or [])
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

        for tool_call in data.required_action.submit_tool_outputs.tool_calls:

            # Pick the tool
            for func_tool in self.tools_set:
                if tool_call.function.name != func_tool.functic_config.name:
                    continue

                # Debugging
                if self.debug:
                    console.print(
                        f"Calling function: '{tool_call.function.name}' "
                        + f"with args: '{tool_call.function.arguments}'"
                    )

                # Create the Functic model
                functic_base_model = func_tool
                functic_model = functic_base_model.from_args_str(
                    tool_call.function.arguments
                )
                functic_model.set_tool_call_id(tool_call.id)

                # Execute the function
                functic_model.sync_execute()
                tool_output_param = functic_model.tool_output_param

                # Add the tool output to the list
                tool_outputs.append(tool_output_param)

                # Debugging
                if self.debug:
                    console.print(f"Tool output: {tool_output_param}")

                break

            else:
                raise ValueError(
                    f"Function name '{tool_call.function.name}' not found, "
                    + "available functions: "
                    + f"{', '.join([t.functic_config.name for t in self.tools_set])}"
                )

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
            event_handler=self.__class__(
                self.client, messages=self.messages, debug=self.debug
            ),
        ) as stream:
            stream.until_done()
