import json
import random
import string
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Coroutine,
    Optional,
    ParamSpec,
    Text,
    Type,
    TypeVar,
)

import openai
from json_repair import repair_json
from openai.types.beta.function_tool import FunctionTool
from openai.types.beta.function_tool_param import FunctionToolParam
from openai.types.beta.threads import run_submit_tool_outputs_params
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared.function_definition import FunctionDefinition
from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from functic.types.chat_completion_tool import ChatCompletionTool
    from functic.types.chat_completion_tool_message import ChatCompletionToolMessage
    from functic.types.tool_output import ToolOutput


R = TypeVar("R")
P = ParamSpec("P")


FunctionType = Callable[P, R]  # Regular function type
CoroutineType = Callable[P, Coroutine[Any, Any, R]]  # Coroutine function type


class FuncticFunctionDefinitionDescriptor:
    def __get__(
        self, instance: None, owner: Type["FuncticBaseModel"]
    ) -> "FunctionDefinition":
        if instance is not None:
            raise AttributeError(
                f"Class property `{self.__class__.__name__}.function_definition` "
                + "cannot be accessed via an instance."
            )
        import functic.utils.function_definition

        return functic.utils.function_definition.from_base_model(owner)


class FuncticChatCompletionToolDescriptor:
    def __get__(
        self, instance: None, owner: Type["FuncticBaseModel"]
    ) -> "ChatCompletionTool":
        if instance is not None:
            raise AttributeError(
                f"Class property `{self.__class__.__name__}.chat_completion_tool` "
                + "cannot be accessed via an instance."
            )
        from functic.types.chat_completion_tool import ChatCompletionTool

        return ChatCompletionTool.model_validate(
            {"function": owner.function_definition}
        )


class FuncticChatCompletionToolParamDescriptor:
    def __get__(
        self, instance: None, owner: Type["FuncticBaseModel"]
    ) -> "ChatCompletionToolParam":
        if instance is not None:
            raise AttributeError(
                "Class property "
                + f"`{self.__class__.__name__}.chat_completion_tool_param` "
                + "cannot be accessed via an instance."
            )
        return owner.chat_completion_tool.model_dump(exclude_none=True)  # type: ignore


class FuncticFunctionToolDescriptor:
    def __get__(
        self, instance: None, owner: Type["FuncticBaseModel"]
    ) -> "FunctionTool":
        if instance is not None:
            raise AttributeError(
                f"Class property `{self.__class__.__name__}.function_tool` "
                + "cannot be accessed via an instance."
            )
        import functic.utils.function_tool

        return functic.utils.function_tool.from_base_model(owner)


class FuncticFunctionToolParamDescriptor:
    def __get__(
        self, instance: None, owner: Type["FuncticBaseModel"]
    ) -> "FunctionToolParam":
        if instance is not None:
            raise AttributeError(
                "Class property "
                + f"`{self.__class__.__name__}.function_tool_param` "
                + "cannot be accessed via an instance."
            )
        return owner.function_tool.model_dump(exclude_none=True)  # type: ignore


class FuncticConfig(BaseModel):
    name: Text = Field(
        ...,
        description="The name of the function.",
        pattern=r"^[a-zA-Z0-9_-]*$",
    )
    description: Text = Field(
        ...,
        description="A description of the function.",
    )
    function: Text = Field(
        ...,
        description="The path of the callable function.",
    )
    error_content: Text = Field(
        default=dedent(
            """
            The service is currently unavailable. Please try again later.
            """
        ).strip(),
        description="The content of the error message.",
    )

    @classmethod
    def is_config_valid(cls, config: "FuncticConfig") -> bool:
        return True  # TODO: Implement validation

    def is_valid(self) -> bool:
        return self.is_config_valid(self)

    def raise_if_invalid(self) -> None:
        if not self.is_valid():
            raise ValueError(f"Invalid configuration: {self}")

    def get_function(self) -> FunctionType | CoroutineType:
        from functic.utils.import_ import import_function

        return import_function(self.function)


class FuncticParser:
    @classmethod
    def parse_content(cls, response: Any) -> Text:
        return str(response)

    @classmethod
    def parse_content_as_openai_tool_message(
        cls, response: Any, *, tool_call_id: Text
    ) -> "ChatCompletionToolMessage":
        from functic.types.chat_completion_tool_message import ChatCompletionToolMessage

        return ChatCompletionToolMessage.model_validate(
            {
                "content": cls.parse_content(response),
                "tool_call_id": tool_call_id,
            }
        )

    @classmethod
    def parse_content_as_openai_tool_message_param(
        cls, response: Any, *, tool_call_id: Text
    ) -> ChatCompletionToolMessageParam:
        return cls.parse_content_as_openai_tool_message(
            response, tool_call_id=tool_call_id
        ).model_dump(
            exclude_none=True
        )  # type: ignore

    @classmethod
    def parse_content_as_assistant_tool_output(
        cls, response: Any, *, tool_call_id: Text
    ) -> "ToolOutput":
        from functic.types.tool_output import ToolOutput

        return ToolOutput.model_validate(
            {
                "output": cls.parse_content(response),
                "tool_call_id": tool_call_id,
            }
        )

    @classmethod
    def parse_content_as_assistant_tool_output_param(
        cls, response: Any, *, tool_call_id: Text
    ) -> "run_submit_tool_outputs_params.ToolOutput":
        return cls.parse_content_as_assistant_tool_output(
            response, tool_call_id=tool_call_id
        ).model_dump(
            exclude_none=True
        )  # type: ignore


class FuncticBaseModel(BaseModel, FuncticParser):
    # Function arguments
    # <function_arguments>

    # Class variables for overrides
    functic_config: ClassVar[FuncticConfig]

    # Class variables for internal use
    function_definition: ClassVar[FuncticFunctionDefinitionDescriptor] = (
        FuncticFunctionDefinitionDescriptor()
    )
    chat_completion_tool: ClassVar[FuncticChatCompletionToolDescriptor] = (
        FuncticChatCompletionToolDescriptor()
    )
    chat_completion_tool_param: ClassVar[FuncticChatCompletionToolParamDescriptor] = (
        FuncticChatCompletionToolParamDescriptor()
    )
    function_tool: ClassVar[FuncticFunctionToolDescriptor] = (
        FuncticFunctionToolDescriptor()
    )
    function_tool_param: ClassVar[FuncticFunctionToolParamDescriptor] = (
        FuncticFunctionToolParamDescriptor()
    )

    # Private attributes
    _tool_call_id: Optional[Text] = PrivateAttr(default=None)
    _content: Optional[Any | openai.NotGiven] = PrivateAttr(default=openai.NOT_GIVEN)

    @classmethod
    def from_args_str(cls, args_str: Optional[Text]):
        func_kwargs = (
            json.loads(repair_json(args_str)) if args_str else {}  # type: ignore
        )
        return cls.model_validate(func_kwargs)

    @classmethod
    def is_base_model_valid(cls, config: Optional[FuncticConfig] = None) -> bool:
        if config is not None:
            return config.is_valid()
        if hasattr(cls, "functic_config"):
            return cls.functic_config.is_valid()
        else:
            raise ValueError(
                "No configuration provided and no default configuration found."
            )

    @property
    def content(self) -> Any:
        if self._content is openai.NOT_GIVEN:
            raise ValueError(
                "Response content is not set, please execute the function first."
            )
        return self._content

    @property
    def content_parsed(self) -> Any:
        return self.parse_content(self.content)

    @property
    def tool_message(self) -> "ChatCompletionToolMessage":
        from functic.config import console

        tool_call_id = self._tool_call_id
        if tool_call_id is None:
            console.print(
                "No tool call id found, you might want to set the tool call id "
                + "provided by the LLM API.",
                style="yellow",
            )
            tool_call_id = "tool_" + "".join(
                random.choices(string.ascii_letters + string.digits, k=12)
            )
        return self.parse_content_as_openai_tool_message(
            self.content, tool_call_id=tool_call_id
        )

    @property
    def tool_message_param(self) -> "ChatCompletionToolMessageParam":
        return self.tool_message.model_dump(exclude_none=True)  # type: ignore

    @property
    def tool_output(self) -> "ToolOutput":
        from functic.config import console

        tool_call_id = self._tool_call_id
        if tool_call_id is None:
            console.print(
                "No tool call id found, you might want to set the tool call id "
                + "provided by the LLM API.",
                style="yellow",
            )
            tool_call_id = "tool_" + "".join(
                random.choices(string.ascii_letters + string.digits, k=12)
            )

        return self.parse_content_as_assistant_tool_output(
            self.content, tool_call_id=tool_call_id
        )

    @property
    def tool_output_param(self) -> "run_submit_tool_outputs_params.ToolOutput":
        return self.tool_output.model_dump(exclude_none=True)  # type: ignore

    def set_tool_call_id(self, tool_call_id: Text) -> None:
        self._tool_call_id = tool_call_id

    def set_content(self, content: Any) -> None:
        self._content = content

    async def execute(self) -> Any:
        from functic.utils.run import run_func

        func = self.functic_config.get_function()

        func_res = await run_func(func, self)
        self.set_content(func_res)
        return func_res

    def sync_execute(self) -> Any:
        from functic.utils.run import sync_run_func

        func = self.functic_config.get_function()
        func_res = sync_run_func(func, self)
        self.set_content(func_res)
        return func_res
