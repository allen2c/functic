import json
from textwrap import dedent
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Text, Type

from json_repair import repair_json
from openai.types.beta.function_tool import FunctionTool
from openai.types.beta.function_tool_param import FunctionToolParam
from openai.types.beta.threads import run_submit_tool_outputs_params
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared.function_definition import FunctionDefinition
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from functic.types.chat_completion_tool import ChatCompletionTool
    from functic.types.chat_completion_tool_message import ChatCompletionToolMessage
    from functic.types.tool_output import ToolOutput


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
        if not self.is_config_valid(self):
            raise ValueError(f"Invalid configuration: {self}")


class FuncticParser:
    @classmethod
    def parse_content(cls, response: Any) -> Text:
        return str(response)

    @classmethod
    def parse_function_return_as_openai_tool_message(
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
    def parse_function_return_as_openai_tool_message_param(
        cls, response: Any, *, tool_call_id: Text
    ) -> ChatCompletionToolMessageParam:
        return cls.parse_function_return_as_openai_tool_message(
            response, tool_call_id=tool_call_id
        ).model_dump(
            exclude_none=True
        )  # type: ignore

    @classmethod
    def parse_function_return_as_assistant_tool_output(
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
    def parse_function_return_as_assistant_tool_output_param(
        cls, response: Any, *, tool_call_id: Text
    ) -> "run_submit_tool_outputs_params.ToolOutput":
        return cls.parse_function_return_as_assistant_tool_output(
            response, tool_call_id=tool_call_id
        ).model_dump(
            exclude_none=True
        )  # type: ignore


class FuncticBaseModel(BaseModel, FuncticParser):
    # Function arguments
    # <function_arguments>

    # Class variables for overrides
    config: ClassVar[FuncticConfig]

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

    @classmethod
    def from_args_str(cls, args_str: Text):
        func_kwargs = (
            json.loads(repair_json(args_str)) if args_str else {}  # type: ignore
        )
        return cls.model_validate(func_kwargs)

    @classmethod
    def is_base_model_valid(cls, config: Optional[FuncticConfig] = None) -> bool:
        if config is not None:
            return config.is_valid()
        if hasattr(cls, "config"):
            return cls.config.is_valid()
        else:
            raise ValueError(
                "No configuration provided and no default configuration found."
            )
