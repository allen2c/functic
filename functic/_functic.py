import json
import typing

import openai
from json_repair import repair_json
from pydantic import BaseModel, Field, PrivateAttr

R = typing.TypeVar("R")
P = typing.ParamSpec("P")

# Regular function type
FunctionType = typing.Callable[P, R]

# Coroutine function type
CoroutineType = typing.Callable[P, typing.Coroutine[typing.Any, typing.Any, R]]


class FuncticConfig(BaseModel):
    name: typing.Text = Field(
        ...,
        description="The name of the function.",
        pattern=r"^[a-zA-Z0-9_-]*$",
    )
    description: typing.Text = Field(
        ...,
        description="A description of the function.",
    )
    function: typing.Text = Field(
        ...,
        description="The path of the callable function.",
    )
    error_output: typing.Text = Field(
        default="The service is currently unavailable. Please try again later.",
        description="The output of the error message.",
    )


class FuncticBaseModel(BaseModel):
    # Function arguments
    # <function_arguments>

    # Class variables for overrides
    functic_config: typing.ClassVar[FuncticConfig]

    # Private attributes
    _tool_call_id: typing.Optional[typing.Text] = PrivateAttr(default=None)
    _output: str | openai.NotGiven = PrivateAttr(default=openai.NOT_GIVEN)

    @classmethod
    def from_args_str(cls, args_str: typing.Optional[typing.Text]):
        func_kwargs = (
            json.loads(repair_json(args_str)) if args_str else {}  # type: ignore
        )
        return cls.model_validate(func_kwargs)
