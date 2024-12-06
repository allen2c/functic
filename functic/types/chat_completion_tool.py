import typing

from openai.types.shared.function_definition import FunctionDefinition
from pydantic import BaseModel, Field


class ChatCompletionTool(BaseModel):
    type: typing.Literal["function"] = Field(default="function")
    function: FunctionDefinition
