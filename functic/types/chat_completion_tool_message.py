import typing

from pydantic import BaseModel, Field


class ChatCompletionToolMessage(BaseModel):
    content: typing.Text
    role: typing.Literal["tool"] = Field(default="tool")
    tool_call_id: typing.Text
