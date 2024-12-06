import typing

from pydantic import BaseModel


class ToolOutput(BaseModel):
    output: typing.Text
    tool_call_id: typing.Text
