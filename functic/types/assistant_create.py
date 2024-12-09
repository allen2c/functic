import json
import typing

import openai
from openai.types.beta.function_tool import FunctionTool
from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder


class AssistantCreate(BaseModel):
    name: typing.Text = Field(...)
    instructions: typing.Text = Field(...)
    model: openai.types.chat_model.ChatModel | typing.Text = Field(...)
    metadata: typing.Optional[typing.Dict] = Field(default_factory=dict)
    tools: typing.List[FunctionTool] = Field(default_factory=list)

    def create(self, client: openai.OpenAI):
        assistant = client.beta.assistants.create(
            name=self.name,
            instructions=self.instructions,
            model=self.model,
            metadata=self.metadata,
            tools=json.loads(json.dumps(self.tools, default=pydantic_encoder)),
        )
        return assistant

    def update(self, assistant_id: typing.Text, client: openai.OpenAI):
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            name=self.name,
            instructions=self.instructions,
            model=self.model,
            metadata=self.metadata,
            tools=json.loads(json.dumps(self.tools, default=pydantic_encoder)),
        )
        return assistant
