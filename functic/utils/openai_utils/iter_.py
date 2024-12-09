import typing

import openai
from openai.types.beta.assistant import Assistant


def iter_assistants(client: openai.OpenAI) -> typing.Generator[Assistant, None, None]:
    after: typing.Text | openai.NotGiven = openai.NOT_GIVEN
    has_more: bool = True
    limit: int = 20
    for _ in range(1000):  # Hard limit 1000 pages
        if not has_more:
            break
        page_assistants = client.beta.assistants.list(after=after, limit=limit)
        if len(page_assistants.data) == 0:
            break
        for assistant in page_assistants.data:
            yield assistant
        after = getattr(page_assistants, "last_id", page_assistants.data[-1].id)
        has_more = getattr(
            page_assistants, "has_more", len(page_assistants.data) >= limit
        )
