import typing

import openai
from openai.types.beta.assistant import Assistant

from functic.config import console

if typing.TYPE_CHECKING:
    import diskcache
    import redis

    from functic.types.assistant_create import AssistantCreate


def ensure_assistant(
    assistant_id_or_name: typing.Text,
    client: openai.OpenAI,
    *,
    cache: typing.Optional[typing.Union["diskcache.Cache", "redis.Redis"]] = None,
    expire: typing.Annotated[int, "Cache expiration time in seconds"] = 15 * 60,
    force: typing.Annotated[bool, "Force to fetch from OpenAI"] = False,
    assistant_create: typing.Optional["AssistantCreate"] = None,
) -> "Assistant":
    from functic.utils.openai_utils.iter_ import iter_assistants

    cache_key = f"openai:assistant:{assistant_id_or_name}"
    # Get from cache
    if force is False and cache is not None:
        might_assistant = cache.get(cache_key)
        if might_assistant:
            console.print(f"Get assistant from cache: '{assistant_id_or_name}'")
            return Assistant.model_validate_json(might_assistant)  # type: ignore

    # Get by assistant id
    assistant: typing.Optional[Assistant] = None
    if assistant_id_or_name.startswith("asst_"):  # asst_<name>
        try:
            assistant = client.beta.assistants.retrieve(assistant_id_or_name)
            console.print(f"Get assistant from OpenAI: '{assistant_id_or_name}'")
        except openai.NotFoundError:
            console.print(
                f"Assistant '{assistant_id_or_name}' not found, try to search by name"
            )

    # Get by name
    if assistant is None:
        for _asst in iter_assistants(client):
            if _asst.name == assistant_id_or_name:
                assistant = _asst
                console.print(f"Get assistant from OpenAI: '{assistant_id_or_name}'")
                break

    # Not found
    if assistant is None:
        if assistant_create is None:
            raise ValueError(
                f"Assistant '{assistant_id_or_name}' not found, "
                + "or please provide `assistant_create` to create it."
            )
        assistant = assistant_create.create(client)
        console.print(f"Create assistant: '{assistant_id_or_name}'")
    else:
        if assistant_create is not None:
            assistant = assistant_create.update(assistant.id, client)
            console.print(f"Update assistant: '{assistant_id_or_name}'")

    # Cache
    if cache is not None:
        cache.set(cache_key, assistant.model_dump_json(), expire)
        console.print(f"Cache assistant: '{assistant_id_or_name}'")

    return assistant
