from typing import TYPE_CHECKING, Type

from openai.types.shared.function_definition import FunctionDefinition

if TYPE_CHECKING:
    import functic


def from_base_model(
    base_model_type: Type["functic.FuncticBaseModel"] | "functic.FuncticBaseModel",
) -> "FunctionDefinition":
    if not base_model_type.is_base_model_valid():
        raise ValueError(f"The base model is invalid: {base_model_type}")

    model_json_schema = base_model_type.model_json_schema()
    model_json_schema.pop("title", None)
    return FunctionDefinition.model_validate(
        {
            "name": base_model_type.functic_config.name,
            "description": base_model_type.functic_config.description,
            "parameters": model_json_schema,
        }
    )
