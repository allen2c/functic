from typing import TYPE_CHECKING, Type

from openai.types.shared.function_definition import FunctionDefinition
from pydantic import BaseModel

if TYPE_CHECKING:
    import functic


def from_base_model(
    base_model_type: (
        Type["functic.FuncticBaseModel"]
        | "functic.FuncticBaseModel"
        | Type[BaseModel]
        | BaseModel
    ),
) -> "FunctionDefinition":
    import functic

    func_name = getattr(base_model_type, functic.FIELD_FUNCTION_NAME, None)
    if not func_name:
        raise ValueError(
            "The class variable `function_name` is not set for the base model: "
            + f"{base_model_type}"
        )
    func_description = getattr(
        base_model_type, functic.FIELD_FUNCTION_DESCRIPTION, None
    )
    model_json_schema = base_model_type.model_json_schema()
    model_json_schema.pop("title", None)
    return FunctionDefinition.model_validate(
        {
            "name": func_name,
            "description": func_description,
            "parameters": model_json_schema,
        }
    )
