from typing import TYPE_CHECKING, List, Sequence, Type

from openai.types.beta.function_tool import FunctionTool
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
) -> "FunctionTool":
    import functic.utils.function_definition

    return FunctionTool.model_validate(
        {
            "function": functic.utils.function_definition.from_base_model(
                base_model_type
            ),
            "type": "function",
        }
    )


def from_base_models(
    base_model_types: Sequence[
        Type["functic.FuncticBaseModel"]
        | "functic.FuncticBaseModel"
        | Type[BaseModel]
        | BaseModel
    ],
) -> List["FunctionTool"]:
    return [from_base_model(base_model_type) for base_model_type in base_model_types]
