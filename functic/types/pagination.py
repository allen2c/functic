import typing

import pydantic

T = typing.TypeVar("T")


class Pagination(pydantic.BaseModel, typing.Generic[T]):
    first_id: typing.Optional[typing.Text] = pydantic.Field(default=None)
    last_id: typing.Optional[typing.Text] = pydantic.Field(default=None)
    has_more: bool = pydantic.Field(default=False)
    data: typing.List[T] = pydantic.Field(default_factory=lambda: list())
