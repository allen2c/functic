import pathlib
import typing

import diskcache
from faker import Faker
from pydantic import Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings
from rich.console import Console

console = Console()
fake = Faker()


class Settings(BaseSettings):
    # LLMs
    OPENAI_API_KEY: typing.Optional[SecretStr] = Field(default=None)

    # Cache
    LOCAL_CACHE_PATH: typing.Text = Field(
        default=str(pathlib.Path.home().joinpath(".functic", "cache"))
    )

    # Private
    _local_cache: typing.Optional[diskcache.Cache] = PrivateAttr(default=None)

    @property
    def local_cache(self) -> diskcache.Cache:
        if self._local_cache is None:
            self._local_cache = diskcache.Cache(self.LOCAL_CACHE_PATH)
        return self._local_cache


settings = Settings()
