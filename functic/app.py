import contextlib
import typing

import fastapi
import pydantic
import pydantic_settings
from loguru import logger


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI) -> typing.AsyncIterator[None]:
    yield


def set_app_config(app: fastapi.FastAPI) -> fastapi.FastAPI:
    import pymongo

    app_settings = AppSettings()
    app.state.settings = app_settings
    app.state.logger = logger
    app.state.db = pymongo.MongoClient(
        app_settings.FUNCTIC_DATABASE_CONNECTION_STRING.get_secret_value(),
    )

    return app


def create_app() -> fastapi.FastAPI:
    logger.debug("Creating application")
    app = fastapi.FastAPI(lifespan=lifespan)

    # Set app states
    set_app_config(app)

    # Add routes
    # TODO:

    return app


class AppSettings(pydantic_settings.BaseSettings):
    FUNCTIC_DATABASE_CONNECTION_STRING: pydantic.SecretStr = pydantic.Field(
        default=pydantic.SecretStr("mongodb://localhost:27017/"),
        description="The connection string to the Functic database",
    )
    FUNCTIC_DATABASE_NAME: str = pydantic.Field(
        default="functic",
        description="The name of the Functic database",
    )
    FUNCTIC_FUNCTIONS_REPOSITORY_TABLE_NAME: str = pydantic.Field(
        default="functions",
        description="The name of the Functic functions repository table",
    )


app = create_app()
