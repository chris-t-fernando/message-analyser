from pydantic import BaseModel
from pydantic_settings import BaseSettings
import boto3
import json


class DatabaseConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    name: str

class Settings(BaseSettings):
    environment: str = "devtest"
    database: DatabaseConfig | None = None

    class Config:
        env_nested_delimiter = "__"


def load_db_config(env: str) -> DatabaseConfig:
    """Load database configuration from AWS SSM."""
    ssm = boto3.client("ssm")
    keys = ["PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"]
    cfg: dict[str, str] = {}
    for key in keys:
        name = f"/stockapp/{env}/{key}"
        resp = ssm.get_parameter(Name=name, WithDecryption=True)
        value = resp["Parameter"]["Value"]
        try:
            cfg[key] = json.loads(value)
        except json.JSONDecodeError:
            cfg[key] = value

    return DatabaseConfig(
        host=cfg["PGHOST"],
        port=int(cfg["PGPORT"]),
        user=cfg["PGUSER"],
        password=cfg["PGPASSWORD"],
        name=cfg["PGDATABASE"],
    )


def get_settings() -> Settings:
    """Return Settings with database populated from SSM if missing."""
    settings = Settings()
    if settings.database is None:
        settings.database = load_db_config(settings.environment)
    return settings
