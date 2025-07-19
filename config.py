from pydantic import BaseSettings, BaseModel

class DatabaseConfig(BaseModel):
    host: str
    port: int

class Settings(BaseSettings):
    environment: str = "devtest"
    database: DatabaseConfig | None = None

    class Config:
        env_nested_delimiter = "__"
