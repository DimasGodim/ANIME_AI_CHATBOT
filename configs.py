from pydantic import BaseSettings, Field

class Config(BaseSettings):
    api_key_voicevox: str = Field("")
    url_database:  str = Field("")

config = Config()