from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central place for all application configuration.
    Values are loaded automatically from the .env file.
    """

    # Groq LLM settings
    # OpenAI LLM settings
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"  # default if not set in .env

    # Maximum clarifying questions limit before forcing strategy generation
    max_questions: int = 6

    # Flag to enable 1-call-per-turn optimization (merged answer processing + next question planning)
    merged_turn_call_enabled: bool = True

    # Flag to enable extra requirement proposal during relevance analysis (M2.1)
    extra_requirement_enabled: bool = True

    # Postgres Database settings
    database_url: str = "postgresql://postgres:postgres@localhost:5432/marketing_agent_db"

    # Apify settings
    apify_api_token: Optional[str] = None

    @property
    def apify_api_key(self) -> Optional[str]:
        return self.apify_api_token

    # Firecrawl settings
    firecrawl_api_key: Optional[str] = None

    # This tells Pydantic Settings where to find the .env file
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# One shared instance, used throughout the app
settings = Settings()