from decimal import Decimal
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
    openai_model: str = "gpt-6-luna"  # default if not set in .env
    llm_max_retries: int = 0  # default retries for failed LLM calls

    # Maximum clarifying questions limit before forcing strategy generation
    max_questions: int = 5

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

    # Tavily settings
    tavily_api_key: Optional[str] = None

    # LLM Pricing settings (USD per 1M tokens)
    llm_input_price_per_1m: Optional[Decimal] = Decimal("0.10")
    llm_cached_input_price_per_1m: Optional[Decimal] = Decimal("0.01")
    llm_cache_write_price_per_1m: Optional[Decimal] = Decimal("0.125")
    llm_output_price_per_1m: Optional[Decimal] = Decimal("0.50")

    # This tells Pydantic Settings where to find the .env file
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# One shared instance, used throughout the app
settings = Settings()