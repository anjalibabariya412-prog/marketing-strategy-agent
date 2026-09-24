from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central place for all application configuration.
    Values are loaded automatically from the .env file.
    """

    # Groq LLM settings
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"  # default if not set in .env

    # Maximum clarifying questions limit before forcing strategy generation
    max_questions: int = 6

    # This tells Pydantic Settings where to find the .env file
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# One shared instance, used throughout the app
settings = Settings()