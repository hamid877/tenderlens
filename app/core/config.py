"""Core configuration for TenderLens, loaded from environment variables."""

import os


class Settings:
    """Application settings read from environment variables."""

    app_name: str = os.getenv("APP_NAME", "TenderLens")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # Milestone 9: Hosted LLM Provider
    # openai_api_key is intentionally not assigned a default so that missing
    # configuration is caught at provider construction time rather than here.
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


settings = Settings()
