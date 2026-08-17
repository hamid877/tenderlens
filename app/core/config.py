"""Core configuration for TenderLens, loaded from environment variables."""

import os


class Settings:
    """Application settings read from environment variables."""

    app_name: str = os.getenv("APP_NAME", "TenderLens")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()


settings = Settings()
