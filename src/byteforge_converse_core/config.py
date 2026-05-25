"""
Database connection configuration for ByteforgeConverse.
"""

import os
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """
    Connection settings for the ByteforgeConverse Postgres database.

    Read from the `BYTEFORGE_CONVERSE_DB_*` environment variables via
    `from_env()`. `port` is an int; everything else is a string.
    """
    host: str
    port: int
    name: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        password = os.environ.get("BYTEFORGE_CONVERSE_DB_PASSWORD")
        if not password:
            raise ValueError(
                "BYTEFORGE_CONVERSE_DB_PASSWORD environment variable is required"
            )

        return cls(
            host=os.environ.get("BYTEFORGE_CONVERSE_DB_HOST", "localhost"),
            port=int(os.environ.get("BYTEFORGE_CONVERSE_DB_PORT", "5432")),
            name=os.environ.get("BYTEFORGE_CONVERSE_DB_NAME", "byteforge_converse"),
            user=os.environ.get("BYTEFORGE_CONVERSE_DB_USER", "byteforge_converse"),
            password=password,
        )


DEFAULT_LLM_MODEL = "anthropic/claude-3.5-sonnet"


@dataclass
class LLMConfig:
    """
    Settings for the OpenRouter-backed LLM proxy.

    Read from `BYTEFORGE_CONVERSE_*` environment variables via `from_env()`.
    `default_model` is used for any conversation that does not pin its own.
    """
    api_key: str
    default_model: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.environ.get("BYTEFORGE_CONVERSE_OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "BYTEFORGE_CONVERSE_OPENROUTER_API_KEY environment variable is required"
            )

        return cls(
            api_key=api_key,
            default_model=os.environ.get("BYTEFORGE_CONVERSE_LLM_MODEL", DEFAULT_LLM_MODEL),
        )
