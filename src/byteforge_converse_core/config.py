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
