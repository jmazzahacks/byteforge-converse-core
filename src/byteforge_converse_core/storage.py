"""Public persistence API. Importing this module never imports an LLM client."""

from .config import DatabaseConfig
from .database import (
    AppendResult,
    ConcurrentMessageChange,
    Database,
    IdempotencyConflict,
    OwnerRepository,
    Repository,
    ResourceNotFound,
)
from .schema import apply_schema, get_schema_sql

__all__ = [
    "Database",
    "DatabaseConfig",
    "Repository",
    "OwnerRepository",
    "ResourceNotFound",
    "AppendResult",
    "IdempotencyConflict",
    "ConcurrentMessageChange",
    "apply_schema",
    "get_schema_sql",
]
