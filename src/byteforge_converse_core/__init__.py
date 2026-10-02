"""Conversation persistence and optional, lazily imported chat orchestration."""

from typing import Any, TYPE_CHECKING

from .config import DatabaseConfig, LLMConfig, DEFAULT_LLM_MODEL
from .database import Database

if TYPE_CHECKING:
    from .chat import ChatService

__all__ = [
    "Database",
    "DatabaseConfig",
    "LLMConfig",
    "DEFAULT_LLM_MODEL",
    "ChatService",
]
__version__ = "0.9.1"


def __getattr__(name: str) -> Any:
    if name == "ChatService":
        from .chat import ChatService

        return ChatService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
