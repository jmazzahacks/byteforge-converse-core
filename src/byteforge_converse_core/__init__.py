"""
Business logic and service layer for ByteforgeConverse — LLM orchestration, conversation state, persistence.
"""

from .config import DatabaseConfig, LLMConfig, DEFAULT_LLM_MODEL
from .database import Database
from .chat import ChatService

__all__ = ["Database", "DatabaseConfig", "LLMConfig", "DEFAULT_LLM_MODEL", "ChatService"]
__version__ = "0.6.0"
