"""
Chat-turn orchestration for ByteforgeConverse.

Ties the Postgres-backed conversation history to the OpenRouter LLM proxy:
persist the user turn, replay history (with the conversation's system prompt),
call the model, persist the assistant turn, and return it. State lives in the
database, not in this service — each turn is stateless.
"""

import time
import logging
from typing import Optional

from openrouter_client import OpenRouterClient

from byteforge_converse_models import Conversation, Message

from .config import LLMConfig
from .database import Database

logger = logging.getLogger(__name__)


class ChatService:
    """Single-turn LLM orchestration over a persisted conversation."""

    def __init__(self, db: Database, config: LLMConfig, client: Optional[OpenRouterClient] = None) -> None:
        self._db = db
        self._config = config
        self._client = client if client is not None else OpenRouterClient(api_key=config.api_key)

    def send_turn(self, conversation_id: str, user_content: str) -> Message:
        """
        Append the user's message, call the LLM with full history, and persist
        and return the assistant reply.

        Raises ValueError if the conversation does not exist.
        """
        conversation = self._db.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation not found: {conversation_id}")

        # Persist the user turn first so it is part of the history we send.
        self._db.create_message(conversation_id, "user", user_content)

        history = self._db.list_messages(conversation_id)
        messages = self._build_messages(conversation, history)
        model = conversation.model or self._config.default_model

        logger.info("LLM turn: conversation=%s model=%s history=%d", conversation_id, model, len(messages))
        response = self._client.chat.create(model=model, messages=messages)

        reply = response.choices[0].message.content or ""
        token_count = response.usage.completion_tokens if response.usage else None

        assistant_message = self._db.create_message(
            conversation_id, "assistant", reply, token_count=token_count
        )
        self._db.touch_conversation(conversation_id, int(time.time()))
        return assistant_message

    def _build_messages(self, conversation: Conversation, history: list[Message]) -> list[dict]:
        messages: list[dict] = []
        if conversation.system_prompt:
            messages.append({"role": "system", "content": conversation.system_prompt})
        for message in history:
            messages.append({"role": message.role, "content": message.content})
        return messages
