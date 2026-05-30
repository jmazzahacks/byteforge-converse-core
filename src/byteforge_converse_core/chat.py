"""
Chat-turn orchestration for ByteforgeConverse.

Ties the Postgres-backed conversation history to the OpenRouter LLM proxy:
persist the user turn, replay history (with the conversation's system prompt
and any persisted tool-call requests / results), call the model, persist the
assistant turn (including any tool_calls the model emitted), and return a
ChatTurn that carries both the persisted message and the tool calls to relay.
State lives in the database, not in this service — each turn is stateless.
"""

import json
import time
import logging
from typing import Any, Optional

from openrouter_client import (
    OpenRouterClient,
    build_json_schema_response_format,
    parse_schema_response,
)

from byteforge_converse_models import ChatTurn, Conversation, Message, ToolCall

from .config import LLMConfig
from .database import Database

logger = logging.getLogger(__name__)


class ChatService:
    """Single-turn LLM orchestration over a persisted conversation."""

    def __init__(self, db: Database, config: LLMConfig, client: Optional[OpenRouterClient] = None) -> None:
        self._db = db
        self._config = config
        self._client = client if client is not None else OpenRouterClient(api_key=config.api_key)

    def send_turn(self, conversation_id: str, user_content: str) -> ChatTurn:
        """
        Append the user's message, call the LLM with full history, persist the
        assistant reply (including any tool_calls), and return a ChatTurn that
        carries both the persisted message and the tool calls to relay.

        The model may emit content, tool_calls, or both in a single turn — all
        three cases are representable in the persisted Message and the
        returned ChatTurn.

        Raises ValueError if the conversation does not exist.
        """
        conversation = self._db.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation not found: {conversation_id}")

        # Persist the user turn first so its created_at precedes the assistant
        # reply (LLM latency separates them). If the LLM call fails we delete
        # it again so a failed turn leaves no orphaned user message behind.
        user_message = self._db.create_message(conversation_id, "user", user_content)
        reply = ""
        token_count: Optional[int] = None
        raw_tool_calls: Optional[list] = None
        tool_calls: Optional[list[ToolCall]] = None
        try:
            history = self._db.list_messages(conversation_id, limit=None)
            messages = self._build_messages(conversation, history)
            model = conversation.model or self._config.default_model

            chat_params: dict = {"model": model, "messages": messages}
            if conversation.response_schema is not None:
                chat_params["response_format"] = build_json_schema_response_format(conversation.response_schema)
            if conversation.tools is not None:
                chat_params["tools"] = conversation.tools

            logger.info(
                "LLM turn: conversation=%s model=%s history=%d structured=%s tools=%d",
                conversation_id, model, len(messages),
                conversation.response_schema is not None,
                len(conversation.tools) if conversation.tools is not None else 0,
            )
            response = self._client.chat.create(**chat_params)
            response_message = response.choices[0].message
            raw_tool_calls = self._raw_tool_calls(response_message)
            tool_calls = self._build_tool_calls(raw_tool_calls) if raw_tool_calls else None
            reply = self._extract_reply(response_message.content, conversation.response_schema)
            token_count = response.usage.completion_tokens if response.usage else None
        except Exception:
            self._db.delete_message(user_message.id)
            raise

        assistant_message = self._db.create_message(
            conversation_id,
            "assistant",
            reply,
            token_count=token_count,
            tool_calls=raw_tool_calls,
        )
        self._db.touch_conversation(conversation_id, int(time.time()))
        return ChatTurn(message=assistant_message, tool_calls=tool_calls)

    def _build_messages(self, conversation: Conversation, history: list[Message]) -> list[dict]:
        """
        Build the OpenAI/OpenRouter `messages` array from persisted history.

        Assistant rows that carried tool_calls replay with the tool_calls
        attached; tool rows replay with their tool_call_id attached. Both are
        required by the OpenAI protocol for the model to correlate a tool
        result with the request that produced it.
        """
        messages: list[dict] = []
        if conversation.system_prompt:
            messages.append({"role": "system", "content": conversation.system_prompt})
        for message in history:
            entry: dict = {"role": message.role, "content": message.content}
            if message.role == "assistant" and message.tool_calls:
                entry["tool_calls"] = message.tool_calls
            if message.role == "tool" and message.tool_call_id:
                entry["tool_call_id"] = message.tool_call_id
            messages.append(entry)
        return messages

    def _extract_reply(self, content: Any, response_schema: Optional[dict]) -> str:
        """
        Reduce the model's reply to the string stored in Message.content.

        For structured-output conversations, validate the reply against the
        schema (raises on invalid JSON / non-object) and store the JSON string.
        When the model emits no content (a pure tool-call turn), returns "".
        Schema validation is intentionally skipped for empty content so a
        structured + tools combination can still emit a tool-only turn.
        """
        if response_schema and content:
            parsed = parse_schema_response(content, response_schema)
            return content if isinstance(content, str) else json.dumps(parsed)
        return content or ""

    def _raw_tool_calls(self, response_message: Any) -> Optional[list]:
        """
        Return the raw tool-call dict list emitted by the model, or None when
        the turn emitted no tool calls. We persist this verbatim so a future
        replay can rebuild the OpenAI-spec message sequence.
        """
        raw = getattr(response_message, "tool_calls", None)
        if not raw:
            return None
        return list(raw)

    def _build_tool_calls(self, raw_tool_calls: list) -> list[ToolCall]:
        """
        Map the raw OpenAI/OpenRouter tool-call dicts onto our `ToolCall`
        model. Raises ValueError on missing id / name / arguments so the
        caller is never handed a ToolCall it cannot correlate or execute.
        """
        calls: list[ToolCall] = []
        for tc in raw_tool_calls:
            function = tc.get("function") or {}
            tc_id = tc.get("id")
            name = function.get("name")
            arguments = function.get("arguments")
            if not tc_id:
                raise ValueError("tool call missing id")
            if not name:
                raise ValueError("tool call missing function.name")
            if arguments is None:
                raise ValueError("tool call missing function.arguments")
            calls.append(ToolCall(id=str(tc_id), name=str(name), arguments=str(arguments)))
        return calls
