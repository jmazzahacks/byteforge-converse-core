from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from byteforge_converse_models import Conversation, Message
from byteforge_converse_core import ChatService, LLMConfig


def test_legacy_chat_import_and_tool_result_replay() -> None:
    from byteforge_converse_core.chat import ChatService as DirectChatService

    assert ChatService is DirectChatService
    db = MagicMock()
    calls = [{"id": "call1", "function": {"name": "lookup", "arguments": "{}"}}]
    conversation = Conversation(
        id="conv",
        user_id="owner",
        title="chat",
        created_at=1,
        system_prompt="help",
        tools=[{"type": "function"}],
    )
    db.get_conversation.return_value = conversation
    assistant = Message("reply", "conv", "assistant", "done", 3)
    db.create_message.side_effect = [
        Message("result", "conv", "tool", "ok", 2, tool_call_id="call1"),
        assistant,
    ]
    db.list_messages.return_value = [
        Message("request", "conv", "assistant", "", 1, tool_calls=calls),
        Message("result", "conv", "tool", "ok", 2, tool_call_id="call1"),
    ]
    client = MagicMock()
    client.chat.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None))
        ],
        usage=None,
    )
    service = ChatService(db, LLMConfig("fake-key", "model"), client=client)
    turn = service.send_tool_result("conv", "call1", "ok")
    assert turn.message == assistant
    params = client.chat.create.call_args.kwargs
    assert params["messages"][1]["tool_calls"] == calls
    assert params["messages"][2]["tool_call_id"] == "call1"
    assert params["tools"] == conversation.tools
    db.list_messages.assert_called_once_with("conv", limit=None)


def test_chat_failure_compensates_input() -> None:
    db, client = MagicMock(), MagicMock()
    db.get_conversation.return_value = Conversation("conv", "owner", "chat", 1)
    db.create_message.return_value = Message("input", "conv", "user", "hello", 1)
    db.list_messages.return_value = []
    client.chat.create.side_effect = RuntimeError("provider down")
    with pytest.raises(RuntimeError, match="provider down"):
        ChatService(db, LLMConfig("fake", "model"), client).send_turn("conv", "hello")
    db.delete_message.assert_called_once_with("input")
