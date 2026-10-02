from unittest.mock import MagicMock
import uuid

import pytest
from psycopg2.extensions import connection as PgConnection
from byteforge_converse_models import ConversationCreate
from byteforge_converse_core.storage import Database, Repository, ResourceNotFound


@pytest.mark.parametrize("owner", [None, "", "  ", 123])
def test_blank_or_invalid_owner_is_rejected(owner: object) -> None:
    conn = MagicMock()
    conn.autocommit = False
    with pytest.raises(ValueError, match="owner_id"):
        Repository(conn).for_owner(owner)


@pytest.mark.parametrize("borrowed", [False, True])
def test_owner_scope_covers_reads_writes_and_retry(
    database: Database, connection: PgConnection, borrowed: bool
) -> None:
    a = database.create_conversation(ConversationCreate(user_id="a", title="A"))
    b = database.create_conversation(ConversationCreate(user_id="b", title="B"))
    bmessage = database.append_message(
        b.id, "user", "private", producer="p", delivery_key="secret"
    )
    repo = Repository(connection) if borrowed else database
    scoped = repo.for_owner("a")
    assert [c.id for c in scoped.list_conversations()] == [a.id]
    assert scoped.get_conversation(a.id).title == "A"
    for hidden in (b.id, str(uuid.uuid4())):
        assert scoped.get_conversation(hidden) is None
        assert scoped.list_messages(hidden) == []
        assert scoped.delete_conversation(hidden) is False
        with pytest.raises(ResourceNotFound, match="Conversation not found"):
            scoped.create_message(hidden, "user", "intrusion")
        with pytest.raises(ResourceNotFound, match="Conversation not found"):
            scoped.append_message(
                hidden, "user", "private", producer="p", delivery_key="secret"
            )
        scoped.touch_conversation(hidden, 99)
    assert scoped.delete_message(bmessage.message.id) is False
    assert scoped.delete_message(str(uuid.uuid4())) is False
    with pytest.raises(ValueError):
        scoped.list_conversations("b")
    with pytest.raises(ValueError):
        scoped.create_conversation(ConversationCreate(user_id="b", title="bad"))
    with pytest.raises(ValueError):
        scoped.for_owner("b")
    local = scoped.create_conversation(ConversationCreate(user_id="a", title="local"))
    message = scoped.create_message(a.id, "user", "hello")
    assert scoped.delete_message(message.id)
    appended = scoped.append_message(
        a.id, "user", "ok", producer="p", delivery_key="secret"
    )
    retry = scoped.append_message(
        a.id, "user", "ok", producer="p", delivery_key="secret"
    )
    assert appended.created and not retry.created
    assert retry.message.id == appended.message.id != bmessage.message.id
    scoped.touch_conversation(a.id, 100)
    assert scoped.get_conversation(a.id).updated_at == 100
    assert scoped.delete_conversation(local.id)
    if borrowed:
        connection.commit()
    assert database.get_conversation(b.id).updated_at is None
    assert [m.id for m in database.list_messages(b.id)] == [bmessage.message.id]


def test_owner_query_contains_predicate_and_append_locks_scope() -> None:
    conn = MagicMock()
    conn.autocommit = False
    cur = conn.cursor.return_value
    cur.fetchone.return_value = None
    cur.fetchall.return_value = []
    cur.rowcount = 0
    scoped = Repository(conn).for_owner("authenticated-owner")
    scoped.get_conversation("c")
    scoped.list_messages("c")
    scoped.delete_conversation("c")
    scoped.delete_message("m")
    with pytest.raises(ResourceNotFound):
        scoped.append_message("c", "user", "secret", producer="p", delivery_key="k")
    for call in cur.execute.call_args_list:
        query, params = call.args
        assert "user_id = %s" in query
        assert "authenticated-owner" in params
    assert "FOR SHARE" in cur.execute.call_args.args[0]
    assert not any("INSERT" in call.args[0] for call in cur.execute.call_args_list)
