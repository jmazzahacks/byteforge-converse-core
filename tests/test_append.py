from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import RealDictCursor
from byteforge_converse_models import ConversationCreate
from byteforge_converse_core.storage import (
    AppendResult,
    IdempotencyConflict,
    Repository,
)
from conftest import connect_test


def make_conversation(connection: PgConnection) -> str:
    cid = (
        Repository(connection)
        .create_conversation(ConversationCreate(user_id="a", title="test"))
        .id
    )
    connection.commit()
    return cid


def test_identity_retries_conflicts_and_namespaces(connection: PgConnection) -> None:
    cid = make_conversation(connection)
    other = make_conversation(connection)
    repo = Repository(connection)
    first = repo.append_message(
        cid, "user", "same", producer="worker", delivery_key="event"
    )
    connection.commit()  # Simulates acknowledgement lost after successful commit.
    retry = repo.append_message(
        cid, "user", "same", producer="worker", delivery_key="event"
    )
    assert first.created and not retry.created
    assert (first.message.id, first.seq) == (retry.message.id, retry.seq)
    with pytest.raises(IdempotencyConflict):
        repo.append_message(
            cid, "user", "different", producer="worker", delivery_key="event"
        )
    distinct = [
        repo.append_message(
            cid, "user", "same", producer="worker", delivery_key="event2"
        ),
        repo.append_message(
            cid, "user", "same", producer="worker2", delivery_key="event"
        ),
        repo.append_message(
            other, "user", "same", producer="worker", delivery_key="event"
        ),
    ]
    assert len({first.message.id, *(r.message.id for r in distinct)}) == 4
    assert len(repo.list_messages(cid)) == 3


@pytest.mark.parametrize(
    "change",
    [
        {"role": "assistant"},
        {"content": "changed"},
        {"token_count": 3},
        {"tool_calls": []},
        {"tool_call_id": "changed"},
    ],
)
def test_conflicting_payload_is_never_overwritten(
    connection: PgConnection, change: dict
) -> None:
    cid = make_conversation(connection)
    repo = Repository(connection)
    payload = dict(
        role="user",
        content="same",
        token_count=None,
        tool_calls=None,
        tool_call_id=None,
    )
    original = repo.append_message(cid, producer="p", delivery_key="k", **payload)
    with pytest.raises(IdempotencyConflict):
        repo.append_message(cid, producer="p", delivery_key="k", **(payload | change))
    assert repo.list_messages(cid)[0] == original.message


def test_json_payload_comparison_uses_jsonb_semantics(connection: PgConnection) -> None:
    cid = make_conversation(connection)
    repo = Repository(connection)
    first = repo.append_message(
        cid,
        "assistant",
        "",
        producer="p",
        delivery_key="k",
        tool_calls=[{"a": 1, "b": True}],
    )
    retry = repo.append_message(
        cid,
        "assistant",
        "",
        producer="p",
        delivery_key="k",
        tool_calls=[{"b": True, "a": 1}],
    )
    assert retry.message.id == first.message.id
    with pytest.raises(IdempotencyConflict):
        repo.append_message(
            cid,
            "assistant",
            "",
            producer="p",
            delivery_key="k",
            tool_calls=[{"a": True, "b": True}],
        )


def test_rollback_retry_and_deletion_retention(connection: PgConnection) -> None:
    cid = make_conversation(connection)
    repo = Repository(connection)
    rolled_back = repo.append_message(
        cid, "tool", "result", producer="p", delivery_key="k", tool_call_id="call"
    )
    connection.rollback()
    committed = repo.append_message(
        cid, "tool", "result", producer="p", delivery_key="k", tool_call_id="call"
    )
    connection.commit()
    assert committed.created and committed.message.id != rolled_back.message.id
    assert len(repo.list_messages(cid)) == 1
    assert repo.delete_message(committed.message.id)
    replacement = repo.append_message(
        cid, "tool", "result", producer="p", delivery_key="k", tool_call_id="call"
    )
    assert replacement.created and replacement.message.id != committed.message.id
    assert repo.delete_conversation(cid)
    assert repo.list_messages(cid) == []


@pytest.mark.parametrize("conflicting", [False, True])
def test_simultaneous_deliveries(
    connection: PgConnection, pg_schema: tuple[str, str], conflicting: bool
) -> None:
    cid = make_conversation(connection)
    barrier = Barrier(2)

    def append(index: int) -> AppendResult | str:
        conn = connect_test(pg_schema)
        try:
            barrier.wait(timeout=5)
            result = Repository(conn).append_message(
                cid,
                "user",
                str(index) if conflicting else "same",
                producer="p",
                delivery_key="k",
            )
            conn.commit()
            return result
        except IdempotencyConflict:
            conn.rollback()
            return "conflict"
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(append, [0, 1]))
    assert len(Repository(connection).list_messages(cid)) == 1
    if conflicting:
        assert results.count("conflict") == 1
    else:
        assert results[0].message.id == results[1].message.id
        assert sorted(r.created for r in results) == [False, True]


def test_snapshot_replay_handles_late_commit_without_watermark_loss(
    connection: PgConnection, pg_schema: tuple[str, str]
) -> None:
    cid = make_conversation(connection)
    slow, fast, reader = [connect_test(pg_schema) for _ in range(3)]
    try:
        first = Repository(slow).create_message(cid, "user", "legacy slow")
        second = Repository(fast).append_message(
            cid, "assistant", "fast", producer="p", delivery_key="k"
        )
        for conn in (slow, fast):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "UPDATE messages SET created_at=100 WHERE conversation_id=%s",
                    (cid,),
                )
        fast.commit()
        reader.set_session(isolation_level="REPEATABLE READ", readonly=True)
        replay = Repository(reader)
        assert [m.id for m in replay.list_messages(cid, limit=1)] == [second.message.id]
        slow.commit()  # Lower seq becomes visible after the reader's snapshot.
        assert replay.list_messages(cid, limit=1, offset=1) == []
        reader.rollback()
        # A new full replay includes both. Never checkpoint on max(seq).
        pages = [replay.list_messages(cid, limit=1, offset=n) for n in range(3)]
        assert [m.id for page in pages for m in page] == [first.id, second.message.id]
        assert pages[-1] == []
    finally:
        for conn in (slow, fast, reader):
            conn.close()
