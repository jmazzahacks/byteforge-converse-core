from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import RealDictCursor
from byteforge_converse_models import ConversationCreate
from byteforge_converse_core.storage import Database, Repository
from conftest import connect_test


def test_borrowed_transaction_commits_and_rolls_back_with_consumer_rows(
    connection: PgConnection,
) -> None:
    repo = Repository(connection)
    with connection.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("CREATE TABLE markers (message_id UUID)")
    connection.commit()
    conversation = repo.create_conversation(
        ConversationCreate(user_id="a", title="test")
    )
    connection.commit()
    for commit in (True, False):
        message = repo.create_message(conversation.id, "user", str(commit))
        with connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("INSERT INTO markers VALUES (%s)", (message.id,))
        assert len(repo.list_messages(conversation.id)) == (1 if commit else 2)
        if commit:
            connection.commit()
        else:
            with pytest.raises(ZeroDivisionError):
                with connection:
                    raise ZeroDivisionError("consumer failed")
        assert connection.closed == 0
    assert [m.content for m in repo.list_messages(conversation.id)] == ["True"]
    with connection.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM markers")
        assert len(cur.fetchall()) == 1


def test_borrowed_never_controls_connection_on_success_or_error() -> None:
    conn = MagicMock()
    conn.autocommit = False
    conn.cursor.return_value.fetchall.return_value = []
    repo = Repository(conn)
    assert repo.list_messages("id") == []
    conn.cursor.return_value.execute.side_effect = ValueError("query failed")
    with pytest.raises(ValueError):
        repo.list_messages("id")
    conn.commit.assert_not_called()
    conn.rollback.assert_not_called()
    conn.close.assert_not_called()


def test_autocommit_rejected(connection: PgConnection) -> None:
    connection.autocommit = True
    with pytest.raises(ValueError, match="autocommit"):
        Repository(connection)


def test_managed_transaction_lifetime_and_rollback(database: Database) -> None:
    with database.transaction() as repo:
        conversation = repo.create_conversation(
            ConversationCreate(user_id="a", title="commit")
        )
        repo.create_message(conversation.id, "user", "keep")
    with pytest.raises(RuntimeError, match="no longer active"):
        repo.list_messages(conversation.id)
    with pytest.raises(ValueError, match="abort"):
        with database.transaction() as failing:
            failing.create_message(conversation.id, "user", "rollback")
            raise ValueError("abort")
    assert [m.content for m in database.list_messages(conversation.id)] == ["keep"]


def test_concurrent_borrowed_transactions_are_independent(
    connection: PgConnection, pg_schema: tuple[str, str]
) -> None:
    def worker(value: str) -> str:
        conn = connect_test(pg_schema)
        try:
            repo = Repository(conn)
            conversation = repo.create_conversation(
                ConversationCreate(user_id=value, title=value)
            )
            repo.create_message(conversation.id, "user", value)
            assert [m.content for m in repo.list_messages(conversation.id)] == [value]
            conn.commit()
            return conversation.id
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(worker, ["a", "b"]))
    assert ids[0] != ids[1]
    for cid, value in zip(ids, ["a", "b"]):
        assert [m.content for m in Repository(connection).list_messages(cid)] == [value]


def test_scoped_append_and_consumer_event_share_commit_and_rollback(
    connection: PgConnection,
) -> None:
    repo = Repository(connection).for_owner("a")
    cid = repo.create_conversation(ConversationCreate(user_id="a", title="atomic")).id
    with connection.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "CREATE TABLE consumer_events (id TEXT PRIMARY KEY, message_id UUID)"
        )
    connection.commit()
    with pytest.raises(ValueError):
        with connection:
            result = repo.append_message(
                cid, "assistant", "answer", producer="worker", delivery_key="event1"
            )
            with connection.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "INSERT INTO consumer_events VALUES ('event1', %s)",
                    (result.message.id,),
                )
            raise ValueError("consumer failed")
    assert repo.list_messages(cid) == []
    with connection.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM consumer_events")
        assert cur.fetchall() == []
    connection.rollback()
    with connection:
        result = repo.append_message(
            cid, "assistant", "answer", producer="worker", delivery_key="event1"
        )
        with connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO consumer_events VALUES ('event1', %s)",
                (result.message.id,),
            )
    assert result.created
    retry = repo.append_message(
        cid, "assistant", "answer", producer="worker", delivery_key="event1"
    )
    assert not retry.created and retry.message.id == result.message.id
    with connection.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT message_id FROM consumer_events WHERE id='event1'")
        assert str(cur.fetchone()["message_id"]) == result.message.id


def test_managed_commit_failure_is_not_replayed() -> None:
    import psycopg2

    db = Database.__new__(Database)
    db._pool = MagicMock()
    conn = db._pool.getconn.return_value
    conn.autocommit = False
    conn.closed = 0
    conn.commit.side_effect = psycopg2.OperationalError("ambiguous acknowledgement")
    db._last_checkout_warn = 0.0
    executions = 0
    with pytest.raises(psycopg2.OperationalError, match="ambiguous"):
        with db.transaction():
            executions += 1
    assert executions == 1
    assert conn.commit.call_count == 1
    assert db._pool.getconn.call_count == 1
    db._pool.putconn.assert_called_once()


def test_managed_transaction_reports_caught_database_failure(
    database: Database,
) -> None:
    import psycopg2
    import uuid

    with pytest.raises(psycopg2.errors.InFailedSqlTransaction):
        with database.transaction() as repo:
            conversation = repo.create_conversation(
                ConversationCreate(user_id="a", title="must roll back")
            )
            try:
                repo.create_message(str(uuid.uuid4()), "user", "invalid FK")
            except psycopg2.IntegrityError:
                pass  # Catching the statement error does not repair the transaction.
    assert database.get_conversation(conversation.id) is None


def test_concurrent_append_then_touch_does_not_upgrade_shared_locks(
    connection: PgConnection, pg_schema: tuple[str, str]
) -> None:
    from conftest import wait_for_lock_or_completion

    cid = (
        Repository(connection)
        .create_conversation(ConversationCreate(user_id="a", title="concurrent"))
        .id
    )
    connection.commit()
    first, second = connect_test(pg_schema), connect_test(pg_schema)
    first_repo = Repository(first).for_owner("a")
    first_repo.append_message(cid, "user", "first", producer="p", delivery_key="1")

    def second_writer() -> None:
        with second:
            repo = Repository(second).for_owner("a")
            repo.append_message(cid, "user", "second", producer="p", delivery_key="2")
            repo.touch_conversation(cid, 101)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(second_writer)
            try:
                wait_for_lock_or_completion(
                    connection,
                    second.get_backend_pid(),
                    first.get_backend_pid(),
                    future,
                )
                first_repo.touch_conversation(cid, 100)
                first.commit()
                future.result(timeout=5)
            finally:
                first.rollback()
        assert len(Repository(connection).list_messages(cid)) == 2
    finally:
        first.close()
        second.close()
