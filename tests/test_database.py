"""Recovery-path tests for `Database` — no live Postgres required.

Covers the critical paths the resilient-driver refactor exists to defend:
  - dead pooled conn on checkout self-heals via retry + close=True
  - pool full of corpses raises psycopg2.OperationalError after MAX retries
  - mid-flight death discards the conn (close=True), never recycles
  - app-level error on a healthy conn recycles (close=False) without churn
  - KeyboardInterrupt during cleanup is not swallowed
  - cursor.close() failure during cleanup does not mask the in-flight error
  - rollback failure during cleanup discards the conn (no aborted-txn recycle)
  - the conn is returned to the pool exactly once on every exit path
"""

from unittest.mock import MagicMock

import psycopg2
import pytest

from byteforge_converse_core.database import Database, MAX_HEALTH_RETRIES


def _make_db_with_mocked_pool(connections: list) -> Database:
    """Build a Database whose pool returns the given pre-built mock conns."""
    db = Database.__new__(Database)
    db._pool = MagicMock()
    db._pool.getconn.side_effect = list(connections)
    db._last_checkout_warn = 0.0
    return db


def _alive_conn() -> MagicMock:
    """Mock conn whose pre-ping succeeds and whose .closed reads as 0."""
    conn = MagicMock()
    conn.cursor.return_value.fetchone.return_value = (1,)
    # Explicit closed=0 is required — MagicMock would otherwise auto-create
    # a truthy attribute, misclassifying every healthy conn as dead.
    conn.closed = 0
    return conn


def _dead_conn() -> MagicMock:
    """Mock conn whose pre-ping SELECT raises OperationalError."""
    conn = MagicMock()
    conn.cursor.return_value.execute.side_effect = psycopg2.OperationalError("dead")
    conn.closed = 2
    return conn


# --- checkout / retry --------------------------------------------------------


def test_dead_conn_on_first_checkout_retries_and_recovers() -> None:
    dead, alive = _dead_conn(), _alive_conn()
    db = _make_db_with_mocked_pool([dead, alive])

    with db._connection() as conn:
        assert conn is alive

    # Dead conn discarded with close=True so the pool refills with a fresh socket.
    db._pool.putconn.assert_any_call(dead, close=True)
    # Healthy conn recycled (close=False) on clean exit.
    db._pool.putconn.assert_any_call(alive, close=False)


def test_pool_full_of_corpses_raises_operational_error_after_max_retries() -> None:
    corpses = [_dead_conn() for _ in range(MAX_HEALTH_RETRIES)]
    db = _make_db_with_mocked_pool(corpses)

    with pytest.raises(psycopg2.OperationalError) as excinfo:
        with db._connection():
            pass

    # Preserves the historical exception class (not RuntimeError) so external
    # `except OperationalError` handlers keep working past the refactor.
    assert isinstance(excinfo.value.__cause__, psycopg2.OperationalError)
    assert db._pool.putconn.call_count == MAX_HEALTH_RETRIES
    for call in db._pool.putconn.call_args_list:
        assert call.kwargs.get("close") is True


def test_keyboard_interrupt_during_pre_ping_propagates_without_retry() -> None:
    """KI during checkout must NOT be silently retried — operator wants out."""
    conn = MagicMock()
    conn.cursor.return_value.execute.side_effect = KeyboardInterrupt()
    conn.closed = 0
    db = _make_db_with_mocked_pool([conn])

    with pytest.raises(KeyboardInterrupt):
        with db._connection():
            pass

    # Conn discarded; no retry attempted past the one getconn call.
    assert db._pool.getconn.call_count == 1
    db._pool.putconn.assert_called_once_with(conn, close=True)


# --- mid-flight cleanup ------------------------------------------------------


def test_mid_flight_dead_conn_error_discards_with_close() -> None:
    """OperationalError mid-query: rollback fails, conn.closed flips → discard."""
    alive = _alive_conn()
    db = _make_db_with_mocked_pool([alive])

    with pytest.raises(psycopg2.OperationalError):
        with db._connection() as conn:
            conn.rollback.side_effect = psycopg2.OperationalError("conn died")
            conn.closed = 2
            raise psycopg2.OperationalError("query failed on dead conn")

    db._pool.putconn.assert_called_once_with(alive, close=True)


def test_serialization_failure_on_healthy_conn_recycles_without_churn() -> None:
    """App-level error on a live conn must recycle (close=False), not destroy."""
    alive = _alive_conn()
    db = _make_db_with_mocked_pool([alive])

    with pytest.raises(psycopg2.errors.SerializationFailure):
        with db._connection():
            raise psycopg2.errors.SerializationFailure("conflict")

    db._pool.putconn.assert_called_once_with(alive, close=False)


def test_value_error_on_silently_dead_conn_discards() -> None:
    """Non-DB exception + silently-dead conn → close=True (don't recycle corpse)."""
    alive = _alive_conn()
    db = _make_db_with_mocked_pool([alive])

    with pytest.raises(ValueError):
        with db._connection() as conn:
            conn.rollback.side_effect = psycopg2.OperationalError("dead")
            conn.closed = 2
            raise ValueError("app error while conn was dying")

    db._pool.putconn.assert_called_once_with(alive, close=True)


def test_rollback_raising_non_dead_error_still_discards() -> None:
    """Rollback raising a non-DEAD_CONN exception must NOT recycle the conn —
    its transaction state is untrustable, the next caller would see
    'current transaction is aborted'."""
    alive = _alive_conn()
    db = _make_db_with_mocked_pool([alive])

    with pytest.raises(ValueError):
        with db._connection() as conn:
            conn.rollback.side_effect = RuntimeError("weird non-dead rollback failure")
            conn.closed = 0
            raise ValueError("original error")

    db._pool.putconn.assert_called_once_with(alive, close=True)


def test_keyboard_interrupt_during_yield_propagates_and_discards() -> None:
    """KI mid-request must propagate cleanly; conn discarded; original KI preserved."""
    alive = _alive_conn()
    db = _make_db_with_mocked_pool([alive])

    with pytest.raises(KeyboardInterrupt):
        with db._connection():
            raise KeyboardInterrupt()

    db._pool.putconn.assert_called_once_with(alive, close=True)


def test_keyboard_interrupt_during_rollback_propagates_and_returns_conn() -> None:
    """KI raised inside cleanup rollback must propagate AND not leak the conn."""
    alive = _alive_conn()
    db = _make_db_with_mocked_pool([alive])

    with pytest.raises(KeyboardInterrupt):
        with db._connection() as conn:
            conn.rollback.side_effect = KeyboardInterrupt()
            raise ValueError("original app error")

    # Outer finally must always putback; close=True because KI signaled untrust.
    db._pool.putconn.assert_called_once_with(alive, close=True)


# --- _cursor wrapping --------------------------------------------------------


def test_cursor_clean_path_commits_and_recycles() -> None:
    alive = _alive_conn()
    db = _make_db_with_mocked_pool([alive])

    with db._cursor(commit=True) as cur:
        cur.execute("INSERT ...")

    alive.commit.assert_called_once()
    db._pool.putconn.assert_called_once_with(alive, close=False)


def test_cursor_read_path_rolls_back_implicit_txn() -> None:
    """Reads must roll back the implicit txn so the conn isn't returned
    to the pool 'idle in transaction'. Pre-ping rolls back once during
    checkout (1), the read body's release adds one more (2)."""
    alive = _alive_conn()
    db = _make_db_with_mocked_pool([alive])

    with db._cursor(commit=False):
        pass

    assert alive.rollback.call_count == 2
    db._pool.putconn.assert_called_once_with(alive, close=False)


def test_cursor_close_failure_does_not_mask_in_flight_exception() -> None:
    """cursor.close() raising during cleanup MUST NOT replace the caller's
    real exception — that was the masking bug fixed in this refactor.
    Pre-ping uses its own cursor mock so its close stays clean; only the
    body cursor's close is rigged to raise."""
    alive = _alive_conn()
    preping_cursor = alive.cursor.return_value
    body_cursor = MagicMock()
    body_cursor.close.side_effect = RuntimeError("cursor.close blew up")
    alive.cursor.side_effect = [preping_cursor, body_cursor]
    db = _make_db_with_mocked_pool([alive])

    with pytest.raises(ValueError, match="real error"):
        with db._cursor() as cur:
            assert cur is body_cursor
            raise ValueError("real error")


def test_cursor_exception_path_rolls_back_exactly_once() -> None:
    """Exception under _cursor must NOT cause double-rollback (one in _cursor,
    one in _connection) — _connection is now the sole rollback owner. Total
    rollback count is pre-ping (1) + _connection cleanup (1) = 2. Before the
    refactor it would have been pre-ping (1) + _cursor.except (1) + _connection.except (1) = 3."""
    alive = _alive_conn()
    db = _make_db_with_mocked_pool([alive])

    with pytest.raises(ValueError):
        with db._cursor() as cur:
            cur.execute("BAD SQL")
            raise ValueError("boom")

    assert alive.rollback.call_count == 2
    alive.commit.assert_not_called()


# --- conn always returned to pool --------------------------------------------


def test_conn_always_returned_on_clean_exit() -> None:
    alive = _alive_conn()
    db = _make_db_with_mocked_pool([alive])
    with db._connection():
        pass
    db._pool.putconn.assert_called_once()


def test_conn_always_returned_on_exception() -> None:
    alive = _alive_conn()
    db = _make_db_with_mocked_pool([alive])
    with pytest.raises(RuntimeError):
        with db._connection():
            raise RuntimeError("boom")
    db._pool.putconn.assert_called_once()


def test_safe_putback_swallows_pool_putconn_failure() -> None:
    """A broken pool must not raise from cleanup — fall back to direct conn.close()."""
    alive = _alive_conn()
    db = _make_db_with_mocked_pool([alive])
    db._pool.putconn.side_effect = Exception("pool internal error")

    with db._connection():
        pass

    alive.close.assert_called_once()
