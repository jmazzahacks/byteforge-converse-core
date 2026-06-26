"""
Postgres persistence layer for ByteforgeConverse.

Owns the only database connection in the product. Reads use `RealDictCursor`
so rows reconstruct directly into models via `Model.from_dict(dict(row))`.
All date/time columns are `BIGINT` unix timestamps; the database generates
ids (`gen_random_uuid()`) and `created_at` defaults, returned via `RETURNING *`.
"""

import logging
import random
import time
from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor, Json
from psycopg2.extensions import connection as PgConnection

from byteforge_converse_models import (
    Conversation,
    ConversationCreate,
    Message,
    Session,
    VALID_ROLES,
)

from .config import DatabaseConfig

logger = logging.getLogger(__name__)


_DEAD_CONN_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)

MAX_HEALTH_RETRIES = 3

# Full-jitter exponential backoff bounds for the checkout retry loop. Keeps
# concurrent workers from stampeding a recovering DB in lockstep.
_RETRY_BACKOFF_BASE_SEC = 0.05
_RETRY_BACKOFF_MAX_SEC = 0.5

# Spam-throttle window for the per-checkout WARNING log: at most one
# WARNING per Database instance per this many seconds; intra-burst retries
# log at DEBUG so a Postgres bounce doesn't flood Loki.
_CHECKOUT_WARN_INTERVAL_SEC = 5.0


class Database:
    """Connection-pooled Postgres access for conversations, messages, and sessions.

    Pre-pings every pooled checkout and recovers transparently from upstream
    Postgres restarts. Build lazily (post-fork) in long-running servers —
    never at import time.
    """

    def __init__(self, config: DatabaseConfig, min_conn: int = 1, max_conn: int = 10) -> None:
        self._pool = ThreadedConnectionPool(
            min_conn,
            max_conn,
            host=config.host,
            port=config.port,
            dbname=config.name,
            user=config.user,
            password=config.password,
            # Bound TCP handshake on fresh connects so a hung DNS / network
            # hop can't pin a worker indefinitely waiting for a new socket.
            connect_timeout=5,
            # OS-level dead-socket detection within ~80s on idle conns,
            # without waiting for the next real query to hit a dead socket.
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
            # Bound stuck-but-alive queries and forgotten open transactions
            # — keepalives alone do nothing against a hung Postgres backend.
            options="-c statement_timeout=30000 -c idle_in_transaction_session_timeout=60000",
        )
        # Monotonic clock of the last WARNING-level checkout failure; used
        # to throttle the warning log during a Postgres bounce.
        self._last_checkout_warn: float = 0.0
        logger.info("Database connection pool initialized (%s:%s/%s)", config.host, config.port, config.name)

    @staticmethod
    def _check_alive(conn: PgConnection) -> None:
        """Pre-ping `conn` with `SELECT 1`. Raises on dead conn."""
        cur = conn.cursor()
        try:
            cur.execute("SELECT 1")
            cur.fetchone()
            # Release the implicit transaction the SELECT opened so the
            # conn isn't returned to the pool 'idle in transaction'. Must
            # be inside the try so a failed rollback short-circuits to the
            # caller via the same finally cleanup as execute/fetchone failures.
            conn.rollback()
        finally:
            cur.close()

    def _safe_putback(self, conn: Optional[PgConnection], close: bool) -> None:
        """Best-effort return-to-pool; falls back to `conn.close()` on pool error."""
        if conn is None:
            return
        try:
            self._pool.putconn(conn, close=close)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass

    def _warn_checkout_failure(self, attempt: int, exc: BaseException) -> None:
        """Throttled WARNING: first failure per burst logs loud, retries quiet."""
        now = time.monotonic()
        if now - self._last_checkout_warn > _CHECKOUT_WARN_INTERVAL_SEC:
            self._last_checkout_warn = now
            log = logger.warning
        else:
            log = logger.debug
        log(
            "DB checkout pre-ping failed (attempt %d/%d): %s",
            attempt + 1, MAX_HEALTH_RETRIES, exc,
        )

    @staticmethod
    def _retry_backoff(attempt: int) -> float:
        """Full-jitter exponential backoff in seconds."""
        base = min(_RETRY_BACKOFF_BASE_SEC * (2 ** attempt), _RETRY_BACKOFF_MAX_SEC)
        return random.uniform(0, base)

    def _acquire_live_conn(self) -> PgConnection:
        """Check out a pre-pinged connection, retrying past dead pooled conns.

        Treats ANY `_DEAD_CONN_ERRORS` during pre-ping as a dead conn rather
        than trusting `conn.closed` here — psycopg2 can raise OperationalError
        on a server-restart race before libpq has flipped the flag. Any other
        exception (KeyboardInterrupt, SystemExit, unexpected) discards the
        conn and propagates without retry.

        Raises `psycopg2.OperationalError` (with the last underlying error
        as `__cause__`) on retry exhaustion — same exception class as a
        direct connect failure so external `except OperationalError` handlers
        stay valid.
        """
        last_err: Optional[BaseException] = None
        for attempt in range(MAX_HEALTH_RETRIES):
            conn: Optional[PgConnection] = None
            try:
                conn = self._pool.getconn()
                self._check_alive(conn)
                return conn
            except _DEAD_CONN_ERRORS as e:
                self._safe_putback(conn, close=True)
                last_err = e
                self._warn_checkout_failure(attempt, e)
                if attempt < MAX_HEALTH_RETRIES - 1:
                    time.sleep(self._retry_backoff(attempt))
            except BaseException:
                self._safe_putback(conn, close=True)
                raise

        raise psycopg2.OperationalError(
            f"Could not acquire a healthy DB connection after "
            f"{MAX_HEALTH_RETRIES} attempts"
        ) from last_err

    @contextmanager
    def _connection(self) -> Iterator[PgConnection]:
        """Yield a healthy pooled connection; own all rollback/recycle decisions.

        Clean exit → recycle (close=False). Exception → best-effort rollback,
        then recycle if rollback succeeded AND `conn.closed` is still 0,
        else discard. KeyboardInterrupt / SystemExit propagate without being
        swallowed, and the outer `finally` guarantees the conn is always
        returned to the pool exactly once.
        """
        conn = self._acquire_live_conn()
        close = False
        try:
            try:
                yield conn
            except (KeyboardInterrupt, SystemExit):
                close = True
                raise
            except BaseException:
                # Roll back so a recycled conn doesn't carry an aborted
                # txn into the next checkout. Classify dead-vs-alive from
                # the rollback outcome — narrow class lists rot, but
                # rollback-succeeds-AND-socket-still-open is a reliable
                # liveness signal.
                try:
                    conn.rollback()
                    close = conn.closed != 0
                except (KeyboardInterrupt, SystemExit):
                    close = True
                    raise
                except BaseException:
                    close = True
                raise
        finally:
            self._safe_putback(conn, close=close)

    @contextmanager
    def _cursor(self, commit: bool = False) -> Iterator[RealDictCursor]:
        """Yield a RealDictCursor inside a managed connection.

        `_connection` owns rollback/discard on exception; this method owns
        commit-on-success and read-side implicit-txn cleanup only — there is
        no exception handler here, so cleanup never masks the caller's error.
        """
        with self._connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            try:
                yield cursor
                if commit:
                    conn.commit()
                else:
                    # End the read's implicit transaction so the connection
                    # isn't returned to the pool 'idle in transaction'.
                    conn.rollback()
            finally:
                # Cursor cleanup is strictly best-effort: any failure here
                # must not replace whatever exception is already in flight.
                try:
                    cursor.close()
                except Exception:
                    pass

    def close(self) -> None:
        self._pool.closeall()

    # --- conversations -----------------------------------------------------

    def create_conversation(self, create: ConversationCreate) -> Conversation:
        response_schema = Json(create.response_schema) if create.response_schema is not None else None
        tools = Json(create.tools) if create.tools is not None else None
        with self._cursor(commit=True) as cursor:
            cursor.execute(
                "INSERT INTO conversations (user_id, title, model, system_prompt, response_schema, tools) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
                (create.user_id, create.title, create.model, create.system_prompt, response_schema, tools),
            )
            row = cursor.fetchone()
        return Conversation.from_dict(dict(row))

    def touch_conversation(self, conversation_id: str, updated_at: int) -> None:
        with self._cursor(commit=True) as cursor:
            cursor.execute(
                "UPDATE conversations SET updated_at = %s WHERE id = %s",
                (updated_at, conversation_id),
            )

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM conversations WHERE id = %s", (conversation_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        return Conversation.from_dict(dict(row))

    def list_conversations(self, user_id: str, limit: int = 100, offset: int = 0) -> list[Conversation]:
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM conversations WHERE user_id = %s "
                "ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (user_id, limit, offset),
            )
            rows = cursor.fetchall()
        return [Conversation.from_dict(dict(row)) for row in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._cursor(commit=True) as cursor:
            cursor.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
            deleted = cursor.rowcount
        return deleted > 0

    # --- messages ----------------------------------------------------------

    def create_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        token_count: Optional[int] = None,
        tool_calls: Optional[list] = None,
        tool_call_id: Optional[str] = None,
    ) -> Message:
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}, got {role!r}")
        tool_calls_json = Json(tool_calls) if tool_calls is not None else None
        with self._cursor(commit=True) as cursor:
            cursor.execute(
                "INSERT INTO messages (conversation_id, role, content, token_count, tool_calls, tool_call_id) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
                (conversation_id, role, content, token_count, tool_calls_json, tool_call_id),
            )
            row = cursor.fetchone()
        return Message.from_dict(dict(row))

    def list_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = 100,
        offset: int = 0,
    ) -> list[Message]:
        """
        List messages in created_at ASC order.

        `limit=None` means no LIMIT (return all rows). The default 100 stays in
        place for paginated read endpoints; chat-turn replay passes `None` so
        long conversations are never silently truncated mid-history.
        """
        with self._cursor() as cursor:
            if limit is None:
                cursor.execute(
                    "SELECT * FROM messages WHERE conversation_id = %s "
                    "ORDER BY created_at ASC OFFSET %s",
                    (conversation_id, offset),
                )
            else:
                cursor.execute(
                    "SELECT * FROM messages WHERE conversation_id = %s "
                    "ORDER BY created_at ASC LIMIT %s OFFSET %s",
                    (conversation_id, limit, offset),
                )
            rows = cursor.fetchall()
        return [Message.from_dict(dict(row)) for row in rows]

    def delete_message(self, message_id: str) -> bool:
        with self._cursor(commit=True) as cursor:
            cursor.execute("DELETE FROM messages WHERE id = %s", (message_id,))
            deleted = cursor.rowcount
        return deleted > 0

    # --- sessions ----------------------------------------------------------

    def create_session(
        self,
        user_id: str,
        expires_at: int,
        conversation_id: Optional[str] = None,
    ) -> Session:
        with self._cursor(commit=True) as cursor:
            cursor.execute(
                "INSERT INTO sessions (user_id, conversation_id, expires_at) "
                "VALUES (%s, %s, %s) RETURNING *",
                (user_id, conversation_id, expires_at),
            )
            row = cursor.fetchone()
        return Session.from_dict(dict(row))

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        return Session.from_dict(dict(row))

    def delete_session(self, session_id: str) -> bool:
        with self._cursor(commit=True) as cursor:
            cursor.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
            deleted = cursor.rowcount
        return deleted > 0
