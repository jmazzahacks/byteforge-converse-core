"""
Postgres persistence layer for ByteforgeConverse.

Provides pooled convenience operations and caller-owned transaction repositories. Reads use `RealDictCursor`
so rows reconstruct directly into models via `Model.from_dict(dict(row))`.
All date/time columns are `BIGINT` unix timestamps; the database generates
ids (`gen_random_uuid()`) and `created_at` defaults, returned via `RETURNING *`.
"""

import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor, Json
from psycopg2.extensions import connection as PgConnection, TRANSACTION_STATUS_INERROR

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


class ResourceNotFound(LookupError):
    """Resource is absent or outside the supplied ownership scope."""


class IdempotencyConflict(ValueError):
    """A delivery key already identifies a different persisted payload."""


class ConcurrentMessageChange(RuntimeError):
    """A conflicting message disappeared; caller may retry its whole transaction."""


@dataclass(frozen=True)
class AppendResult:
    """Storage metadata; seq is an ordering tiebreaker, NOT a commit watermark."""

    message: Message
    created: bool
    seq: int
    producer: str
    delivery_key: str


class Repository:
    """Conversation/message operations on a caller-owned psycopg2 connection.

    The caller owns commit, rollback, close and pool return. Construct one per
    transaction; never share a connection/repository between concurrent tasks.
    No pre-ping or automatic retry is performed on a borrowed transaction.
    """

    _owner_id: Optional[str] = None

    def for_owner(self, owner_id: str) -> "OwnerRepository":
        """Scope operations to an already authenticated opaque owner ID."""
        if self._owner_id is not None and owner_id != self._owner_id:
            raise ValueError("Cannot change an existing repository's owner scope")
        return OwnerRepository(self, owner_id)

    def _require_conversation(
        self, cursor: RealDictCursor, conversation_id: str
    ) -> None:
        # Acquire the writer lock up front: two FOR SHARE holders that both
        # touch/delete the conversation later would deadlock on lock upgrade.
        # NO KEY UPDATE also holds ownership stable, while allowing FK checks
        # from legacy unscoped inserts (which acquire KEY SHARE).
        cursor.execute(
            "SELECT id FROM conversations WHERE id = %s "
            "AND (%s::text IS NULL OR user_id = %s) FOR NO KEY UPDATE",
            (conversation_id, self._owner_id, self._owner_id),
        )
        if cursor.fetchone() is None:
            raise ResourceNotFound("Conversation not found")

    def __init__(self, connection: PgConnection) -> None:
        if connection.autocommit:
            raise ValueError("Repository requires a non-autocommit connection")
        self._borrowed_connection = connection
        self._active = True

    @contextmanager
    def _cursor(self, commit: bool = False) -> Iterator[RealDictCursor]:
        if not self._active:
            raise RuntimeError("Transaction repository is no longer active")
        if self._borrowed_connection.autocommit:
            raise ValueError("Repository requires a non-autocommit connection")
        cursor = self._borrowed_connection.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    # --- conversations -----------------------------------------------------

    def create_conversation(self, create: ConversationCreate) -> Conversation:
        if self._owner_id is not None and create.user_id != self._owner_id:
            raise ValueError("Conversation owner does not match repository scope")
        response_schema = (
            Json(create.response_schema) if create.response_schema is not None else None
        )
        tools = Json(create.tools) if create.tools is not None else None
        with self._cursor(commit=True) as cursor:
            cursor.execute(
                "INSERT INTO conversations (user_id, title, model, system_prompt, response_schema, tools) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
                (
                    create.user_id,
                    create.title,
                    create.model,
                    create.system_prompt,
                    response_schema,
                    tools,
                ),
            )
            row = cursor.fetchone()
        return Conversation.from_dict(dict(row))

    def touch_conversation(self, conversation_id: str, updated_at: int) -> None:
        with self._cursor(commit=True) as cursor:
            cursor.execute(
                "UPDATE conversations SET updated_at = %s WHERE id = %s "
                "AND (%s::text IS NULL OR user_id = %s)",
                (updated_at, conversation_id, self._owner_id, self._owner_id),
            )

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM conversations WHERE id = %s AND (%s::text IS NULL OR user_id = %s)",
                (conversation_id, self._owner_id, self._owner_id),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return Conversation.from_dict(dict(row))

    def list_conversations(
        self, user_id: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> list[Conversation]:
        if self._owner_id is not None:
            if user_id is not None and user_id != self._owner_id:
                raise ValueError("Requested owner does not match repository scope")
            user_id = self._owner_id
        elif user_id is None:
            raise ValueError("user_id is required for an unscoped repository")
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
            cursor.execute(
                "DELETE FROM conversations WHERE id = %s AND (%s::text IS NULL OR user_id = %s)",
                (conversation_id, self._owner_id, self._owner_id),
            )
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
            if self._owner_id is not None:
                self._require_conversation(cursor, conversation_id)
            cursor.execute(
                "INSERT INTO messages (conversation_id, role, content, token_count, tool_calls, tool_call_id) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
                (
                    conversation_id,
                    role,
                    content,
                    token_count,
                    tool_calls_json,
                    tool_call_id,
                ),
            )
            row = cursor.fetchone()
        return Message.from_dict(dict(row))

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        producer: str,
        delivery_key: str,
        token_count: Optional[int] = None,
        tool_calls: Optional[list] = None,
        tool_call_id: Optional[str] = None,
    ) -> AppendResult:
        """Idempotently append; key namespace is (conversation, producer, key).

        Every persisted payload field participates in conflict detection,
        including token_count. No existing payload is ever overwritten.
        REPEATABLE READ/SERIALIZABLE conflicts propagate to the transaction
        owner; the repository never retries a borrowed transaction.
        """
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}, got {role!r}")
        for name, value, maximum in (
            ("producer", producer, 128),
            ("delivery_key", delivery_key, 512),
        ):
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value.encode("utf-8")) > maximum
            ):
                raise ValueError(
                    f"{name} must be nonblank and at most {maximum} UTF-8 bytes"
                )
        payload = (
            role,
            content,
            token_count,
            Json(tool_calls) if tool_calls is not None else None,
            tool_call_id,
        )
        with self._cursor(commit=True) as cursor:
            self._require_conversation(cursor, conversation_id)
            cursor.execute(
                "INSERT INTO messages (conversation_id, role, content, token_count, tool_calls, tool_call_id, producer, delivery_key) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (conversation_id, producer, delivery_key) WHERE delivery_key IS NOT NULL "
                "DO NOTHING RETURNING *",
                (conversation_id, *payload, producer, delivery_key),
            )
            row = cursor.fetchone()
            created = row is not None
            if row is None:
                # A separate statement gets a fresh READ COMMITTED snapshot
                # after ON CONFLICT waits for a concurrent writer to commit.
                cursor.execute(
                    "SELECT *, (role = %s AND content = %s "
                    "AND token_count IS NOT DISTINCT FROM %s "
                    "AND tool_calls IS NOT DISTINCT FROM %s::jsonb "
                    "AND tool_call_id IS NOT DISTINCT FROM %s) AS same_payload "
                    "FROM messages WHERE conversation_id = %s AND producer = %s AND delivery_key = %s FOR SHARE",
                    (*payload, conversation_id, producer, delivery_key),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ConcurrentMessageChange(
                        "Message changed during retry lookup; retry the transaction"
                    )
                if not row["same_payload"]:
                    raise IdempotencyConflict(
                        "Delivery key already identifies a different message payload"
                    )
            return AppendResult(
                Message.from_dict(dict(row)),
                created,
                int(row["seq"]),
                producer,
                delivery_key,
            )

    def list_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = 100,
        offset: int = 0,
    ) -> list[Message]:
        """
        List messages in the order they were written.

        created_at is epoch SECONDS, so messages written within the same
        second tie on it, and Postgres returns ties in no particular order.
        A chat turn routinely writes several rows inside one second (a tool
        call, its result, the next reply; or a fast reply followed by the
        user's next message), and replaying them out of order breaks the LLM
        protocol: a history ending on an assistant row is rejected by models
        that do not support prefill ("the conversation must end with a user
        message"), and a tool row replayed before its tool call is invalid.
        seq (BIGSERIAL, insertion order) breaks the tie.

        `limit=None` means no LIMIT (return all rows). The default 100 stays in
        place for paginated read endpoints; chat-turn replay passes `None` so
        long conversations are never silently truncated mid-history.
        """
        owner_filter = ""
        params: tuple = (conversation_id,)
        if self._owner_id is not None:
            owner_filter = "AND EXISTS (SELECT 1 FROM conversations c WHERE c.id = messages.conversation_id AND c.user_id = %s) "
            params += (self._owner_id,)
        query = (
            "SELECT * FROM messages WHERE conversation_id = %s "
            + owner_filter
            + "ORDER BY created_at ASC, seq ASC "
        )
        if limit is not None:
            query += "LIMIT %s "
            params += (limit,)
        query += "OFFSET %s"
        params += (offset,)
        with self._cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return [Message.from_dict(dict(row)) for row in rows]

    def delete_message(self, message_id: str) -> bool:
        with self._cursor(commit=True) as cursor:
            if self._owner_id is not None:
                # DELETE ... USING locks only the message, so its join can
                # otherwise authorize against a stale conversation owner.
                # Lock the parent first, just as scoped appends do, and let
                # Postgres recheck the owner if a concurrent transfer wins.
                cursor.execute(
                    "SELECT c.id FROM conversations c JOIN messages m "
                    "ON m.conversation_id = c.id WHERE m.id = %s "
                    "AND c.user_id = %s FOR NO KEY UPDATE OF c",
                    (message_id, self._owner_id),
                )
                if cursor.fetchone() is None:
                    return False
            cursor.execute(
                "DELETE FROM messages m USING conversations c WHERE m.id = %s "
                "AND c.id = m.conversation_id AND (%s::text IS NULL OR c.user_id = %s)",
                (message_id, self._owner_id, self._owner_id),
            )
            deleted = cursor.rowcount
        return deleted > 0


class OwnerRepository(Repository):
    """Explicit owner facade; authentication and identity resolution stay upstream."""

    def __init__(self, repository: Repository, owner_id: str) -> None:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must be a nonblank authenticated identity")
        if repository._owner_id is not None and repository._owner_id != owner_id:
            raise ValueError("Cannot change an existing repository's owner scope")
        self._repository = repository
        self._owner_id = owner_id

    @contextmanager
    def _cursor(self, commit: bool = False) -> Iterator[RealDictCursor]:
        with self._repository._cursor(commit=commit) as cursor:
            yield cursor


class Database(Repository):
    """Connection-pooled Postgres access for conversations, messages, and sessions.

    Pre-pings every pooled checkout and recovers transparently from upstream
    Postgres restarts. Build lazily (post-fork) in long-running servers —
    never at import time.
    """

    def __init__(
        self, config: DatabaseConfig, min_conn: int = 1, max_conn: int = 10
    ) -> None:
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
        logger.info(
            "Database connection pool initialized (%s:%s/%s)",
            config.host,
            config.port,
            config.name,
        )

    @staticmethod
    def _check_alive(conn: PgConnection) -> None:
        """Pre-ping `conn` with `SELECT 1`. Raises on dead conn."""
        cur = conn.cursor(cursor_factory=RealDictCursor)
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
            attempt + 1,
            MAX_HEALTH_RETRIES,
            exc,
        )

    @staticmethod
    def _retry_backoff(attempt: int) -> float:
        """Full-jitter exponential backoff in seconds."""
        base = min(_RETRY_BACKOFF_BASE_SEC * (2**attempt), _RETRY_BACKOFF_MAX_SEC)
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

    @contextmanager
    def transaction(self) -> Iterator[Repository]:
        """Commit a complete unit of storage work, or roll it back on error.

        Checkout recovery happens before the transaction starts. The body and
        commit are never replayed; an ambiguous commit error propagates.
        """
        with self._connection() as connection:
            repository = Repository(connection)
            try:
                yield repository
                # PostgreSQL accepts COMMIT on an aborted transaction as a
                # ROLLBACK. Do not report success if the body caught a SQL
                # error without recovering to a savepoint.
                if connection.get_transaction_status() == TRANSACTION_STATUS_INERROR:
                    raise psycopg2.errors.InFailedSqlTransaction(
                        "Cannot commit an aborted transaction; its writes were not saved"
                    )
                connection.commit()
            finally:
                repository._active = False

    def close(self) -> None:
        self._pool.closeall()

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
