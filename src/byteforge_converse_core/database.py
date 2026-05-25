"""
Postgres persistence layer for ByteforgeConverse.

Owns the only database connection in the product. Reads use `RealDictCursor`
so rows reconstruct directly into models via `Model.from_dict(dict(row))`.
All date/time columns are `BIGINT` unix timestamps; the database generates
ids (`gen_random_uuid()`) and `created_at` defaults, returned via `RETURNING *`.
"""

import logging
from contextlib import contextmanager
from typing import Iterator, Optional

from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor

from byteforge_converse_models import Conversation, Message, Session, VALID_ROLES

from .config import DatabaseConfig

logger = logging.getLogger(__name__)


class Database:
    """
    Connection-pooled Postgres access for conversations, messages, and sessions.

    The pool is created when the `Database` is constructed, so build it lazily
    (post-fork) in long-running servers — never at import time.
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
        )
        logger.info("Database connection pool initialized (%s:%s/%s)", config.host, config.port, config.name)

    @contextmanager
    def _cursor(self, commit: bool = False) -> Iterator[RealDictCursor]:
        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                yield cursor
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        self._pool.closeall()

    # --- conversations -----------------------------------------------------

    def create_conversation(self, user_id: str, title: str) -> Conversation:
        with self._cursor(commit=True) as cursor:
            cursor.execute(
                "INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING *",
                (user_id, title),
            )
            row = cursor.fetchone()
        return Conversation.from_dict(dict(row))

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
    ) -> Message:
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}, got {role!r}")
        with self._cursor(commit=True) as cursor:
            cursor.execute(
                "INSERT INTO messages (conversation_id, role, content, token_count) "
                "VALUES (%s, %s, %s, %s) RETURNING *",
                (conversation_id, role, content, token_count),
            )
            row = cursor.fetchone()
        return Message.from_dict(dict(row))

    def list_messages(self, conversation_id: str, limit: int = 100, offset: int = 0) -> list[Message]:
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM messages WHERE conversation_id = %s "
                "ORDER BY created_at ASC LIMIT %s OFFSET %s",
                (conversation_id, limit, offset),
            )
            rows = cursor.fetchall()
        return [Message.from_dict(dict(row)) for row in rows]

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
