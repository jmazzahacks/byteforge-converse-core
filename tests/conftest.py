"""Opt-in integration tests against an explicitly supplied disposable database."""

import os
import uuid
from collections.abc import Iterator

import psycopg2
import pytest
from psycopg2 import sql
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import RealDictCursor


@pytest.fixture
def pg_schema() -> Iterator[tuple[str, str]]:
    dsn = os.environ.get("CONVERSE_TEST_DSN")
    if not dsn:
        pytest.skip("Set CONVERSE_TEST_DSN to an isolated test Postgres database")
    schema = "test_" + uuid.uuid4().hex
    admin = psycopg2.connect(dsn)
    admin.autocommit = True
    with admin.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield dsn, schema
    finally:
        with admin.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )
        admin.close()


def connect_test(pg_schema: tuple[str, str]) -> PgConnection:
    dsn, schema = pg_schema
    return psycopg2.connect(dsn, options=f"-c search_path={schema},public")


@pytest.fixture
def connection(pg_schema: tuple[str, str]) -> Iterator[PgConnection]:
    from byteforge_converse_core.schema import apply_schema

    conn = connect_test(pg_schema)
    try:
        apply_schema(conn)
        conn.commit()
        yield conn
    finally:
        conn.close()


@pytest.fixture
def database(
    connection: PgConnection,
    pg_schema: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[object]:
    from byteforge_converse_core.storage import Database, DatabaseConfig
    from psycopg2.pool import ThreadedConnectionPool

    def test_pool(
        minconn: int, maxconn: int, **kwargs: object
    ) -> ThreadedConnectionPool:
        kwargs["options"] = f"-c search_path={pg_schema[1]},public"
        return ThreadedConnectionPool(minconn, maxconn, **kwargs)

    monkeypatch.setattr(
        "byteforge_converse_core.database.ThreadedConnectionPool", test_pool
    )
    params = connection.get_dsn_parameters()
    db = Database(
        DatabaseConfig(
            params["host"], int(params["port"]), params["dbname"], params["user"], ""
        )
    )
    try:
        yield db
    finally:
        db.close()
