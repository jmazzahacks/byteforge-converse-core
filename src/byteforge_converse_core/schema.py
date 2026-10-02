"""Canonical, installed schema and additive upgrades; caller owns the transaction."""

import sys
from importlib.resources import files

from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import RealDictCursor


def get_schema_sql() -> str:
    """Return re-runnable bootstrap/upgrade SQL from installed package resources."""
    return (
        files("byteforge_converse_core")
        .joinpath("sql/schema.sql")
        .read_text(encoding="utf-8")
    )


def apply_schema(connection: PgConnection) -> None:
    """Apply SQL using caller credentials; never commit, roll back or close connection.

    Use a non-autocommit connection and commit explicitly after success. The
    caller needs DDL privileges in its chosen search_path and database.
    """
    with connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(get_schema_sql())


if __name__ == "__main__":
    sys.stdout.write(get_schema_sql())
