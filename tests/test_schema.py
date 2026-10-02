import os
from pathlib import Path
import subprocess
import sys
import zipfile

from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import RealDictCursor

from byteforge_converse_core.schema import apply_schema
from conftest import connect_test


def test_installed_wheel_storage_import_without_llm(tmp_path: Path) -> None:
    project = Path(__file__).parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hatchling",
            "build",
            "-t",
            "wheel",
            "-d",
            str(tmp_path),
        ],
        cwd=project,
        check=True,
        capture_output=True,
    )
    wheel = next(tmp_path.glob("*.whl"))
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
    env = os.environ.copy()
    env.pop("BYTEFORGE_CONVERSE_OPENROUTER_API_KEY", None)
    env["PYTHONPATH"] = str(installed)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class NoLLM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname.startswith("openrouter_client"):
            raise AssertionError("storage imported an LLM client")
sys.meta_path.insert(0, NoLLM())
from byteforge_converse_core.storage import Database, DatabaseConfig, get_schema_sql
import byteforge_converse_core
assert "/installed/" in byteforge_converse_core.__file__
assert "CREATE TABLE IF NOT EXISTS messages" in get_schema_sql()
assert "openrouter_client" not in sys.modules
""",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )


def test_upgrade_preserves_history_and_is_rerunnable(
    pg_schema: tuple[str, str],
) -> None:
    conn = connect_test(pg_schema)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                (Path(__file__).parent / "fixtures/schema_0_8_1.sql").read_text()
            )
            cur.execute(
                "INSERT INTO conversations (user_id, title) VALUES ('owner', 'old') RETURNING id"
            )
            cid = cur.fetchone()["id"]
            for role, content in [
                ("user", "hello"),
                ("assistant", ""),
                ("tool", "result"),
            ]:
                cur.execute(
                    "INSERT INTO messages (conversation_id, role, content, created_at, tool_calls, tool_call_id) VALUES (%s,%s,%s,100,%s,%s)",
                    (
                        cid,
                        role,
                        content,
                        (
                            '[{"id":"call1","function":{"name":"lookup","arguments":"{}"}}]'
                            if role == "assistant"
                            else None
                        ),
                        "call1" if role == "tool" else None,
                    ),
                )
            cur.execute("SELECT * FROM messages ORDER BY created_at, seq")
            before = cur.fetchall()
        conn.commit()
        apply_schema(conn)
        apply_schema(conn)
        conn.commit()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM messages ORDER BY created_at, seq")
            after = cur.fetchall()
        assert [{key: row[key] for key in before[0]} for row in after] == before
    finally:
        conn.close()


def test_schema_does_not_commit(connection: PgConnection) -> None:
    with connection.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("INSERT INTO conversations (user_id,title) VALUES ('a','rollback')")
    apply_schema(connection)
    connection.rollback()
    with connection.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM conversations")
        assert cur.fetchall() == []


def test_fresh_process_storage_use_never_imports_llm(
    pg_schema: tuple[str, str],
) -> None:
    env = os.environ.copy()
    env["STORAGE_TEST_DSN"], env["STORAGE_TEST_SCHEMA"] = pg_schema
    env.pop("BYTEFORGE_CONVERSE_OPENROUTER_API_KEY", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import os
import sys
class NoLLM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname.startswith("openrouter_client"):
            raise AssertionError("storage imported an LLM client")
sys.meta_path.insert(0, NoLLM())
import psycopg2
from byteforge_converse_core.storage import Repository, apply_schema
from byteforge_converse_models import ConversationCreate
conn = psycopg2.connect(os.environ["STORAGE_TEST_DSN"], options="-c search_path=" + os.environ["STORAGE_TEST_SCHEMA"] + ",public")
try:
    with conn:
        apply_schema(conn)
        repo = Repository(conn).for_owner("owner")
        conversation = repo.create_conversation(ConversationCreate(user_id="owner", title="storage only"))
        result = repo.append_message(conversation.id, "user", "hello", producer="test", delivery_key="k")
        assert repo.list_messages(conversation.id)[0].id == result.message.id
finally:
    conn.close()
assert "openrouter_client" not in sys.modules
""",
        ],
        env=env,
        check=True,
        capture_output=True,
    )
