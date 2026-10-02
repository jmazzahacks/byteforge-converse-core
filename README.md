# byteforge-converse-core

Postgres conversation storage and optional OpenRouter chat orchestration. Version
0.9.0 adds embedded storage without changing existing HTTP/wire models or chat
imports. Authentication, consumer run/event tables and provider-native transcripts
remain the consuming application's responsibility.

## Install and import

```bash
pip install 'git+https://github.com/jmazzahacks/byteforge-converse-core.git'
```

All dependencies are public GitHub/PyPI packages; no token is needed. Existing
OpenRouter dependencies remain installed for compatibility. There is no lean
installation extra in this release. Storage imports do not load OpenRouter,
read an LLM key, instantiate a client, or make network requests. Only explicitly
constructing `Database` connects to the configured database.

```python
from byteforge_converse_core.storage import Database, DatabaseConfig, Repository

# Legacy chat imports still work, and load the chat client lazily:
from byteforge_converse_core import ChatService, LLMConfig
```

## Bootstrap and upgrades

The installed `byteforge_converse_core/sql/schema.sql` is the single maintained
schema source, including re-runnable additive upgrades. It creates conversations,
messages and frontend handshake sessions. Those sessions are **not** provider-native
sessions; Claude-native transcripts and SDK dependencies do not belong here.

The caller provisions its database/role and supplies DDL-capable credentials.
`apply_schema` neither commits, rolls back, closes connections nor creates roles
or databases. Run setup explicitly during provisioning, outside application
transactions. It takes normal PostgreSQL DDL/index locks; schedule upgrades
appropriately for database size and traffic. The caller controls `search_path`.
Use a UTF-8 database. PostgreSQL 18 is covered by the integration suite.

```python
import psycopg2
from byteforge_converse_core.storage import DatabaseConfig, apply_schema

config = DatabaseConfig.from_env()  # BYTEFORGE_CONVERSE_DB_* only
connection = psycopg2.connect(
    host=config.host, port=config.port, dbname=config.name,
    user=config.user, password=config.password,
)
try:
    with connection:  # psycopg2 caller context owns commit/rollback
        apply_schema(connection)
finally:
    connection.close()
```

For consumer-managed pools (including a PgCat endpoint), pass that pool's normal
psycopg2 connection instead. No hard-coded host, credentials or provider settings
are introduced. `get_schema_sql()` returns the installed SQL as text; export it
without any backend checkout:

```bash
python -m byteforge_converse_core.schema > /tmp/converse-schema.sql
psql -v ON_ERROR_STOP=1 -f /tmp/converse-schema.sql
```

Release order: install core >=0.9.0, apply its schema once, then enable new storage
operations or update the backend setup script. The backend's
`dev_scripts/setup_database.py` calls this packaged API directly; its old
`database/schema.sql` now fails with migration guidance rather than maintaining
a drifting copy. Existing legacy CRUD still works before the additive upgrade;
`append_message` requires the new columns/index. Existing IDs, content, tool
history, timestamps and seq values are preserved. Old deployments that predate
seq retain the historical physical-order backfill limitation for tied timestamps.
No production migration is performed by installing/importing this package.

## Pooled convenience and managed transactions

```python
from byteforge_converse_models import ConversationCreate
from byteforge_converse_core.storage import Database, DatabaseConfig

storage = Database(DatabaseConfig.from_env())  # construct after worker fork
try:
    owned = storage.for_owner("authenticated-user-id")
    conversation = owned.create_conversation(
        ConversationCreate(user_id="authenticated-user-id", title="Discussion")
    )
    with storage.transaction() as transaction:
        repo = transaction.for_owner("authenticated-user-id")
        repo.create_message(conversation.id, "user", "Hello")
        repo.touch_conversation(conversation.id, 1790938000)
finally:
    storage.close()
```

Ordinary `Database` calls commit writes independently and end read transactions.
`Database.transaction()` checks out/pre-pings once, commits on successful exit,
rolls back on failure, and returns/discards the connection through the existing
recovery path. Its repository and derived owner facades expire on context exit.
Neither its body nor an ambiguous commit is automatically replayed.

## Borrow a consumer-owned transaction

`Repository(connection)` accepts a non-autocommit psycopg2 connection, including
one with writes already in progress. It opens/closes only its own RealDictCursor;
it never pre-pings, commits, rolls back, closes the connection, changes isolation,
or returns it to a pool. Errors propagate. The caller owns the entire lifetime
and must discard the repository when its transaction ends. Borrowed cursors are
not a separate public API; supply their connection instead.

For example, a Rivet-style worker resolves authentication first, then atomically
maps its event to a portable message in its **own** event table:

```python
from psycopg2.extras import RealDictCursor
from byteforge_converse_core.storage import Repository

connection = consumer_pool.getconn()
try:
    with connection:
        repo = Repository(connection).for_owner(authenticated_owner_id)
        result = repo.append_message(
            conversation_id, "assistant", response_text,
            producer="worker", delivery_key=event_id,
        )
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "UPDATE consumer_events SET message_id = %s WHERE id = %s",
                (result.message.id, event_id),
            )
finally:
    consumer_pool.putconn(connection)
```

Each task must borrow its own connection/repository. Calls are synchronous: async
workers should offload the **whole transaction** to a thread with a connection
acquired/returned there. Do not scatter individual calls across workers sharing a
transaction. Finish LLM calls and external tools before opening these transactions.
Do not pass a transaction-bound repository to `ChatService`.

## Idempotent append

`create_message` still creates a fresh message every call. Opt-in `append_message`
uses the exact `(conversation_id, producer, delivery_key)` as its identity.
Producer/key are nonblank opaque strings, limited to 128/512 UTF-8 bytes. Keys
are case-sensitive and are not trimmed. Equal text under different keys is distinct.

The returned storage-only `AppendResult` has `message` (the existing wire model),
`created`, `seq`, `producer`, and `delivery_key`. Save `message.id` in consumer
records for event correlation; producer/key are a durable retry lookup while that
message exists. Neither identity columns nor seq are added to HTTP/client models.

An identical retry returns the same message/seq with `created=False`; uniqueness
is enforced by a PostgreSQL index. Different role, content, token_count, tool_calls,
or tool_call_id raises `IdempotencyConflict` without overwriting history. JSONB
comparison ignores object key ordering but preserves array order and JSON types.
A changed token count is a conflict too; retries must carry the original payload.

There is no TTL/tombstone: deleting a message frees its key; deleting a conversation
cascades to its messages/keys. A new delivery after deletion can create a new ID.
Concurrent deletion during retry lookup can raise `ConcurrentMessageChange`.
Deadlock/serialization errors propagate; callers may retry the **whole** transaction
where safe. An ambiguous acknowledgement can be reconciled with the same key and
payload. Converse never automatically repeats consumer work or external effects.

## Replay and concurrency

History remains ordered by `(created_at ASC, seq ASC)`. `created_at` is integer
epoch seconds (Postgres transaction-start time); seq breaks same-second ties and
can contain gaps. It is **not** commit order or a safe incremental watermark.
Concurrent transactions can commit in a different order from sequence allocation.

`list_messages(..., limit=None)` reads a complete statement snapshot. For multiple
pages, use one read-only REPEATABLE READ transaction, with no writes by that reader,
and increment offset by the number of rows returned. Configure isolation before
starting the transaction, and restore pool settings before returning the connection.

```python
connection.set_session(isolation_level="REPEATABLE READ", readonly=True)
try:
    with connection:
        repo = Repository(connection).for_owner(authenticated_owner_id)
        offset = 0
        while True:
            page = repo.list_messages(conversation_id, limit=100, offset=offset)
            if not page:
                break
            consume_page(page)
            offset += len(page)
finally:
    connection.set_session(isolation_level="READ COMMITTED", readonly=False)
```

That replay includes exactly the rows visible to its snapshot. Later commits
appear in a new full replay, which consumers can reconcile by message UUID.
Separate READ COMMITTED pages may shift as other transactions commit/delete rows;
do not use them as an exactly-once live feed or checkpoint at max(seq).
See PostgreSQL's [transaction isolation documentation](https://www.postgresql.org/docs/current/transaction-iso.html).

## Owner-scoped access and compatibility

`for_owner(owner_id)` scopes conversation create/get/list/touch/delete and message
list/create/append/delete, including duplicate lookup. Blank/missing identity is
rejected; a scoped facade cannot be re-scoped to a different owner. The consumer
must authenticate that identity first. A user-supplied header, body or query value
alone is not authentication. No passwords, tokens, Aegis or OAuth checks are added.

Owner predicates are evaluated in SQL. Appends hold a conversation `FOR SHARE`
lock through insertion/retry lookup so ownership cannot change between the check
and write. Keep transactions short; transactions that also upgrade those locks
(e.g. touch/delete the same conversation) may deadlock with concurrent writers,
so callers should handle PostgreSQL deadlock errors at the transaction boundary.

Absent and cross-owner resources have the same behavior: conversation get returns
None, message list returns [], delete returns False, touch is a no-op, and message
create/append raises `ResourceNotFound`. Scoped conversation creation or listing
with a different explicit owner raises ValueError. Owner IDs are opaque and compared
exactly. This facade covers conversations/messages; frontend handshake session
methods remain on the trusted-service `Database` API.

Unscoped legacy methods and HTTP routes remain available for trusted-service
compatibility. They are not automatically converted into authenticated owner access.
Consumers still enforce authenticated identity, any shared-resource/tenant policy,
allowed providers/tools, and permissions for their own run/event records. A holder
of database credentials can use unscoped SQL; this API is not PostgreSQL RLS.

## Development and validation

```bash
python -m venv .
source bin/activate
pip install -r dev-requirements.txt
pip install -e .
pytest -q
# Use an explicitly disposable database; tests create/drop isolated schemas:
CONVERSE_TEST_DSN='host=/tmp/test-pg port=55439 dbname=converse_test' pytest -q
```

Cross-repo dependencies always install from GitHub. Tests cover installed-wheel
storage imports, chat/tool-result regression, pooled recovery, schema upgrades,
consumer transaction atomicity, concurrent duplicate delivery, payload conflicts,
late-commit snapshot replay and owner-scoped writes/retries. The historical SQL
under tests/fixtures is an immutable upgrade input, not a maintained schema copy.

## License

O'Saasy License — see [LICENSE](LICENSE).
