-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- conversations: one row per chat thread between a user and the LLM.
-- user_id is an opaque identifier from the consuming app (auth is delegated).
CREATE TABLE IF NOT EXISTS conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL,
    title           TEXT NOT NULL,
    model           TEXT,
    system_prompt   TEXT,
    response_schema JSONB,
    tools           JSONB,
    created_at      BIGINT NOT NULL DEFAULT extract(epoch from now())::bigint,
    updated_at      BIGINT
);

-- Idempotent upgrades for databases created before these columns existed
-- (there is no migration framework yet; the setup script is re-runnable).
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS model           TEXT;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS system_prompt   TEXT;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS response_schema JSONB;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS tools           JSONB;

-- Defense-in-depth: tools must be a JSON array (or NULL). Catches manual
-- writes and future code paths that would otherwise propagate a malformed
-- value straight to the LLM client. Idempotent via DO block.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'conversations_tools_is_array'
    ) THEN
        ALTER TABLE conversations
            ADD CONSTRAINT conversations_tools_is_array
            CHECK (tools IS NULL OR jsonb_typeof(tools) = 'array');
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_conversations_user_id    ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);

COMMENT ON TABLE  conversations            IS 'A single chat thread between a user and the LLM';
COMMENT ON COLUMN conversations.user_id    IS 'Opaque user id from the consuming app — auth is delegated';
COMMENT ON COLUMN conversations.model      IS 'OpenRouter model id for this conversation; NULL falls back to BYTEFORGE_CONVERSE_LLM_MODEL';
COMMENT ON COLUMN conversations.system_prompt   IS 'Per-conversation system prompt encoding its purpose (form-filling, freeform, etc.)';
COMMENT ON COLUMN conversations.response_schema IS 'JSON schema for structured-output (JSON construction) conversations; NULL for freeform';
COMMENT ON COLUMN conversations.tools           IS 'Opaque list of OpenAI/OpenRouter tool definitions advertised to the model; tool calls are relayed back to the caller, never executed by Converse';
COMMENT ON COLUMN conversations.created_at IS 'Unix timestamp (epoch seconds) of creation';
COMMENT ON COLUMN conversations.updated_at IS 'Unix timestamp (epoch seconds) of last update';

-- messages: ordered turns within a conversation.
-- role mirrors the LLM convention (user / assistant / system / tool).
--
-- tool_calls (assistant rows only) stores the raw OpenAI/OpenRouter tool-call
-- list the model emitted on this turn, so a future replay carries the
-- request alongside the corresponding tool result.
-- tool_call_id (tool rows only) ties a tool result back to a specific
-- assistant tool_call, as required by the OpenAI protocol.
CREATE TABLE IF NOT EXISTS messages (
    id              UUID   PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID   NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT   NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content         TEXT   NOT NULL,
    token_count     INTEGER,
    tool_calls      JSONB,
    tool_call_id    TEXT,
    created_at      BIGINT NOT NULL DEFAULT extract(epoch from now())::bigint
);

-- Idempotent upgrades for databases created before these columns existed.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS tool_calls   JSONB;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS tool_call_id TEXT;
-- Insertion order. created_at is epoch seconds and a chat turn writes several
-- rows within one second; history is replayed ORDER BY created_at, seq so a
-- tie can never put an assistant row last or a tool result before its call.
-- Rows that predate the column (production: the first 56, added 2026-09-27)
-- were numbered in PHYSICAL heap order, not write order, and one conversation
-- is visibly shuffled. Never order by seq alone; it is only a tiebreaker
-- within one created_at second, and those old rows have no such ties.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS seq BIGSERIAL NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'messages_tool_calls_is_array'
    ) THEN
        ALTER TABLE messages
            ADD CONSTRAINT messages_tool_calls_is_array
            CHECK (tool_calls IS NULL OR jsonb_typeof(tool_calls) = 'array');
    END IF;
END $$;

-- One index serves every messages lookup: by conversation (incl. the FK
-- cascade from conversations), and history replay ordered by (created_at,
-- seq). The older (conversation_id) and (conversation_id, created_at)
-- indexes are strict prefixes of it and are dropped (athena, 2026-09-27).
CREATE INDEX IF NOT EXISTS idx_messages_conversation_replay ON messages(conversation_id, created_at, seq);
DROP INDEX IF EXISTS idx_messages_conversation_id;
DROP INDEX IF EXISTS idx_messages_conversation_id_created_at;

COMMENT ON TABLE  messages              IS 'A single message within a conversation';
COMMENT ON COLUMN messages.role         IS 'One of: user, assistant, system, tool';
COMMENT ON COLUMN messages.token_count  IS 'Tokens consumed by this message, if known';
COMMENT ON COLUMN messages.tool_calls   IS 'Raw OpenAI/OpenRouter tool-call list emitted by the assistant on this turn (NULL on non-assistant rows or assistant rows that emitted no tools)';
COMMENT ON COLUMN messages.tool_call_id IS 'For tool-role rows, the assistant tool_call id this row responds to';
COMMENT ON COLUMN messages.created_at   IS 'Unix timestamp (epoch seconds) of creation';
COMMENT ON COLUMN messages.seq          IS 'Tiebreaker for rows created in the same second when replaying history: insertion order for rows written after the column was added, physical order for older rows. Never order by seq alone.';

-- sessions: short-lived frontend handshake records.
-- conversation_id may be NULL until the user starts a chat.
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID   PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT   NOT NULL,
    conversation_id UUID            REFERENCES conversations(id) ON DELETE SET NULL,
    created_at      BIGINT NOT NULL DEFAULT extract(epoch from now())::bigint,
    expires_at      BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id    ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);

COMMENT ON TABLE  sessions            IS 'Short-lived frontend handshake sessions';
COMMENT ON COLUMN sessions.user_id    IS 'Opaque user id from the consuming app — auth is delegated';
COMMENT ON COLUMN sessions.created_at IS 'Unix timestamp (epoch seconds) of creation';
COMMENT ON COLUMN sessions.expires_at IS 'Unix timestamp (epoch seconds) after which the session is invalid';
