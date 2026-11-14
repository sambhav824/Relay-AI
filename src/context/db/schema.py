"""Relay AI Universal Context Engine — Database Schema & DDL.

Defines the relational tables, FTS5 virtual tables, triggers, and indexes
for persistent projects, conversations, messages, memories, documents, and chunks.
"""

SCHEMA_VERSION = 1

CORE_TABLES_DDL = """
-- Schema Migrations Tracking
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT NOT NULL
);

-- 1. Projects
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    root_path TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT DEFAULT '{}'
);

-- 2. Conversations
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT 'global',
    title TEXT NOT NULL DEFAULT 'New Conversation',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_id TEXT,
    system_prompt TEXT,
    is_archived INTEGER DEFAULT 0,
    metadata_json TEXT DEFAULT '{}',
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- 3. Messages
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    token_count INTEGER DEFAULT 0,
    model_id TEXT,
    metadata_json TEXT DEFAULT '{}',
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

-- 4. Persistent Memories
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT 'global',
    category TEXT NOT NULL CHECK(category IN ('USER_PREFERENCE', 'PROJECT_FACT', 'DOMAIN_TERM', 'AGENT_FEEDBACK')),
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    provenance TEXT NOT NULL CHECK(provenance IN ('USER_EXPLICIT', 'AI_INFERRED')),
    confidence REAL DEFAULT 1.0,
    is_approved INTEGER DEFAULT 1,
    source_attribution TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- 5. Documents
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT 'global',
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    mime_type TEXT NOT NULL,
    page_count INTEGER DEFAULT 1,
    status TEXT NOT NULL CHECK(status IN ('PENDING', 'PROCESSING', 'INDEXED', 'FAILED', 'NEEDS_OCR')),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT DEFAULT '{}',
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- 6. Document Chunks
CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT 'global',
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_estimate INTEGER DEFAULT 0,
    page_number INTEGER DEFAULT 1,
    metadata_json TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
"""

FTS5_TABLES_DDL = """
-- 7. FTS5 Virtual Tables for BM25 Search
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    project_id UNINDEXED,
    content,
    tokenize = 'porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    project_id UNINDEXED,
    category UNINDEXED,
    content,
    tokenize = 'porter unicode61'
);
"""

TRIGGERS_DDL = """
-- Triggers for document_chunks <-> chunks_fts synchronization
CREATE TRIGGER IF NOT EXISTS trg_chunks_insert AFTER INSERT ON document_chunks BEGIN
    INSERT INTO chunks_fts(chunk_id, document_id, project_id, content)
    VALUES (new.id, new.document_id, new.project_id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS trg_chunks_delete AFTER DELETE ON document_chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_chunks_update AFTER UPDATE ON document_chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.id;
    INSERT INTO chunks_fts(chunk_id, document_id, project_id, content)
    VALUES (new.id, new.document_id, new.project_id, new.content);
END;

-- Triggers for memories <-> memories_fts synchronization
CREATE TRIGGER IF NOT EXISTS trg_memories_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(memory_id, project_id, category, content)
    VALUES (new.id, new.project_id, new.category, new.content);
END;

CREATE TRIGGER IF NOT EXISTS trg_memories_delete AFTER DELETE ON memories BEGIN
    DELETE FROM memories_fts WHERE memory_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_memories_update AFTER UPDATE ON memories BEGIN
    DELETE FROM memories_fts WHERE memory_id = old.id;
    INSERT INTO memories_fts(memory_id, project_id, category, content)
    VALUES (new.id, new.project_id, new.category, new.content);
END;
"""

INDEXES_DDL = """
-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_conversations_project ON conversations(project_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id, category, is_approved);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id, status);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_project ON document_chunks(project_id);
"""

DEFAULT_DATA_DML = """
-- Default Global Project
INSERT OR IGNORE INTO projects (id, name, description)
VALUES ('global', 'Global / General', 'Default context workspace across non-project conversations');
"""
