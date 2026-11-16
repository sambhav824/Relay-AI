-- Relay AI Universal Context Engine: Rollback migration for Milestone 10.4 full-text search RPC
-- Drops public.search_document_chunks_fts. Does not alter table columns or stored document chunks.

-- 1. Drop Full-Text Search RPC function
DROP FUNCTION IF EXISTS public.search_document_chunks_fts(
  text, integer, uuid, uuid
);
