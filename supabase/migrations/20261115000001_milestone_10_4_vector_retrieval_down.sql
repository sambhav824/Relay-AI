-- Relay AI Universal Context Engine: Rollback migration for Milestone 10.4 vector retrieval
-- Enforces a fail-closed data safety guard to prevent accidental deletion of embeddings.
--
-- OPERATOR ROLLBACK INSTRUCTIONS:
-- When populated vector embeddings exist, this migration fails closed by default (SQLSTATE 55000).
-- Connection Pooler Notice:
-- In Supabase environments utilizing connection poolers (PgBouncer or Supavisor) in transaction
-- pooling mode (e.g., port 6543), session-level `SET` commands do not persist reliably across statements.
-- Operators MUST execute this rollback within an explicit transaction block using `SET LOCAL`:
--
--   BEGIN;
--   SET LOCAL relay.force_vector_rollback = 'true';
--   \i supabase/migrations/20261115000001_milestone_10_4_vector_retrieval_down.sql
--   COMMIT;
--
-- Or execute as a single transactional script within the Supabase SQL editor or migration runner.

-- 1. DATA SAFETY GUARD: Abort if vector embeddings currently exist in document_chunks
DO $$
DECLARE
  v_chunk_count bigint;
  v_force_param text;
BEGIN
  -- Check if any chunk currently possesses an embedding vector
  SELECT count(*) INTO v_chunk_count
  FROM public.document_chunks
  WHERE embedding_1536 IS NOT NULL OR embedding_768 IS NOT NULL;

  -- Check for explicit operator bypass setting in current transaction/session
  BEGIN
    v_force_param := current_setting('relay.force_vector_rollback', true);
  EXCEPTION WHEN OTHERS THEN
    v_force_param := NULL;
  END;

  IF v_chunk_count > 0 AND (v_force_param IS NULL OR lower(v_force_param) NOT IN ('true', '1', 'yes')) THEN
    RAISE EXCEPTION 'DATA SAFETY GUARD ABORT: public.document_chunks contains % row(s) with populated vector embeddings. '
                    'Executing this rollback migration will permanently destroy all stored vector embeddings. '
                    'To force rollback under connection pooling (PgBouncer/Supavisor), wrap execution in an explicit transaction: '
                    'BEGIN; SET LOCAL relay.force_vector_rollback = ''true''; \i path/to/down.sql; COMMIT;',
                    v_chunk_count
      USING ERRCODE = '55000'; -- object_not_in_prerequisite_state
  END IF;
END $$;

-- 2. Drop vector similarity RPC function
DROP FUNCTION IF EXISTS public.match_document_chunks(
  text, integer, extensions.vector, extensions.vector, double precision, integer, uuid, uuid
);

-- 3. Drop canonical model-specific HNSW indexes
DROP INDEX IF EXISTS public.idx_chunks_hnsw_openai_1536;
DROP INDEX IF EXISTS public.idx_chunks_hnsw_openai_768;
DROP INDEX IF EXISTS public.idx_chunks_hnsw_nomic_768;

-- 4. Drop check constraints
ALTER TABLE public.document_chunks
  DROP CONSTRAINT IF EXISTS chk_document_chunks_vector_consistency,
  DROP CONSTRAINT IF EXISTS chk_document_chunks_embedding_dimension;

-- 5. Drop vector and metadata columns
ALTER TABLE public.document_chunks
  DROP COLUMN IF EXISTS embedding_1536,
  DROP COLUMN IF EXISTS embedding_768,
  DROP COLUMN IF EXISTS embedding_model,
  DROP COLUMN IF EXISTS embedding_dimension;

-- Note: We intentionally preserve the vector extension because other schemas or extensions may depend on it.
