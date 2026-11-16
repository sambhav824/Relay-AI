-- Relay AI Universal Context Engine: Milestone 10.4 pgvector retrieval migration
-- Enables pgvector, adds dual-dimension vector columns, check constraints,
-- model-specific partial HNSW indexes, and secure vector similarity search RPC.

-- 1. Enable pgvector extension safely
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- 2. Add dual-dimension embedding columns and metadata to public.document_chunks
ALTER TABLE public.document_chunks
  ADD COLUMN IF NOT EXISTS embedding_1536 extensions.vector(1536),
  ADD COLUMN IF NOT EXISTS embedding_768 extensions.vector(768),
  ADD COLUMN IF NOT EXISTS embedding_model text,
  ADD COLUMN IF NOT EXISTS embedding_dimension integer;

-- 3. Add integrity constraints
-- Validates dimension value is either 768, 1536, or NULL
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'chk_document_chunks_embedding_dimension'
  ) THEN
    ALTER TABLE public.document_chunks
      ADD CONSTRAINT chk_document_chunks_embedding_dimension
      CHECK (embedding_dimension IN (768, 1536) OR embedding_dimension IS NULL);
  END IF;
END $$;

-- Validates vector-dimension consistency (prevents mismatched dimensions or dual population)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'chk_document_chunks_vector_consistency'
  ) THEN
    ALTER TABLE public.document_chunks
      ADD CONSTRAINT chk_document_chunks_vector_consistency
      CHECK (
        (embedding_dimension = 1536 AND embedding_1536 IS NOT NULL AND embedding_768 IS NULL) OR
        (embedding_dimension = 768 AND embedding_768 IS NOT NULL AND embedding_1536 IS NULL) OR
        (embedding_dimension IS NULL AND embedding_1536 IS NULL AND embedding_768 IS NULL)
      );
  END IF;
END $$;

-- 4. Canonical Model-Specific Partial HNSW Indexes
-- Note: Avoids overlapping generic fallback indexes to eliminate double write-amplification and index bloat.
-- 4.1 OpenAI text-embedding-3-small (1536 dimensions)
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_openai_1536
  ON public.document_chunks
  USING hnsw (embedding_1536 extensions.vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE embedding_model = 'openai/text-embedding-3-small'
    AND embedding_dimension = 1536
    AND embedding_1536 IS NOT NULL;

-- 4.2 OpenAI text-embedding-3-small (768 dimensions via MRL)
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_openai_768
  ON public.document_chunks
  USING hnsw (embedding_768 extensions.vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE embedding_model = 'openai/text-embedding-3-small'
    AND embedding_dimension = 768
    AND embedding_768 IS NOT NULL;

-- 4.3 Ollama nomic-embed-text (768 dimensions)
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_nomic_768
  ON public.document_chunks
  USING hnsw (embedding_768 extensions.vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE embedding_model = 'ollama/nomic-embed-text'
    AND embedding_dimension = 768
    AND embedding_768 IS NOT NULL;

-- 5. Secure Vector Similarity Search RPC
CREATE OR REPLACE FUNCTION public.match_document_chunks(
  p_model_id text,
  p_dimension int,
  query_embedding_1536 extensions.vector(1536) DEFAULT NULL,
  query_embedding_768 extensions.vector(768) DEFAULT NULL,
  match_threshold double precision DEFAULT 0.3,
  match_count int DEFAULT 10,
  filter_project_id uuid DEFAULT NULL,
  filter_document_id uuid DEFAULT NULL
)
RETURNS TABLE (
  id uuid,
  document_id uuid,
  project_id uuid,
  chunk_index int,
  content text,
  metadata jsonb,
  similarity double precision
)
LANGUAGE plpgsql
STABLE
SECURITY INVOKER
SET search_path = public, extensions
AS $$
DECLARE
  v_user_id uuid;
BEGIN
  -- 1. Authentication Check: Ensure caller has a valid authenticated session
  v_user_id := auth.uid();
  IF v_user_id IS NULL THEN
    RAISE EXCEPTION 'Authentication required: auth.uid() is null.'
      USING ERRCODE = '28000';
  END IF;

  -- 2. Input Validation: Validate model_id and dimension parameters
  IF p_model_id IS NULL OR length(trim(p_model_id)) = 0 THEN
    RAISE EXCEPTION 'p_model_id must not be empty.'
      USING ERRCODE = '22023';
  END IF;

  IF p_dimension NOT IN (768, 1536) THEN
    RAISE EXCEPTION 'Invalid dimension: %. Allowed dimensions are 768 and 1536.', p_dimension
      USING ERRCODE = '22023';
  END IF;

  -- 3. Input Validation: Mutually exclusive query vector input
  IF query_embedding_1536 IS NOT NULL AND query_embedding_768 IS NOT NULL THEN
    RAISE EXCEPTION 'Cannot provide both query_embedding_1536 and query_embedding_768. Pass exactly one matching p_dimension.'
      USING ERRCODE = '22023';
  END IF;

  IF query_embedding_1536 IS NULL AND query_embedding_768 IS NULL THEN
    RAISE EXCEPTION 'Missing query vector. Pass query_embedding_1536 for dimension 1536 or query_embedding_768 for dimension 768.'
      USING ERRCODE = '22023';
  END IF;

  -- 4. Input Validation: Dimension-Vector compatibility
  IF p_dimension = 1536 AND query_embedding_1536 IS NULL THEN
    RAISE EXCEPTION 'Dimension mismatch: p_dimension is 1536 but query_embedding_1536 is NULL.'
      USING ERRCODE = '22023';
  END IF;

  IF p_dimension = 768 AND query_embedding_768 IS NULL THEN
    RAISE EXCEPTION 'Dimension mismatch: p_dimension is 768 but query_embedding_768 is NULL.'
      USING ERRCODE = '22023';
  END IF;

  -- 5. Input Validation: Model-Dimension compatibility
  IF p_model_id = 'ollama/nomic-embed-text' AND p_dimension = 1536 THEN
    RAISE EXCEPTION 'Model % does not support 1536 dimensions. Nomic natively supports 768.', p_model_id
      USING ERRCODE = '22023';
  END IF;

  -- 6. Bounds Validation: Threshold and limits
  IF match_threshold < -1.0 OR match_threshold > 1.0 THEN
    RAISE EXCEPTION 'match_threshold must be between -1.0 and 1.0, got %.', match_threshold
      USING ERRCODE = '22023';
  END IF;

  IF match_count < 1 OR match_count > 100 THEN
    RAISE EXCEPTION 'match_count must be between 1 and 100, got %.', match_count
      USING ERRCODE = '22023';
  END IF;

  -- 7. Configure local HNSW exploration depth
  SET LOCAL hnsw.ef_search = 100;

  -- 8. Execute similarity search on the validated vector column using static branches
  -- Static model literal predicates guarantee PostgreSQL's query planner utilizes
  -- the corresponding partial HNSW index, avoiding generic-plan parameterization de-optimizations.
  IF p_dimension = 1536 AND p_model_id = 'openai/text-embedding-3-small' THEN
    -- Canonical Branch 1: OpenAI 1536d (statically matches idx_chunks_hnsw_openai_1536)
    RETURN QUERY
      SELECT
        c.id,
        c.document_id,
        d.project_id,
        c.chunk_index,
        c.content,
        c.metadata,
        (1.0 - (c.embedding_1536 <=> query_embedding_1536))::double precision AS similarity
      FROM public.document_chunks c
      JOIN public.documents d ON d.id = c.document_id AND d.user_id = c.user_id
      WHERE c.user_id = v_user_id
        AND c.embedding_model = 'openai/text-embedding-3-small'
        AND c.embedding_dimension = 1536
        AND c.embedding_1536 IS NOT NULL
        AND (filter_project_id IS NULL OR d.project_id = filter_project_id)
        AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
        AND (1.0 - (c.embedding_1536 <=> query_embedding_1536)) >= match_threshold
      ORDER BY c.embedding_1536 <=> query_embedding_1536 ASC
      LIMIT match_count;

  ELSIF p_dimension = 768 AND p_model_id = 'openai/text-embedding-3-small' THEN
    -- Canonical Branch 2: OpenAI 768d MRL (statically matches idx_chunks_hnsw_openai_768)
    RETURN QUERY
      SELECT
        c.id,
        c.document_id,
        d.project_id,
        c.chunk_index,
        c.content,
        c.metadata,
        (1.0 - (c.embedding_768 <=> query_embedding_768))::double precision AS similarity
      FROM public.document_chunks c
      JOIN public.documents d ON d.id = c.document_id AND d.user_id = c.user_id
      WHERE c.user_id = v_user_id
        AND c.embedding_model = 'openai/text-embedding-3-small'
        AND c.embedding_dimension = 768
        AND c.embedding_768 IS NOT NULL
        AND (filter_project_id IS NULL OR d.project_id = filter_project_id)
        AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
        AND (1.0 - (c.embedding_768 <=> query_embedding_768)) >= match_threshold
      ORDER BY c.embedding_768 <=> query_embedding_768 ASC
      LIMIT match_count;

  ELSIF p_dimension = 768 AND p_model_id = 'ollama/nomic-embed-text' THEN
    -- Canonical Branch 3: Ollama Nomic 768d (statically matches idx_chunks_hnsw_nomic_768)
    RETURN QUERY
      SELECT
        c.id,
        c.document_id,
        d.project_id,
        c.chunk_index,
        c.content,
        c.metadata,
        (1.0 - (c.embedding_768 <=> query_embedding_768))::double precision AS similarity
      FROM public.document_chunks c
      JOIN public.documents d ON d.id = c.document_id AND d.user_id = c.user_id
      WHERE c.user_id = v_user_id
        AND c.embedding_model = 'ollama/nomic-embed-text'
        AND c.embedding_dimension = 768
        AND c.embedding_768 IS NOT NULL
        AND (filter_project_id IS NULL OR d.project_id = filter_project_id)
        AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
        AND (1.0 - (c.embedding_768 <=> query_embedding_768)) >= match_threshold
      ORDER BY c.embedding_768 <=> query_embedding_768 ASC
      LIMIT match_count;

  ELSIF p_dimension = 1536 THEN
    -- Fallback Branch 4: Custom or newly provisioned 1536d models
    RETURN QUERY
      SELECT
        c.id,
        c.document_id,
        d.project_id,
        c.chunk_index,
        c.content,
        c.metadata,
        (1.0 - (c.embedding_1536 <=> query_embedding_1536))::double precision AS similarity
      FROM public.document_chunks c
      JOIN public.documents d ON d.id = c.document_id AND d.user_id = c.user_id
      WHERE c.user_id = v_user_id
        AND c.embedding_model = p_model_id
        AND c.embedding_dimension = 1536
        AND c.embedding_1536 IS NOT NULL
        AND (filter_project_id IS NULL OR d.project_id = filter_project_id)
        AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
        AND (1.0 - (c.embedding_1536 <=> query_embedding_1536)) >= match_threshold
      ORDER BY c.embedding_1536 <=> query_embedding_1536 ASC
      LIMIT match_count;

  ELSIF p_dimension = 768 THEN
    -- Fallback Branch 5: Custom or newly provisioned 768d models
    RETURN QUERY
      SELECT
        c.id,
        c.document_id,
        d.project_id,
        c.chunk_index,
        c.content,
        c.metadata,
        (1.0 - (c.embedding_768 <=> query_embedding_768))::double precision AS similarity
      FROM public.document_chunks c
      JOIN public.documents d ON d.id = c.document_id AND d.user_id = c.user_id
      WHERE c.user_id = v_user_id
        AND c.embedding_model = p_model_id
        AND c.embedding_dimension = 768
        AND c.embedding_768 IS NOT NULL
        AND (filter_project_id IS NULL OR d.project_id = filter_project_id)
        AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
        AND (1.0 - (c.embedding_768 <=> query_embedding_768)) >= match_threshold
      ORDER BY c.embedding_768 <=> query_embedding_768 ASC
      LIMIT match_count;
  END IF;
END;
$$;

-- 6. Permissions: Grant execution to authenticated role only
REVOKE ALL ON FUNCTION public.match_document_chunks(
  text, integer, extensions.vector, extensions.vector, double precision, integer, uuid, uuid
) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION public.match_document_chunks(
  text, integer, extensions.vector, extensions.vector, double precision, integer, uuid, uuid
) TO authenticated;
