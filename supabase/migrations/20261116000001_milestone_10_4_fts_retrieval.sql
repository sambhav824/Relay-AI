-- Relay AI Universal Context Engine: Full-Text Search RPC for Document Chunks (Milestone 10.4, Phase 4)
-- Enables secure, authenticated sparse keyword retrieval with project filtering and ranking.

CREATE OR REPLACE FUNCTION public.search_document_chunks_fts(
  p_query text,
  match_count integer DEFAULT 20,
  filter_project_id uuid DEFAULT NULL,
  filter_document_id uuid DEFAULT NULL
)
RETURNS TABLE (
  id uuid,
  document_id uuid,
  project_id uuid,
  chunk_index integer,
  content text,
  metadata jsonb,
  score double precision
)
LANGUAGE plpgsql
STABLE
SECURITY INVOKER
SET search_path = public, extensions
AS $$
DECLARE
  v_user_id uuid;
  v_tsquery tsquery;
BEGIN
  -- 1. Authentication Check: enforce caller identity
  v_user_id := auth.uid();
  IF v_user_id IS NULL THEN
    RAISE EXCEPTION 'Authentication required: search_document_chunks_fts requires an authenticated caller.'
      USING ERRCODE = '28000'; -- invalid_authorization_specification
  END IF;

  -- 2. Input Validation: query must not be empty or whitespace
  IF p_query IS NULL OR length(trim(p_query)) = 0 THEN
    RETURN;
  END IF;

  -- 3. Input Validation: match_count bounds [1, 100]
  IF match_count < 1 OR match_count > 100 THEN
    RAISE EXCEPTION 'match_count must be between 1 and 100, got %.', match_count
      USING ERRCODE = '22023'; -- invalid_parameter_value
  END IF;

  -- 4. Construct safe tsquery using websearch syntax (safe for unclosed quotes, minus, OR)
  BEGIN
    v_tsquery := websearch_to_tsquery('english', p_query);
  EXCEPTION WHEN OTHERS THEN
    v_tsquery := NULL;
  END;

  -- Fallback to plainto_tsquery if websearch produced an empty query
  IF v_tsquery IS NULL OR v_tsquery = ''::tsquery THEN
    BEGIN
      v_tsquery := plainto_tsquery('english', p_query);
    EXCEPTION WHEN OTHERS THEN
      v_tsquery := NULL;
    END;
  END IF;

  -- If neither parsed valid query tokens, return empty set safely
  IF v_tsquery IS NULL OR v_tsquery = ''::tsquery THEN
    RETURN;
  END IF;

  -- 5. Execute full-text search with composite document join for project filtering & tenant isolation
  RETURN QUERY
    SELECT
      c.id,
      c.document_id,
      d.project_id,
      c.chunk_index,
      c.content,
      c.metadata,
      ts_rank_cd(c.tsv_content, v_tsquery)::double precision AS score
    FROM public.document_chunks c
    JOIN public.documents d ON d.id = c.document_id AND d.user_id = c.user_id
    WHERE c.user_id = v_user_id
      AND c.tsv_content @@ v_tsquery
      AND (filter_project_id IS NULL OR d.project_id = filter_project_id)
      AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
    ORDER BY score DESC
    LIMIT match_count;
END;
$$;

-- Explicitly revoke from PUBLIC and grant solely to authenticated role
REVOKE ALL ON FUNCTION public.search_document_chunks_fts(text, integer, uuid, uuid) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.search_document_chunks_fts(text, integer, uuid, uuid) TO authenticated;
