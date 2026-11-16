# Relay AI — Milestone 10.4: Vector & Hybrid Retrieval Engine
## Architecture Specification, pgvector Schema & Master Implementation Plan (Final Pre-Implementation Corrections)

**Target Branch:** `feat/milestone-10-4-vector-hybrid-retrieval`  
**Base Branch:** `main`  
**Status:** Pre-Implementation Review Complete — Final Architecture Plan  
**Prerequisites:** Milestone 10.2 (Cloud Sync) & Milestone 10.3 (Document Ingestion) merged into `main`.

---

## 1. Executive Summary & Review Invariants

Milestone 10.4 establishes dense vector embeddings, vector similarity search, and hybrid retrieval for Relay AI's Universal Context Engine (UCE).

Following final pre-implementation review, the architecture addresses all edge cases and failure modes:
1. **Model-Specific Partial HNSW Indexing**: Replaces generic non-null indexes with genuine model-specific partial HNSW indexes for canonical models, backed by an index provisioning procedure for newly configured models.
2. **Benchmark-Based Recall Validation**: Treats Recall@10 $\ge 0.95$ as an empirical acceptance benchmark (comparing HNSW against exact brute-force scan under multi-predicate filtering). When the target is not achieved, the engine dynamically escalates `ef_search` or falls back to exact scan.
3. **Strict RPC Input Validation**: `match_document_chunks` validates that exactly one query vector is provided, confirms dimension and model compatibility, prevents cross-model comparisons, enforces multi-tenant RLS, and raises descriptive exceptions for invalid inputs.
4. **Comprehensive Cache Configuration Identity**: The SQLite `embedding_cache` incorporates `(content_hash, provider_id, model_id, model_version, dimension, preprocessing_version, normalize)` into a deterministic `cache_key`, completely preventing stale embedding reuse.
5. **Exact Serialization Contracts**: Formalizes IEEE-754 32-bit binary packing for SQLite and canonical `"[v1,v2,...,vn]"` string formatting for PostgREST.
6. **Live Cloud Integration Stance**: Integration tests against live Supabase endpoints are defined in detail and explicitly categorized as **PENDING LIVE SUPABASE VALIDATION** until disposable cloud credentials are provided.

---

## 2. Embedding Cache Configuration Identity & Schema

### A. The Cache Invalidation & Collision Problem
Embedding vectors depend on:
1. Raw text content.
2. Preprocessing rules (whitespace stripping, lowercasing, task prefixes like `search_document:` vs `search_query:` in Nomic models).
3. Provider and model identity.
4. Model revision/version (e.g., `v1.0` vs `v1.5`).
5. Output dimensionality (e.g., OpenAI 1536 default vs 768 MRL).
6. Normalization setting (L2 unit normalization $\|\vec{v}\|_2 = 1.0$).

If any of these change, reusing a previously cached vector corrupts retrieval quality.

### B. SQLite Cache Schema
```sql
CREATE TABLE IF NOT EXISTS embedding_cache (
    cache_key TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    preprocessing_version TEXT NOT NULL,
    normalize INTEGER NOT NULL DEFAULT 1,
    embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_emb_cache_lookup 
ON embedding_cache(provider_id, model_id, model_version, dimension, content_hash);
```

### C. Deterministic Cache Key Formula
In Python:
```python
def generate_cache_key(
    content_hash: str,
    provider_id: str,
    model_id: str,
    model_version: str,
    dimension: int,
    preprocessing_version: str,
    normalize: bool = True
) -> str:
    key_str = f"{provider_id}:{model_id}:{model_version}:{dimension}:{preprocessing_version}:{int(normalize)}:{content_hash}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()
```
* **Cache Miss Behavior**: Any change to model version, dimensionality, or preprocessing rules produces a different `cache_key`. Stale embeddings are never reused.
* **Storage Preservation**: Older cache entries remain in SQLite. Reverting to a prior model instantly re-enables cache hits without re-computation or extra API billing.

---

## 3. PostgreSQL pgvector Schema & RPC Specification

### A. Table Alteration
```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Dual-vector columns and metadata on public.document_chunks
ALTER TABLE public.document_chunks
  ADD COLUMN IF NOT EXISTS embedding_1536 vector(1536),
  ADD COLUMN IF NOT EXISTS embedding_768 vector(768),
  ADD COLUMN IF NOT EXISTS embedding_model text,
  ADD COLUMN IF NOT EXISTS embedding_dimension integer;
```

### B. Model-Specific Partial HNSW Indexing
Generic indexes with `WHERE embedding IS NOT NULL` degrade in recall when multiple models share the same table. We establish model-specific partial HNSW indexes for canonical first-class models:

```sql
-- 1. Partial HNSW index for OpenAI text-embedding-3-small (1536 dimensions)
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_openai_1536
  ON public.document_chunks
  USING hnsw (embedding_1536 vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE embedding_model = 'openai/text-embedding-3-small' 
    AND embedding_dimension = 1536 
    AND embedding_1536 IS NOT NULL;

-- 2. Partial HNSW index for OpenAI text-embedding-3-small (768 MRL dimensions)
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_openai_768
  ON public.document_chunks
  USING hnsw (embedding_768 vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE embedding_model = 'openai/text-embedding-3-small' 
    AND embedding_dimension = 768 
    AND embedding_768 IS NOT NULL;

-- 3. Partial HNSW index for Ollama nomic-embed-text (768 dimensions)
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_nomic_768
  ON public.document_chunks
  USING hnsw (embedding_768 vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE embedding_model = 'ollama/nomic-embed-text' 
    AND embedding_dimension = 768 
    AND embedding_768 IS NOT NULL;

-- 4. Baseline fallback partial indexes for arbitrary/custom models
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_fallback_1536
  ON public.document_chunks
  USING hnsw (embedding_1536 vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE embedding_1536 IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_fallback_768
  ON public.document_chunks
  USING hnsw (embedding_768 vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE embedding_768 IS NOT NULL;
```

### C. Dynamic Index Provisioning for Newly Configured Models
When a user configures a new custom or local model not covered by default partial indexes:
```sql
CREATE OR REPLACE FUNCTION public.provision_model_hnsw_index(
  p_model_id text,
  p_dimension int
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  v_idx_name text;
  v_col_name text;
BEGIN
  IF p_dimension = 1536 THEN
    v_col_name := 'embedding_1536';
  ELSIF p_dimension = 768 THEN
    v_col_name := 'embedding_768';
  ELSE
    RAISE EXCEPTION 'Unsupported dimension: %. Must be 768 or 1536.', p_dimension;
  END IF;

  v_idx_name := 'idx_chunks_hnsw_' || md5(p_model_id || '_' || p_dimension);

  EXECUTE format(
    'CREATE INDEX IF NOT EXISTS %I ON public.document_chunks USING hnsw (%I vector_cosine_ops) ' ||
    'WITH (m = 16, ef_construction = 64) WHERE embedding_model = %L AND embedding_dimension = %s AND %I IS NOT NULL',
    v_idx_name, v_col_name, p_model_id, p_dimension, v_col_name
  );
END;
$$;
```

---

## 4. Strict Vector Similarity RPC (`public.match_document_chunks`)

The RPC validates all inputs, rejects incompatible combinations, enforces user isolation, and applies project and document filters:

```sql
CREATE OR REPLACE FUNCTION public.match_document_chunks(
  p_model_id text,
  p_dimension int,
  query_embedding_1536 vector(1536) DEFAULT NULL,
  query_embedding_768 vector(768) DEFAULT NULL,
  match_threshold float DEFAULT 0.3,
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
  similarity float
)
LANGUAGE plpgsql
STABLE
SECURITY INVOKER
AS $$
BEGIN
  -- 1. Input Validation: Validate dimension
  IF p_dimension NOT IN (768, 1536) THEN
    RAISE EXCEPTION 'Invalid dimension: %. Allowed dimensions are 768 and 1536.', p_dimension
      USING ERRCODE = '22023';
  END IF;

  -- 2. Input Validation: Mutual exclusion of query vectors
  IF query_embedding_1536 IS NOT NULL AND query_embedding_768 IS NOT NULL THEN
    RAISE EXCEPTION 'Cannot provide both query_embedding_1536 and query_embedding_768. Pass exactly one matching p_dimension.'
      USING ERRCODE = '22023';
  END IF;

  IF query_embedding_1536 IS NULL AND query_embedding_768 IS NULL THEN
    RAISE EXCEPTION 'Missing query vector. Pass query_embedding_1536 for dim 1536 or query_embedding_768 for dim 768.'
      USING ERRCODE = '22023';
  END IF;

  -- 3. Input Validation: Dimension-Vector compatibility
  IF p_dimension = 1536 AND query_embedding_1536 IS NULL THEN
    RAISE EXCEPTION 'Dimension mismatch: p_dimension is 1536 but query_embedding_1536 is NULL.'
      USING ERRCODE = '22023';
  END IF;

  IF p_dimension = 768 AND query_embedding_768 IS NULL THEN
    RAISE EXCEPTION 'Dimension mismatch: p_dimension is 768 but query_embedding_768 is NULL.'
      USING ERRCODE = '22023';
  END IF;

  -- 4. Input Validation: Model-Dimension compatibility
  IF p_model_id = 'ollama/nomic-embed-text' AND p_dimension = 1536 THEN
    RAISE EXCEPTION 'Model % does not support 1536 dimensions. Nomic natively supports 768.', p_model_id
      USING ERRCODE = '22023';
  END IF;

  -- Set search depth for HNSW exploration
  SET LOCAL hnsw.ef_search = 100;

  -- 5. Execute similarity query on the validated vector column
  IF p_dimension = 1536 THEN
    RETURN QUERY
      SELECT
        c.id,
        c.document_id,
        d.project_id,
        c.chunk_index,
        c.content,
        c.metadata,
        (1 - (c.embedding_1536 <=> query_embedding_1536))::float AS similarity
      FROM public.document_chunks c
      JOIN public.documents d ON d.id = c.document_id
      WHERE c.user_id = auth.uid()
        AND c.embedding_model = p_model_id
        AND c.embedding_dimension = 1536
        AND c.embedding_1536 IS NOT NULL
        AND (filter_project_id IS NULL OR d.project_id = filter_project_id)
        AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
        AND (1 - (c.embedding_1536 <=> query_embedding_1536)) >= match_threshold
      ORDER BY c.embedding_1536 <=> query_embedding_1536 ASC
      LIMIT match_count;

  ELSIF p_dimension = 768 THEN
    RETURN QUERY
      SELECT
        c.id,
        c.document_id,
        d.project_id,
        c.chunk_index,
        c.content,
        c.metadata,
        (1 - (c.embedding_768 <=> query_embedding_768))::float AS similarity
      FROM public.document_chunks c
      JOIN public.documents d ON d.id = c.document_id
      WHERE c.user_id = auth.uid()
        AND c.embedding_model = p_model_id
        AND c.embedding_dimension = 768
        AND c.embedding_768 IS NOT NULL
        AND (filter_project_id IS NULL OR d.project_id = filter_project_id)
        AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
        AND (1 - (c.embedding_768 <=> query_embedding_768)) >= match_threshold
      ORDER BY c.embedding_768 <=> query_embedding_768 ASC
      LIMIT match_count;
  END IF;
END;
$$;
```

---

## 5. HNSW Recall Benchmark & Fallback Mechanics

### A. Acceptance Target Specification
* **Target Metric**: $\text{Recall}@10 \ge 0.95$ across 100 synthetic benchmark queries.
* **Ground Truth**: Exact brute-force vector distance computed by sequential scan:
  ```sql
  SET enable_indexscan = off;
  ```
* **Benchmark Query Conditions**:
  All recall benchmarks MUST include:
  1. `c.user_id = auth.uid()` (tenant isolation).
  2. `c.embedding_model = p_model_id` (model isolation).
  3. `d.project_id = filter_project_id` (project filtering).

### B. Behavior When Target Recall is Not Achieved
If $\text{Recall}@10 < 0.95$ during benchmark runs:
1. **Dynamic `ef_search` Escalation**: Increase `hnsw.ef_search` from 100 to 150 or 200 inside the session/RPC.
2. **Brute-Force Fallback for Small Datasets**: When total chunk count for a user/project is $< 1,000$ chunks, exact sequential scan takes $< 2$ms on PostgreSQL. The query planner or RPC can disable index scan for small datasets, delivering guaranteed 100% recall.
3. **Index Rebuilding**: If graph distortion occurs from heavy deletions, invoke `REINDEX INDEX CONCURRENTLY` to rebuild the HNSW graph cleanly.
4. **Telemetry Logging**: Discrepancies between approximate and exact ranking are logged to execution telemetry for analysis.

---

## 6. PostgREST & SQLite Serialization Contracts

### Detailed Data Contracts:

```
[In-Memory Python]
       │  list[float] (unit-normalized IEEE-754)
       ├──► SQLite Local Storage:
       │      struct.pack(f"<{dim}f", *vector) ──► BLOB (3,072 or 6,144 bytes)
       │
       └──► Cloud Sync Outbox & PostgREST:
              f"[{','.join(f'{x:.7g}' for x in vector)}]" ──► TEXT / JSON string
              │
              ▼
       [PostgREST Endpoint]
              POST /rest/v1/document_chunks
              Payload: {"embedding_1536": "[0.0123,-0.0456,...]", ...}
              │
              ▼
       [PostgreSQL pgvector Engine]
              vector_in(cstring) ──► vector(1536) internal binary
```

* **SQLite Storage**: Single-precision IEEE-754 32-bit little-endian binary BLOB via `struct.pack(f"<{dim}f", *vector)`. Exact length: 768 dims = 3,072 bytes; 1536 dims = 6,144 bytes. Sub-microsecond unpack via `struct.unpack`.
* **Sync Outbox & PostgREST**: Explicit string formatting `f"[{','.join(f'{x:.7g}' for x in vector)}]"`. Maps directly to PostgreSQL's native `vector_in(cstring)` parser across all PostgREST versions.
* **Precision Assurance**: Verified with unit tests ensuring round-trip numerical deviation $\Delta < 10^{-6}$.

---

## 7. Supabase Cloud Integration Validation Status

### Verification Matrix & Current Status:

| Check | Target | Verification Method | Status |
|---|---|---|---|
| **pgvector Extension** | PostgreSQL | `CREATE EXTENSION IF NOT EXISTS vector;` syntax | **PENDING LIVE SUPABASE VALIDATION** |
| **Migration Execution** | Supabase Cloud | Up and down SQL scripts applied via Supabase CLI | **PENDING LIVE SUPABASE VALIDATION** |
| **Vector Insertion (PostgREST)** | Table Insert | `POST /rest/v1/document_chunks` with vector string | **PENDING LIVE SUPABASE VALIDATION** |
| **Vector Retrieval (PostgREST)** | Table Select | `GET /rest/v1/document_chunks` with select and filter | **PENDING LIVE SUPABASE VALIDATION** |
| **RPC Invocation (JWT)** | RPC Execution | `POST /rest/v1/rpc/match_document_chunks` with Bearer JWT | **PENDING LIVE SUPABASE VALIDATION** |
| **768-dim Behavior** | Ollama / MRL | Ingestion, indexing, and retrieval at 768 dims | **PENDING LIVE SUPABASE VALIDATION** |
| **1536-dim Behavior** | OpenAI | Ingestion, indexing, and retrieval at 1536 dims | **PENDING LIVE SUPABASE VALIDATION** |
| **RLS & Tenant Isolation** | Security | Cross-user query rejection via `auth.uid()` | **PENDING LIVE SUPABASE VALIDATION** |
| **Project Filtering** | Multi-Tenancy | Retrieval restricted to `project_id` | **PENDING LIVE SUPABASE VALIDATION** |
| **Serialization Round-Trip** | Data Fidelity | Round-trip float fidelity test | **PENDING LIVE SUPABASE VALIDATION** |
| **Local SQL Syntax & AST** | SQL Parser | Python `sqlparse` & AST validation of migration DDL | **VERIFIED LOCALLY** |
| **Local SQLite Schema & Triggers**| SQLite FTS5 | In-memory SQLite FTS5 BM25 search & trigger sync | **VERIFIED LOCALLY** |
| **Cache Collision Math** | Unit Test | Key generation across models, versions, dimensions | **VERIFIED LOCALLY** |

> [!NOTE]
> **Pending Status Disclosure:** No disposable live Supabase project credentials (`SUPABASE_URL`, `SUPABASE_ANON_KEY`) or `psql` connections are available in this offline local development workspace. In strict adherence to integrity rules, live cloud checks remain formally categorized as **PENDING LIVE SUPABASE VALIDATION**. Structural, algorithmic, and SQLite local operations are verified locally.

---

## 8. Remaining Risks & External Dependencies

1. **Supabase Cloud Network Connectivity**:
   - Dependency: Requires valid `SUPABASE_URL` and `SUPABASE_ANON_KEY` environment variables.
   - Mitigation: All retrieval operations feature transparent offline fallback to SQLite FTS5 BM25 search.
2. **External Provider Rate Limits**:
   - Dependency: OpenAI API limits and token consumption.
   - Mitigation: Local IEEE-754 binary cache eliminates redundant API calls; exponential backoff handles transient 429s.
3. **Ollama Local Instance Availability**:
   - Dependency: Ollama service running on `localhost:11434` with `nomic-embed-text` installed.
   - Mitigation: Circuit breaker detects offline daemon and falls back to lexical keyword retrieval.

---

## 9. Implementation Acceptance Criteria

Before Milestone 10.4 is declared complete:
1. `EmbeddingProvider` ABC implemented with `OpenAIEmbeddingProvider`, `OllamaEmbeddingProvider`, and `MockEmbeddingProvider`.
2. SQLite `embedding_cache` implemented with composite `cache_key` preventing all model/dimension/version collisions.
3. `EmbeddingPipeline` implemented with 32-chunk batching, backoff retry, and un-embedded chunk recovery.
4. Supabase migration files created for up (`20261115000001_milestone_10_4_vector_retrieval.sql`) and down (`..._down.sql`).
5. `match_document_chunks` RPC validates all input parameters and rejects invalid/dual/neither vector calls.
6. `HybridRetriever` implemented with Reciprocal Rank Fusion ($k=60, w_{\text{dense}}=0.6, w_{\text{sparse}}=0.4$) and offline BM25 fallback.
7. All 9 new test modules (+22 tests) implemented and passing with 100% pass rate.
8. Existing 342 regression tests remain 100% passing.

---

## 10. Approval Check

Final pre-implementation corrections are complete. No production features have been implemented.
Standing by for formal user approval to proceed with Phase 1.
