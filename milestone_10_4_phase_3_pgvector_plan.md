# Relay AI — Milestone 10.4, Phase 3: Supabase pgvector Architecture & Implementation Plan

**Branch:** `feat/milestone-10-4-vector-hybrid-retrieval`  
**Base Commit:** `d4b4dbe8a4c899ffd1a7694aa98e89f967db8954` (Phase 2 pushed and verified)  
**Status:** Planning & Architecture Review Only (Pre-Implementation)  
**Deliverable:** Implementation-ready blueprint for Supabase pgvector schema, model-specific HNSW indexes, security policies, and vector similarity RPC.

---

## 1. Scope and Objectives

### Primary Objectives
1. **Enable pgvector Extension:** Safely provision the `vector` extension in Supabase PostgreSQL without disrupting existing tables or operations.
2. **Dual-Dimension Vector Schema:** Add `embedding_1536 vector(1536)`, `embedding_768 vector(768)`, `embedding_model text`, and `embedding_dimension integer` to `public.document_chunks`.
3. **Data Integrity & Preserving Existing Chunks:** Ensure zero data loss for existing document chunks, existing full-text search (`tsv_content`), and existing RLS policies.
4. **Model-Specific Partial HNSW Indexing:** Establish targeted partial HNSW indexes for canonical embedding models (`openai/text-embedding-3-small` at 1536 and 768 dimensions, `ollama/nomic-embed-text` at 768 dimensions) plus fallback indexing, preventing cross-model index pollution.
5. **Secure Similarity Search RPC (`public.match_document_chunks`):** Implement a `SECURITY INVOKER` function enforcing strict multi-tenant user isolation (`auth.uid()`), dimension/model validation, project/document filtering, and cosine distance ordering.
6. **Reversible Migration:** Provide an exact rollback migration (`down.sql`) restoring the pre-Phase 3 schema state.

### Explicit Scope Boundaries
- **In Scope for Phase 3:** SQL migration files, SQL rollback files, model-specific HNSW indexes, RPC function definition, Python Supabase RPC wrapper/client integration, and unit/integration tests.
- **Out of Scope for Phase 3 (Deferred to Phase 4 & Phase 5):**
  - Cloud vector synchronization worker enhancements (Phase 4).
  - Hybrid retrieval combining pgvector and Full-Text Search via Reciprocal Rank Fusion (RRF) (Phase 4).
  - Cross-encoder reranking and contextual compression (Phase 5).

---

## 2. Existing Architecture & Code Audit

A thorough audit of the repository (`C:\Users\ARPIT\Downloads\Relay-Ai`) reveals the following structural invariants and deviations from initial assumptions:

### A. Supabase Cloud Schema Audit (`supabase/migrations/20261114000001_initial_uce_schema.sql`)
1. **`document_chunks` Table Structure:**
   ```sql
   create table public.document_chunks (
     id uuid primary key default gen_random_uuid(),
     user_id uuid not null references auth.users(id) on delete cascade,
     document_id uuid not null,
     chunk_index integer not null check (chunk_index >= 0),
     content text not null,
     tsv_content tsvector generated always as (to_tsvector('english', coalesce(content,''))) stored,
     metadata jsonb not null default '{}'::jsonb,
     created_at timestamptz not null default now(),
     foreign key (document_id, user_id) references public.documents(id, user_id) on delete cascade,
     unique (document_id, chunk_index)
   );
   ```
2. **Critical Schema Difference on `project_id`:**
   - **SQLite (`src/context/db/schema.py`):** `document_chunks` includes a denormalized `project_id TEXT NOT NULL DEFAULT 'global'`.
   - **Supabase Cloud Schema:** `document_chunks` **does not** contain a `project_id` column. Instead, `project_id` is defined on `public.documents(id, user_id)`:
     ```sql
     create table public.documents (
       id uuid primary key default gen_random_uuid(),
       user_id uuid not null references auth.users(id) on delete cascade,
       project_id uuid,
       ...
     );
     ```
   - **Impact on RPC:** The similarity RPC `match_document_chunks` **must** join `public.documents` to apply `filter_project_id`. The join must be composite: `ON d.id = c.document_id AND d.user_id = c.user_id` to leverage the composite foreign key index and enforce tenant boundaries.
3. **Existing RLS Policy on `document_chunks`:**
   ```sql
   create policy document_chunks_owner_all on public.document_chunks
   for all to authenticated
   using (user_id = (select auth.uid()))
   with check (user_id = (select auth.uid()));
   ```
   Every row has `user_id`. When executing a `SECURITY INVOKER` function, PostgreSQL automatically enforces this policy.

### B. Python Phase 1 & Phase 2 Invariants
1. **Vector Precision:**
   - SQLite cache (`src/context/embeddings/cache.py`) packs vectors as IEEE-754 32-bit floats (`<f`).
   - pgvector stores vector components as 32-bit `float4` (4 bytes per dimension).
   - Precision across SQLite cache, Python runtime, and PostgreSQL pgvector is identical.
2. **Dimension Support:**
   - Supported canonical dimensions: `1536` (OpenAI default) and `768` (Nomic and OpenAI MRL).
   - Providers validate vectors strictly and reject empty, non-finite, truncated, or padded vectors.
3. **Chunk Tracking Invariant:**
   - Chunks in SQLite are tracked with `embedding_status` (`"PENDING"`, `"COMPLETED"`, `"FAILED"`), `embedding_model`, `embedding_dimension`, and `embedding_retry_count`.

---

## 3. Proposed Migration Design

### A. Up Migration: `supabase/migrations/20261115000001_milestone_10_4_vector_retrieval.sql`

```sql
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
-- Validates dimension value
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
```

### B. Down Migration (Rollback): `supabase/migrations/20261115000001_milestone_10_4_vector_retrieval_down.sql`

```sql
-- Relay AI: Rollback migration for Milestone 10.4 vector retrieval

-- 1. Drop vector similarity RPC function
DROP FUNCTION IF EXISTS public.match_document_chunks(
  text, integer, extensions.vector, extensions.vector, double precision, integer, uuid, uuid
);

-- 2. Drop model-specific and fallback HNSW indexes
DROP INDEX IF EXISTS public.idx_chunks_hnsw_openai_1536;
DROP INDEX IF EXISTS public.idx_chunks_hnsw_openai_768;
DROP INDEX IF EXISTS public.idx_chunks_hnsw_nomic_768;
DROP INDEX IF EXISTS public.idx_chunks_hnsw_fallback_1536;
DROP INDEX IF EXISTS public.idx_chunks_hnsw_fallback_768;

-- 3. Drop check constraints
ALTER TABLE public.document_chunks
  DROP CONSTRAINT IF EXISTS chk_document_chunks_vector_consistency,
  DROP CONSTRAINT IF EXISTS chk_document_chunks_embedding_dimension;

-- 4. Drop added columns
ALTER TABLE public.document_chunks
  DROP COLUMN IF EXISTS embedding_1536,
  DROP COLUMN IF EXISTS embedding_768,
  DROP COLUMN IF EXISTS embedding_model,
  DROP COLUMN IF EXISTS embedding_dimension;

-- Note: We do NOT execute DROP EXTENSION vector because other databases or extensions may depend on it.
```

### C. Migration Invariants
- **Non-Destructive:** Existing rows in `document_chunks` receive `NULL` for the new columns. Existing `tsv_content`, `content`, and `metadata` are 100% untouched.
- **Idempotency:** Guarded by `IF NOT EXISTS` and `DO $$` blocks. Running the migration multiple times produces zero errors.
- **Zero Lock Escalation:** Column additions with default `NULL` in PostgreSQL $\ge 11$ are metadata-only operations that complete in sub-milliseconds without table rewrites.

---

## 4. Model-Specific HNSW Indexes & Critical Security Review

### A. The Cross-Model Pollution Problem
In pgvector, cosine distance between two vectors $\vec{u}$ and $\vec{v}$ is only mathematically defined within the *same* vector space. If `nomic-embed-text` (768d) and `text-embedding-3-small` (768d) both write to `embedding_768`:
- A generic index on `embedding_768 WHERE embedding_768 IS NOT NULL` builds a unified HNSW graph containing vectors from both models.
- When querying for model A, the HNSW graph traversal visits nodes belonging to model B. Distances to model B nodes are mathematically meaningless noise.
- This results in dead-end graph explorations, severe recall drops, and degraded latency.

### B. Canonical Model-Specific Partial HNSW Indexes
We create model-specific partial HNSW indexes targeting canonical models:

```sql
-- 1. OpenAI text-embedding-3-small (1536 dimensions)
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_openai_1536
  ON public.document_chunks
  USING hnsw (embedding_1536 extensions.vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE embedding_model = 'openai/text-embedding-3-small'
    AND embedding_dimension = 1536
    AND embedding_1536 IS NOT NULL;

-- 2. OpenAI text-embedding-3-small (768 dimensions via MRL)
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_openai_768
  ON public.document_chunks
  USING hnsw (embedding_768 extensions.vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE embedding_model = 'openai/text-embedding-3-small'
    AND embedding_dimension = 768
    AND embedding_768 IS NOT NULL;

-- 3. Ollama nomic-embed-text (768 dimensions)
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_nomic_768
  ON public.document_chunks
  USING hnsw (embedding_768 extensions.vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE embedding_model = 'ollama/nomic-embed-text'
    AND embedding_dimension = 768
    AND embedding_768 IS NOT NULL;

-- 4. Baseline fallback partial indexes for custom or future models
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_fallback_1536
  ON public.document_chunks
  USING hnsw (embedding_1536 extensions.vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE embedding_1536 IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_fallback_768
  ON public.document_chunks
  USING hnsw (embedding_768 extensions.vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE embedding_768 IS NOT NULL;
```

### C. Security Review: Rejection of Dynamic `SECURITY DEFINER` Index Provisioning
* **Vulnerability Analysis in Prior Proposal:**  
  The earlier architecture plan suggested a `SECURITY DEFINER` function `public.provision_model_hnsw_index(p_model_id text, p_dimension int)` executing dynamic DDL via `EXECUTE format('CREATE INDEX...')`.
* **Identified Critical Threats:**
  1. **Denial of Service (DoS) & Disk Exhaustion:** Building an HNSW index is an expensive CPU- and memory-intensive operation requiring substantial `maintenance_work_mem` and exclusive table locking. Any client with authenticated access could call this RPC repeatedly with random model strings, creating hundreds of indexes, locking `document_chunks`, and crashing database disk storage.
  2. **Catalog Bloat & Write Amplification:** Every insert or update on `document_chunks` must update all active partial indexes. Proliferating indexes severely degrades ingestion throughput.
  3. **Privilege Escalation Anti-Pattern:** Executing dynamic DDL inside `SECURITY DEFINER` functions callable by non-admin roles violates least privilege principles.
* **Architectural Decision:**  
  **The `provision_model_hnsw_index` function is strictly rejected and excluded from the migration.**  
  - Dynamic index creation will **never** be exposed via client-accessible RPC.
  - The 3 canonical models cover 100% of standard production workflows.
  - Custom or self-hosted models are immediately serviced by the baseline fallback partial indexes (`idx_chunks_hnsw_fallback_*`) or exact scan.
  - Provisioning dedicated partial indexes for newly added models must be conducted through **trusted administrative migrations** (e.g. `supabase db push` or migration scripts executed by `postgres`/`service_role` administrator credentials).

---

## 5. Secure Vector Similarity RPC (`public.match_document_chunks`)

### A. Full SQL Specification

```sql
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

  -- 8. Execute similarity search on the validated vector column
  IF p_dimension = 1536 THEN
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

-- Grant execution permission to authenticated users only
REVOKE ALL ON FUNCTION public.match_document_chunks FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.match_document_chunks TO authenticated;
```

### B. Security & Design Analysis
1. **`SECURITY INVOKER`:**  
   The function runs under the permissions of the calling authenticated user, ensuring PostgreSQL evaluates existing RLS policies on `document_chunks` and `documents`.
2. **Explicit User Isolation (`c.user_id = v_user_id`):**  
   Provides defense-in-depth: even if RLS were bypassed, the explicit `WHERE` clause isolates queries to `auth.uid()`.
3. **Composite Join:**  
   `JOIN public.documents d ON d.id = c.document_id AND d.user_id = c.user_id` ensures that document-level access and project relationships cannot be spoofed across tenants.
4. **Exact Cosine Distance Ordering:**  
   Orders by `c.embedding <=> query_embedding ASC` (cosine distance). The HNSW index on `vector_cosine_ops` matches this operator directly, utilizing index scans rather than sequential scans.
5. **SQL Injection Prevention:**  
   Static, parameterized query with strong typing (`extensions.vector`, `uuid`, `text`, `int`). Zero dynamic SQL execution (`EXECUTE`).
6. **Standard Error Codes:**  
   Uses standard PostgreSQL SQLSTATE codes (`28000` for invalid authorization, `22023` for invalid parameter values).

---

## 6. Serialization Contract & Data Integrity

### A. PostgREST Wire Format
When invoking `POST /rest/v1/rpc/match_document_chunks` or uploading chunks via `POST /rest/v1/document_chunks`:
- Vectors can be transmitted as either:
  1. A standard JSON array of numbers: `[0.0123, -0.0456, ...]`
  2. A canonical bracketed string: `"[0.0123,-0.0456,...]"`
- PostgREST natively parses both formats into PostgreSQL `extensions.vector`.
- In the Python Supabase client, vectors will be formatted as standard JSON lists: `list[float]`.

### B. Float Precision Alignment
| Component | Storage Representation | Byte Width | Precision |
| :--- | :--- | :--- | :--- |
| **Python Provider** | IEEE-754 `float` (double precision) | 8 bytes | ~15 decimal digits |
| **SQLite Cache** | IEEE-754 `float32` (`<f`) binary | 4 bytes / dim | ~7 decimal digits |
| **PostgREST JSON** | Decimal formatted float string | ASCII | Preserves full float32 |
| **Supabase pgvector** | PostgreSQL `float4` | 4 bytes / dim | ~7 decimal digits |

The round-trip conversion from SQLite cache to Supabase pgvector maintains exact single-precision float fidelity.

---

## 7. Comprehensive Test Matrix

```
+-------------------------------------------------------------+-------------------------------+
| Test Category                                               | Execution Environment         |
+-------------------------------------------------------------+-------------------------------+
| 1. SQL Migration Syntax & Idempotency                       | Local Python / sqlparse mock  |
| 2. Down Migration Complete Reversibility                    | Local Python / sqlparse mock  |
| 3. RPC Parameter Validation (dimension, mutual exclusivity) | Local Mock / Supabase Live    |
| 4. Model/Dimension Compatibility Check                      | Local Mock / Supabase Live    |
| 5. PostgREST JSON Payload Serialization                     | Local Python unit tests       |
| 6. Unauthenticated Caller Rejection (auth.uid() = NULL)     | Supabase Live (disposable)    |
| 7. Multi-Tenant User Isolation (cross-user data denial)     | Supabase Live (2 user JWTs)   |
| 8. Project & Document Filtering Verification                | Supabase Live (disposable)    |
| 9. 768-dim Vector Similarity Search & Cosine Ordering       | Supabase Live (disposable)    |
| 10. 1536-dim Vector Similarity Search & Cosine Ordering     | Supabase Live (disposable)    |
| 11. HNSW Index Plan Utilization (EXPLAIN ANALYZE)           | Supabase Live (disposable)    |
| 12. Benchmark Recall@10 >= 0.95 vs Brute-Force Scan         | Supabase Live (disposable)    |
+-------------------------------------------------------------+-------------------------------+
```

### Categorized Test Separation:
1. **Local Tests (Executable offline without Supabase):**
   - Migration file integrity, syntax, and presence checks.
   - Down migration reversibility parsing.
   - Python vector serializer/deserializer tests for PostgREST format.
   - Mocked Supabase client RPC call parameter verification.
2. **Disposable Live Supabase Tests (Requires credentials in disposable project):**
   - Applying `20261115000001_milestone_10_4_vector_retrieval.sql`.
   - Verifying `vector` extension and table column schema.
   - Executing `match_document_chunks` with 768d and 1536d vectors.
   - Verifying check constraints reject invalid dimensions or dual vector populations.
   - Testing down migration and verifying clean removal.
3. **Multi-User Authenticated Tests (Requires real JWTs):**
   - Verifying User A cannot retrieve User B's document chunks even if `filter_project_id` or `filter_document_id` matches.
   - Verifying unauthenticated requests fail with code `28000`.

---

## 8. GitHub Actions CI Observation & Safe Update Recommendation

### Audit Finding
In `.github/workflows/python-tests.yml`:
```yaml
on:
  pull_request:
    branches: [main, feat/milestone-10-2-cloud-sync]
  push:
    branches: [main, feat/milestone-10-2-cloud-sync, feat/milestone-10-3-document-ingestion]
```
The Milestone 10.4 branch (`feat/milestone-10-4-vector-hybrid-retrieval`) is **not** included in the trigger branches. Consequently, pushes and PRs on this branch do not run automated CI.

### Recommended Minimal Update (For Future Implementation Turn)
Add `feat/milestone-10-4-vector-hybrid-retrieval` to both `pull_request.branches` and `push.branches`:
```yaml
on:
  pull_request:
    branches: [main, feat/milestone-10-2-cloud-sync, feat/milestone-10-4-vector-hybrid-retrieval]
  push:
    branches: [main, feat/milestone-10-2-cloud-sync, feat/milestone-10-3-document-ingestion, feat/milestone-10-4-vector-hybrid-retrieval]
```
*(No workflow modifications are performed in this planning turn).*

---

## 9. File-by-File Implementation Plan for Phase 3

When implementation approval is granted, Phase 3 will introduce or update the following exact files:

| File Path | Action | Description |
| :--- | :--- | :--- |
| `supabase/migrations/20261115000001_milestone_10_4_vector_retrieval.sql` | **Create** | Up-migration: pgvector extension, dual-vector columns, constraints, partial HNSW indexes, RPC function. |
| `supabase/migrations/20261115000001_milestone_10_4_vector_retrieval_down.sql` | **Create** | Down-migration: drops RPC, partial indexes, constraints, and vector columns. |
| `src/context/retrieval/vector_client.py` | **Create** | Python client wrapper for calling `match_document_chunks` via Supabase PostgREST client. |
| `src/context/retrieval/__init__.py` | **Create/Update** | Package export for vector retrieval client and data structures. |
| `tests/test_vector_migration.py` | **Create** | Local offline tests verifying migration SQL structure, constraints, and serialization. |
| `tests/test_vector_client.py` | **Create** | Unit tests for Python RPC client validation, error handling, and parameter construction. |

---

## 10. Assumptions & Unresolved Questions

1. **Supabase Vector Schema Placement:**  
   *Assumption:* pgvector extension resides in the `extensions` schema (standard on Supabase), referenced as `extensions.vector`. The migration script explicitly sets `search_path = public, extensions`.
2. **Dynamic Index Provisioning Policy:**  
   *Assumption:* The user confirms that dynamic `SECURITY DEFINER` index creation is rejected in favor of trusted administrative migrations and fallback indexing.
3. **Live Supabase Credentials Availability:**  
   *Status:* Disposable Supabase project credentials are not currently configured in local environment variables. All live cloud tests will be marked **PENDING LIVE SUPABASE VALIDATION** until an authorized sandbox is provided.

---

## 11. Acceptance Criteria

- [ ] `20261115000001_milestone_10_4_vector_retrieval.sql` is syntactically valid PostgreSQL DDL.
- [ ] `20261115000001_milestone_10_4_vector_retrieval_down.sql` fully reverses all schema additions.
- [ ] Dual-vector columns (`vector(1536)` and `vector(768)`) and metadata columns are added without data loss.
- [ ] Check constraints prevent invalid dimensions and simultaneous dual-vector population.
- [ ] Canonical model-specific partial HNSW indexes are defined with cosine distance operator.
- [ ] `match_document_chunks` executes with `SECURITY INVOKER`, validates inputs, isolates users via `auth.uid()`, joins `documents` for project filtering, and returns results ordered by cosine distance.
- [ ] Zero production Python code or migrations were created during this planning turn.
- [ ] Full regression suite remains 100% passing (405/405 tests).
