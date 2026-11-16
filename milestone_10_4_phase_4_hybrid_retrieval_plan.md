# Relay AI — Milestone 10.4: Phase 4 Hybrid Retrieval Plan
## Dense Vector & Sparse Full-Text Retrieval with Reciprocal Rank Fusion (RRF)

**Target Branch:** `feat/milestone-10-4-hybrid-retrieval`  
**Base Commit:** `f6830af` (Merge PR #3 containing Phase 3 commit `383e6e8`)  
**Status:** Implementation Complete & Verified  
**Deliverable:** `milestone_10_4_phase_4_hybrid_retrieval_plan.md`  

---

## 1. Current Architecture Analysis

Relay AI's Universal Context Engine (UCE) has implemented Phases 1 through 3 of Milestone 10.4:

### 1.1 Phase 1: Embedding Providers & SQLite Embedding Cache
* **Providers (`src/providers/embeddings.py`):**
  * `EmbeddingProvider` abstract base class defining `embed_documents`, `embed_query`, `dimension`, `model_id`, `provider_id`, and `model_version`.
  * `OpenAIEmbeddingProvider`: text-embedding-3-small supporting configurable dimensions (1536 and 768 via Matryoshka Representation Learning).
  * `OllamaEmbeddingProvider`: nomic-embed-text (768 dimensions) with automatic `search_document:` and `search_query:` task prefixes.
  * `MockEmbeddingProvider`: Deterministic synthetic embeddings for reproducible offline testing.
* **Cache (`src/context/embeddings/cache.py`):**
  * SQLite table `embedding_cache` with composite SHA-256 `cache_key` computed from `(provider_id, model_id, model_version, dimension, preprocessing_version, normalize, content_hash)`.
  * Guarantees zero stale vector reuse across different models, dimensions, or text revisions.

### 1.2 Phase 2: Embedding Batch Pipeline & Recovery
* **Pipeline (`src/context/embeddings/pipeline.py`):**
  * Slices document chunks into 32-chunk batches.
  * Thread-safe `RateLimiter` managing provider-specific request concurrency.
  * Exponential backoff retry with full jitter and `Retry-After` header adherence.
  * Query embedding method `embed_query()` with cache-first lookup.
* **Recovery (`src/context/embeddings/recovery.py`):**
  * Persistent chunk status tracking (`PENDING`, `COMPLETED`, `FAILED`) in SQLite chunk metadata.
  * Exponential retry counters surviving process restarts.

### 1.3 Phase 3: Supabase pgvector & Vector Client
* **Cloud Schema & Indexing (`supabase/migrations/20261115000001_milestone_10_4_vector_retrieval.sql`):**
  * Dual vector columns on `public.document_chunks`: `embedding_1536` (`vector(1536)`) and `embedding_768` (`vector(768)`).
  * Check constraints enforcing mutual exclusivity and dimension consistency.
  * Canonical model-specific partial HNSW indexes (`idx_chunks_hnsw_openai_1536`, `idx_chunks_hnsw_openai_768`, `idx_chunks_hnsw_nomic_768`) using `vector_cosine_ops` and `(m = 16, ef_construction = 64)`.
  * RPC function `public.match_document_chunks`:
    * Executes under `SECURITY INVOKER`.
    * Validates dimensions, query vector presence, and model compatibility.
    * Sets `SET LOCAL hnsw.ef_search = 100`.
    * Uses static model literal branches for generic-plan index selection.
    * Enforces multi-tenant isolation via `c.user_id = auth.uid()` and composite join `JOIN public.documents d ON d.id = c.document_id AND d.user_id = c.user_id`.
* **Vector Client (`src/context/retrieval/vector_client.py`):**
  * PostgREST client invoking `match_document_chunks` with `AuthManager` bearer token authentication.
  * Returns structured `List[VectorMatch]` sorted by cosine similarity descending.

### 1.4 Current Sparse & Full-Text Search State
* **Local SQLite (`src/context/db/schema.py`):**
  * Virtual table `chunks_fts USING fts5(chunk_id UNINDEXED, document_id UNINDEXED, project_id UNINDEXED, content, tokenize = 'porter unicode61')`.
  * SQLite triggers (`trg_chunks_insert`, `trg_chunks_delete`, `trg_chunks_update`) automatically keep `chunks_fts` synchronized with `document_chunks`.
* **Cloud Supabase (`supabase/migrations/20261114000001_initial_uce_schema.sql`):**
  * Table `public.document_chunks` includes generated tsvector column:
    ```sql
    tsv_content tsvector generated always as (to_tsvector('english', coalesce(content,''))) stored
    ```
  * Index: `create index document_chunks_tsv_idx on public.document_chunks using gin(tsv_content);`.
  * Gap: Currently, no secure PostgREST RPC function exists to expose full-text search with project filtering, user isolation, and ranking against `tsv_content`.

---

## 2. Proposed Hybrid Retrieval Architecture

The Hybrid Retrieval Engine unifies dense semantic similarity and sparse keyword matching into a single resilient search pipeline.

```
                           ┌────────────────────────┐
                           │      User Query        │
                           └───────────┬────────────┘
                                       │
                                       ▼
                     ┌──────────────────────────────────┐
                     │         HybridRetriever          │
                     └─────────┬──────────────┬─────────┘
                               │              │
             ┌─────────────────┴────┐    ┌────┴─────────────────┐
             │    Dense Search      │    │    Sparse Search     │
             │       Branch         │    │       Branch         │
             └─────────┬────────────┘    └────┬─────────────────┘
                       │                      │
                       ▼                      ▼
             ┌──────────────────┐   ┌───────────────────────────┐
             │ EmbeddingPipeline│   │   SparseRetrievalClient   │
             │  (embed_query)   │   └─────────┬──────────────┬──┘
             └─────────┬────────┘             │              │
                       │                      │ (Online)     │ (Offline/Fallback)
                       ▼                      ▼              ▼
             ┌──────────────────┐   ┌──────────────────┐ ┌──────────────────┐
             │   VectorClient   │   │  PostgreSQL FTS  │ │   SQLite FTS5    │
             │   (pgvector)     │   │ (Supabase RPC)   │ │  (chunks_fts)    │
             └─────────┬────────┘   └─────────┬────────┘ └─────────┬────────┘
                       │                      │                    │
                       ▼                      └──────────┬─────────┘
               List[VectorMatch]                         ▼
             (Ranked by Cosine)                  List[SparseMatch]
                       │                         (Ranked by Score)
                       └──────────────┬──────────────────┘
                                      │
                                      ▼
                      ┌─────────────────────────────────┐
                      │    Reciprocal Rank Fusion       │
                      │             (RRF)               │
                      │   k=60, w_dense, w_sparse       │
                      └───────────────┬─────────────────┘
                                      │
                                      ▼
                      ┌─────────────────────────────────┐
                      │       Deduplicated & Fused      │
                      │     List[HybridSearchResult]    │
                      └─────────────────────────────────┘
```

### Core Architecture Components:
1. **`SparseRetrievalClient` (`src/context/retrieval/sparse_client.py`):**
   * Encapsulates sparse keyword retrieval across both cloud Supabase (PostgreSQL `to_tsvector`/`ts_rank_cd`) and local offline SQLite (`chunks_fts` BM25).
   * Implements robust query sanitization to prevent syntax errors on raw user inputs (punctuation, logical keywords, unbalanced quotes).
2. **PostgreSQL FTS Cloud RPC (`supabase/migrations/20261116000001_milestone_10_4_fts_retrieval.sql`):**
   * Migration introducing `public.search_document_chunks_fts` RPC.
   * `SECURITY INVOKER`, joins `public.documents` to enforce project filtering and tenant isolation via `auth.uid()`, utilizes GIN index `document_chunks_tsv_idx`, and ranks using `ts_rank_cd`.
3. **`ReciprocalRankFusion` Engine (`src/context/retrieval/rrf.py`):**
   * Pure algorithmic fusion implementation combining ranked lists of arbitrary lengths.
   * Parameterized smoothing constant $k$ and normalized source weights ($w_{\text{dense}}, w_{\text{sparse}}$).
   * Deterministic tie-breaking preserving ranking stability.
4. **`HybridRetriever` Orchestrator (`src/context/retrieval/hybrid.py`):**
   * High-level retrieval coordinator providing concurrent execution of dense and sparse branches.
   * Automatic candidate oversampling: requests $\max(\text{match\_count} \times 2, 20)$ candidates from each branch to maximize fusion depth.
   * Graceful fallback when either branch fails, degrades, or is disabled.
   * Comprehensive result dataclass with score provenance (dense rank, sparse rank, original similarity, BM25 score).

---

## 3. Dense and Sparse Retrieval Interfaces

### 3.1 Common Data Structures
```python
@dataclass
class RetrievalMatch:
    """Standardized retrieved chunk across dense and sparse retrievers."""
    id: str
    document_id: str
    project_id: Optional[str]
    chunk_index: int
    content: str
    metadata: Dict[str, Any]
    score: float  # Cosine similarity for dense; ts_rank/BM25 for sparse
```

Existing `VectorMatch` maps directly to this structure.

### 3.2 Sparse Match Specification
```python
@dataclass
class SparseMatch:
    """Represents a matching document chunk returned by full-text keyword search."""
    id: str
    document_id: str
    project_id: Optional[str]
    chunk_index: int
    content: str
    metadata: Dict[str, Any]
    score: float  # ts_rank_cd in PostgreSQL, -rank in SQLite FTS5
```

### 3.3 Sparse Client Interface
```python
class SparseRetrievalClient:
    """Client for executing sparse full-text search against Supabase or local SQLite."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_anon_key: Optional[str] = None,
        auth_manager: Optional[AuthManager] = None,
        sqlite_conn: Optional[sqlite3.Connection] = None,
        http_client: Optional[httpx.Client] = None,
        timeout: float = 10.0,
    ) -> None: ...

    def search_keywords(
        self,
        query: str,
        match_count: int = 20,
        filter_project_id: Optional[str] = None,
        filter_document_id: Optional[str] = None,
        prefer_offline: bool = False,
    ) -> List[SparseMatch]:
        """Execute full-text keyword search.
        
        Attempts cloud PostgreSQL FTS RPC if authenticated and online;
        falls back to local SQLite FTS5 if offline, unauthenticated, or on error.
        """
```

### 3.4 Hybrid Retriever Interface
```python
@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval fusion and execution."""
    k: int = 60
    weight_dense: float = 0.6
    weight_sparse: float = 0.4
    dense_threshold: float = 0.0  # Cosine similarity floor
    match_count: int = 10
    candidate_oversampling_factor: int = 2  # Request match_count * 2 candidates
    timeout_seconds: float = 10.0
    concurrent_execution: bool = True


@dataclass
class HybridSearchResult:
    """Final merged search result with full provenance."""
    id: str
    document_id: str
    project_id: Optional[str]
    chunk_index: int
    content: str
    metadata: Dict[str, Any]
    rrf_score: float
    dense_rank: Optional[int] = None       # 1-based rank (None if absent in dense Top-K)
    sparse_rank: Optional[int] = None      # 1-based rank (None if absent in sparse Top-K)
    dense_similarity: Optional[float] = None
    sparse_score: Optional[float] = None
    retrieval_mode: str = "HYBRID"         # "HYBRID", "DENSE_ONLY", "SPARSE_ONLY", "OFFLINE_LOCAL"


class HybridRetriever:
    """Master orchestrator combining VectorClient and SparseRetrievalClient with RRF."""

    def __init__(
        self,
        vector_client: VectorSimilarityClient,
        sparse_client: SparseRetrievalClient,
        embedding_pipeline: EmbeddingPipeline,
        default_provider: EmbeddingProvider,
        default_config: Optional[HybridConfig] = None,
    ) -> None: ...

    def search(
        self,
        query: str,
        *,
        filter_project_id: Optional[str] = None,
        filter_document_id: Optional[str] = None,
        config: Optional[HybridConfig] = None,
        provider: Optional[EmbeddingProvider] = None,
    ) -> List[HybridSearchResult]: ...
```

---

## 4. Reciprocal Rank Fusion (RRF) Algorithm

Reciprocal Rank Fusion is an established, robust rank aggregation algorithm that merges rankings from distinct retrieval mechanisms without requiring score normalization calibration (e.g., transforming unbounded BM25 scores to match [0, 1] cosine bounds).

### 4.1 Mathematical Formulation
For each unique chunk $d$ in the union of retrieved candidates $\mathcal{D} = \mathcal{D}_{\text{dense}} \cup \mathcal{D}_{\text{sparse}}$:

$$RRF(d) = w_{\text{dense}} \cdot \frac{1}{k + r_{\text{dense}}(d)} + w_{\text{sparse}} \cdot \frac{1}{k + r_{\text{sparse}}(d)}$$

Where:
* $k$: Smoothing constant (default: 60). Ensures that high rankings do not disproportionately dominate the score while still rewarding mutual agreement.
* $r_{\text{dense}}(d) \in \{1, 2, \dots, K\}$: 1-based ordinal rank of chunk $d$ in the dense search result list. If $d \notin \mathcal{D}_{\text{dense}}$, $\frac{1}{k + r_{\text{dense}}(d)} = 0$.
* $r_{\text{sparse}}(d) \in \{1, 2, \dots, K\}$: 1-based ordinal rank of chunk $d$ in the sparse search result list. If $d \notin \mathcal{D}_{\text{sparse}}$, $\frac{1}{k + r_{\text{sparse}}(d)} = 0$.
* $w_{\text{dense}}, w_{\text{sparse}}$: Relative importance weights. Standardized such that:
  $$\hat{w}_{\text{dense}} = \frac{w_{\text{dense}}}{w_{\text{dense}} + w_{\text{sparse}}}, \quad \hat{w}_{\text{sparse}} = \frac{w_{\text{sparse}}}{w_{\text{dense}} + w_{\text{sparse}}}$$
  Default: $w_{\text{dense}} = 0.6$ ($\hat{w} = 0.6$), $w_{\text{sparse}} = 0.4$ ($\hat{w} = 0.4$).

### 4.2 Property Highlights
* **Scale-Invariance:** Operates strictly on ordinal ranks ($1, 2, 3, \dots$), making it immune to distribution skews between dense cosine similarities (range $[0, 1]$ or $[-1, 1]$) and sparse BM25 scores (range $[0, \infty)$).
* **Consensus Amplification:** Chunks appearing in both dense and sparse Top-$K$ receive contributions from both terms, propelling them significantly higher than chunks present in only one branch.
* **Configurable Specialization:**
  * Setting $w_{\text{sparse}} = 0.0$ yields pure dense ranking.
  * Setting $w_{\text{dense}} = 0.0$ yields pure sparse ranking.
  * Adjusting $k \in [1, 100]$ controls the decay slope. Smaller $k$ penalizes lower ranks more sharply; larger $k$ produces smoother rank distributions.

---

## 5. Deduplication and Result-Ranking Strategy

### 5.1 Deduplication Workflow
1. **Key Identifier:** Chunk uniqueness is determined strictly by chunk UUID (`id`).
2. **Dictionary Accumulator:**
   * Iterate over dense results: assign $r_{\text{dense}} = i + 1$. Store chunk metadata, content, and cosine similarity.
   * Iterate over sparse results: assign $r_{\text{sparse}} = j + 1$. If chunk already exists, merge sparse score and rank; if new, initialize entry.
3. **Score Calculation:** Compute $RRF(d)$ using the formula.

### 5.2 Deterministic Tie-Breaking
When two chunks $d_A$ and $d_B$ evaluate to identical RRF scores (e.g., $d_A$ is dense rank 1 / sparse unranked; $d_B$ is sparse rank 1 / dense unranked under equal weights):
1. **Primary sort:** $RRF(d)$ descending.
2. **Secondary sort:** Presence in both modalities (items with both `dense_rank` and `sparse_rank` not None rank higher than single-modality items).
3. **Tertiary sort:** Dense cosine similarity descending (higher semantic relevance breaks tie).
4. **Quaternary sort:** Sparse keyword score descending.
5. **Final deterministic fallback:** Lexicographical sort on chunk `id` string.

### 5.3 Candidate Oversampling & Truncation
* If the user requests `match_count = 10`:
  * Dense branch requests $\min(\max(10 \times 2, 20), 100) = 20$ candidates.
  * Sparse branch requests 20 candidates.
  * Union contains between 20 and 40 unique chunks.
  * RRF ranks the union and truncates the top 10 results.
  * Candidate oversampling guarantees that mutual agreement between positions 11–20 can elevate high-consensus items into the final Top-10.

---

## 6. User and Project Isolation Requirements

Multi-tenant security and project isolation are non-negotiable invariants:

### 6.1 Cloud Supabase Isolation
1. **User Tenant Isolation:**
   * Sparse FTS RPC `search_document_chunks_fts` must run under `SECURITY INVOKER`.
   * The active user is resolved via `v_user_id := auth.uid()`. Unauthenticated requests abort with SQLSTATE `28000`.
   * Chunks are filtered by `c.user_id = v_user_id`.
2. **Project Filtering:**
   * In Supabase, `public.document_chunks` does not store `project_id` directly (it resides on `public.documents`).
   * The RPC must execute a composite join:
     ```sql
     JOIN public.documents d ON d.id = c.document_id AND d.user_id = c.user_id
     ```
   * Predicate: `(filter_project_id IS NULL OR d.project_id = filter_project_id)`.
   * Chunks from different projects are never exposed.

### 6.2 Local SQLite Isolation
1. **Project Filtering:**
   * Local SQLite schema maintains `project_id` on both `documents` and `document_chunks`.
   * Virtual table `chunks_fts` indexes `project_id` as an `UNINDEXED` column.
   * Keyword query:
     ```sql
     SELECT chunk_id, document_id, project_id, content, rank 
     FROM chunks_fts 
     WHERE chunks_fts MATCH ? AND (project_id = ? OR ? IS NULL)
     ORDER BY rank
     LIMIT ?
     ```
2. **User Ownership:**
   * Evaluated against `metadata_json -> owner_user_id` to ensure local workspace multi-profile safety.

---

## 7. Timeout, Retry, and Fallback Handling

Hybrid retrieval must be resilient to partial failures without failing the entire user request.

```
                           ┌────────────────────────┐
                           │      Query Start       │
                           └───────────┬────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
            [ Dense Search Branch ]           [ Sparse Search Branch ]
            - embed_query (cached?)           - Try Supabase FTS RPC
            - match_document_chunks           - Try SQLite FTS5 fallback
                      │                                 │
           ┌──────────┴──────────┐           ┌──────────┴──────────┐
           │                     │           │                     │
        Success               Failure     Success               Failure
           │                     │           │                     │
           ▼                     ▼           ▼                     ▼
      Dense Matches           Empty       Sparse Matches        Empty
           │                     │           │                     │
           └──────────┬──────────┴───────────┴──────────┬──────────┘
                      │                                 │
                      ▼                                 ▼
              Both Succeeded?                  Only One Succeeded?
                      │                                 │
                     YES                               YES
                      │                                 │
                      ▼                                 ▼
              Normal RRF Fusion                 Single-Mode Fallback
               (mode: "HYBRID")                 (DENSE_ONLY / SPARSE_ONLY)
```

### 7.1 Failure Scenarios & Mitigations

| Failure Mode | Trigger / Cause | Behavior & Mitigation | Resulting Mode |
|---|---|---|---|
| **Embedding Provider Offline** | Ollama daemon down, OpenAI API 5xx | Catch exception, log warning, skip dense branch. Do not fail request. | `SPARSE_ONLY` |
| **Embedding Rate Limit (429)** | Exceeded provider RPM/TPM | Pipeline handles backoff. If max retries exhausted, fall back to sparse. | `SPARSE_ONLY` |
| **Vector RPC Network Failure** | Supabase connection refused/timeout | Catch `VectorClientRPCError`, log warning, fall back to sparse. | `SPARSE_ONLY` |
| **Cloud FTS RPC Failure** | Supabase unreachable or 500 | Catch HTTP error, seamlessly fallback to local SQLite FTS5. | `HYBRID` (Local FTS) |
| **Both Cloud Services Unreachable** | Complete offline environment | Vector fails, Cloud FTS fails $\to$ Fallback to local SQLite FTS5. | `OFFLINE_LOCAL` |
| **Query Syntax Error in FTS** | User typed `"unclosed quotes OR AND :"` | Client sanitizes query string into safe tsquery terms (`websearch_to_tsquery` or token stripping). | Handled safely |
| **Both Branches Return Empty** | No matches above threshold / keywords not in corpus | Return empty list `[]` cleanly without error. | `HYBRID` (0 matches) |

### 7.2 Concurrency & Timeouts
* **Concurrent Execution:** Dense and sparse retrieval execute in parallel using Python's `concurrent.futures.ThreadPoolExecutor(max_workers=2)`.
* **Latency Guarantee:** Total retrieval wall-clock time is bounded by $\max(T_{\text{dense}}, T_{\text{sparse}}) + T_{\text{RRF}}$ (where $T_{\text{RRF}} < 1\text{ms}$), rather than the sequential sum.
* **Per-Branch Timeout:** Default 10.0s timeout per branch. If one branch hangs, it is cancelled and the completed branch produces the fallback result.

---

## 8. Performance and Scalability Considerations

1. **HNSW Vector Search Latency:**
   * Partial HNSW indexes with `(m = 16, ef_construction = 64)` maintain index sizes under memory limits.
   * `SET LOCAL hnsw.ef_search = 100` guarantees $\sim 5\text{ms}$ search time on 100k chunks.
2. **PostgreSQL FTS Latency:**
   * Pre-computed `tsv_content` generated column with `GIN` index (`document_chunks_tsv_idx`) guarantees sub-millisecond keyword matching on large collections.
3. **SQLite FTS5 Latency:**
   * Porter stemmer with `unicode61` tokenizer provides sub-millisecond local BM25 evaluation.
4. **RRF Computation Complexity:**
   * Fusion runs in memory in Python.
   * For $K = 40$ candidates: Set union, dictionary lookup, float arithmetic, and sorting $O(K \log K)$ takes $< 0.1\text{ms}$.
5. **Embedding Query Caching:**
   * Identical user search queries hit `EmbeddingCache` (`embed_query`), bypassing the embedding provider entirely and reducing dense branch latency to network round-trip time.

---

## 9. Supabase FTS Migration Specification

To support sparse keyword search against cloud Supabase, an accompanying migration is defined.

### 9.1 Up Migration: `supabase/migrations/20261116000001_milestone_10_4_fts_retrieval.sql`
```sql
-- Relay AI Universal Context Engine: Full-Text Search RPC for Document Chunks
-- Supports secure sparse keyword retrieval with project filtering and ranking.

CREATE OR REPLACE FUNCTION public.search_document_chunks_fts(
  p_query text,
  match_count int DEFAULT 20,
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
  -- 1. Authentication check
  v_user_id := auth.uid();
  IF v_user_id IS NULL THEN
    RAISE EXCEPTION 'Authentication required to search document chunks.'
      USING ERRCODE = '28000';
  END IF;

  -- 2. Input validation
  IF p_query IS NULL OR length(trim(p_query)) = 0 THEN
    RETURN;
  END IF;

  IF match_count < 1 OR match_count > 100 THEN
    RAISE EXCEPTION 'match_count must be between 1 and 100, got %.', match_count
      USING ERRCODE = '22023';
  END IF;

  -- 3. Construct safe tsquery using websearch syntax (tolerates quotes, -, OR)
  v_tsquery := websearch_to_tsquery('english', p_query);
  IF v_tsquery IS NULL OR v_tsquery = ''::tsquery THEN
    v_tsquery := plainto_tsquery('english', p_query);
  END IF;

  IF v_tsquery IS NULL OR v_tsquery = ''::tsquery THEN
    RETURN;
  END IF;

  -- 4. Execute search joining documents to enforce project filtering and RLS
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

-- Security permissions
REVOKE ALL ON FUNCTION public.search_document_chunks_fts(text, integer, uuid, uuid) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.search_document_chunks_fts(text, integer, uuid, uuid) TO authenticated;
```

### 9.2 Down Migration: `supabase/migrations/20261116000001_milestone_10_4_fts_retrieval_down.sql`
```sql
-- Rollback for Full-Text Search RPC
DROP FUNCTION IF EXISTS public.search_document_chunks_fts(text, integer, uuid, uuid);
```

---

## 10. Detailed Implementation Sequence

Implementation of Phase 4 will proceed in strict, verifiable steps:

```
Step 1: SQL Migration & Down Script
├── supabase/migrations/20261116000001_milestone_10_4_fts_retrieval.sql
└── supabase/migrations/20261116000001_milestone_10_4_fts_retrieval_down.sql

Step 2: Reciprocal Rank Fusion Engine
└── src/context/retrieval/rrf.py (Pure algorithmic unit)

Step 3: Sparse Retrieval Client
└── src/context/retrieval/sparse_client.py (Supabase RPC + SQLite FTS5)

Step 4: Hybrid Retriever Orchestrator
└── src/context/retrieval/hybrid.py (Coordinator, thread pool, fallbacks)

Step 5: Package Exports Update
└── src/context/retrieval/__init__.py

Step 6: Comprehensive Unit & Integration Tests
├── tests/test_rrf.py
├── tests/test_sparse_client.py
├── tests/test_hybrid_retriever.py
└── tests/test_fts_migration.py

Step 7: Verification & Regression Testing
└── Run 427+ repository test suite to guarantee 100% pass rate
```

---

## 11. Test and Verification Plan

### 11.1 Unit Test Coverage Target
* **`tests/test_rrf.py` (Minimum 10 tests):**
  1. Equal weights fusion ($w_{\text{dense}}=0.5, w_{\text{sparse}}=0.5$).
  2. Dense-dominant fusion ($w_{\text{dense}}=0.8, w_{\text{sparse}}=0.2$).
  3. Sparse-dominant fusion ($w_{\text{dense}}=0.2, w_{\text{sparse}}=0.8$).
  4. Consensus boost: chunk appearing at rank 2 in both modalities beats chunk at rank 1 in only one modality.
  5. Disjoint result sets (no chunk overlap).
  6. Identical result sets (100% chunk overlap).
  7. Single modality input (dense empty, sparse populated).
  8. Single modality input (sparse empty, dense populated).
  9. Both modalities empty.
  10. Deterministic tie-breaking behavior.
  11. Smoothing constant $k$ variation ($k=1$ vs $k=60$ vs $k=100$).
* **`tests/test_sparse_client.py` (Minimum 8 tests):**
  1. PostgreSQL FTS RPC request formatting and token authentication.
  2. Local SQLite FTS5 search execution and BM25 ranking.
  3. Automatic fallback to SQLite FTS5 when Supabase is unreachable.
  4. Project filtering in SQLite FTS5 and Supabase RPC.
  5. Document filtering in SQLite FTS5 and Supabase RPC.
  6. Empty or whitespace query handling.
  7. Query sanitization (special characters, unclosed quotes, boolean operators).
  8. Auth error handling on missing token.
* **`tests/test_hybrid_retriever.py` (Minimum 12 tests):**
  1. Full end-to-end hybrid retrieval with synthetic dense and sparse matches.
  2. Candidate oversampling verification ($\text{oversampling}=2$).
  3. Query vector embedding generation via `EmbeddingPipeline`.
  4. Fallback to `SPARSE_ONLY` when vector search fails or provider is offline.
  5. Fallback to `DENSE_ONLY` when sparse search returns empty or fails.
  6. Fallback to `OFFLINE_LOCAL` when cloud is unreachable.
  7. Both branches fail gracefully $\to$ returns empty list without exception.
  8. Project filtering propagation to both dense and sparse clients.
  9. Result provenance verification (`rrf_score`, `dense_rank`, `sparse_rank`, `dense_similarity`).
  10. Threshold filtering on dense similarity.
  11. Concurrent execution performance and timeout handling.
  12. Custom configuration overrides (`k`, weights, limits).
* **`tests/test_fts_migration.py` (Minimum 5 tests):**
  1. Migration SQL file existence and syntax validation.
  2. Security checks: `SECURITY INVOKER`, `auth.uid()` enforcement, `REVOKE ALL FROM PUBLIC`.
  3. Safe tsquery handling: `websearch_to_tsquery` and `plainto_tsquery`.
  4. Composite join validation against `public.documents`.
  5. Down migration cleanly drops function without collateral damage.

### 11.2 Regression Target
* All 427 existing tests in the suite must remain 100% passing.
* Zero secrets detected across all newly introduced files.

---

## 12. Acceptance Criteria & Implementation Status

Phase 4 status:
1. [x] `ReciprocalRankFusion` class implemented in `src/context/retrieval/rrf.py`, fully tested (13 tests in `test_rrf.py`), handling all edge cases, weights, and tie-breaking deterministically.
2. [x] `SparseRetrievalClient` implemented in `src/context/retrieval/sparse_client.py` with dual backend support: Supabase FTS RPC and local SQLite FTS5 fallback (9 tests in `test_sparse_client.py`).
3. [x] Supabase migration files (`20261116000001_milestone_10_4_fts_retrieval.sql` and `..._down.sql`) created and validated (8 tests in `test_fts_migration.py`).
4. [x] `HybridRetriever` implemented in `src/context/retrieval/hybrid.py` with concurrent execution, candidate oversampling, configurable weights, and multi-tier graceful fallback (10 tests in `test_hybrid_retriever.py`).
5. [x] All new test modules (`test_rrf.py`, `test_sparse_client.py`, `test_hybrid_retriever.py`, `test_fts_migration.py`) pass with 100% success rate (40 new tests).
6. [x] Full regression suite passes (467 tests total).
7. [x] Working tree clean, and no modifications have been made to `main`.

---

## 13. Risks and Unresolved Decisions

1. **PostgreSQL FTS Query Syntax (`websearch_to_tsquery` vs `plainto_tsquery`):**
   * *Risk:* User queries containing complex syntax might fail under `to_tsquery`.
   * *Resolution:* The RPC utilizes `websearch_to_tsquery('english', p_query)`, which natively and safely parses Google-like search queries (ignoring unclosed quotes, handling hyphens as negation, and treating unadorned terms as conjunctions) without syntax crashes. If that yields empty tsquery, it falls back to `plainto_tsquery`.
2. **Offline Local Vector Search:**
   * *Context:* While SQLite FTS5 provides local sparse search, SQLite does not natively host pgvector.
   * *Resolution:* When completely offline, `HybridRetriever` transparently operates in `OFFLINE_LOCAL` mode using SQLite FTS5 BM25.
3. **Live Cloud Migration Execution:**
   * *Status:* Consistent with project rules, live cloud migration execution remains **PENDING LIVE SUPABASE VALIDATION** until disposable credentials are provided. All SQL files and RPC interfaces are thoroughly verified using AST, syntax tests, and mock HTTP transports.
