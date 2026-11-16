# Relay AI Universal Context Engine — Hybrid Retrieval System
## Milestone 10.4, Phase 4 Architecture & Developer Guide

The Hybrid Retrieval Engine combines dense semantic vector similarity search and sparse full-text keyword search into a unified, resilient retrieval system using Reciprocal Rank Fusion (RRF).

---

## 1. System Architecture

```
                               ┌────────────────────────┐
                               │       User Query       │
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
                          │    Deduplicated, Fused Top-N    │
                          │    List[HybridSearchResult]     │
                          └─────────────────────────────────┘
```

---

## 2. Component Responsibilities

| Component | Module | Responsibility |
|---|---|---|
| **`ReciprocalRankFusion`** | `src/context/retrieval/rrf.py` | Pure, scale-invariant rank aggregation. Computes combined scores, deduplicates chunks, normalizes weights, and enforces deterministic tie-breaking. |
| **`SparseRetrievalClient`** | `src/context/retrieval/sparse_client.py` | Full-text keyword search client. Queries cloud Supabase RPC (`search_document_chunks_fts`) when online, with transparent fallback to local SQLite FTS5 (`chunks_fts`) when offline or on network failure. |
| **`VectorSimilarityClient`** | `src/context/retrieval/vector_client.py` | Dense vector similarity client. Queries Supabase pgvector RPC (`match_document_chunks`) using authenticated JWT tokens, multi-tenant checks, and cosine distance ordering. |
| **`EmbeddingPipeline`** | `src/context/embeddings/pipeline.py` | Query embedding generator (`embed_query`). Checks local SQLite `embedding_cache` first before invoking external provider API. |
| **`HybridRetriever`** | `src/context/retrieval/hybrid.py` | Master orchestrator. Dispatches concurrent dense and sparse searches, applies candidate oversampling ($\max(2N, 20)$), fuses results via RRF, tracks provenance metadata, and handles multi-tier fallbacks. |

---

## 3. Reciprocal Rank Fusion (RRF) Formulation

RRF aggregates ranked lists without requiring score normalization calibration across disparate metric scales (cosine similarity $[-1, 1]$ vs unbounded BM25 scores $[0, \infty)$):

$$RRF(d) = w_{\text{dense}} \cdot \frac{1}{k + r_{\text{dense}}(d)} + w_{\text{sparse}} \cdot \frac{1}{k + r_{\text{sparse}}(d)}$$

### Default Parameters:
* $k = 60$: Smoothing constant preventing high ranks from disproportionately overpowering the distribution.
* $w_{\text{dense}} = 0.6$: Weight allocated to dense semantic similarity.
* $w_{\text{sparse}} = 0.4$: Weight allocated to exact keyword matching.
* Normalization: Weights are automatically normalized: $\hat{w}_d = \frac{w_d}{w_d + w_s}$, $\hat{w}_s = \frac{w_s}{w_d + w_s}$.

### Ranking & Tie-Breaking:
1. Primary sort: $RRF(d)$ descending.
2. Secondary sort: Consensus presence (chunks appearing in both modalities beat single-modality chunks).
3. Tertiary sort: Dense cosine similarity descending.
4. Quaternary sort: Sparse keyword score descending.
5. Deterministic fallback: Chunk ID string ascending.

---

## 4. Explicit Retrieval Modes

Every `HybridSearchResult` contains an explicit `retrieval_mode` field reflecting how the result was formed:

* **`HYBRID`**: Both dense vector similarity and sparse keyword search executed successfully and were fused via RRF.
* **`DENSE_ONLY`**: Dense retrieval succeeded, but sparse retrieval failed or returned zero matches.
* **`SPARSE_ONLY`**: Sparse keyword retrieval succeeded, but dense retrieval was unavailable (e.g. embedding provider offline, rate limits, or network timeout).
* **`OFFLINE_SPARSE_ONLY`**: Cloud services were unavailable or offline mode was requested (`prefer_offline=True`); results were retrieved from local SQLite FTS5 (`chunks_fts`).
* **`EMPTY`**: No matching chunks were found in either modality.

> [!IMPORTANT]
> Authentication, permission, or tenant-isolation failures (`VectorClientAuthError`, `SparseClientAuthError`) are **never silently converted** into degraded fallback results. They are immediately propagated to the caller to prevent unauthorized access and alert the operator.

---

## 5. Security & Multi-Tenant Model

1. **Authentication Enforcement:**
   * Supabase RPCs enforce `v_user_id := auth.uid()`. Unauthenticated requests immediately fail with SQLSTATE `28000`.
   * Clients pass Bearer JWT tokens dynamically supplied by `AuthManager`.
2. **Security Invoker:**
   * Both vector and FTS RPCs execute with `SECURITY INVOKER`, running strictly within the caller's row-level security permissions.
   * `SECURITY DEFINER` is strictly rejected.
3. **Composite Join for Project Filtering:**
   * In Supabase, `public.document_chunks` does not store `project_id`. The RPCs perform a composite join:
     ```sql
     JOIN public.documents d ON d.id = c.document_id AND d.user_id = c.user_id
     ```
   * Enforces cross-user isolation and filters by `d.project_id = filter_project_id`.
4. **Local SQLite Parameterization:**
   * All SQLite FTS queries use parameterized SQL (`?`) and regex token sanitization, preventing SQL injection and FTS5 syntax crashes.

---

## 6. Database Migrations

### Up Migration: `supabase/migrations/20261116000001_milestone_10_4_fts_retrieval.sql`
Creates `public.search_document_chunks_fts` RPC with:
* `websearch_to_tsquery` / `plainto_tsquery` safe parsing.
* GIN index acceleration via `document_chunks_tsv_idx`.
* `ts_rank_cd` scoring.
* `SECURITY INVOKER`.
* Grants execution solely to `authenticated` role.

### Down Migration: `supabase/migrations/20261116000001_milestone_10_4_fts_retrieval_down.sql`
Drops `public.search_document_chunks_fts`. Does not alter table columns or delete stored document chunks.

---

## 7. Example Usage

```python
import sqlite3
from src.context.auth.manager import AuthManager
from src.context.embeddings.cache import EmbeddingCache
from src.context.embeddings.pipeline import EmbeddingPipeline
from src.context.retrieval import (
    HybridConfig,
    HybridRetriever,
    SparseRetrievalClient,
    VectorSimilarityClient,
)
from src.providers.embeddings import OpenAIEmbeddingProvider

# 1. Initialize clients
auth_mgr = AuthManager(...)
conn = sqlite3.connect("relay_context.db")

vector_client = VectorSimilarityClient(auth_manager=auth_mgr)
sparse_client = SparseRetrievalClient(auth_manager=auth_mgr, sqlite_connection=conn)

cache = EmbeddingCache(conn)
pipeline = EmbeddingPipeline(cache=cache)
provider = OpenAIEmbeddingProvider(dimension=1536)

# 2. Configure retriever
config = HybridConfig(
    k=60,
    weight_dense=0.6,
    weight_sparse=0.4,
    match_count=10,
    concurrent_execution=True,
)

retriever = HybridRetriever(
    vector_client=vector_client,
    sparse_client=sparse_client,
    embedding_pipeline=pipeline,
    default_provider=provider,
    default_config=config,
)

# 3. Execute hybrid search
results = retriever.search(
    query="quarterly revenue projections and EBITDA",
    filter_project_id="project-uuid-1234",
)

for item in results:
    print(f"[{item.retrieval_mode}] RRF: {item.rrf_score:.4f} | Dense Rank: {item.dense_rank} | Sparse Rank: {item.sparse_rank}")
    print(f"Content: {item.content[:100]}...\n")
```

---

## 8. Known Limitations & Verification Status

* **Live Supabase Validation:** Offline local workspace lacks active disposable Supabase credentials. All SQL syntax, RPC contracts, PostgREST payloads, and RLS behaviors are thoroughly verified via AST and mock transport suites (**PENDING LIVE SUPABASE VALIDATION**).
* **Local Offline Vector Search:** SQLite does not host native pgvector. When working completely offline, the system seamlessly falls back to `OFFLINE_SPARSE_ONLY` using SQLite FTS5 BM25 search.
