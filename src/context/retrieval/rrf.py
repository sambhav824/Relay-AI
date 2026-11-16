"""Relay AI Universal Context Engine — Reciprocal Rank Fusion (RRF).

Combines ranked results from distinct retrieval modalities (e.g. dense vector search
and sparse keyword search) using Reciprocal Rank Fusion.

Formula:
    RRF(d) = (w_dense / (k + r_dense(d))) + (w_sparse / (k + r_sparse(d)))

Defaults:
    k = 60
    weight_dense = 0.6
    weight_sparse = 0.4
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger("RelayAI.RRF")


@dataclass
class FusedResult:
    """Represents a deduplicated, RRF-scored retrieval result with full provenance."""

    id: str
    rrf_score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    document_id: Optional[str] = None
    project_id: Optional[str] = None
    chunk_index: int = 0
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReciprocalRankFusion:
    """Pure, scale-invariant rank aggregation engine for hybrid retrieval."""

    def __init__(
        self,
        k: int = 60,
        weight_dense: float = 0.6,
        weight_sparse: float = 0.4,
    ) -> None:
        if k <= 0:
            raise ValueError(f"Smoothing constant k must be positive, got {k}.")
        if weight_dense < 0.0 or weight_sparse < 0.0:
            raise ValueError(
                f"Weights must be non-negative, got weight_dense={weight_dense}, weight_sparse={weight_sparse}."
            )
        if weight_dense == 0.0 and weight_sparse == 0.0:
            raise ValueError("At least one retrieval weight must be positive.")

        self.k = int(k)
        # Normalize weights so they sum to 1.0
        total_w = weight_dense + weight_sparse
        self.weight_dense = float(weight_dense) / total_w
        self.weight_sparse = float(weight_sparse) / total_w

    def fuse(
        self,
        dense_results: Sequence[Any],
        sparse_results: Sequence[Any],
        limit: Optional[int] = None,
    ) -> List[FusedResult]:
        """Merge and rank dense and sparse results using Reciprocal Rank Fusion.

        Args:
            dense_results: Sequence of dense search matches ordered best-to-worst.
            sparse_results: Sequence of sparse search matches ordered best-to-worst.
            limit: Maximum number of fused results to return. If None, returns all.

        Returns:
            List of FusedResult sorted by rrf_score descending with deterministic tie-breaking.
        """
        if limit is not None and limit < 1:
            raise ValueError(f"Limit must be positive if specified, got {limit}.")

        # Helper to extract attribute or dict key safely
        def _get_field(obj: Any, name: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        def _get_score(obj: Any) -> Optional[float]:
            # Dense matches might have 'similarity' or 'score'
            val = _get_field(obj, "similarity")
            if val is None:
                val = _get_field(obj, "score")
            try:
                return float(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        # Accumulator: chunk_id -> dict of properties
        accum: Dict[str, Dict[str, Any]] = {}

        # 1. Process Dense Results (1-based ordinal rank)
        dense_seen_ids = set()
        dense_rank = 1
        for item in dense_results:
            cid = str(_get_field(item, "id", "") or "").strip()
            if not cid:
                continue
            # Deduplicate multiple occurrences within the same modality
            if cid in dense_seen_ids:
                continue
            dense_seen_ids.add(cid)

            score = _get_score(item)
            accum[cid] = {
                "id": cid,
                "document_id": str(_get_field(item, "document_id") or "") or None,
                "project_id": str(_get_field(item, "project_id") or "") or None,
                "chunk_index": int(_get_field(item, "chunk_index", 0) or 0),
                "content": str(_get_field(item, "content", "") or ""),
                "metadata": dict(_get_field(item, "metadata", {}) or {}),
                "dense_rank": dense_rank,
                "sparse_rank": None,
                "dense_score": score,
                "sparse_score": None,
            }
            dense_rank += 1

        # 2. Process Sparse Results (1-based ordinal rank)
        sparse_seen_ids = set()
        sparse_rank = 1
        for item in sparse_results:
            cid = str(_get_field(item, "id", "") or "").strip()
            if not cid:
                continue
            if cid in sparse_seen_ids:
                continue
            sparse_seen_ids.add(cid)

            score = _get_score(item)
            if cid in accum:
                accum[cid]["sparse_rank"] = sparse_rank
                accum[cid]["sparse_score"] = score
                # If document_id/project_id were missing in dense, populate from sparse
                if not accum[cid]["document_id"]:
                    accum[cid]["document_id"] = str(_get_field(item, "document_id") or "") or None
                if not accum[cid]["project_id"]:
                    accum[cid]["project_id"] = str(_get_field(item, "project_id") or "") or None
                if not accum[cid]["content"] and _get_field(item, "content"):
                    accum[cid]["content"] = str(_get_field(item, "content"))
            else:
                accum[cid] = {
                    "id": cid,
                    "document_id": str(_get_field(item, "document_id") or "") or None,
                    "project_id": str(_get_field(item, "project_id") or "") or None,
                    "chunk_index": int(_get_field(item, "chunk_index", 0) or 0),
                    "content": str(_get_field(item, "content", "") or ""),
                    "metadata": dict(_get_field(item, "metadata", {}) or {}),
                    "dense_rank": None,
                    "sparse_rank": sparse_rank,
                    "dense_score": None,
                    "sparse_score": score,
                }
            sparse_rank += 1

        # 3. Calculate RRF Score for Each Unique Chunk
        fused_items: List[FusedResult] = []
        for entry in accum.values():
            rrf_score = 0.0
            r_d = entry["dense_rank"]
            r_s = entry["sparse_rank"]

            if r_d is not None:
                rrf_score += self.weight_dense / (self.k + r_d)
            if r_s is not None:
                rrf_score += self.weight_sparse / (self.k + r_s)

            fused_items.append(
                FusedResult(
                    id=entry["id"],
                    rrf_score=rrf_score,
                    dense_rank=r_d,
                    sparse_rank=r_s,
                    dense_score=entry["dense_score"],
                    sparse_score=entry["sparse_score"],
                    document_id=entry["document_id"],
                    project_id=entry["project_id"],
                    chunk_index=entry["chunk_index"],
                    content=entry["content"],
                    metadata=entry["metadata"],
                )
            )

        # 4. Deterministic Tie-Breaking Sort
        # Sort Key:
        # 1. rrf_score descending (-rrf_score)
        # 2. Both modalities present (0 if both present, 1 if single modality)
        # 3. Dense score descending (if present, else float('-inf'))
        # 4. Sparse score descending (if present, else float('-inf'))
        # 5. Chunk ID string ascending (lexicographical)
        def _sort_key(item: FusedResult) -> tuple:
            has_both = 0 if (item.dense_rank is not None and item.sparse_rank is not None) else 1
            d_score = item.dense_score if item.dense_score is not None else float("-inf")
            s_score = item.sparse_score if item.sparse_score is not None else float("-inf")
            return (-item.rrf_score, has_both, -d_score, -s_score, item.id)

        fused_items.sort(key=_sort_key)

        if limit is not None:
            return fused_items[:limit]
        return fused_items
