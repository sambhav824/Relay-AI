"""Unit tests for Relay AI Reciprocal Rank Fusion (Milestone 10.4, Phase 4).

Validates:
- Mathematical correctness of RRF scoring formula
- Default weights (0.6, 0.4) and k=60
- Custom weights and parameter validation
- Weight normalization
- Missing rankings handling
- Empty and single-modality result handling
- Intra-modality duplicate ID deduplication
- Deterministic tie-breaking behavior
- Scale invariance (ordinal ranks vs raw scores)
- Consensus amplification
"""

from __future__ import annotations

from unittest import TestCase

from src.context.retrieval.rrf import FusedResult, ReciprocalRankFusion


class TestReciprocalRankFusion(TestCase):
    """Test suite for ReciprocalRankFusion rank aggregation engine."""

    def test_default_weights_and_k(self):
        """Verify default parameters: k=60, w_dense=0.6, w_sparse=0.4."""
        rrf = ReciprocalRankFusion()
        self.assertEqual(rrf.k, 60)
        self.assertAlmostEqual(rrf.weight_dense, 0.6)
        self.assertAlmostEqual(rrf.weight_sparse, 0.4)

    def test_parameter_validation(self):
        """Verify parameter bounds enforcement on k, weights, and limit."""
        with self.assertRaises(ValueError):
            ReciprocalRankFusion(k=0)

        with self.assertRaises(ValueError):
            ReciprocalRankFusion(k=-5)

        with self.assertRaises(ValueError):
            ReciprocalRankFusion(weight_dense=-0.1)

        with self.assertRaises(ValueError):
            ReciprocalRankFusion(weight_sparse=-0.1)

        with self.assertRaises(ValueError):
            ReciprocalRankFusion(weight_dense=0.0, weight_sparse=0.0)

        rrf = ReciprocalRankFusion()
        with self.assertRaises(ValueError):
            rrf.fuse([], [], limit=0)

        with self.assertRaises(ValueError):
            rrf.fuse([], [], limit=-1)

    def test_weight_normalization(self):
        """Verify un-normalized weights (e.g. 6.0 and 4.0) normalize to 0.6 and 0.4."""
        rrf = ReciprocalRankFusion(k=60, weight_dense=6.0, weight_sparse=4.0)
        self.assertAlmostEqual(rrf.weight_dense, 0.6)
        self.assertAlmostEqual(rrf.weight_sparse, 0.4)

    def test_mathematical_scoring_single_item_both_modalities(self):
        """Verify exact RRF formula: w_d / (k + r_d) + w_s / (k + r_s)."""
        rrf = ReciprocalRankFusion(k=60, weight_dense=0.6, weight_sparse=0.4)
        dense = [{"id": "chunk_1", "similarity": 0.95}]
        sparse = [{"id": "chunk_1", "score": 0.42}]

        results = rrf.fuse(dense, sparse)
        self.assertEqual(len(results), 1)

        item = results[0]
        # rank 1 for both: 0.6 / (60 + 1) + 0.4 / (60 + 1) = 1.0 / 61
        expected_score = 1.0 / 61.0
        self.assertAlmostEqual(item.rrf_score, expected_score, places=7)
        self.assertEqual(item.dense_rank, 1)
        self.assertEqual(item.sparse_rank, 1)
        self.assertAlmostEqual(item.dense_score, 0.95)
        self.assertAlmostEqual(item.sparse_score, 0.42)

    def test_consensus_amplification(self):
        """Verify chunk appearing at rank 2 in both modalities beats rank 1 in only one modality."""
        rrf = ReciprocalRankFusion(k=60, weight_dense=0.5, weight_sparse=0.5)

        # chunk_mutual: rank 2 in dense, rank 2 in sparse
        # chunk_dense_only: rank 1 in dense, absent in sparse
        # chunk_sparse_only: rank 1 in sparse, absent in dense
        dense = [
            {"id": "chunk_dense_only", "similarity": 0.99},
            {"id": "chunk_mutual", "similarity": 0.85},
        ]
        sparse = [
            {"id": "chunk_sparse_only", "score": 10.0},
            {"id": "chunk_mutual", "score": 5.0},
        ]

        results = rrf.fuse(dense, sparse)
        # Expected scores:
        # chunk_dense_only: 0.5 / (60 + 1) = 0.5 / 61 ≈ 0.0081967
        # chunk_sparse_only: 0.5 / (60 + 1) = 0.5 / 61 ≈ 0.0081967
        # chunk_mutual: 0.5 / (60 + 2) + 0.5 / (60 + 2) = 1.0 / 62 ≈ 0.016129
        self.assertEqual(results[0].id, "chunk_mutual")
        self.assertGreater(results[0].rrf_score, results[1].rrf_score)

    def test_disjoint_result_sets(self):
        """Verify correct merging and ranking when dense and sparse have zero chunk overlap."""
        rrf = ReciprocalRankFusion(k=60, weight_dense=0.6, weight_sparse=0.4)
        dense = [
            {"id": "d1", "similarity": 0.9},
            {"id": "d2", "similarity": 0.8},
        ]
        sparse = [
            {"id": "s1", "score": 2.0},
            {"id": "s2", "score": 1.0},
        ]

        results = rrf.fuse(dense, sparse)
        self.assertEqual(len(results), 4)

        # d1: 0.6 / 61 ≈ 0.009836
        # s1: 0.4 / 61 ≈ 0.006557
        # d2: 0.6 / 62 ≈ 0.009677
        # s2: 0.4 / 62 ≈ 0.006451
        expected_order = ["d1", "d2", "s1", "s2"]
        actual_order = [r.id for r in results]
        self.assertEqual(actual_order, expected_order)

    def test_dense_only_results(self):
        """Verify behavior when sparse results are completely empty."""
        rrf = ReciprocalRankFusion(k=60, weight_dense=0.6, weight_sparse=0.4)
        dense = [
            {"id": "d1", "similarity": 0.9},
            {"id": "d2", "similarity": 0.8},
        ]
        results = rrf.fuse(dense, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "d1")
        self.assertEqual(results[0].dense_rank, 1)
        self.assertIsNone(results[0].sparse_rank)
        self.assertEqual(results[1].id, "d2")

    def test_sparse_only_results(self):
        """Verify behavior when dense results are completely empty."""
        rrf = ReciprocalRankFusion(k=60, weight_dense=0.6, weight_sparse=0.4)
        sparse = [
            {"id": "s1", "score": 5.0},
            {"id": "s2", "score": 2.0},
        ]
        results = rrf.fuse([], sparse)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "s1")
        self.assertIsNone(results[0].dense_rank)
        self.assertEqual(results[0].sparse_rank, 1)
        self.assertEqual(results[1].id, "s2")

    def test_both_empty_results(self):
        """Verify empty input produces empty output list."""
        rrf = ReciprocalRankFusion()
        results = rrf.fuse([], [])
        self.assertEqual(results, [])

    def test_duplicate_chunk_ids_within_modality(self):
        """Verify duplicate chunk IDs in input list are deduplicated to their best rank."""
        rrf = ReciprocalRankFusion()
        dense = [
            {"id": "d1", "similarity": 0.9},
            {"id": "d1", "similarity": 0.7},  # duplicate
            {"id": "d2", "similarity": 0.6},
        ]
        results = rrf.fuse(dense, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "d1")
        self.assertEqual(results[0].dense_rank, 1)
        self.assertEqual(results[1].id, "d2")
        self.assertEqual(results[1].dense_rank, 2)

    def test_limit_truncation(self):
        """Verify limit parameter properly slices top-k items."""
        rrf = ReciprocalRankFusion()
        dense = [{"id": f"d{i}", "similarity": 0.9 - i * 0.1} for i in range(10)]
        results = rrf.fuse(dense, [], limit=3)
        self.assertEqual(len(results), 3)
        self.assertEqual([r.id for r in results], ["d0", "d1", "d2"])

    def test_deterministic_tie_breaking(self):
        """Verify deterministic tie-breaking when two items have identical RRF scores."""
        # Under equal weights, rank 1 in dense-only has score 0.5/61,
        # rank 1 in sparse-only has score 0.5/61.
        rrf = ReciprocalRankFusion(k=60, weight_dense=0.5, weight_sparse=0.5)

        # Item A: dense rank 1, dense score 0.8
        # Item B: sparse rank 1, sparse score 0.9
        # Both single-modality. Tie-breaker preference:
        # 1. both modalities (neither has both)
        # 2. dense score (Item A has dense score 0.8, Item B has None -> Item A wins)
        dense = [{"id": "chunk_A", "similarity": 0.8}]
        sparse = [{"id": "chunk_B", "score": 0.9}]

        results = rrf.fuse(dense, sparse)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].rrf_score, results[1].rrf_score)
        self.assertEqual(results[0].id, "chunk_A")
        self.assertEqual(results[1].id, "chunk_B")

    def test_scale_invariance_ordinal_ranking(self):
        """Verify that extreme raw scores do not distort rank-based fusion."""
        rrf = ReciprocalRankFusion(k=60, weight_dense=0.5, weight_sparse=0.5)

        # Dense similarities are between 0 and 1
        dense = [{"id": "d1", "similarity": 0.9999}]
        # Sparse scores can be arbitrarily large (e.g. 50,000)
        sparse = [{"id": "s1", "score": 50000.0}]

        results = rrf.fuse(dense, sparse)
        # Both are rank 1 in their respective modalities; their RRF scores must be identical
        self.assertAlmostEqual(results[0].rrf_score, results[1].rrf_score)
