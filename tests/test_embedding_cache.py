"""Unit tests for Relay AI Vector Embedding Cache (Milestone 10.4, Phase 1).

Tests collision-free cache key identity, IEEE-754 little-endian binary serialization,
corrupted BLOB detection, duplicate prevention, and dimensional isolation.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
from unittest import TestCase

from src.context.embeddings.cache import (
    EmbeddingCache,
    compute_cache_key,
    deserialize_embedding,
    serialize_embedding,
)


class TestEmbeddingSerialization(TestCase):
    """Tests for binary float32 vector serialization and deserialization."""

    def test_serialize_and_deserialize_768_dim(self):
        dim = 768
        original = [float(i) / 1000.0 for i in range(dim)]
        blob = serialize_embedding(original, dimension=dim)

        # 768 float32s * 4 bytes = 3072 bytes
        self.assertEqual(len(blob), 3072)

        recovered = deserialize_embedding(blob, expected_dimension=dim)
        self.assertEqual(len(recovered), dim)
        for orig, rec in zip(original, recovered):
            self.assertAlmostEqual(orig, rec, places=5)

    def test_serialize_and_deserialize_1536_dim(self):
        dim = 1536
        original = [math.sin(float(i)) for i in range(dim)]
        blob = serialize_embedding(original, dimension=dim)

        # 1536 float32s * 4 bytes = 6144 bytes
        self.assertEqual(len(blob), 6144)

        recovered = deserialize_embedding(blob, expected_dimension=dim)
        self.assertEqual(len(recovered), dim)
        for orig, rec in zip(original, recovered):
            self.assertAlmostEqual(orig, rec, places=5)

    def test_serialize_dimension_mismatch_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            serialize_embedding([0.1, 0.2], dimension=768)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_serialize_non_finite_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            serialize_embedding([0.1, float("nan"), 0.3], dimension=3)
        self.assertIn("non-finite", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            serialize_embedding([0.1, float("inf"), 0.3], dimension=3)
        self.assertIn("non-finite", str(ctx.exception).lower())

    def test_deserialize_truncated_blob_rejected(self):
        dim = 768
        blob = serialize_embedding([0.1] * dim, dimension=dim)
        # Truncate by 4 bytes
        truncated = blob[:-4]
        with self.assertRaises(ValueError) as ctx:
            deserialize_embedding(truncated, expected_dimension=dim)
        self.assertIn("corrupted cache blob", str(ctx.exception).lower())

    def test_deserialize_oversized_blob_rejected(self):
        dim = 768
        blob = serialize_embedding([0.1] * dim, dimension=dim) + b"\x00\x00\x00\x00"
        with self.assertRaises(ValueError) as ctx:
            deserialize_embedding(blob, expected_dimension=dim)
        self.assertIn("corrupted cache blob", str(ctx.exception).lower())


class TestEmbeddingCache(TestCase):
    """Tests for SQLite EmbeddingCache."""

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.cache = EmbeddingCache(self.conn)

    def tearDown(self):
        self.conn.close()

    def test_table_and_index_created(self):
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_cache'"
        )
        self.assertIsNotNone(cursor.fetchone())

        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_emb_cache_lookup'"
        )
        self.assertIsNotNone(cursor.fetchone())

    def test_cache_set_and_get_hit(self):
        content = "Important document text"
        c_hash = hashlib.sha256(content.encode()).hexdigest()
        vector = [0.12345] * 768

        inserted = self.cache.set(
            content_hash=c_hash,
            provider_id="openai",
            model_id="text-embedding-3-small",
            model_version="default",
            dimension=768,
            embedding=vector,
        )
        self.assertTrue(inserted)
        self.assertEqual(self.cache.count(), 1)

        retrieved = self.cache.get(
            content_hash=c_hash,
            provider_id="openai",
            model_id="text-embedding-3-small",
            model_version="default",
            dimension=768,
        )
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved), 768)
        self.assertAlmostEqual(retrieved[0], 0.12345, places=5)

    def test_cache_miss_on_unknown(self):
        miss = self.cache.get(
            content_hash="unknown_hash",
            provider_id="openai",
            model_id="text-embedding-3-small",
            model_version="default",
            dimension=768,
        )
        self.assertIsNone(miss)

    def test_collision_prevention_different_dimensions(self):
        """Same content, same model, different dimensions (1536 vs 768) must not collide."""
        c_hash = "shared_hash_1"
        vec_1536 = [0.1536] * 1536
        vec_768 = [0.768] * 768

        self.cache.set(
            content_hash=c_hash,
            provider_id="openai",
            model_id="text-embedding-3-small",
            model_version="default",
            dimension=1536,
            embedding=vec_1536,
        )
        self.cache.set(
            content_hash=c_hash,
            provider_id="openai",
            model_id="text-embedding-3-small",
            model_version="default",
            dimension=768,
            embedding=vec_768,
        )
        self.assertEqual(self.cache.count(), 2)

        res_1536 = self.cache.get(
            content_hash=c_hash,
            provider_id="openai",
            model_id="text-embedding-3-small",
            model_version="default",
            dimension=1536,
        )
        res_768 = self.cache.get(
            content_hash=c_hash,
            provider_id="openai",
            model_id="text-embedding-3-small",
            model_version="default",
            dimension=768,
        )

        self.assertIsNotNone(res_1536)
        self.assertIsNotNone(res_768)
        self.assertEqual(len(res_1536), 1536)
        self.assertEqual(len(res_768), 768)
        self.assertAlmostEqual(res_1536[0], 0.1536, places=4)
        self.assertAlmostEqual(res_768[0], 0.768, places=4)

    def test_collision_prevention_different_models(self):
        """Same content, same dimension (768), different models must not collide."""
        c_hash = "shared_hash_2"
        vec_openai = [0.111] * 768
        vec_nomic = [0.222] * 768

        self.cache.set(
            content_hash=c_hash,
            provider_id="openai",
            model_id="text-embedding-3-small",
            model_version="default",
            dimension=768,
            embedding=vec_openai,
        )
        self.cache.set(
            content_hash=c_hash,
            provider_id="ollama",
            model_id="nomic-embed-text",
            model_version="default",
            dimension=768,
            embedding=vec_nomic,
        )
        self.assertEqual(self.cache.count(), 2)

        res_openai = self.cache.get(
            content_hash=c_hash,
            provider_id="openai",
            model_id="text-embedding-3-small",
            model_version="default",
            dimension=768,
        )
        res_nomic = self.cache.get(
            content_hash=c_hash,
            provider_id="ollama",
            model_id="nomic-embed-text",
            model_version="default",
            dimension=768,
        )
        self.assertAlmostEqual(res_openai[0], 0.111, places=4)
        self.assertAlmostEqual(res_nomic[0], 0.222, places=4)

    def test_collision_prevention_different_versions(self):
        """Same content and model, different versions (v1.0 vs v1.5) must not collide."""
        c_hash = "shared_hash_3"
        vec_v1 = [0.1] * 768
        vec_v15 = [0.2] * 768

        self.cache.set(
            content_hash=c_hash,
            provider_id="ollama",
            model_id="nomic-embed-text",
            model_version="v1.0",
            dimension=768,
            embedding=vec_v1,
        )
        self.cache.set(
            content_hash=c_hash,
            provider_id="ollama",
            model_id="nomic-embed-text",
            model_version="v1.5",
            dimension=768,
            embedding=vec_v15,
        )
        self.assertEqual(self.cache.count(), 2)

        res_v1 = self.cache.get(
            content_hash=c_hash,
            provider_id="ollama",
            model_id="nomic-embed-text",
            model_version="v1.0",
            dimension=768,
        )
        res_v15 = self.cache.get(
            content_hash=c_hash,
            provider_id="ollama",
            model_id="nomic-embed-text",
            model_version="v1.5",
            dimension=768,
        )
        self.assertAlmostEqual(res_v1[0], 0.1, places=4)
        self.assertAlmostEqual(res_v15[0], 0.2, places=4)

    def test_collision_prevention_preprocessing_and_normalization(self):
        c_hash = "shared_hash_4"
        vec_v1 = [0.5] * 768
        vec_v2 = [0.6] * 768

        self.cache.set(
            content_hash=c_hash,
            provider_id="mock",
            model_id="mock-model",
            model_version="v1",
            dimension=768,
            embedding=vec_v1,
            preprocessing_version="raw",
            normalize=True,
        )
        self.cache.set(
            content_hash=c_hash,
            provider_id="mock",
            model_id="mock-model",
            model_version="v1",
            dimension=768,
            embedding=vec_v2,
            preprocessing_version="prefixed",
            normalize=True,
        )
        self.assertEqual(self.cache.count(), 2)

        res_raw = self.cache.get(
            content_hash=c_hash,
            provider_id="mock",
            model_id="mock-model",
            model_version="v1",
            dimension=768,
            preprocessing_version="raw",
            normalize=True,
        )
        res_prefixed = self.cache.get(
            content_hash=c_hash,
            provider_id="mock",
            model_id="mock-model",
            model_version="v1",
            dimension=768,
            preprocessing_version="prefixed",
            normalize=True,
        )
        self.assertAlmostEqual(res_raw[0], 0.5, places=4)
        self.assertAlmostEqual(res_prefixed[0], 0.6, places=4)

    def test_duplicate_write_ignored(self):
        c_hash = "dup_hash"
        vec = [0.1] * 768
        first = self.cache.set(
            content_hash=c_hash,
            provider_id="openai",
            model_id="text-embedding-3-small",
            model_version="default",
            dimension=768,
            embedding=vec,
        )
        self.assertTrue(first)

        # Second write with exact same key must return False and not duplicate
        second = self.cache.set(
            content_hash=c_hash,
            provider_id="openai",
            model_id="text-embedding-3-small",
            model_version="default",
            dimension=768,
            embedding=vec,
        )
        self.assertFalse(second)
        self.assertEqual(self.cache.count(), 1)

    def test_corrupted_blob_purging(self):
        """Corrupted BLOB in SQLite should be safely handled and purged upon lookup."""
        cache_key = compute_cache_key(
            content_hash="corrupt_hash",
            provider_id="mock",
            model_id="m1",
            model_version="v1",
            dimension=768,
        )
        # Manually insert corrupted bytes (only 10 bytes instead of 3072)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO embedding_cache
                (cache_key, content_hash, provider_id, model_id, model_version,
                 dimension, preprocessing_version, normalize, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (cache_key, "corrupt_hash", "mock", "m1", "v1", 768, "v1", 1, b"corrupted!"),
            )
        self.assertEqual(self.cache.count(), 1)

        # Attempting to get must return None and delete the bad entry
        result = self.cache.get(
            content_hash="corrupt_hash",
            provider_id="mock",
            model_id="m1",
            model_version="v1",
            dimension=768,
        )
        self.assertIsNone(result)
        self.assertEqual(self.cache.count(), 0)

    def test_get_record(self):
        c_hash = "rec_hash"
        vec = [0.777] * 768
        self.cache.set(
            content_hash=c_hash,
            provider_id="openai",
            model_id="text-embedding-3-small",
            model_version="default",
            dimension=768,
            embedding=vec,
        )
        rec = self.cache.get_record(
            content_hash=c_hash,
            provider_id="openai",
            model_id="text-embedding-3-small",
            model_version="default",
            dimension=768,
        )
        self.assertIsNotNone(rec)
        self.assertEqual(rec.content_hash, c_hash)
        self.assertEqual(rec.provider_id, "openai")
        self.assertEqual(rec.dimension, 768)
        self.assertAlmostEqual(rec.embedding[0], 0.777, places=4)

    def test_clear_and_delete(self):
        self.cache.set("h1", "p1", "m1", "v1", 768, [0.1] * 768)
        self.cache.set("h2", "p1", "m1", "v1", 768, [0.2] * 768)
        self.assertEqual(self.cache.count(), 2)

        key1 = compute_cache_key("h1", "p1", "m1", "v1", 768)
        deleted = self.cache.delete(key1)
        self.assertTrue(deleted)
        self.assertEqual(self.cache.count(), 1)

        cleared = self.cache.clear()
        self.assertEqual(cleared, 1)
        self.assertEqual(self.cache.count(), 0)
