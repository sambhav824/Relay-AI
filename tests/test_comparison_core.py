import os
import time
import unittest
from unittest.mock import MagicMock

# Run Qt offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from src.core.events import ResponseMetadata, StreamChunk
from src.core.message import Message, MessageRole
from src.core.comparison import (
    ComparisonMetrics,
    ComparisonSlotConfig,
    ComparisonSlotResult,
    ComparisonCoordinator,
    ModelCompareWorker,
    estimate_api_cost,
)
from src.providers.base import AIProvider
from src.providers.manager import ProviderManager

app = QApplication.instance() or QApplication([])


class MockProvider(AIProvider):
    def __init__(self, pid: str = "mock_p", deltas=None, delay=0.0, fail_with=None):
        self.provider_id = pid
        self.display_name = f"Mock {pid.title()}"
        self.is_local = (pid == "ollama")
        self.deltas = deltas or ["chunk1 ", "chunk2"]
        self.delay = delay
        self.fail_with = fail_with

    def generate(self, messages, model, **kwargs):
        if self.fail_with:
            raise self.fail_with
        return "".join(self.deltas)

    def stream(self, messages, model, **kwargs):
        if self.fail_with:
            raise self.fail_with
        for d in self.deltas:
            if self.delay > 0:
                time.sleep(self.delay)
            meta = ResponseMetadata(
                model=model,
                provider_id=self.provider_id,
                total_tokens=20,
                prompt_tokens=8,
                completion_tokens=12,
            )
            yield StreamChunk(delta=d, metadata=meta)

    def validate_connection(self):
        return True

    def get_available_models(self):
        return []


class TestComparisonCore(unittest.TestCase):
    def test_estimate_api_cost(self):
        # Local provider is always $0.00
        cost_local = estimate_api_cost("ollama", "llama3.2:3b", 100, 200)
        self.assertEqual(cost_local, 0.0)

        # OpenAI gpt-4o-mini ($0.15 / 1M prompt, $0.60 / 1M completion)
        cost_openai = estimate_api_cost("openai", "gpt-4o-mini", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost_openai, 0.75, places=4)

        # None tokens
        self.assertIsNone(estimate_api_cost("openai", "gpt-4o-mini", None, None))

    def test_metrics_formatting(self):
        metrics = ComparisonMetrics(
            elapsed_sec=1.456,
            total_tokens=150,
            completion_tokens=100,
            tokens_per_sec=68.7,
            estimated_cost_usd=0.00025,
        )
        self.assertEqual(metrics.formatted_elapsed(), "1.46s")
        self.assertEqual(metrics.formatted_tokens(), "150 tokens")
        self.assertEqual(metrics.formatted_speed(), "68.7 t/s")
        self.assertEqual(metrics.formatted_cost(is_local=False), "$0.0003")
        self.assertEqual(metrics.formatted_cost(is_local=True), "$0.00 (Local)")

    def test_worker_success(self):
        prov = MockProvider("groq", deltas=["Speed ", "rocks!"])
        worker = ModelCompareWorker(
            slot_id=1,
            provider=prov,
            messages=[Message("user", "Hello")],
            model="openai/gpt-oss-120b",
        )

        received_chunks = []
        statuses = []
        finished_results = []

        worker.chunk_received.connect(lambda s_id, delta: received_chunks.append((s_id, delta)))
        worker.status_changed.connect(lambda s_id, st: statuses.append((s_id, st)))
        worker.slot_finished.connect(lambda s_id, res: finished_results.append((s_id, res)))

        worker.start()
        worker.wait(5000)
        app.processEvents()

        self.assertEqual(len(finished_results), 1)
        res = finished_results[0][1]
        self.assertTrue(res.is_success)
        self.assertEqual(res.content, "Speed rocks!")
        self.assertEqual(res.model, "openai/gpt-oss-120b")
        self.assertEqual(len(received_chunks), 2)
        self.assertIn("Completed", [s[1] for s in statuses])

    def test_worker_cooperative_cancellation(self):
        prov = MockProvider("openai", deltas=["a", "b", "c", "d"], delay=0.1)
        worker = ModelCompareWorker(
            slot_id=2,
            provider=prov,
            messages=[Message("user", "Slow stream")],
            model="gpt-4o",
        )

        finished_results = []
        worker.slot_finished.connect(lambda s_id, res: finished_results.append(res))

        worker.start()
        time.sleep(0.05)
        worker.abort()
        worker.wait(5000)
        app.processEvents()

        self.assertEqual(len(finished_results), 1)
        res = finished_results[0]
        self.assertTrue(res.is_canceled)
        self.assertFalse(res.is_success)

    def test_worker_timeout(self):
        prov = MockProvider("gemini", deltas=["1", "2", "3"], delay=0.15)
        worker = ModelCompareWorker(
            slot_id=3,
            provider=prov,
            messages=[Message("user", "Timeout test")],
            model="gemini-3.5-flash",
            timeout_sec=0.1,  # Short timeout
        )

        failed_results = []
        worker.slot_failed.connect(lambda s_id, err, res: failed_results.append((err, res)))

        worker.start()
        worker.wait(5000)
        app.processEvents()

        self.assertEqual(len(failed_results), 1)
        err_msg, res = failed_results[0]
        self.assertIn("timed out", err_msg.lower())
        self.assertFalse(res.is_success)

    def test_coordinator_parallel_execution_and_isolation(self):
        mgr = ProviderManager()
        prov1 = MockProvider("ollama", deltas=["Local 1 ", "Local 2"])
        prov2 = MockProvider("groq", deltas=["Cloud 1 ", "Cloud 2"])
        prov3 = MockProvider("openai", fail_with=RuntimeError("Quota exceeded 429"))

        mgr.register_provider(prov1)
        mgr.register_provider(prov2)
        mgr.register_provider(prov3)

        coord = ComparisonCoordinator()
        slots = [
            ComparisonSlotConfig(slot_id=0, provider_id="ollama", model="llama3.2:3b"),
            ComparisonSlotConfig(slot_id=1, provider_id="groq", model="openai/gpt-oss-120b"),
            ComparisonSlotConfig(slot_id=2, provider_id="openai", model="gpt-4o-mini"),
        ]

        all_completed_data = []
        coord.all_completed.connect(lambda results: all_completed_data.append(results))

        messages = [Message("user", "Compare these three models")]
        coord.start_comparison(slots, messages, mgr)

        # Wait for all workers to finish
        for w in coord.workers.values():
            w.wait(5000)
        app.processEvents()

        self.assertEqual(len(all_completed_data), 1)
        results = all_completed_data[0]
        self.assertEqual(len(results), 3)

        # Verify results: Slot 0 and 1 succeed, Slot 2 fails with error message without crashing the others
        self.assertTrue(results[0].is_success)
        self.assertEqual(results[0].content, "Local 1 Local 2")

        self.assertTrue(results[1].is_success)
        self.assertEqual(results[1].content, "Cloud 1 Cloud 2")

        self.assertFalse(results[2].is_success)
        self.assertIn("Quota exceeded 429", results[2].error_message)


if __name__ == "__main__":
    unittest.main()
