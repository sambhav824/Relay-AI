import unittest
from src.core.message import Message, MessageRole
from src.routing.classifier import TaskClassifier
from src.routing.types import (
    ClassificationResult,
    ModelPricing,
    ModelProfile,
    RoutingStrategy,
    TaskType,
)


class TestTaskClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = TaskClassifier(confidence_threshold=0.60)

    def test_coding_classification_fenced_block(self):
        prompt = (
            "Here is my Python code:\n"
            "```python\n"
            "def calculate_total(items):\n"
            "    return sum(item.price for item in items)\n"
            "```\n"
            "Why is this throwing an AttributeError?"
        )
        res = self.classifier.classify(prompt)
        self.assertEqual(res.task_type, TaskType.CODING)
        self.assertGreaterEqual(res.confidence, 0.60)
        self.assertFalse(res.is_fallback_general)

    def test_coding_classification_sql_and_keywords(self):
        prompt = "Write a SQL query: SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id WHERE o.status = 'active';"
        res = self.classifier.classify(prompt)
        self.assertEqual(res.task_type, TaskType.CODING)
        self.assertGreaterEqual(res.confidence, 0.60)

    def test_math_classification_derivatives_and_equations(self):
        prompt = "Calculate the derivative of f(x) = 3x^3 + 5x^2 - 11x + 4 and find its roots."
        res = self.classifier.classify(prompt)
        self.assertEqual(res.task_type, TaskType.MATH)
        self.assertGreaterEqual(res.confidence, 0.60)

    def test_math_classification_latex_symbols(self):
        prompt = "Evaluate the definite integral \\int_{0}^{1} \\sqrt{1 - x^2} dx using standard trigonometry."
        res = self.classifier.classify(prompt)
        self.assertEqual(res.task_type, TaskType.MATH)
        self.assertGreaterEqual(res.confidence, 0.60)

    def test_summarization_classification(self):
        prompt = "Please summarize the following quarterly earnings report and give me a brief tl;dr with key takeaways."
        res = self.classifier.classify(prompt)
        self.assertEqual(res.task_type, TaskType.SUMMARIZATION)
        self.assertGreaterEqual(res.confidence, 0.60)

    def test_creative_writing_classification(self):
        prompt = "Write a poem about the quiet silence of a snowfall in the mountains using metaphor and rhyme."
        res = self.classifier.classify(prompt)
        self.assertEqual(res.task_type, TaskType.CREATIVE)
        self.assertGreaterEqual(res.confidence, 0.60)

    def test_complex_reasoning_classification(self):
        prompt = (
            "Compare and contrast the trade-offs between distributed consensus protocols like Raft and Paxos. "
            "Perform a root cause analysis of network partition vulnerabilities."
        )
        res = self.classifier.classify(prompt)
        self.assertEqual(res.task_type, TaskType.COMPLEX_REASONING)
        self.assertGreaterEqual(res.confidence, 0.60)

    def test_document_analysis_long_text(self):
        long_body = "The company reported earnings across diverse regional departments. " * 80
        prompt = f"Analyze the following text and extract information from this report:\n\n{long_body}"
        res = self.classifier.classify(prompt)
        self.assertEqual(res.task_type, TaskType.DOCUMENT_ANALYSIS)
        self.assertGreaterEqual(res.confidence, 0.60)

    def test_general_conversational_classification(self):
        prompt = "Hello! What is the capital city of Australia?"
        res = self.classifier.classify(prompt)
        self.assertEqual(res.task_type, TaskType.GENERAL)

    def test_low_confidence_fallback_to_general(self):
        # A mixed, vague prompt with low distinct evidence should safely fall back to GENERAL
        prompt = "Maybe some things are fast and nice today."
        res = self.classifier.classify(prompt)
        self.assertEqual(res.task_type, TaskType.GENERAL)
        self.assertTrue(res.is_fallback_general)

    def test_context_aware_continuation_coding(self):
        history = [
            Message(MessageRole.USER, "How do I build this?"),
            Message(
                MessageRole.ASSISTANT,
                "```python\ndef run_server():\n    app.listen(8080)\n```"
            ),
        ]
        # Brief follow up without explicit language keywords
        prompt = "How do I fix the error when running it?"
        res = self.classifier.classify(prompt, history=history)
        self.assertEqual(res.task_type, TaskType.CODING)
        self.assertIn("context_history_coding_continuation", res.matched_cues)

    def test_model_pricing_formatting(self):
        # Local inference explicitly distinguishes zero API charge from power/hardware
        local_pricing = ModelPricing(is_local=True)
        self.assertIn("API charge", local_pricing.formatted_pricing())
        self.assertIn("hardware/power", local_pricing.formatted_pricing())

        # Cloud pricing with verified numbers
        cloud_pricing = ModelPricing(
            input_cost_per_m=0.15,
            output_cost_per_m=0.60,
            last_verified="2026-11-12",
            source="Official Groq Pricing",
            is_local=False,
        )
        self.assertEqual(cloud_pricing.formatted_pricing(), "$0.15 in / $0.60 out per 1M tokens")

        # Unknown pricing
        unknown_pricing = ModelPricing(input_cost_per_m=None, is_local=False)
        self.assertEqual(unknown_pricing.formatted_pricing(), "Pricing unavailable")


if __name__ == "__main__":
    unittest.main()

