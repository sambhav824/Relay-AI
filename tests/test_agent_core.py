import concurrent.futures
import threading
import time
import unittest
from unittest.mock import MagicMock

from src.agents import (
    AgentBase,
    AgentConfig,
    AgentConfigurationError,
    AgentExecutionError,
    AgentRegistry,
    AgentResult,
    AgentRole,
    AgentStatus,
    DuplicateAgentIdError,
    TaskContext,
)
from src.core.events import ResponseMetadata, StreamChunk
from src.core.message import Message, MessageRole
from src.core.security import CredentialStore
from src.providers.base import AIProvider
from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from src.providers.manager import ProviderManager
from src.providers.models import ModelInfo


class MockAgentProvider(AIProvider):
    """Mock AIProvider for isolated Agent Core unit testing without network or credentials."""

    def __init__(
        self,
        provider_id: str = "mock_provider",
        stream_chunks=None,
        generate_response: str = "Mock response",
        exception_to_raise: Exception = None,
        delay_per_chunk: float = 0.0,
    ):
        self.provider_id = provider_id
        self.display_name = provider_id.title()
        self.is_local = (provider_id == "ollama")
        self.stream_chunks = stream_chunks or [
            StreamChunk("Chunk 1 "),
            StreamChunk(
                "Chunk 2.",
                metadata=ResponseMetadata(
                    model="mock-model",
                    provider_id=provider_id,
                    prompt_tokens=15,
                    completion_tokens=25,
                    total_tokens=40,
                ),
            ),
        ]
        self.generate_response = generate_response
        self.exception_to_raise = exception_to_raise
        self.delay_per_chunk = delay_per_chunk
        self.calls = []

    def stream(self, messages, model, **kwargs):
        self.calls.append({"messages": list(messages), "model": model, "kwargs": kwargs})
        if self.exception_to_raise:
            raise self.exception_to_raise

        for chunk in self.stream_chunks:
            if self.delay_per_chunk > 0:
                time.sleep(self.delay_per_chunk)
            yield chunk

    def generate(self, messages, model, **kwargs):
        self.calls.append({"messages": list(messages), "model": model, "kwargs": kwargs})
        if self.exception_to_raise:
            raise self.exception_to_raise
        return self.generate_response

    def validate_connection(self):
        return True

    def get_available_models(self):
        return [ModelInfo("mock-model", "Mock Model", self.provider_id)]


class TestAgent(AgentBase):
    """Concrete agent subclass for lifecycle testing."""
    pass


class TestAgentCore(unittest.TestCase):
    def setUp(self):
        self.mock_cred_store = MagicMock(spec=CredentialStore)
        self.provider_mgr = ProviderManager(credential_store=self.mock_cred_store)
        self.provider_mgr._providers.clear()

        self.mock_provider = MockAgentProvider("mock_provider")
        self.provider_mgr.register_provider(self.mock_provider)

        self.config = AgentConfig(
            provider_id="mock_provider",
            model_id="mock-model",
            temperature=0.5,
            timeout_sec=5.0,
            system_instruction="You are a helpful test agent.",
        )

        self.agent = TestAgent(
            agent_id="test_agent_1",
            name="Test Agent",
            role=AgentRole.CREATOR,
            config=self.config,
            description="Testing agent implementation",
            provider_manager=self.provider_mgr,
        )

    def test_agent_successful_execution(self):
        context = TaskContext(
            user_prompt="Write a hello world function",
            conversation_history=[Message(MessageRole.USER, "Prior turn")],
        )
        self.assertEqual(self.agent.status, AgentStatus.IDLE)

        result = self.agent.execute(context)

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertTrue(result.is_success)
        self.assertEqual(result.output, "Chunk 1 Chunk 2.")
        self.assertIsNone(result.error)
        self.assertEqual(result.agent_id, "test_agent_1")
        self.assertEqual(result.agent_role, AgentRole.CREATOR)
        self.assertEqual(result.metadata.provider_id, "mock_provider")
        self.assertEqual(result.metadata.model_id, "mock-model")
        self.assertEqual(result.metadata.prompt_tokens, 15)
        self.assertEqual(result.metadata.completion_tokens, 25)
        self.assertEqual(result.metadata.total_tokens, 40)
        self.assertGreater(result.metadata.duration_sec, 0.0)
        self.assertEqual(self.agent.status, AgentStatus.COMPLETED)

    def test_concurrent_execution_prohibited(self):
        # Configure provider with slight delay to simulate in-flight execution
        slow_provider = MockAgentProvider("slow_provider", delay_per_chunk=0.3)
        self.provider_mgr.register_provider(slow_provider)

        slow_config = AgentConfig(provider_id="slow_provider", model_id="mock-model", timeout_sec=5.0)
        agent = TestAgent("slow_agent", "Slow Agent", AgentRole.PLANNER, slow_config, provider_manager=self.provider_mgr)

        context1 = TaskContext(user_prompt="Task 1")
        context2 = TaskContext(user_prompt="Task 2")

        thread_errors = []

        def run_agent():
            agent.execute(context1)

        t1 = threading.Thread(target=run_agent)
        t1.start()

        # Brief sleep to ensure t1 transitions agent to RUNNING
        time.sleep(0.05)

        try:
            # Second concurrent call MUST raise AgentExecutionError
            agent.execute(context2)
        except AgentExecutionError as exc:
            thread_errors.append(exc)

        t1.join()
        self.assertEqual(len(thread_errors), 1)
        self.assertIn("already RUNNING", str(thread_errors[0]))

    def test_safe_reset(self):
        context = TaskContext(user_prompt="Quick task")
        result = self.agent.execute(context)
        self.assertEqual(self.agent.status, AgentStatus.COMPLETED)

        self.agent.reset()
        self.assertEqual(self.agent.status, AgentStatus.IDLE)
        self.assertFalse(self.agent.is_cancelled())

        # Attempting reset while RUNNING raises AgentExecutionError
        self.agent.status = AgentStatus.RUNNING
        with self.assertRaises(AgentExecutionError):
            self.agent.reset()
        self.agent.status = AgentStatus.IDLE

    def test_cancellation_before_execution(self):
        context = TaskContext(user_prompt="Do something")
        self.agent.cancel()  # Signal cancellation prior to execute()

        result = self.agent.execute(context)

        self.assertEqual(result.status, AgentStatus.CANCELLED)
        self.assertFalse(result.is_success)
        self.assertIn("cancelled before invocation", result.error.lower())
        self.assertEqual(len(self.mock_provider.calls), 0)

    def test_cancellation_during_execution(self):
        slow_provider = MockAgentProvider("slow_p", delay_per_chunk=0.2)
        self.provider_mgr.register_provider(slow_provider)
        config = AgentConfig(provider_id="slow_p", model_id="mock-model")
        agent = TestAgent("cancelling_agent", "Agent", AgentRole.RESEARCHER, config, provider_manager=self.provider_mgr)

        context = TaskContext(user_prompt="Long stream task")

        def trigger_cancel():
            time.sleep(0.1)
            agent.cancel()

        cancel_thread = threading.Thread(target=trigger_cancel)
        cancel_thread.start()

        result = agent.execute(context)
        cancel_thread.join()

        self.assertEqual(result.status, AgentStatus.CANCELLED)
        self.assertFalse(result.is_success)

    def test_timeout_handling(self):
        # Configure provider with delay exceeding timeout_sec
        slow_provider = MockAgentProvider("timeout_provider", delay_per_chunk=0.2)
        self.provider_mgr.register_provider(slow_provider)

        short_timeout_config = AgentConfig(
            provider_id="timeout_provider",
            model_id="mock-model",
            timeout_sec=0.1,  # Strict short timeout
            max_retries=0,
        )
        agent = TestAgent("timeout_agent", "Timeout Agent", AgentRole.CRITIC, short_timeout_config, provider_manager=self.provider_mgr)

        context = TaskContext(user_prompt="Will timeout")
        result = agent.execute(context)

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertFalse(result.is_success)
        self.assertIn("Timeout", result.error)

    def test_provider_timeout_error_mapping(self):
        self.mock_provider.exception_to_raise = ProviderTimeoutError("Gateway timed out", provider_id="mock_provider")
        context = TaskContext(user_prompt="Prompt")
        self.agent.config.max_retries = 0

        result = self.agent.execute(context)

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("Timeout", result.error)

    def test_provider_rate_limit_error_mapping(self):
        self.mock_provider.exception_to_raise = ProviderRateLimitError("Rate limit exceeded", provider_id="mock_provider")
        self.agent.config.max_retries = 0
        context = TaskContext(user_prompt="Prompt")

        result = self.agent.execute(context)

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("Rate limit", result.error)

    def test_provider_auth_error_mapping(self):
        self.mock_provider.exception_to_raise = ProviderAuthenticationError("Invalid API key", provider_id="mock_provider")
        self.agent.config.max_retries = 0
        context = TaskContext(user_prompt="Prompt")

        result = self.agent.execute(context)

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("Invalid API key", result.error)

    def test_missing_provider_fails_gracefully(self):
        invalid_config = AgentConfig(provider_id="unregistered_p", model_id="some-model")
        agent = TestAgent("missing_prov_agent", "Name", AgentRole.FINALIZER, invalid_config, provider_manager=self.provider_mgr)

        context = TaskContext(user_prompt="Prompt")
        result = agent.execute(context)

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("not registered", result.error)

    def test_context_mutation_isolation(self):
        original_history = [Message(MessageRole.USER, "Original User Message")]
        original_inputs = {"nested": {"key": "original_val"}}
        original_prev_outputs = {"planner": "Phase 1 plan"}
        original_metadata = {"tag": "audit"}

        context = TaskContext(
            user_prompt="Analyze code",
            conversation_history=original_history,
            input_data=original_inputs,
            previous_outputs=original_prev_outputs,
            shared_metadata=original_metadata,
        )

        # Mutate caller structures after TaskContext instantiation
        original_history.append(Message(MessageRole.USER, "Rogue added turn"))
        original_inputs["nested"]["key"] = "tampered_val"
        original_prev_outputs["planner"] = "tampered plan"
        original_metadata["tag"] = "tampered"

        # TaskContext MUST retain pristine isolated copies
        self.assertEqual(len(context.conversation_history), 1)
        self.assertEqual(context.input_data["nested"]["key"], "original_val")
        self.assertEqual(context.previous_outputs["planner"], "Phase 1 plan")
        self.assertEqual(context.shared_metadata["tag"], "audit")

    def test_context_message_assembly(self):
        context = TaskContext(
            user_prompt="Implement quicksort",
            instructions="Output only valid Python 3 code.",
            previous_outputs={"planner": "Step 1: Partition. Step 2: Recurse."},
        )
        msgs = context.build_messages(system_instruction="System prompt.")

        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].role, MessageRole.SYSTEM)
        self.assertEqual(msgs[0].content, "System prompt.")
        self.assertEqual(msgs[1].role, MessageRole.USER)
        self.assertIn("Overall Task Objective:", msgs[1].content)
        self.assertIn("Previous Stage Outputs:", msgs[1].content)
        self.assertIn("Step Instructions:", msgs[1].content)

    def test_registry_registration_and_retrieval(self):
        registry = AgentRegistry()
        self.assertEqual(registry.count(), 0)

        registry.register(self.agent)
        self.assertEqual(registry.count(), 1)

        retrieved = registry.get("test_agent_1")
        self.assertIs(retrieved, self.agent)

        required = registry.get_required("test_agent_1")
        self.assertIs(required, self.agent)

        # Listing by role
        creators = registry.list_agents(AgentRole.CREATOR)
        self.assertEqual(len(creators), 1)
        planners = registry.list_agents(AgentRole.PLANNER)
        self.assertEqual(len(planners), 0)

    def test_registry_duplicate_rejection(self):
        registry = AgentRegistry()
        registry.register(self.agent)

        # Duplicate ID without overwrite=True raises DuplicateAgentIdError
        dup_agent = TestAgent(
            agent_id="test_agent_1",
            name="Duplicate Name",
            role=AgentRole.ARCHITECT,
            config=self.config,
            provider_manager=self.provider_mgr,
        )
        with self.assertRaises(DuplicateAgentIdError):
            registry.register(dup_agent, overwrite=False)

        # With overwrite=True, replaces cleanly
        registry.register(dup_agent, overwrite=True)
        self.assertEqual(registry.get("test_agent_1").name, "Duplicate Name")

    def test_registry_cannot_unregister_or_overwrite_running_agent(self):
        registry = AgentRegistry()
        registry.register(self.agent)

        # Mark agent as RUNNING
        self.agent.status = AgentStatus.RUNNING

        # Overwrite attempt raises AgentConfigurationError
        with self.assertRaises(AgentConfigurationError):
            registry.register(self.agent, overwrite=True)

        # Unregister attempt raises AgentConfigurationError
        with self.assertRaises(AgentConfigurationError):
            registry.unregister("test_agent_1")

        self.agent.status = AgentStatus.IDLE
        self.assertTrue(registry.unregister("test_agent_1"))
        self.assertIsNone(registry.get("test_agent_1"))

    def test_registry_thread_safety(self):
        registry = AgentRegistry()

        def register_worker(agent_idx: int):
            cfg = AgentConfig(provider_id="mock_provider", model_id="mock-model")
            agent = TestAgent(
                agent_id=f"thread_agent_{agent_idx}",
                name=f"Agent {agent_idx}",
                role=AgentRole.RESEARCHER,
                config=cfg,
                provider_manager=self.provider_mgr,
            )
            registry.register(agent)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(register_worker, i) for i in range(25)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        self.assertEqual(registry.count(), 25)
        self.assertEqual(len(registry.list_agents(AgentRole.RESEARCHER)), 25)

    def test_invalid_agent_configs(self):
        # Empty provider ID
        with self.assertRaises(AgentConfigurationError):
            AgentConfig(provider_id="", model_id="model")

        # Empty model ID
        with self.assertRaises(AgentConfigurationError):
            AgentConfig(provider_id="prov", model_id="")

        # Non-positive timeout
        with self.assertRaises(AgentConfigurationError):
            AgentConfig(provider_id="prov", model_id="model", timeout_sec=0.0)

        # Negative retries
        with self.assertRaises(AgentConfigurationError):
            AgentConfig(provider_id="prov", model_id="model", max_retries=-1)


if __name__ == "__main__":
    unittest.main()

