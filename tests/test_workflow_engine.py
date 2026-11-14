import threading
import time
import unittest
from unittest.mock import MagicMock

from src.agents import (
    AgentBase,
    AgentConfig,
    AgentRegistry,
    AgentRole,
    AgentStatus,
    TaskContext,
)
from src.core.events import ResponseMetadata, StreamChunk
from src.core.message import Message, MessageRole
from src.core.security import CredentialStore
from src.providers.base import AIProvider
from src.providers.exceptions import ProviderConnectionError, ProviderError
from src.providers.manager import ProviderManager
from src.providers.models import ModelInfo
from src.workflow import (
    CriticReviewResult,
    StepFailurePolicy,
    WorkflowConfig,
    WorkflowConfigurationError,
    WorkflowEngine,
    WorkflowExecutionError,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStepConfig,
)


class MockWorkflowProvider(AIProvider):
    """Mock AIProvider for workflow engine testing."""

    def __init__(
        self,
        provider_id: str,
        response_text: str = "Default output",
        tokens: int = 20,
        exception_to_raise: Exception = None,
        delay_per_chunk: float = 0.0,
    ):
        self.provider_id = provider_id
        self.display_name = provider_id.title()
        self.is_local = (provider_id == "ollama")
        self.response_text = response_text
        self.tokens = tokens
        self.exception_to_raise = exception_to_raise
        self.delay_per_chunk = delay_per_chunk
        self.calls = []

    def stream(self, messages, model, **kwargs):
        self.calls.append({"messages": list(messages), "model": model, "kwargs": kwargs})
        if self.exception_to_raise:
            raise self.exception_to_raise

        if self.delay_per_chunk > 0:
            time.sleep(self.delay_per_chunk)

        yield StreamChunk(
            delta=self.response_text,
            is_final=True,
            metadata=ResponseMetadata(
                model=model,
                provider_id=self.provider_id,
                prompt_tokens=self.tokens // 2 if self.tokens else None,
                completion_tokens=self.tokens // 2 if self.tokens else None,
                total_tokens=self.tokens,
            ) if self.tokens is not None else None,
        )

    def generate(self, messages, model, **kwargs):
        self.calls.append({"messages": list(messages), "model": model, "kwargs": kwargs})
        if self.exception_to_raise:
            raise self.exception_to_raise
        return self.response_text

    def validate_connection(self):
        return True

    def get_available_models(self):
        return [ModelInfo("model-1", "Model 1", self.provider_id)]


class ConcreteWorkflowAgent(AgentBase):
    """Concrete agent for workflow execution testing."""
    def __init__(self, *args, critic_review_to_attach: CriticReviewResult = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic_review_to_attach = critic_review_to_attach

    def _run(self, context: TaskContext):
        output, meta = super()._run(context)
        if self.critic_review_to_attach and meta:
            meta.critic_review = self.critic_review_to_attach
        return output, meta


class FunctionalWorkflowAgent(AgentBase):
    """Agent whose behavior and metadata are driven dynamically by a callable for strict contract testing."""
    def __init__(self, *args, response_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_fn = response_fn
        self.executed_contexts = []

    def _run(self, context: TaskContext):
        self.executed_contexts.append(context)
        if self.response_fn:
            output, review = self.response_fn(context)
            meta = ResponseMetadata(
                model=self.config.model_id,
                provider_id=self.config.provider_id,
                prompt_tokens=10,
                completion_tokens=15,
                total_tokens=25,
            )
            if review:
                meta.critic_review = review
            return output, meta
        return super()._run(context)


class TestWorkflowEngine(unittest.TestCase):
    def setUp(self):
        self.mock_cred_store = MagicMock(spec=CredentialStore)
        self.provider_mgr = ProviderManager(credential_store=self.mock_cred_store)
        self.provider_mgr._providers.clear()

        # Build two mock providers
        self.prov_a = MockWorkflowProvider("provider_a", response_text="Output from Provider A", tokens=25)
        self.prov_b = MockWorkflowProvider("provider_b", response_text="Output from Provider B", tokens=35)
        self.prov_c = MockWorkflowProvider("provider_c", response_text="Final polish output", tokens=10)

        self.provider_mgr.register_provider(self.prov_a)
        self.provider_mgr.register_provider(self.prov_b)
        self.provider_mgr.register_provider(self.prov_c)

        self.registry = AgentRegistry()

        # Register 3 agents across roles and providers
        self.agent_planner = ConcreteWorkflowAgent(
            agent_id="planner_agent",
            name="Planner",
            role=AgentRole.PLANNER,
            config=AgentConfig("provider_a", "model-1", system_instruction="You are Planner."),
            provider_manager=self.provider_mgr,
        )
        self.agent_creator = ConcreteWorkflowAgent(
            agent_id="creator_agent",
            name="Creator",
            role=AgentRole.CREATOR,
            config=AgentConfig("provider_b", "model-1", system_instruction="You are Creator."),
            provider_manager=self.provider_mgr,
        )
        self.agent_finalizer = ConcreteWorkflowAgent(
            agent_id="finalizer_agent",
            name="Finalizer",
            role=AgentRole.FINALIZER,
            config=AgentConfig("provider_c", "model-1", system_instruction="You are Finalizer."),
            provider_manager=self.provider_mgr,
        )

        self.registry.register(self.agent_planner)
        self.registry.register(self.agent_creator)
        self.registry.register(self.agent_finalizer)

    def test_sequential_execution_and_context_transfer(self):
        config = WorkflowConfig(
            workflow_id="test_wf_seq",
            name="Sequential Test",
            steps=[
                WorkflowStepConfig(step_id="step_plan", agent_id="planner_agent", step_name="Plan"),
                WorkflowStepConfig(step_id="step_create", agent_id="creator_agent", step_name="Create", dependencies=["step_plan"]),
                WorkflowStepConfig(step_id="step_finalize", agent_id="finalizer_agent", step_name="Finalize", dependencies=["step_create"]),
            ],
        )
        engine = WorkflowEngine(config=config, registry=self.registry)
        base_context = TaskContext(user_prompt="Build an authentication microservice")

        result = engine.execute(base_context)

        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertTrue(result.is_success)
        self.assertEqual(len(result.step_records), 3)
        self.assertEqual(result.final_output, "Final polish output")
        self.assertEqual(result.total_tokens, 70)  # 25 + 35 + 10

        # Verify Creator received Planner's output in previous_outputs
        creator_call_messages = self.prov_b.calls[0]["messages"]
        user_message_to_creator = next(m.content for m in creator_call_messages if m.role == MessageRole.USER)
        self.assertIn("Build an authentication microservice", user_message_to_creator)
        self.assertIn("Output from Provider A", user_message_to_creator)

    def test_original_objective_preservation(self):
        config = WorkflowConfig(
            workflow_id="test_wf_obj",
            name="Objective Test",
            steps=[
                WorkflowStepConfig(step_id="s1", agent_id="planner_agent"),
                WorkflowStepConfig(step_id="s2", agent_id="creator_agent"),
            ],
        )
        engine = WorkflowEngine(config, self.registry)
        original_prompt = "Do not mutate this prompt"
        original_history = [Message(MessageRole.USER, "Prior Turn 1")]
        base_context = TaskContext(user_prompt=original_prompt, conversation_history=original_history)

        result = engine.execute(base_context)
        self.assertEqual(result.status, WorkflowStatus.COMPLETED)

        # Base context remains untouched
        self.assertEqual(base_context.user_prompt, original_prompt)
        self.assertEqual(len(base_context.conversation_history), 1)
        self.assertEqual(base_context.conversation_history[0].content, "Prior Turn 1")

    def test_multiple_provider_assignments(self):
        config = WorkflowConfig(
            workflow_id="test_wf_prov",
            name="Multi Provider",
            steps=[
                WorkflowStepConfig(step_id="s_a", agent_id="planner_agent"),  # uses provider_a
                WorkflowStepConfig(step_id="s_b", agent_id="creator_agent"),  # uses provider_b
            ],
        )
        engine = WorkflowEngine(config, self.registry)
        result = engine.execute(TaskContext(user_prompt="Multi provider query"))

        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertEqual(len(self.prov_a.calls), 1)
        self.assertEqual(len(self.prov_b.calls), 1)

    def test_dependency_validation_missing_dependency(self):
        config = WorkflowConfig(
            workflow_id="test_wf_dep",
            name="Missing Dep",
            steps=[
                WorkflowStepConfig(step_id="s2", agent_id="creator_agent", dependencies=["non_existent_s1"]),
            ],
        )
        engine = WorkflowEngine(config, self.registry)
        result = engine.execute(TaskContext(user_prompt="Query"))

        self.assertEqual(result.status, WorkflowStatus.FAILED)
        self.assertIn("missing required dependencies", result.error_summary)

    def test_workflow_cooperative_cancellation(self):
        slow_prov = MockWorkflowProvider("slow_prov", response_text="Slow", delay_per_chunk=0.3)
        self.provider_mgr.register_provider(slow_prov)
        slow_agent = ConcreteWorkflowAgent(
            agent_id="slow_agent",
            name="Slow",
            role=AgentRole.RESEARCHER,
            config=AgentConfig("slow_prov", "model-1"),
            provider_manager=self.provider_mgr,
        )
        self.registry.register(slow_agent)

        config = WorkflowConfig(
            workflow_id="test_wf_cancel",
            name="Cancel Test",
            steps=[
                WorkflowStepConfig(step_id="s_slow", agent_id="slow_agent"),
                WorkflowStepConfig(step_id="s_never", agent_id="finalizer_agent"),
            ],
        )
        engine = WorkflowEngine(config, self.registry)

        def trigger_cancel():
            time.sleep(0.1)
            engine.cancel()

        ct = threading.Thread(target=trigger_cancel)
        ct.start()

        result = engine.execute(TaskContext(user_prompt="Will be cancelled"))
        ct.join()

        self.assertEqual(result.status, WorkflowStatus.CANCELLED)
        self.assertFalse(result.is_success)
        # S_never step was never reached
        self.assertEqual(len(result.step_records), 1)

    def test_workflow_timeout(self):
        slow_prov = MockWorkflowProvider("timeout_p", response_text="Slow", delay_per_chunk=0.2)
        self.provider_mgr.register_provider(slow_prov)
        slow_agent2 = ConcreteWorkflowAgent(
            agent_id="slow_agent2",
            name="Slow2",
            role=AgentRole.RESEARCHER,
            config=AgentConfig("timeout_p", "model-1", timeout_sec=10.0),
            provider_manager=self.provider_mgr,
        )
        self.registry.register(slow_agent2)

        config = WorkflowConfig(
            workflow_id="test_wf_timeout",
            name="Timeout Test",
            timeout_sec=0.1,  # Short global timeout
            steps=[WorkflowStepConfig(step_id="s1", agent_id="slow_agent2")],
        )
        engine = WorkflowEngine(config, self.registry)
        result = engine.execute(TaskContext(user_prompt="Timeout task"))

        self.assertEqual(result.status, WorkflowStatus.FAILED)
        self.assertIn("timeout", result.error_summary.lower())

    def test_pause_and_resume_semantics(self):
        config = WorkflowConfig(
            workflow_id="test_wf_pause",
            name="Pause Test",
            steps=[
                WorkflowStepConfig(step_id="s1", agent_id="planner_agent"),
                WorkflowStepConfig(step_id="s2", agent_id="creator_agent"),
            ],
        )
        engine = WorkflowEngine(config, self.registry)

        # Trigger pause immediately
        engine.pause()
        self.assertTrue(engine.is_paused())

        def delayed_resume():
            time.sleep(0.15)
            engine.resume()

        rt = threading.Thread(target=delayed_resume)
        rt.start()

        result = engine.execute(TaskContext(user_prompt="Pause resume query"))
        rt.join()

        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertEqual(len(result.step_records), 2)

    def test_failure_policy_abort(self):
        self.prov_a.exception_to_raise = ProviderError("Fatal host down", provider_id="provider_a")
        config = WorkflowConfig(
            workflow_id="test_wf_abort",
            name="Abort Policy",
            steps=[
                WorkflowStepConfig(step_id="s1", agent_id="planner_agent", failure_policy=StepFailurePolicy.ABORT, max_retries=0),
                WorkflowStepConfig(step_id="s2", agent_id="creator_agent"),
            ],
        )
        engine = WorkflowEngine(config, self.registry)
        result = engine.execute(TaskContext(user_prompt="Task"))

        self.assertEqual(result.status, WorkflowStatus.FAILED)
        self.assertIn("s1", result.failed_steps)
        self.assertEqual(len(result.step_records), 1)

    def test_failure_policy_skip_step_partial_success(self):
        # Step 2 fails, but is optional with SKIP_STEP policy
        self.prov_b.exception_to_raise = ProviderError("Temporary glitch", provider_id="provider_b")
        config = WorkflowConfig(
            workflow_id="test_wf_skip",
            name="Skip Policy",
            steps=[
                WorkflowStepConfig(step_id="s1", agent_id="planner_agent"),
                WorkflowStepConfig(step_id="s2", agent_id="creator_agent", failure_policy=StepFailurePolicy.SKIP_STEP, is_optional=True, max_retries=0),
                WorkflowStepConfig(step_id="s3", agent_id="finalizer_agent"),
            ],
        )
        engine = WorkflowEngine(config, self.registry)
        result = engine.execute(TaskContext(user_prompt="Task"))

        self.assertEqual(result.status, WorkflowStatus.PARTIAL_SUCCESS)
        self.assertTrue(result.is_success)
        self.assertIn("s2", result.skipped_steps)
        self.assertEqual(len(result.step_records), 3)

    def test_failure_policy_fallback_step(self):
        # Primary agent fails, fallback agent succeeds
        self.prov_a.exception_to_raise = ProviderConnectionError("Offline", provider_id="provider_a")
        config = WorkflowConfig(
            workflow_id="test_wf_fallback",
            name="Fallback Policy",
            steps=[
                WorkflowStepConfig(
                    step_id="s1",
                    agent_id="planner_agent",
                    failure_policy=StepFailurePolicy.FALLBACK_STEP,
                    fallback_agent_id="finalizer_agent",
                    max_retries=0,
                ),
            ],
        )
        engine = WorkflowEngine(config, self.registry)
        result = engine.execute(TaskContext(user_prompt="Fallback task"))

        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertEqual(len(result.step_records), 2)  # Initial failed step + successful fallback step
        self.assertEqual(result.step_records[1].agent_id, "finalizer_agent")

    def test_critic_revision_loop_enforcement(self):
        # Create a Critic agent that requests revisions
        critic_review = CriticReviewResult(
            revision_required=True,
            feedback="Fix security vulnerability in SQL query",
            target_step_id="step_create",
            reason="Unsanitized input detected",
        )
        agent_critic = ConcreteWorkflowAgent(
            agent_id="critic_agent",
            name="Critic",
            role=AgentRole.CRITIC,
            config=AgentConfig("provider_a", "model-1"),
            critic_review_to_attach=critic_review,
            provider_manager=self.provider_mgr,
        )
        self.registry.register(agent_critic)

        config = WorkflowConfig(
            workflow_id="test_wf_critic",
            name="Critic Revision Test",
            max_revision_loops=2,  # Maximum 2 revision loops
            max_steps=10,
            steps=[
                WorkflowStepConfig(step_id="step_create", agent_id="creator_agent"),
                WorkflowStepConfig(
                    step_id="step_critic",
                    agent_id="critic_agent",
                    is_critic_step=True,
                    allowed_revision_targets=["step_create"],
                ),
                WorkflowStepConfig(step_id="step_finalize", agent_id="finalizer_agent"),
            ],
        )
        engine = WorkflowEngine(config, self.registry)
        result = engine.execute(TaskContext(user_prompt="Generate secure query"))

        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        # Should execute: create (0) -> critic -> create (rev 1) -> critic -> create (rev 2) -> critic -> finalize
        # Notice that when revision limit 2 is reached, it halts revisions and progresses to finalize
        create_executions = [r for r in result.step_records if r.step_id == "step_create"]
        self.assertEqual(len(create_executions), 3)  # Initial + 2 revisions
        self.assertEqual(create_executions[1].revision_iteration, 1)
        self.assertEqual(create_executions[2].revision_iteration, 2)

    def test_global_max_steps_guard(self):
        # Critic always requests revision
        critic_review = CriticReviewResult(
            revision_required=True,
            feedback="Always revise",
            target_step_id="step_create",
        )
        agent_critic = ConcreteWorkflowAgent(
            agent_id="infinite_critic",
            name="Critic",
            role=AgentRole.CRITIC,
            config=AgentConfig("provider_a", "model-1"),
            critic_review_to_attach=critic_review,
            provider_manager=self.provider_mgr,
        )
        self.registry.register(agent_critic)

        config = WorkflowConfig(
            workflow_id="test_wf_max_steps",
            name="Max Steps Guard",
            max_steps=4,  # Hard global limit: 4 step executions max
            max_revision_loops=10,
            steps=[
                WorkflowStepConfig(step_id="step_create", agent_id="creator_agent"),
                WorkflowStepConfig(
                    step_id="step_critic",
                    agent_id="infinite_critic",
                    is_critic_step=True,
                    allowed_revision_targets=["step_create"],
                ),
            ],
        )
        engine = WorkflowEngine(config, self.registry)
        result = engine.execute(TaskContext(user_prompt="Loop task"))

        # Bounded by max_steps=4
        self.assertLessEqual(len(result.step_records), 4)

    def test_token_budget_enforcement(self):
        config = WorkflowConfig(
            workflow_id="test_wf_tokens",
            name="Token Budget",
            token_budget=30,  # Step 1 consumes 25, Step 2 consumes 35 -> total exceeds 30
            steps=[
                WorkflowStepConfig(step_id="s1", agent_id="planner_agent"),
                WorkflowStepConfig(step_id="s2", agent_id="creator_agent"),
            ],
        )
        engine = WorkflowEngine(config, self.registry)
        result = engine.execute(TaskContext(user_prompt="Token query"))

        self.assertEqual(result.status, WorkflowStatus.FAILED)
        self.assertIn("token budget", result.error_summary.lower())

    def test_unknown_token_usage_preserved_as_none(self):
        # Provider with tokens=None
        untracked_prov = MockWorkflowProvider("untracked_p", response_text="Answer", tokens=None)
        self.provider_mgr.register_provider(untracked_prov)
        untracked_agent = ConcreteWorkflowAgent(
            agent_id="untracked_agent",
            name="Untracked",
            role=AgentRole.CREATOR,
            config=AgentConfig("untracked_p", "model-1"),
            provider_manager=self.provider_mgr,
        )
        self.registry.register(untracked_agent)

        config = WorkflowConfig(
            workflow_id="test_wf_untracked",
            name="Untracked Tokens",
            steps=[WorkflowStepConfig(step_id="s1", agent_id="untracked_agent")],
        )
        engine = WorkflowEngine(config, self.registry)
        result = engine.execute(TaskContext(user_prompt="Task"))

        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        # Total tokens remains None, not fabricated as 0
        self.assertIsNone(result.total_tokens)

    def test_concurrent_workflow_execution_prohibited(self):
        slow_prov = MockWorkflowProvider("slow_p_conc", response_text="Slow", delay_per_chunk=0.3)
        self.provider_mgr.register_provider(slow_prov)
        agent = ConcreteWorkflowAgent(
            agent_id="slow_agent_conc",
            name="Slow",
            role=AgentRole.CREATOR,
            config=AgentConfig("slow_p_conc", "model-1"),
            provider_manager=self.provider_mgr,
        )
        self.registry.register(agent)

        config = WorkflowConfig(
            workflow_id="test_wf_concurrent",
            name="Concurrent Check",
            steps=[WorkflowStepConfig(step_id="s1", agent_id="slow_agent_conc")],
        )
        engine = WorkflowEngine(config, self.registry)

        errors = []

        def run_wf():
            engine.execute(TaskContext(user_prompt="Run 1"))

        t1 = threading.Thread(target=run_wf)
        t1.start()
        time.sleep(0.05)

        try:
            # Second call on same engine instance while RUNNING MUST fail
            engine.execute(TaskContext(user_prompt="Run 2"))
        except WorkflowExecutionError as exc:
            errors.append(exc)

        t1.join()
        self.assertEqual(len(errors), 1)
        self.assertIn("already RUNNING", str(errors[0]))

    def test_invalid_revision_target_rejected(self):
        critic_review = CriticReviewResult(
            revision_required=True,
            feedback="Fix plan",
            target_step_id="step_plan",  # Exists in workflow, but is NOT in allowed_revision_targets!
        )
        critic = ConcreteWorkflowAgent(
            agent_id="critic_unpermitted",
            name="Critic",
            role=AgentRole.CRITIC,
            config=AgentConfig("provider_a", "model-1"),
            critic_review_to_attach=critic_review,
            provider_manager=self.provider_mgr,
        )
        self.registry.register(critic)

        config = WorkflowConfig(
            workflow_id="test_wf_inv_rev",
            name="Invalid Revision Target",
            steps=[
                WorkflowStepConfig(step_id="step_plan", agent_id="planner_agent"),
                WorkflowStepConfig(step_id="step_create", agent_id="creator_agent"),
                WorkflowStepConfig(
                    step_id="step_critic",
                    agent_id="critic_unpermitted",
                    is_critic_step=True,
                    allowed_revision_targets=["step_create"],  # "step_plan" is NOT permitted
                ),
            ],
        )
        engine = WorkflowEngine(config, self.registry)
        result = engine.execute(TaskContext(user_prompt="Task"))

        self.assertEqual(result.status, WorkflowStatus.FAILED)
        self.assertIn("not in allowed_revision_targets", result.error_summary)

    def test_selective_context_inputs(self):
        config = WorkflowConfig(
            workflow_id="test_wf_selective",
            name="Selective Context",
            steps=[
                WorkflowStepConfig(step_id="step_plan", agent_id="planner_agent"),
                WorkflowStepConfig(step_id="step_create", agent_id="creator_agent"),
                WorkflowStepConfig(
                    step_id="step_finalize",
                    agent_id="finalizer_agent",
                    required_inputs=["step_plan"],  # Explicitly excludes step_create!
                ),
            ],
        )
        engine = WorkflowEngine(config, self.registry)
        result = engine.execute(TaskContext(user_prompt="Selective test"))

        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        # Check messages to finalizer (provider_c)
        finalizer_call = self.prov_c.calls[0]["messages"]
        user_msg = next(m.content for m in finalizer_call if m.role == MessageRole.USER)
        self.assertIn("Output from Provider A", user_msg)
        self.assertNotIn("Output from Provider B", user_msg)  # Filtered out cleanly!

    def test_exact_critic_revision_downstream_invalidation_and_finalizer(self):
        """Test exact prompt scenario:
        1. Planner produces output A.
        2. Creator consumes A and produces output B.
        3. Independent Researcher produces R (situated in pipeline, depends only on Planner).
        4. Critic reviews B and requests revision targeting Creator.
        5. Creator executes again and produces revised output B2.
        6. Independent Researcher is NOT repeated.
        7. Finalizer consumes B2 and R, never stale B.
        """
        planner_calls = []
        creator_calls = []
        researcher_calls = []
        critic_calls = []
        finalizer_calls = []

        def planner_run(ctx: TaskContext):
            planner_calls.append(ctx)
            return "Architecture Blueprint A", None

        def creator_run(ctx: TaskContext):
            creator_calls.append(ctx)
            # Check if instructions contain critique feedback
            if "CRITIQUE" in ctx.instructions:
                return "Output B2: Secured with bcrypt", None
            return "Output B: Plaintext passwords", None

        def researcher_run(ctx: TaskContext):
            researcher_calls.append(ctx)
            return "Research Data R: Industry standards", None

        def critic_run(ctx: TaskContext):
            critic_calls.append(ctx)
            creator_out = ctx.previous_outputs.get("step_create", "")
            if "Plaintext" in creator_out:
                review = CriticReviewResult(
                    revision_required=True,
                    feedback="Security flaw: Replace plaintext passwords with bcrypt",
                    target_step_id="step_create",
                    reason="Insecure password storage",
                )
                return "Critic Evaluation: Revision required", review
            else:
                review = CriticReviewResult(
                    revision_required=False,
                    feedback="Approved: Bcrypt hashing verified",
                    target_step_id="step_create",
                    reason="Secure",
                )
                return "Critic Evaluation: Approved", review

        def finalizer_run(ctx: TaskContext):
            finalizer_calls.append(ctx)
            # Crucial verification: Finalizer must receive B2, never stale B
            creator_val = ctx.previous_outputs.get("step_create", "")
            if "Output B2" in creator_val and "Output B:" not in creator_val:
                return f"Final Polish: Successfully integrated {creator_val}", None
            return f"Final Polish Error: Stale or missing output ({creator_val})", None

        # Build dynamic agents
        dyn_planner = FunctionalWorkflowAgent(
            agent_id="dyn_planner",
            name="Planner",
            role=AgentRole.PLANNER,
            config=AgentConfig("provider_a", "model-1"),
            response_fn=planner_run,
            provider_manager=self.provider_mgr,
        )
        dyn_creator = FunctionalWorkflowAgent(
            agent_id="dyn_creator",
            name="Creator",
            role=AgentRole.CREATOR,
            config=AgentConfig("provider_b", "model-1"),
            response_fn=creator_run,
            provider_manager=self.provider_mgr,
        )
        dyn_researcher = FunctionalWorkflowAgent(
            agent_id="dyn_researcher",
            name="Researcher",
            role=AgentRole.RESEARCHER,
            config=AgentConfig("provider_a", "model-1"),
            response_fn=researcher_run,
            provider_manager=self.provider_mgr,
        )
        dyn_critic = FunctionalWorkflowAgent(
            agent_id="dyn_critic",
            name="Critic",
            role=AgentRole.CRITIC,
            config=AgentConfig("provider_c", "model-1"),
            response_fn=critic_run,
            provider_manager=self.provider_mgr,
        )
        dyn_finalizer = FunctionalWorkflowAgent(
            agent_id="dyn_finalizer",
            name="Finalizer",
            role=AgentRole.FINALIZER,
            config=AgentConfig("provider_c", "model-1"),
            response_fn=finalizer_run,
            provider_manager=self.provider_mgr,
        )

        reg = AgentRegistry()
        reg.register(dyn_planner)
        reg.register(dyn_creator)
        reg.register(dyn_researcher)
        reg.register(dyn_critic)
        reg.register(dyn_finalizer)

        config = WorkflowConfig(
            workflow_id="wf_exact_scenario",
            name="Exact Revision Verification",
            max_steps=15,
            max_revision_loops=2,
            steps=[
                WorkflowStepConfig(step_id="step_plan", agent_id="dyn_planner"),
                WorkflowStepConfig(step_id="step_create", agent_id="dyn_creator", dependencies=["step_plan"]),
                WorkflowStepConfig(step_id="step_research", agent_id="dyn_researcher", dependencies=["step_plan"]),  # Independent of creator!
                WorkflowStepConfig(
                    step_id="step_critic",
                    agent_id="dyn_critic",
                    is_critic_step=True,
                    allowed_revision_targets=["step_create"],
                    dependencies=["step_create"],
                ),
                WorkflowStepConfig(
                    step_id="step_finalize",
                    agent_id="dyn_finalizer",
                    dependencies=["step_create", "step_research"],
                ),
            ],
        )

        engine = WorkflowEngine(config=config, registry=reg)
        original_prompt = "Build an enterprise user account service"
        original_history = [Message(MessageRole.USER, "Initial prompt"), Message(MessageRole.ASSISTANT, "Acknowledged")]
        base_context = TaskContext(user_prompt=original_prompt, conversation_history=original_history)

        result = engine.execute(base_context)

        # 1. Workflow completes successfully
        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertTrue(result.is_success)

        # 2. Final output contains B2 polish
        self.assertIn("Successfully integrated Output B2: Secured with bcrypt", result.final_output)

        # 3. Finalizer received B2, never stale B
        self.assertEqual(len(finalizer_calls), 1)
        finalizer_received_creator = finalizer_calls[0].previous_outputs.get("step_create")
        self.assertEqual(finalizer_received_creator, "Output B2: Secured with bcrypt")

        # 4. Independent step (Researcher) was executed ONCE only, not repeated!
        self.assertEqual(len(researcher_calls), 1)

        # 5. Planner was executed ONCE only
        self.assertEqual(len(planner_calls), 1)

        # 6. Creator executed TWICE (initial + 1 revision)
        self.assertEqual(len(creator_calls), 2)
        # First execution did not have critique instructions
        self.assertNotIn("CRITIQUE", creator_calls[0].instructions)
        # Second execution contained the exact critique feedback
        self.assertIn("Replace plaintext passwords with bcrypt", creator_calls[1].instructions)

        # 7. Critic executed TWICE (requested revision + approved)
        self.assertEqual(len(critic_calls), 2)

        # 8. All execution attempts accurately recorded
        self.assertEqual(len(result.step_records), 7)
        # step records order: plan, create(0), research, critic(req), create(1), critic(app), finalize
        self.assertEqual(result.step_records[0].step_id, "step_plan")
        self.assertEqual(result.step_records[1].step_id, "step_create")
        self.assertEqual(result.step_records[1].revision_iteration, 0)
        self.assertEqual(result.step_records[2].step_id, "step_research")
        self.assertEqual(result.step_records[3].step_id, "step_critic")
        self.assertEqual(result.step_records[4].step_id, "step_create")
        self.assertEqual(result.step_records[4].revision_iteration, 1)
        self.assertEqual(result.step_records[5].step_id, "step_critic")
        self.assertEqual(result.step_records[6].step_id, "step_finalize")

        # 9. Latest valid step outputs in WorkflowResult
        self.assertEqual(result.step_outputs["step_create"], "Output B2: Secured with bcrypt")
        self.assertEqual(result.step_outputs["step_research"], "Research Data R: Industry standards")
        self.assertIn("Successfully integrated Output B2", result.step_outputs["step_finalize"])

        # 10. Immutability of original objective and conversation history
        self.assertEqual(base_context.user_prompt, original_prompt)
        self.assertEqual(len(base_context.conversation_history), 2)
        self.assertEqual(base_context.conversation_history[0].content, "Initial prompt")

    def test_critic_revision_intermediate_downstream_invalidation(self):
        """Test that an intermediate step depending on Creator is also invalidated and re-computed."""
        formatter_calls = []

        def creator_fn(ctx: TaskContext):
            if "CRITIQUE" in ctx.instructions:
                return "Raw Data B2", None
            return "Raw Data B", None

        def formatter_fn(ctx: TaskContext):
            formatter_calls.append(ctx)
            raw = ctx.previous_outputs.get("step_create", "")
            return f"Formatted({raw})", None

        def critic_fn(ctx: TaskContext):
            fmt_out = ctx.previous_outputs.get("step_format", "")
            if "Formatted(Raw Data B)" in fmt_out:
                return "Need revision", CriticReviewResult(
                    revision_required=True,
                    feedback="Revise Raw Data",
                    target_step_id="step_create",
                )
            return "Approved", CriticReviewResult(revision_required=False)

        def finalizer_fn(ctx: TaskContext):
            fmt_out = ctx.previous_outputs.get("step_format", "")
            return f"Final: {fmt_out}", None

        c_agent = FunctionalWorkflowAgent("c_ag", "Creator", AgentRole.CREATOR, AgentConfig("provider_b", "m1"), response_fn=creator_fn, provider_manager=self.provider_mgr)
        f_agent = FunctionalWorkflowAgent("f_ag", "Formatter", AgentRole.CREATOR, AgentConfig("provider_b", "m1"), response_fn=formatter_fn, provider_manager=self.provider_mgr)
        cr_agent = FunctionalWorkflowAgent("cr_ag", "Critic", AgentRole.CRITIC, AgentConfig("provider_c", "m1"), response_fn=critic_fn, provider_manager=self.provider_mgr)
        fin_agent = FunctionalWorkflowAgent("fin_ag", "Finalizer", AgentRole.FINALIZER, AgentConfig("provider_c", "m1"), response_fn=finalizer_fn, provider_manager=self.provider_mgr)

        reg = AgentRegistry()
        reg.register(c_agent)
        reg.register(f_agent)
        reg.register(cr_agent)
        reg.register(fin_agent)

        config = WorkflowConfig(
            workflow_id="wf_intermediate_inv",
            name="Intermediate Invalidation",
            steps=[
                WorkflowStepConfig(step_id="step_create", agent_id="c_ag"),
                WorkflowStepConfig(step_id="step_format", agent_id="f_ag", dependencies=["step_create"]),
                WorkflowStepConfig(step_id="step_critic", agent_id="cr_ag", is_critic_step=True, allowed_revision_targets=["step_create"], dependencies=["step_format"]),
                WorkflowStepConfig(step_id="step_finalize", agent_id="fin_ag", dependencies=["step_format"]),
            ],
        )
        engine = WorkflowEngine(config, reg)
        result = engine.execute(TaskContext(user_prompt="Format data"))

        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        # Formatter was executed TWICE: once for B, once for B2!
        self.assertEqual(len(formatter_calls), 2)
        self.assertEqual(formatter_calls[0].previous_outputs.get("step_create"), "Raw Data B")
        self.assertEqual(formatter_calls[1].previous_outputs.get("step_create"), "Raw Data B2")
        self.assertEqual(result.final_output, "Final: Formatted(Raw Data B2)")

    def test_critic_revision_failed_target_step_follows_policy(self):
        """Test that a failed revision step follows the configured failure policy (ABORT)."""
        call_count = [0]

        def flaky_creator(ctx: TaskContext):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Initial draft B", None
            # On revision, provider crashes
            raise ProviderError("Unrecoverable GPU out of memory on revision", provider_id="provider_b")

        def critic_fn(ctx: TaskContext):
            return "Revise", CriticReviewResult(
                revision_required=True,
                feedback="Needs polish",
                target_step_id="step_flaky",
            )

        flaky_agent = FunctionalWorkflowAgent("flaky_ag", "Creator", AgentRole.CREATOR, AgentConfig("provider_b", "m1"), response_fn=flaky_creator, provider_manager=self.provider_mgr)
        critic_agent = FunctionalWorkflowAgent("crit_ag", "Critic", AgentRole.CRITIC, AgentConfig("provider_c", "m1"), response_fn=critic_fn, provider_manager=self.provider_mgr)

        reg = AgentRegistry()
        reg.register(flaky_agent)
        reg.register(critic_agent)

        config = WorkflowConfig(
            workflow_id="wf_rev_fail",
            name="Revision Failure Test",
            steps=[
                WorkflowStepConfig(step_id="step_flaky", agent_id="flaky_ag", failure_policy=StepFailurePolicy.ABORT, max_retries=0),
                WorkflowStepConfig(step_id="step_critic", agent_id="crit_ag", is_critic_step=True, allowed_revision_targets=["step_flaky"]),
            ],
        )
        engine = WorkflowEngine(config, reg)
        result = engine.execute(TaskContext(user_prompt="Task"))

        # ABORT policy stops workflow on revision failure
        self.assertEqual(result.status, WorkflowStatus.FAILED)
        self.assertIn("step_flaky", result.failed_steps)
        # Step records preserved: 1st create, 1st critic, 2nd create (failed)
        self.assertEqual(len(result.step_records), 3)
        self.assertEqual(result.step_records[2].status, AgentStatus.FAILED)
        self.assertEqual(result.step_records[2].revision_iteration, 1)

    def test_token_budget_post_request_detection_and_pre_request_enforcement(self):
        """Verify token budget accounting: verified usage only, pre-request checks, and post-request detection."""
        # Step 1 consumes 25 tokens. Budget is 20 tokens.
        # Step 1 completes -> post-step check detects 25 > 20 and halts before Step 2.
        config = WorkflowConfig(
            workflow_id="wf_token_verify",
            name="Token Verification",
            token_budget=20,
            steps=[
                WorkflowStepConfig(step_id="step_1", agent_id="planner_agent"),  # prov_a returns 25 tokens
                WorkflowStepConfig(step_id="step_2", agent_id="creator_agent"),
            ],
        )
        engine = WorkflowEngine(config, self.registry)
        result = engine.execute(TaskContext(user_prompt="Token query"))

        self.assertEqual(result.status, WorkflowStatus.FAILED)
        self.assertIn("token budget", result.error_summary.lower())
        # Step 1 record exists with verified tokens = 25
        self.assertEqual(len(result.step_records), 1)
        self.assertEqual(result.step_records[0].result.metadata.total_tokens, 25)
        # Step 2 never launched
        self.assertEqual(len(self.prov_b.calls), 0)


if __name__ == "__main__":
    unittest.main()
