import json
import unittest
from unittest.mock import MagicMock

from src.agents import (
    AgentConfig,
    AgentRegistry,
    AgentRole,
    AgentStatus,
    CreatorAgent,
    CriticAgent,
    FinalizerAgent,
    PlanMilestone,
    PlanOutput,
    PlannerAgent,
    ResearcherAgent,
    ResearchOutput,
    StructuredContractError,
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
    WorkflowConfig,
    WorkflowEngine,
    WorkflowStatus,
    WorkflowStepConfig,
)


class MockSpecializedProvider(AIProvider):
    """Mock AIProvider delivering configurable responses or errors for agent testing."""

    def __init__(self, provider_id: str = "mock_provider", response_text: str = "", tokens: int = 20):
        self.provider_id = provider_id
        self.display_name = provider_id.title()
        self.is_local = False
        self.response_text = response_text
        self.tokens = tokens
        self.exception_to_raise = None
        self.calls = []

    def stream(self, messages, model, **kwargs):
        self.calls.append({"messages": list(messages), "model": model, "kwargs": kwargs})
        if self.exception_to_raise:
            raise self.exception_to_raise

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


class TestSpecializedAgents(unittest.TestCase):
    def setUp(self):
        self.mock_cred_store = MagicMock(spec=CredentialStore)
        self.provider_mgr = ProviderManager(credential_store=self.mock_cred_store)
        self.provider_mgr._providers.clear()

        self.mock_provider = MockSpecializedProvider("mock_prov", tokens=30)
        self.provider_mgr.register_provider(self.mock_provider)

    def test_distinct_roles_and_system_instructions(self):
        """Verify each of the 5 agents has the correct role and a distinct system instruction."""
        planner = PlannerAgent(provider_manager=self.provider_mgr)
        researcher = ResearcherAgent(provider_manager=self.provider_mgr)
        creator = CreatorAgent(provider_manager=self.provider_mgr)
        critic = CriticAgent(provider_manager=self.provider_mgr)
        finalizer = FinalizerAgent(provider_manager=self.provider_mgr)

        self.assertEqual(planner.role, AgentRole.PLANNER)
        self.assertEqual(researcher.role, AgentRole.RESEARCHER)
        self.assertEqual(creator.role, AgentRole.CREATOR)
        self.assertEqual(critic.role, AgentRole.CRITIC)
        self.assertEqual(finalizer.role, AgentRole.FINALIZER)

        instructions = [
            planner.config.system_instruction,
            researcher.config.system_instruction,
            creator.config.system_instruction,
            critic.config.system_instruction,
            finalizer.config.system_instruction,
        ]

        # Verify all instructions are non-empty and unique
        self.assertEqual(len(instructions), len(set(instructions)))
        for instr in instructions:
            self.assertTrue(len(instr.strip()) > 50)

    # -------------------------------------------------------------------------
    # PlannerAgent Tests
    # -------------------------------------------------------------------------

    def test_planner_valid_json_contract(self):
        """Planner successfully parses and validates a structured JSON plan."""
        valid_plan_json = {
            "task_objective": "Build secure JWT authentication microservice",
            "deliverable": "FastAPI service with /login and /refresh endpoints",
            "milestones": [
                {"step_number": 1, "title": "Database Schema", "description": "Design user table", "dependencies": []},
                {"step_number": 2, "title": "JWT Utility", "description": "Implement token signing", "dependencies": [1]},
                {"step_number": 3, "title": "Endpoints", "description": "Create routes", "dependencies": [2]},
            ],
            "constraints": ["Use argon2 for password hashing", "Stateless tokens"],
            "risks": ["Token expiration clock drift"],
        }
        self.mock_provider.response_text = f"```json\n{json.dumps(valid_plan_json)}\n```"

        planner = PlannerAgent(
            config=AgentConfig("mock_prov", "model-1"),
            provider_manager=self.provider_mgr,
        )
        ctx = TaskContext(user_prompt="Build JWT service")
        result = planner.execute(ctx)

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertTrue(result.is_success)
        plan_obj: PlanOutput = result.metadata.extra.get("plan")
        self.assertIsNotNone(plan_obj)
        self.assertEqual(plan_obj.task_objective, "Build secure JWT authentication microservice")
        self.assertEqual(len(plan_obj.milestones), 3)
        self.assertEqual(plan_obj.milestones[1].dependencies, [1])

    def test_planner_markdown_fallback_parsing(self):
        """Planner falls back to markdown section parsing when JSON is omitted."""
        markdown_text = """
# Execution Plan

### Deliverable
FastAPI production service

### Milestones
1. Initialize repository and virtualenv
2. Implement user database model
3. Configure authentication routes

### Constraints
- Must run in Docker
- Python 3.12+

### Risks
- High database connection latency
"""
        self.mock_provider.response_text = markdown_text
        planner = PlannerAgent(
            config=AgentConfig("mock_prov", "model-1"),
            provider_manager=self.provider_mgr,
        )
        ctx = TaskContext(user_prompt="Build FastAPI service")
        result = planner.execute(ctx)

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        plan_obj: PlanOutput = result.metadata.extra.get("plan")
        self.assertIsNotNone(plan_obj)
        self.assertEqual(plan_obj.deliverable, "FastAPI production service")
        self.assertEqual(len(plan_obj.milestones), 3)

    def test_planner_invalid_contract_fails_cleanly(self):
        """Planner fails with StructuredContractError when plan references missing dependencies."""
        invalid_plan = {
            "task_objective": "Task",
            "deliverable": "App",
            "milestones": [
                {"step_number": 1, "title": "A", "description": "A", "dependencies": [999]},  # 999 does not exist!
            ],
        }
        self.mock_provider.response_text = f"```json\n{json.dumps(invalid_plan)}\n```"
        planner = PlannerAgent(config=AgentConfig("mock_prov", "model-1"), provider_manager=self.provider_mgr)
        result = planner.execute(TaskContext(user_prompt="Task"))

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("non-existent milestone dependency", result.error)

    def test_planner_empty_response_fails(self):
        """Planner fails cleanly if provider produces empty output."""
        self.mock_provider.response_text = "   "
        planner = PlannerAgent(config=AgentConfig("mock_prov", "model-1"), provider_manager=self.provider_mgr)
        result = planner.execute(TaskContext(user_prompt="Task"))

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("empty or whitespace", result.error.lower())

    # -------------------------------------------------------------------------
    # ResearcherAgent Tests
    # -------------------------------------------------------------------------

    def test_researcher_valid_json_contract(self):
        """Researcher parses verified facts and assumptions into ResearchOutput."""
        res_data = {
            "verified_facts": ["PostgreSQL 16 is specified by user", "Auth service must be stateless"],
            "assumptions": ["Redis will be used for token blacklisting", "Traffic is under 5k RPS"],
            "information_gaps": ["Session timeout policy not defined"],
            "recommendations": ["Use connection pooling via pgvector or asyncpg"],
        }
        self.mock_provider.response_text = f"```json\n{json.dumps(res_data)}\n```"

        researcher = ResearcherAgent(config=AgentConfig("mock_prov", "model-1"), provider_manager=self.provider_mgr)
        result = researcher.execute(TaskContext(user_prompt="Research architecture"))

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        res_obj: ResearchOutput = result.metadata.extra.get("research")
        self.assertIsNotNone(res_obj)
        self.assertEqual(len(res_obj.verified_facts), 2)
        self.assertEqual(len(res_obj.assumptions), 2)
        self.assertEqual(len(res_obj.information_gaps), 1)

    def test_researcher_markdown_fallback(self):
        """Researcher parses markdown bullet lists if JSON is omitted."""
        md = """
### Verified Facts
- Client requires OAuth2 compatibility
- Database is SQLite for development

### Assumptions
- Deployment target is Linux x86_64

### Information Gaps
- TLS termination proxy not specified

### Recommendations
- Use Argon2id for password hashing
"""
        self.mock_provider.response_text = md
        researcher = ResearcherAgent(config=AgentConfig("mock_prov", "model-1"), provider_manager=self.provider_mgr)
        result = researcher.execute(TaskContext(user_prompt="Task"))

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        res_obj: ResearchOutput = result.metadata.extra.get("research")
        self.assertIsNotNone(res_obj)
        self.assertEqual(len(res_obj.verified_facts), 2)
        self.assertEqual(len(res_obj.assumptions), 1)

    def test_researcher_empty_findings_fails(self):
        """Researcher fails if output has no facts, assumptions, or recommendations."""
        self.mock_provider.response_text = "I am a researcher. Nothing else to say."
        researcher = ResearcherAgent(config=AgentConfig("mock_prov", "model-1"), provider_manager=self.provider_mgr)
        result = researcher.execute(TaskContext(user_prompt="Task"))

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("Could not extract valid structured facts", result.error)

    # -------------------------------------------------------------------------
    # CreatorAgent Tests
    # -------------------------------------------------------------------------

    def test_creator_deliverable_generation(self):
        """Creator generates code deliverable and handles context."""
        code_output = "def authenticate_user(username, password):\n    return verify_hash(password)"
        self.mock_provider.response_text = code_output

        creator = CreatorAgent(config=AgentConfig("mock_prov", "model-1"), provider_manager=self.provider_mgr)
        ctx = TaskContext(
            user_prompt="Write auth function",
            previous_outputs={"planner": "Step 1: Write auth function"},
        )
        result = creator.execute(ctx)

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.output, code_output)

    def test_creator_incorporates_critique_feedback(self):
        """Creator receives critique feedback in instructions and messages."""
        self.mock_provider.response_text = "Revised code with bcrypt"

        creator = CreatorAgent(config=AgentConfig("mock_prov", "model-1"), provider_manager=self.provider_mgr)
        ctx = TaskContext(
            user_prompt="Write auth function",
            instructions="### CRITIQUE & REVISION FEEDBACK:\nUse bcrypt instead of md5",
        )
        result = creator.execute(ctx)

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        # Verify provider call received critique instructions
        sent_messages = self.mock_provider.calls[0]["messages"]
        user_msg = next(m.content for m in sent_messages if m.role == MessageRole.USER)
        self.assertIn("Use bcrypt instead of md5", user_msg)

    # -------------------------------------------------------------------------
    # CriticAgent Tests (Strict Fail-Safe Behavior)
    # -------------------------------------------------------------------------

    def test_critic_valid_approval(self):
        """Critic outputs valid JSON approval with revision_required=False."""
        approval_json = {
            "revision_required": False,
            "target_step_id": None,
            "reason": "Code meets security standards and test coverage",
            "feedback": "Approved. Excellent implementation.",
        }
        self.mock_provider.response_text = f"```json\n{json.dumps(approval_json)}\n```"

        critic = CriticAgent(config=AgentConfig("mock_prov", "model-1"), provider_manager=self.provider_mgr)
        result = critic.execute(TaskContext(user_prompt="Review code"))

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        review: CriticReviewResult = getattr(result.metadata, "critic_review")
        self.assertIsNotNone(review)
        self.assertFalse(review.revision_required)
        self.assertEqual(review.reason, "Code meets security standards and test coverage")

    def test_critic_valid_revision_request(self):
        """Critic outputs valid JSON revision request with revision_required=True."""
        revision_json = {
            "revision_required": True,
            "target_step_id": "step_create",
            "reason": "Hardcoded secret detected",
            "feedback": "Extract JWT_SECRET from environment variable rather than hardcoding string.",
        }
        self.mock_provider.response_text = f"```json\n{json.dumps(revision_json)}\n```"

        critic = CriticAgent(config=AgentConfig("mock_prov", "model-1"), provider_manager=self.provider_mgr)
        result = critic.execute(TaskContext(user_prompt="Review code"))

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        review: CriticReviewResult = getattr(result.metadata, "critic_review")
        self.assertIsNotNone(review)
        self.assertTrue(review.revision_required)
        self.assertEqual(review.target_step_id, "step_create")
        self.assertIn("JWT_SECRET", review.feedback)

    def test_critic_malformed_json_never_silently_approves(self):
        """Mandatory requirement: Malformed or unparseable JSON must fail the step, NEVER silently approve."""
        self.mock_provider.response_text = "This code looks kinda good but maybe fix the syntax: {invalid json here"

        critic = CriticAgent(config=AgentConfig("mock_prov", "model-1"), provider_manager=self.provider_mgr)
        result = critic.execute(TaskContext(user_prompt="Review code"))

        # Must fail, NOT complete with revision_required=False!
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("rejected to prevent silent approval", result.error)
        self.assertIsNone(result.metadata.critic_review)

    def test_critic_revision_without_target_fails(self):
        """Critic fails if revision_required=True but target_step_id is omitted."""
        invalid_rev = {
            "revision_required": True,
            "target_step_id": "",  # Empty target!
            "reason": "Fix bug",
            "feedback": "Fix the bug",
        }
        self.mock_provider.response_text = f"```json\n{json.dumps(invalid_rev)}\n```"

        critic = CriticAgent(config=AgentConfig("mock_prov", "model-1"), provider_manager=self.provider_mgr)
        result = critic.execute(TaskContext(user_prompt="Review"))

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("empty or invalid 'target_step_id'", result.error)

    def test_critic_non_boolean_revision_required_fails(self):
        """Critic fails if revision_required is not a boolean."""
        invalid_rev = {
            "revision_required": "yes",  # String instead of bool!
            "target_step_id": "step_create",
            "feedback": "Do revision",
        }
        self.mock_provider.response_text = f"```json\n{json.dumps(invalid_rev)}\n```"

        critic = CriticAgent(config=AgentConfig("mock_prov", "model-1"), provider_manager=self.provider_mgr)
        result = critic.execute(TaskContext(user_prompt="Review"))

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("must be a boolean", result.error)

    # -------------------------------------------------------------------------
    # FinalizerAgent Tests
    # -------------------------------------------------------------------------

    def test_finalizer_synthesis(self):
        """Finalizer synthesizes final answer and eliminates noise."""
        clean_final = "# Final API Documentation\n\nAll endpoints verified and operational."
        self.mock_provider.response_text = clean_final

        finalizer = FinalizerAgent(config=AgentConfig("mock_prov", "model-1"), provider_manager=self.provider_mgr)
        ctx = TaskContext(
            user_prompt="Produce final document",
            previous_outputs={
                "planner": "Step 1: plan",
                "creator": "Raw code",
                "critic": "Approved",
            },
        )
        result = finalizer.execute(ctx)

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.output, clean_final)

    def test_finalizer_empty_response_fails(self):
        """Finalizer fails if provider returns empty string."""
        self.mock_provider.response_text = ""
        finalizer = FinalizerAgent(config=AgentConfig("mock_prov", "model-1"), provider_manager=self.provider_mgr)
        result = finalizer.execute(TaskContext(user_prompt="Finalize"))

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("empty or whitespace", result.error)

    # -------------------------------------------------------------------------
    # End-to-End Workflow with Specialized Agents & Revision Loop
    # -------------------------------------------------------------------------

    def test_full_workflow_integration_with_critic_revision(self):
        """Complete 5-agent pipeline in WorkflowEngine with a Critic revision turn:
        Planner -> Researcher -> Creator (v1) -> Critic (requests rev) -> Creator (v2) -> Critic (approves) -> Finalizer
        """
        # Providers for different agents
        p_planner = MockSpecializedProvider(
            "p_plan",
            response_text=json.dumps({
                "task_objective": "Create auth module",
                "deliverable": "auth.py",
                "milestones": [{"step_number": 1, "title": "Implement", "description": "Implement auth", "dependencies": []}],
            }),
        )
        p_research = MockSpecializedProvider(
            "p_res",
            response_text=json.dumps({
                "verified_facts": ["OAuth2 requirement"],
                "assumptions": ["FastAPI framework"],
                "information_gaps": [],
                "recommendations": ["Use bcrypt"],
            }),
        )

        # Creator responds with v1 on first call, v2 on revision
        creator_call_count = [0]
        creator_responses = [
            "def auth():\n    return 'v1 plaintext'",
            "def auth():\n    return 'v2 bcrypt hashed'",
        ]

        class DynamicCreatorProvider(MockSpecializedProvider):
            def stream(self, messages, model, **kwargs):
                resp = creator_responses[min(creator_call_count[0], len(creator_responses) - 1)]
                creator_call_count[0] += 1
                yield StreamChunk(delta=resp, is_final=True, metadata=ResponseMetadata(model=model, provider_id=self.provider_id, total_tokens=15))

        p_create = DynamicCreatorProvider("p_create")

        # Critic requests revision on v1, approves on v2
        class DynamicCriticProvider(MockSpecializedProvider):
            def stream(self, messages, model, **kwargs):
                all_text = " ".join(m.content for m in messages)
                if "v1 plaintext" in all_text:
                    resp = json.dumps({
                        "revision_required": True,
                        "target_step_id": "step_create",
                        "reason": "Plaintext insecure",
                        "feedback": "Use bcrypt hashing instead of plaintext",
                    })
                else:
                    resp = json.dumps({
                        "revision_required": False,
                        "target_step_id": None,
                        "reason": "Approved: Bcrypt hashing verified",
                        "feedback": "Looks secure",
                    })
                yield StreamChunk(delta=resp, is_final=True, metadata=ResponseMetadata(model=model, provider_id=self.provider_id, total_tokens=15))

        p_critic = DynamicCriticProvider("p_critic")
        p_final = MockSpecializedProvider("p_fin", response_text="Final Deliverable: Production Auth Module with Bcrypt Hashing")

        # Register providers
        for p in [p_planner, p_research, p_create, p_critic, p_final]:
            self.provider_mgr.register_provider(p)

        reg = AgentRegistry()
        reg.register(PlannerAgent(agent_id="plan_ag", config=AgentConfig("p_plan", "m1"), provider_manager=self.provider_mgr))
        reg.register(ResearcherAgent(agent_id="res_ag", config=AgentConfig("p_res", "m1"), provider_manager=self.provider_mgr))
        reg.register(CreatorAgent(agent_id="create_ag", config=AgentConfig("p_create", "m1"), provider_manager=self.provider_mgr))
        reg.register(CriticAgent(agent_id="critic_ag", config=AgentConfig("p_critic", "m1"), provider_manager=self.provider_mgr))
        reg.register(FinalizerAgent(agent_id="final_ag", config=AgentConfig("p_fin", "m1"), provider_manager=self.provider_mgr))

        config = WorkflowConfig(
            workflow_id="wf_5_agents",
            name="Five Agent Collaboration",
            max_steps=15,
            max_revision_loops=2,
            steps=[
                WorkflowStepConfig(step_id="step_plan", agent_id="plan_ag"),
                WorkflowStepConfig(step_id="step_research", agent_id="res_ag", dependencies=["step_plan"]),
                WorkflowStepConfig(step_id="step_create", agent_id="create_ag", dependencies=["step_plan", "step_research"]),
                WorkflowStepConfig(
                    step_id="step_critic",
                    agent_id="critic_ag",
                    is_critic_step=True,
                    allowed_revision_targets=["step_create"],
                    dependencies=["step_create"],
                ),
                WorkflowStepConfig(step_id="step_finalize", agent_id="final_ag", dependencies=["step_create"]),
            ],
        )

        engine = WorkflowEngine(config=config, registry=reg)
        result = engine.execute(TaskContext(user_prompt="Build authentication module"))

        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertTrue(result.is_success)
        self.assertIn("Production Auth Module with Bcrypt Hashing", result.final_output)

        # Creator was called twice (v1, then v2 on revision)
        self.assertEqual(creator_call_count[0], 2)
        # Latest valid step output for creator is v2
        self.assertEqual(result.step_outputs["step_create"], "def auth():\n    return 'v2 bcrypt hashed'")

    def test_workflow_engine_rejects_unauthorized_critic_target(self):
        """WorkflowEngine rejects revision if Critic requests target not in allowed_revision_targets."""
        p_plan_ok = MockSpecializedProvider(
            "p_plan_ok",
            response_text=json.dumps({
                "task_objective": "Plan",
                "deliverable": "Plan deliverable",
                "milestones": [{"step_number": 1, "title": "M1", "description": "Step 1", "dependencies": []}],
            }),
        )
        self.provider_mgr.register_provider(p_plan_ok)

        p_create_ok = MockSpecializedProvider("p_create_ok", response_text="def foo(): pass")
        self.provider_mgr.register_provider(p_create_ok)

        p_critic_bad = MockSpecializedProvider(
            "p_crit_bad",
            response_text=json.dumps({
                "revision_required": True,
                "target_step_id": "step_plan",  # NOT in allowed_revision_targets!
                "reason": "Change plan",
                "feedback": "Rewrite plan",
            }),
        )
        self.provider_mgr.register_provider(p_critic_bad)

        reg = AgentRegistry()
        reg.register(PlannerAgent(agent_id="plan_ag_bad", config=AgentConfig("p_plan_ok", "m1"), provider_manager=self.provider_mgr))
        reg.register(CreatorAgent(agent_id="create_ag2", config=AgentConfig("p_create_ok", "m1"), provider_manager=self.provider_mgr))
        reg.register(CriticAgent(agent_id="critic_ag2", config=AgentConfig("p_crit_bad", "m1"), provider_manager=self.provider_mgr))

        config = WorkflowConfig(
            workflow_id="wf_bad_target",
            name="Unauthorized Target",
            steps=[
                WorkflowStepConfig(step_id="step_plan", agent_id="plan_ag_bad"),
                WorkflowStepConfig(step_id="step_create", agent_id="create_ag2", dependencies=["step_plan"]),
                WorkflowStepConfig(
                    step_id="step_critic",
                    agent_id="critic_ag2",
                    is_critic_step=True,
                    allowed_revision_targets=["step_create"],  # "step_plan" is known, but NOT in allowed_revision_targets!
                ),
            ],
        )
        engine = WorkflowEngine(config=config, registry=reg)
        result = engine.execute(TaskContext(user_prompt="Task"))

        self.assertEqual(result.status, WorkflowStatus.FAILED)
        self.assertIn("not in allowed_revision_targets", result.error_summary)


if __name__ == "__main__":
    unittest.main()
