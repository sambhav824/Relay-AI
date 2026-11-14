import json
import threading
import time
import unittest
from unittest.mock import MagicMock

from src.agents import (
    AgentRegistry,
    AgentRole,
    AgentStatus,
    TaskContext,
)
from src.core.events import ResponseMetadata, StreamChunk
from src.core.message import Message, MessageRole
from src.core.security import CredentialStore
from src.providers.base import AIProvider
from src.providers.exceptions import ProviderError
from src.providers.manager import ProviderManager
from src.providers.models import ModelInfo
from src.teams import (
    TeamConfig,
    TeamConfigurationError,
    TeamError,
    TeamExecutionError,
    TeamExecutionObserver,
    TeamExecutionResult,
    TeamExecutionSnapshot,
    TeamFactory,
    TeamMemberConfig,
    TeamOrchestrator,
    create_default_team_config,
)
from src.workflow.models import StepFailurePolicy, WorkflowStatus


class MockTeamProvider(AIProvider):
    """Configurable mock AIProvider for team orchestration testing."""

    def __init__(
        self,
        provider_id: str,
        response_text: str = "",
        tokens: int = 20,
        exception_to_raise: Exception = None,
        delay_sec: float = 0.0,
    ):
        self.provider_id = provider_id
        self.display_name = provider_id.title()
        self.is_local = (provider_id == "ollama")
        self.response_text = response_text
        self.tokens = tokens
        self.exception_to_raise = exception_to_raise
        self.delay_sec = delay_sec
        self.calls = []

    def stream(self, messages, model, **kwargs):
        self.calls.append({"messages": list(messages), "model": model, "kwargs": kwargs})
        if self.exception_to_raise:
            raise self.exception_to_raise

        if self.delay_sec > 0:
            time.sleep(self.delay_sec)

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
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        return self.response_text

    def validate_connection(self):
        return True

    def get_available_models(self):
        return [ModelInfo("model-1", "Model 1", self.provider_id)]


class RecordingTeamObserver(TeamExecutionObserver):
    """Observer capturing lifecycle events and telemetry snapshots for assertion verification."""

    def __init__(self, raise_on_event: bool = False):
        self.raise_on_event = raise_on_event
        self.events: list = []
        self.started_team_ids: list = []
        self.step_started_snapshots: list = []
        self.step_completed_snapshots: list = []
        self.revisions: list = []
        self.completed_results: list = []

    def on_team_started(self, team_id: str, task_id: str):
        self.events.append(("team_started", team_id, task_id))
        self.started_team_ids.append(team_id)
        if self.raise_on_event:
            raise RuntimeError("Observer deliberate crash during on_team_started")

    def on_step_started(self, snapshot: TeamExecutionSnapshot):
        self.events.append(("step_started", snapshot.step_id, snapshot.role))
        self.step_started_snapshots.append(snapshot)
        if self.raise_on_event:
            raise RuntimeError("Observer deliberate crash during on_step_started")

    def on_step_completed(self, snapshot: TeamExecutionSnapshot):
        self.events.append(("step_completed", snapshot.step_id, snapshot.role))
        self.step_completed_snapshots.append(snapshot)
        if self.raise_on_event:
            raise RuntimeError("Observer deliberate crash during on_step_completed")

    def on_revision_requested(self, target_step_id: str, reason: str, feedback: str, loop_count: int):
        self.events.append(("revision_requested", target_step_id, loop_count))
        self.revisions.append((target_step_id, reason, feedback, loop_count))
        if self.raise_on_event:
            raise RuntimeError("Observer deliberate crash during on_revision_requested")

    def on_team_completed(self, result: TeamExecutionResult):
        self.events.append(("team_completed", result.team_id, result.status))
        self.completed_results.append(result)
        if self.raise_on_event:
            raise RuntimeError("Observer deliberate crash during on_team_completed")


class TestTeamOrchestration(unittest.TestCase):
    def setUp(self):
        self.mock_cred_store = MagicMock(spec=CredentialStore)
        self.mock_cred_store.has_api_key.return_value = True
        self.provider_mgr = ProviderManager(credential_store=self.mock_cred_store)
        self.provider_mgr._providers.clear()

        # Standard default responses for agents
        self.plan_json = json.dumps({
            "task_objective": "Build user auth service",
            "deliverable": "auth.py",
            "milestones": [{"step_number": 1, "title": "Design", "description": "Design auth", "dependencies": []}],
        })
        self.research_json = json.dumps({
            "verified_facts": ["OAuth2 requirement"],
            "assumptions": ["FastAPI framework"],
            "information_gaps": [],
            "recommendations": ["Use bcrypt"],
        })
        self.creator_output = "def authenticate():\n    return 'auth_v1_ok'"
        self.critic_approval_json = json.dumps({
            "revision_required": False,
            "target_step_id": None,
            "reason": "Approved: meets requirements",
            "feedback": "Clean code",
        })
        self.finalizer_output = "# Final User Auth Deliverable\n\nProduction-ready authentication module."

        # Register providers
        self.prov_plan = MockTeamProvider("prov_plan", response_text=self.plan_json, tokens=20)
        self.prov_res = MockTeamProvider("prov_res", response_text=self.research_json, tokens=20)
        self.prov_create = MockTeamProvider("prov_create", response_text=self.creator_output, tokens=30)
        self.prov_critic = MockTeamProvider("prov_critic", response_text=self.critic_approval_json, tokens=15)
        self.prov_fin = MockTeamProvider("prov_fin", response_text=self.finalizer_output, tokens=25)

        for p in [self.prov_plan, self.prov_res, self.prov_create, self.prov_critic, self.prov_fin]:
            self.provider_mgr.register_provider(p)

        self.registry = AgentRegistry()
        self.factory = TeamFactory(provider_manager=self.provider_mgr, registry=self.registry)

    # -------------------------------------------------------------------------
    # Configuration & Factory Tests
    # -------------------------------------------------------------------------

    def test_default_team_config_creation(self):
        """Verify default team creation with all 5 specialized agents."""
        config = create_default_team_config(team_id="team_def", name="Default Team", default_provider_id="prov_plan")
        self.assertEqual(config.team_id, "team_def")
        self.assertEqual(len(config.members), 5)
        self.assertTrue(all(m.enabled for m in config.members.values()))
        self.assertEqual(config.members[AgentRole.CREATOR].role, AgentRole.CREATOR)

    def test_custom_provider_model_per_agent(self):
        """Verify assigning distinct providers and models per specialized agent."""
        config = create_default_team_config(
            team_id="team_custom",
            name="Custom Team",
            custom_member_configs={
                AgentRole.PLANNER: {"provider_id": "prov_plan", "model_id": "model-plan", "temperature": 0.1},
                AgentRole.RESEARCHER: {"provider_id": "prov_res", "model_id": "model-res", "temperature": 0.2},
                AgentRole.CREATOR: {"provider_id": "prov_create", "model_id": "model-create", "temperature": 0.6},
                AgentRole.CRITIC: {"provider_id": "prov_critic", "model_id": "model-crit", "temperature": 0.0},
                AgentRole.FINALIZER: {"provider_id": "prov_fin", "model_id": "model-fin", "temperature": 0.3},
            },
        )
        self.factory.validate_team_readiness(config)
        agents = self.factory.instantiate_and_register_agents(config)
        self.assertEqual(len(agents), 5)
        self.assertEqual(agents[0].config.provider_id, "prov_plan")
        self.assertEqual(agents[2].config.provider_id, "prov_create")

    def test_missing_provider_rejection(self):
        """Verify team readiness fails fast if an enabled agent specifies an unregistered provider."""
        config = create_default_team_config(
            team_id="team_missing",
            default_provider_id="prov_plan",
            custom_member_configs={
                AgentRole.CREATOR: {"provider_id": "non_existent_provider", "model_id": "m1"},
            },
        )
        with self.assertRaises(TeamConfigurationError) as ctx:
            self.factory.validate_team_readiness(config)
        self.assertIn("non_existent_provider", str(ctx.exception))
        # Ensure no secrets or API keys are exposed
        self.assertNotIn("sk-", str(ctx.exception))

    def test_disabled_agent_without_provider_succeeds(self):
        """Disabled agents do NOT require a registered provider or credentials."""
        config = create_default_team_config(
            team_id="team_disabled_prov",
            custom_member_configs={
                AgentRole.PLANNER: {"provider_id": "unregistered_prov", "model_id": "m1", "enabled": False},
                AgentRole.RESEARCHER: {"provider_id": "prov_res", "model_id": "m1"},
                AgentRole.CREATOR: {"provider_id": "prov_create", "model_id": "m1"},
                AgentRole.CRITIC: {"provider_id": "prov_critic", "model_id": "m1"},
                AgentRole.FINALIZER: {"provider_id": "prov_fin", "model_id": "m1"},
            },
        )
        # Must succeed because Planner is disabled
        self.factory.validate_team_readiness(config)
        agents = self.factory.instantiate_and_register_agents(config)
        self.assertEqual(len(agents), 4)
        self.assertNotIn("team_disabled_prov_planner", [a.agent_id for a in agents])

    def test_credential_readiness_cloud_provider_missing_key_rejected(self):
        """Pre-flight check rejects cloud providers missing credentials without leaking secrets or calling inference."""
        cloud_no_cred = MockTeamProvider("cloud_no_cred")
        cloud_no_cred.is_local = False
        self.provider_mgr.register_provider(cloud_no_cred)

        # Configure credential store to report no key for this provider
        self.mock_cred_store.has_api_key.side_effect = lambda pid: pid != "cloud_no_cred"

        config = create_default_team_config(
            team_id="t_cred_missing",
            default_provider_id="prov_plan",
            custom_member_configs={
                AgentRole.CREATOR: {"provider_id": "cloud_no_cred", "model_id": "m1"},
            },
        )

        with self.assertRaises(TeamConfigurationError) as ctx:
            self.factory.validate_team_readiness(config)

        err_msg = str(ctx.exception)
        self.assertIn("missing required API credentials", err_msg)
        self.assertIn("cloud_no_cred", err_msg)
        self.assertNotIn("sk-", err_msg)
        self.assertNotIn("Bearer", err_msg)

        # Strictly verify zero inference requests were made
        self.assertEqual(len(cloud_no_cred.calls), 0)

    def test_credential_readiness_local_provider_needs_no_key(self):
        """Local providers (e.g. Ollama with is_local=True) require no credentials in CredentialStore."""
        local_prov = MockTeamProvider("ollama_local")
        local_prov.is_local = True
        self.provider_mgr.register_provider(local_prov)

        # Credential store reports no key for ollama_local
        self.mock_cred_store.has_api_key.side_effect = lambda pid: pid != "ollama_local"

        config = create_default_team_config(
            team_id="t_local_cred",
            default_provider_id="ollama_local",
        )
        # Must succeed without error
        self.factory.validate_team_readiness(config)

    def test_credential_readiness_cloud_provider_with_manual_key_succeeds(self):
        """Cloud provider with manual key override succeeds even if CredentialStore is empty."""
        cloud_manual = MockTeamProvider("cloud_manual")
        cloud_manual.is_local = False
        cloud_manual._manual_api_key = "user-supplied-key"
        self.provider_mgr.register_provider(cloud_manual)

        self.mock_cred_store.has_api_key.return_value = False

        config = create_default_team_config(
            team_id="t_manual_cred",
            default_provider_id="cloud_manual",
        )
        self.factory.validate_team_readiness(config)

    def test_credential_readiness_disabled_agent_missing_cred_succeeds(self):
        """Disabled agents missing credentials are never validated or instantiated."""
        cloud_no_cred = MockTeamProvider("cloud_unauth")
        cloud_no_cred.is_local = False
        self.provider_mgr.register_provider(cloud_no_cred)
        self.mock_cred_store.has_api_key.return_value = False

        config = create_default_team_config(
            team_id="t_disabled_unauth",
            custom_member_configs={
                AgentRole.PLANNER: {"provider_id": "cloud_unauth", "model_id": "m1", "enabled": False},
                AgentRole.RESEARCHER: {"provider_id": "prov_res", "model_id": "m1"},
                AgentRole.CREATOR: {"provider_id": "prov_create", "model_id": "m1"},
                AgentRole.CRITIC: {"provider_id": "prov_critic", "model_id": "m1"},
                AgentRole.FINALIZER: {"provider_id": "prov_fin", "model_id": "m1"},
            },
        )
        # Make sure other providers pass
        self.mock_cred_store.has_api_key.side_effect = lambda pid: pid != "cloud_unauth"

        self.factory.validate_team_readiness(config)
        agents = self.factory.instantiate_and_register_agents(config)
        self.assertNotIn("t_disabled_unauth_planner", [a.agent_id for a in agents])

    def test_mandatory_creator_cannot_be_disabled(self):
        """Creator agent is strictly mandatory and cannot be disabled."""
        config = create_default_team_config(team_id="team_no_creator")
        config.members[AgentRole.CREATOR].enabled = False
        with self.assertRaises(TeamConfigurationError) as ctx:
            config.validate()
        self.assertIn("CREATOR' agent is mandatory", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Team Isolation & Lifecycle Safety Tests
    # -------------------------------------------------------------------------

    def test_team_scoped_agent_ids_prevent_collision(self):
        """Verify two independent teams register with distinct namespaced agent IDs without collision."""
        cfg_alpha = create_default_team_config(team_id="team_alpha", default_provider_id="prov_plan")
        cfg_beta = create_default_team_config(team_id="team_beta", default_provider_id="prov_plan")

        agents_alpha = self.factory.instantiate_and_register_agents(cfg_alpha)
        agents_beta = self.factory.instantiate_and_register_agents(cfg_beta)

        self.assertEqual(len(agents_alpha), 5)
        self.assertEqual(len(agents_beta), 5)
        self.assertEqual(self.registry.count(), 10)
        self.assertIsNotNone(self.registry.get("team_alpha_creator"))
        self.assertIsNotNone(self.registry.get("team_beta_creator"))

    def test_cleanup_unregisters_only_owned_agents(self):
        """Cleanup removes only the agents belonging to that team, leaving other teams intact."""
        member_map = {
            AgentRole.PLANNER: {"provider_id": "prov_plan", "model_id": "m1"},
            AgentRole.RESEARCHER: {"provider_id": "prov_res", "model_id": "m1"},
            AgentRole.CREATOR: {"provider_id": "prov_create", "model_id": "m1"},
            AgentRole.CRITIC: {"provider_id": "prov_critic", "model_id": "m1"},
            AgentRole.FINALIZER: {"provider_id": "prov_fin", "model_id": "m1"},
        }
        cfg_alpha = create_default_team_config(team_id="alpha_t", custom_member_configs=member_map)
        cfg_beta = create_default_team_config(team_id="beta_t", custom_member_configs=member_map)

        orch_alpha = TeamOrchestrator(cfg_alpha, factory=self.factory, registry=self.registry)
        orch_beta = TeamOrchestrator(cfg_beta, factory=self.factory, registry=self.registry)

        orch_alpha.execute("Task Alpha")
        orch_beta.execute("Task Beta")

        self.assertEqual(self.registry.count(), 10)

        # Cleanup Alpha only
        orch_alpha.cleanup()
        self.assertEqual(self.registry.count(), 5)
        self.assertIsNone(self.registry.get("alpha_t_creator"))
        self.assertIsNotNone(self.registry.get("beta_t_creator"))

        # Cleanup Beta
        orch_beta.cleanup()
        self.assertEqual(self.registry.count(), 0)

    def test_concurrent_execution_on_same_instance_rejected(self):
        """Same TeamOrchestrator instance prohibits concurrent execution."""
        slow_prov = MockTeamProvider("slow_p", response_text=self.creator_output, delay_sec=0.2)
        self.provider_mgr.register_provider(slow_prov)

        cfg = create_default_team_config(
            team_id="team_slow",
            custom_member_configs={
                AgentRole.PLANNER: {"enabled": False},
                AgentRole.RESEARCHER: {"enabled": False},
                AgentRole.CREATOR: {"provider_id": "slow_p", "model_id": "m1"},
                AgentRole.CRITIC: {"enabled": False},
                AgentRole.FINALIZER: {"enabled": False},
            },
        )
        orch = TeamOrchestrator(cfg, factory=self.factory, registry=self.registry)

        errors = []

        def run_task():
            try:
                orch.execute("Slow task")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=run_task)
        t1.start()
        time.sleep(0.04)

        try:
            orch.execute("Concurrent task")
        except TeamExecutionError as exc:
            errors.append(exc)

        t1.join()
        self.assertEqual(len(errors), 1)
        self.assertIn("already executing a task", str(errors[0]))

    def test_multiple_independent_teams_execute_concurrently(self):
        """Two different TeamOrchestrator instances can execute concurrently without state leakage."""
        cfg_1 = create_default_team_config(
            team_id="t1",
            custom_member_configs={
                AgentRole.PLANNER: {"provider_id": "prov_plan", "model_id": "m1"},
                AgentRole.CREATOR: {"provider_id": "prov_create", "model_id": "m1"},
                AgentRole.RESEARCHER: {"enabled": False},
                AgentRole.CRITIC: {"enabled": False},
                AgentRole.FINALIZER: {"enabled": False},
            },
        )
        cfg_2 = create_default_team_config(
            team_id="t2",
            custom_member_configs={
                AgentRole.PLANNER: {"provider_id": "prov_plan", "model_id": "m1"},
                AgentRole.CREATOR: {"provider_id": "prov_create", "model_id": "m1"},
                AgentRole.RESEARCHER: {"enabled": False},
                AgentRole.CRITIC: {"enabled": False},
                AgentRole.FINALIZER: {"enabled": False},
            },
        )
        orch_1 = TeamOrchestrator(cfg_1, factory=self.factory, registry=self.registry)
        orch_2 = TeamOrchestrator(cfg_2, factory=self.factory, registry=self.registry)

        results = {}

        def run_1():
            results["r1"] = orch_1.execute("Task 1")

        def run_2():
            results["r2"] = orch_2.execute("Task 2")

        th1 = threading.Thread(target=run_1)
        th2 = threading.Thread(target=run_2)
        th1.start()
        th2.start()
        th1.join()
        th2.join()

        self.assertTrue(results["r1"].is_success)
        self.assertTrue(results["r2"].is_success)
        self.assertEqual(results["r1"].team_id, "t1")
        self.assertEqual(results["r2"].team_id, "t2")

    # -------------------------------------------------------------------------
    # Sequential Execution & Context Transfer Tests
    # -------------------------------------------------------------------------

    def test_full_five_agent_pipeline_execution(self):
        """Execute standard Planner -> Researcher -> Creator -> Critic -> Finalizer pipeline."""
        config = create_default_team_config(
            team_id="team_5_full",
            custom_member_configs={
                AgentRole.PLANNER: {"provider_id": "prov_plan", "model_id": "m1"},
                AgentRole.RESEARCHER: {"provider_id": "prov_res", "model_id": "m1"},
                AgentRole.CREATOR: {"provider_id": "prov_create", "model_id": "m1"},
                AgentRole.CRITIC: {"provider_id": "prov_critic", "model_id": "m1"},
                AgentRole.FINALIZER: {"provider_id": "prov_fin", "model_id": "m1"},
            },
        )
        orch = TeamOrchestrator(config, factory=self.factory, registry=self.registry)
        observer = RecordingTeamObserver()
        orch.add_observer(observer)

        result = orch.execute("Build complete auth service")

        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertTrue(result.is_success)
        self.assertEqual(len(result.step_records), 5)
        self.assertEqual(result.final_deliverable, self.finalizer_output)
        self.assertEqual(result.total_tokens, 110)  # 20 + 20 + 30 + 15 + 25

        # Check selective context: Creator received Plan and Research
        creator_call = self.prov_create.calls[0]["messages"]
        creator_prompt = next(m.content for m in creator_call if m.role == MessageRole.USER)
        self.assertIn("Build user auth service", creator_prompt)
        self.assertIn("OAuth2 requirement", creator_prompt)

    def test_critic_revision_and_finalizer_receives_latest_output(self):
        """Critic requests Creator revision -> Creator produces v2 -> Finalizer receives v2 (never v1)."""
        call_count = [0]
        creator_v1 = "def auth():\n    return 'insecure_plaintext'"
        creator_v2 = "def auth():\n    return 'secure_bcrypt_hash'"

        class RevisionCreatorProvider(MockTeamProvider):
            def stream(self, messages, model, **kwargs):
                self.calls.append({"messages": list(messages)})
                resp = creator_v1 if call_count[0] == 0 else creator_v2
                call_count[0] += 1
                yield StreamChunk(delta=resp, is_final=True, metadata=ResponseMetadata(model=model, provider_id=self.provider_id, total_tokens=15))

        prov_dyn_create = RevisionCreatorProvider("prov_dyn_create")

        class RevisionCriticProvider(MockTeamProvider):
            def stream(self, messages, model, **kwargs):
                self.calls.append({"messages": list(messages)})
                all_text = " ".join(m.content for m in messages)
                if "insecure_plaintext" in all_text:
                    resp = json.dumps({
                        "revision_required": True,
                        "target_step_id": "team_rev_step_create",
                        "reason": "Plaintext insecure",
                        "feedback": "Use bcrypt hashing",
                    })
                else:
                    resp = json.dumps({
                        "revision_required": False,
                        "target_step_id": None,
                        "reason": "Approved",
                        "feedback": "Bcrypt verified",
                    })
                yield StreamChunk(delta=resp, is_final=True, metadata=ResponseMetadata(model=model, provider_id=self.provider_id, total_tokens=15))

        prov_dyn_critic = RevisionCriticProvider("prov_dyn_critic")
        self.provider_mgr.register_provider(prov_dyn_create)
        self.provider_mgr.register_provider(prov_dyn_critic)

        config = create_default_team_config(
            team_id="team_rev",
            custom_member_configs={
                AgentRole.PLANNER: {"provider_id": "prov_plan", "model_id": "m1"},
                AgentRole.RESEARCHER: {"enabled": False},
                AgentRole.CREATOR: {"provider_id": "prov_dyn_create", "model_id": "m1"},
                AgentRole.CRITIC: {"provider_id": "prov_dyn_critic", "model_id": "m1"},
                AgentRole.FINALIZER: {"provider_id": "prov_fin", "model_id": "m1"},
            },
        )
        orch = TeamOrchestrator(config, factory=self.factory, registry=self.registry)
        observer = RecordingTeamObserver()
        orch.add_observer(observer)

        result = orch.execute("Build secure auth")

        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertEqual(result.revision_count, 1)

        # Creator executed twice
        self.assertEqual(call_count[0], 2)
        # Finalizer received v2, not v1
        finalizer_call = self.prov_fin.calls[0]["messages"]
        finalizer_prompt = next(m.content for m in finalizer_call if m.role == MessageRole.USER)
        self.assertIn("secure_bcrypt_hash", finalizer_prompt)
        self.assertNotIn("insecure_plaintext", finalizer_prompt)

        # Observer recorded revision
        self.assertEqual(len(observer.revisions), 1)
        self.assertEqual(observer.revisions[0][0], "team_rev_step_create")

    # -------------------------------------------------------------------------
    # Enable / Disable Combinations Tests
    # -------------------------------------------------------------------------

    def test_pipeline_planner_disabled(self):
        """Researcher -> Creator -> Critic -> Finalizer (Planner disabled)."""
        cfg = create_default_team_config(
            team_id="t_no_plan",
            custom_member_configs={
                AgentRole.PLANNER: {"enabled": False},
                AgentRole.RESEARCHER: {"provider_id": "prov_res", "model_id": "m1"},
                AgentRole.CREATOR: {"provider_id": "prov_create", "model_id": "m1"},
                AgentRole.CRITIC: {"provider_id": "prov_critic", "model_id": "m1"},
                AgentRole.FINALIZER: {"provider_id": "prov_fin", "model_id": "m1"},
            },
        )
        orch = TeamOrchestrator(cfg, factory=self.factory, registry=self.registry)
        result = orch.execute("Task without plan")
        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertEqual(len(result.step_records), 4)

    def test_pipeline_researcher_disabled(self):
        """Planner -> Creator -> Critic -> Finalizer (Researcher disabled)."""
        cfg = create_default_team_config(
            team_id="t_no_res",
            custom_member_configs={
                AgentRole.PLANNER: {"provider_id": "prov_plan", "model_id": "m1"},
                AgentRole.RESEARCHER: {"enabled": False},
                AgentRole.CREATOR: {"provider_id": "prov_create", "model_id": "m1"},
                AgentRole.CRITIC: {"provider_id": "prov_critic", "model_id": "m1"},
                AgentRole.FINALIZER: {"provider_id": "prov_fin", "model_id": "m1"},
            },
        )
        orch = TeamOrchestrator(cfg, factory=self.factory, registry=self.registry)
        result = orch.execute("Task without research")
        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertEqual(len(result.step_records), 4)

    def test_pipeline_critic_disabled(self):
        """Planner -> Researcher -> Creator -> Finalizer (Critic disabled)."""
        cfg = create_default_team_config(
            team_id="t_no_crit",
            custom_member_configs={
                AgentRole.PLANNER: {"provider_id": "prov_plan", "model_id": "m1"},
                AgentRole.RESEARCHER: {"provider_id": "prov_res", "model_id": "m1"},
                AgentRole.CREATOR: {"provider_id": "prov_create", "model_id": "m1"},
                AgentRole.CRITIC: {"enabled": False},
                AgentRole.FINALIZER: {"provider_id": "prov_fin", "model_id": "m1"},
            },
        )
        orch = TeamOrchestrator(cfg, factory=self.factory, registry=self.registry)
        result = orch.execute("Task without critic")
        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertEqual(len(result.step_records), 4)

    def test_pipeline_finalizer_disabled(self):
        """Planner -> Researcher -> Creator -> Critic (Finalizer disabled). Creator output is final."""
        cfg = create_default_team_config(
            team_id="t_no_fin",
            custom_member_configs={
                AgentRole.PLANNER: {"provider_id": "prov_plan", "model_id": "m1"},
                AgentRole.RESEARCHER: {"provider_id": "prov_res", "model_id": "m1"},
                AgentRole.CREATOR: {"provider_id": "prov_create", "model_id": "m1"},
                AgentRole.CRITIC: {"provider_id": "prov_critic", "model_id": "m1"},
                AgentRole.FINALIZER: {"enabled": False},
            },
        )
        orch = TeamOrchestrator(cfg, factory=self.factory, registry=self.registry)
        result = orch.execute("Task without finalizer")
        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertEqual(len(result.step_records), 4)
        # Final deliverable must be Creator's output
        self.assertEqual(result.final_deliverable, self.creator_output)

    def test_minimal_pipeline_creator_only(self):
        """Only Creator is enabled. Root prompt goes directly to Creator and its output is final."""
        cfg = create_default_team_config(
            team_id="t_creator_only",
            custom_member_configs={
                AgentRole.PLANNER: {"enabled": False},
                AgentRole.RESEARCHER: {"enabled": False},
                AgentRole.CREATOR: {"provider_id": "prov_create", "model_id": "m1"},
                AgentRole.CRITIC: {"enabled": False},
                AgentRole.FINALIZER: {"enabled": False},
            },
        )
        orch = TeamOrchestrator(cfg, factory=self.factory, registry=self.registry)
        result = orch.execute("Generate single script")
        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertEqual(len(result.step_records), 1)
        self.assertEqual(result.final_deliverable, self.creator_output)

    # -------------------------------------------------------------------------
    # Observer Resilience & Failure Handling Tests
    # -------------------------------------------------------------------------

    def test_flaky_observer_exceptions_do_not_halt_team(self):
        """Observer callback raising an unhandled exception does NOT crash or abort the team."""
        flaky_observer = RecordingTeamObserver(raise_on_event=True)
        config = create_default_team_config(
            team_id="t_flaky_obs",
            custom_member_configs={
                AgentRole.PLANNER: {"enabled": False},
                AgentRole.RESEARCHER: {"enabled": False},
                AgentRole.CREATOR: {"provider_id": "prov_create", "model_id": "m1"},
                AgentRole.CRITIC: {"enabled": False},
                AgentRole.FINALIZER: {"enabled": False},
            },
        )
        orch = TeamOrchestrator(config, factory=self.factory, registry=self.registry)
        orch.add_observer(flaky_observer)

        result = orch.execute("Task with flaky observer")
        # Must still succeed cleanly despite observer crashes
        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertTrue(result.is_success)

    def test_terminal_event_fires_exactly_once_on_failure(self):
        """Observer's on_team_completed event fires exactly once even when a step fails."""
        failing_prov = MockTeamProvider("fail_p", exception_to_raise=ProviderError("Network offline", provider_id="fail_p"))
        self.provider_mgr.register_provider(failing_prov)

        config = create_default_team_config(
            team_id="t_fail",
            custom_member_configs={
                AgentRole.PLANNER: {"enabled": False},
                AgentRole.RESEARCHER: {"enabled": False},
                AgentRole.CREATOR: {"provider_id": "fail_p", "model_id": "m1", "failure_policy": StepFailurePolicy.ABORT},
                AgentRole.CRITIC: {"enabled": False},
                AgentRole.FINALIZER: {"enabled": False},
            },
        )
        orch = TeamOrchestrator(config, factory=self.factory, registry=self.registry)
        observer = RecordingTeamObserver()
        orch.add_observer(observer)

        result = orch.execute("Failing task")
        self.assertEqual(result.status, WorkflowStatus.FAILED)
        self.assertEqual(len(observer.completed_results), 1)
        self.assertEqual(observer.completed_results[0].status, WorkflowStatus.FAILED)

    def test_cooperative_cancellation(self):
        """Cancelling the team orchestrator halts execution cleanly with status CANCELLED."""
        slow_prov = MockTeamProvider("slow_canc", response_text=self.creator_output, delay_sec=0.4)
        self.provider_mgr.register_provider(slow_canc := slow_prov)

        config = create_default_team_config(
            team_id="t_canc",
            custom_member_configs={
                AgentRole.PLANNER: {"enabled": False},
                AgentRole.RESEARCHER: {"enabled": False},
                AgentRole.CREATOR: {"provider_id": "slow_canc", "model_id": "m1"},
                AgentRole.CRITIC: {"enabled": False},
                AgentRole.FINALIZER: {"enabled": False},
            },
        )
        orch = TeamOrchestrator(config, factory=self.factory, registry=self.registry)
        observer = RecordingTeamObserver()
        orch.add_observer(observer)

        def trigger_cancel():
            time.sleep(0.08)
            orch.cancel()

        ct = threading.Thread(target=trigger_cancel)
        ct.start()
        result = orch.execute("Cancel this task")
        ct.join()

        self.assertEqual(result.status, WorkflowStatus.CANCELLED)
        self.assertEqual(len(observer.completed_results), 1)
        self.assertEqual(observer.completed_results[0].status, WorkflowStatus.CANCELLED)

    def test_exactly_once_terminal_observer_event_on_timeout(self):
        """Team execution exceeding timeout returns FAILED, fires on_team_completed exactly once, and releases lock."""
        # Provider with 0.3s delay
        timeout_prov = MockTeamProvider("prov_timeout", response_text=self.creator_output, delay_sec=0.25)
        self.provider_mgr.register_provider(timeout_prov)

        config = create_default_team_config(
            team_id="t_timeout",
            timeout_sec=0.1,  # Strict short timeout
            custom_member_configs={
                AgentRole.PLANNER: {"enabled": False},
                AgentRole.RESEARCHER: {"enabled": False},
                AgentRole.CREATOR: {"provider_id": "prov_timeout", "model_id": "m1"},
                AgentRole.CRITIC: {"enabled": False},
                AgentRole.FINALIZER: {"enabled": False},
            },
        )
        orch = TeamOrchestrator(config, factory=self.factory, registry=self.registry)
        observer = RecordingTeamObserver()
        orch.add_observer(observer)

        result = orch.execute("Long running task that will time out")

        # 1. Terminal status FAILED with typed timeout distinction
        self.assertEqual(result.status, WorkflowStatus.FAILED)
        self.assertFalse(result.is_success)
        self.assertTrue(result.is_timeout)
        self.assertIn("timeout", result.error_summary.lower())
        self.assertEqual(result.failed_steps, [])  # Timeout is a global pipeline deadline, not a specific step logic failure

        # 2. on_team_completed fired exactly once
        self.assertEqual(len(observer.completed_results), 1)
        self.assertEqual(observer.completed_results[0].status, WorkflowStatus.FAILED)
        self.assertTrue(observer.completed_results[0].is_timeout)
        self.assertIn("timeout", observer.completed_results[0].error_summary.lower())

        # 3. Exactly one "team_completed" event in the events log
        completion_events = [ev for ev in observer.events if ev[0] == "team_completed"]
        self.assertEqual(len(completion_events), 1)

        # 4. Execution lock and active engine state are released after timeout
        self.assertFalse(orch._is_running)
        self.assertIsNone(orch._active_engine)

        # 5. The SAME TeamOrchestrator can execute a new task successfully after a timeout
        # Remove the delay and restore standard timeout so the subsequent call finishes quickly
        timeout_prov.delay_sec = 0.0
        config.timeout_sec = 10.0

        result2 = orch.execute("Fast follow-up task on the exact same orchestrator instance")
        self.assertEqual(result2.status, WorkflowStatus.COMPLETED)
        self.assertTrue(result2.is_success)
        self.assertFalse(result2.is_timeout)
        self.assertEqual(result2.final_deliverable, self.creator_output)
        self.assertFalse(orch._is_running)
        self.assertIsNone(orch._active_engine)

        # Observer recorded second completion event for the second run on the same orchestrator
        self.assertEqual(len(observer.completed_results), 2)
        self.assertEqual(observer.completed_results[1].status, WorkflowStatus.COMPLETED)

    def test_exactly_once_terminal_observer_event_across_all_four_outcomes(self):
        """Verify on_team_completed fires exactly once across COMPLETED, FAILED, CANCELLED, and TIMEOUT."""
        # Outcome 1: COMPLETED
        cfg_ok = create_default_team_config(
            team_id="t_all_ok",
            custom_member_configs={
                AgentRole.PLANNER: {"enabled": False},
                AgentRole.RESEARCHER: {"enabled": False},
                AgentRole.CREATOR: {"provider_id": "prov_create", "model_id": "m1"},
                AgentRole.CRITIC: {"enabled": False},
                AgentRole.FINALIZER: {"enabled": False},
            },
        )
        orch_ok = TeamOrchestrator(cfg_ok, factory=self.factory, registry=self.registry)
        obs_ok = RecordingTeamObserver()
        orch_ok.add_observer(obs_ok)
        res_ok = orch_ok.execute("Ok task")
        self.assertEqual(res_ok.status, WorkflowStatus.COMPLETED)
        self.assertEqual(len([e for e in obs_ok.events if e[0] == "team_completed"]), 1)

        # Outcome 2: FAILED (step execution error)
        fail_p = MockTeamProvider("prov_err_4", exception_to_raise=RuntimeError("Step crash"))
        self.provider_mgr.register_provider(fail_p)
        cfg_fail = create_default_team_config(
            team_id="t_all_fail",
            custom_member_configs={
                AgentRole.PLANNER: {"enabled": False},
                AgentRole.RESEARCHER: {"enabled": False},
                AgentRole.CREATOR: {"provider_id": "prov_err_4", "model_id": "m1"},
                AgentRole.CRITIC: {"enabled": False},
                AgentRole.FINALIZER: {"enabled": False},
            },
        )
        orch_fail = TeamOrchestrator(cfg_fail, factory=self.factory, registry=self.registry)
        obs_fail = RecordingTeamObserver()
        orch_fail.add_observer(obs_fail)
        res_fail = orch_fail.execute("Fail task")
        self.assertEqual(res_fail.status, WorkflowStatus.FAILED)
        self.assertEqual(len([e for e in obs_fail.events if e[0] == "team_completed"]), 1)

        # Outcome 3: CANCELLED
        slow_canc_p = MockTeamProvider("prov_slow_canc", response_text="ok", delay_sec=0.3)
        self.provider_mgr.register_provider(slow_canc_p)
        cfg_canc = create_default_team_config(
            team_id="t_all_canc",
            custom_member_configs={
                AgentRole.PLANNER: {"enabled": False},
                AgentRole.RESEARCHER: {"enabled": False},
                AgentRole.CREATOR: {"provider_id": "prov_slow_canc", "model_id": "m1"},
                AgentRole.CRITIC: {"enabled": False},
                AgentRole.FINALIZER: {"enabled": False},
            },
        )
        orch_canc = TeamOrchestrator(cfg_canc, factory=self.factory, registry=self.registry)
        obs_canc = RecordingTeamObserver()
        orch_canc.add_observer(obs_canc)

        th = threading.Thread(target=lambda: (time.sleep(0.05), orch_canc.cancel()))
        th.start()
        res_canc = orch_canc.execute("Canc task")
        th.join()
        self.assertEqual(res_canc.status, WorkflowStatus.CANCELLED)
        self.assertEqual(len([e for e in obs_canc.events if e[0] == "team_completed"]), 1)

        # Outcome 4: TIMEOUT
        slow_to_p = MockTeamProvider("prov_slow_to", response_text="ok", delay_sec=0.25)
        self.provider_mgr.register_provider(slow_to_p)
        cfg_to = create_default_team_config(
            team_id="t_all_to",
            timeout_sec=0.1,
            custom_member_configs={
                AgentRole.PLANNER: {"enabled": False},
                AgentRole.RESEARCHER: {"enabled": False},
                AgentRole.CREATOR: {"provider_id": "prov_slow_to", "model_id": "m1"},
                AgentRole.CRITIC: {"enabled": False},
                AgentRole.FINALIZER: {"enabled": False},
            },
        )
        orch_to = TeamOrchestrator(cfg_to, factory=self.factory, registry=self.registry)
        obs_to = RecordingTeamObserver()
        orch_to.add_observer(obs_to)
        res_to = orch_to.execute("Timeout task")
        self.assertEqual(res_to.status, WorkflowStatus.FAILED)
        self.assertIn("timeout", res_to.error_summary.lower())
        self.assertEqual(len([e for e in obs_to.events if e[0] == "team_completed"]), 1)


if __name__ == "__main__":
    unittest.main()
