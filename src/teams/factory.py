import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from src.agents import (
    AgentBase,
    AgentConfig,
    AgentRegistry,
    AgentRole,
    CreatorAgent,
    CriticAgent,
    FinalizerAgent,
    PlannerAgent,
    ResearcherAgent,
)
from src.providers.manager import ProviderManager, get_default_provider_manager
from src.teams.exceptions import TeamConfigurationError
from src.teams.models import TeamConfig, TeamMemberConfig
from src.workflow.models import (
    StepFailurePolicy,
    WorkflowConfig,
    WorkflowStepConfig,
)

logger = logging.getLogger("RelayAI.TeamFactory")


def create_default_team_config(
    team_id: str = "default_team",
    name: str = "Default 5-Agent Collaboration Team",
    default_provider_id: str = "ollama",
    default_model_id: str = "llama3.2:3b",
    custom_member_configs: Optional[Dict[AgentRole, Dict[str, Any]]] = None,
    timeout_sec: float = 300.0,
    max_revision_loops: int = 2,
) -> TeamConfig:
    """Generate a standard TeamConfig covering the full Planner -> Researcher -> Creator -> Critic -> Finalizer pipeline."""
    member_specs: Dict[AgentRole, Dict[str, Any]] = {
        AgentRole.PLANNER: {
            "temperature": 0.3,
            "timeout_sec": 60.0,
        },
        AgentRole.RESEARCHER: {
            "temperature": 0.2,
            "timeout_sec": 60.0,
        },
        AgentRole.CREATOR: {
            "temperature": 0.5,
            "timeout_sec": 90.0,
        },
        AgentRole.CRITIC: {
            "temperature": 0.1,
            "timeout_sec": 60.0,
        },
        AgentRole.FINALIZER: {
            "temperature": 0.3,
            "timeout_sec": 60.0,
        },
    }

    members: Dict[AgentRole, TeamMemberConfig] = {}
    for role, default_params in member_specs.items():
        prov = default_provider_id
        model = default_model_id
        enabled = True
        temp = default_params["temperature"]
        timeout = default_params["timeout_sec"]
        sys_instr = None
        fail_pol = StepFailurePolicy.ABORT
        fallback_id = None
        extra = {}

        if custom_member_configs and role in custom_member_configs:
            custom = custom_member_configs[role]
            prov = custom.get("provider_id", prov)
            model = custom.get("model_id", model)
            enabled = custom.get("enabled", enabled)
            temp = custom.get("temperature", temp)
            timeout = custom.get("timeout_sec", timeout)
            sys_instr = custom.get("system_instruction", sys_instr)
            fail_pol = custom.get("failure_policy", fail_pol)
            fallback_id = custom.get("fallback_agent_id", fallback_id)
            extra = custom.get("extra_params", extra)

        members[role] = TeamMemberConfig(
            role=role,
            provider_id=prov,
            model_id=model,
            agent_id=f"{team_id}_{role.value}",
            enabled=enabled,
            temperature=temp,
            timeout_sec=timeout,
            system_instruction=sys_instr,
            failure_policy=fail_pol,
            fallback_agent_id=fallback_id,
            extra_params=extra,
        )

    return TeamConfig(
        team_id=team_id,
        name=name,
        members=members,
        timeout_sec=timeout_sec,
        max_revision_loops=max_revision_loops,
    )


class TeamFactory:
    """Instantiates, validates, and registers specialized AI agents for a team configuration."""

    def __init__(
        self,
        provider_manager: Optional[ProviderManager] = None,
        registry: Optional[AgentRegistry] = None,
    ):
        self.provider_manager = provider_manager or get_default_provider_manager()
        self.registry = registry or AgentRegistry()

    def validate_team_readiness(self, config: TeamConfig) -> None:
        """Verify provider availability, model configuration, and credential readiness for enabled members.
        
        Strict constraints:
        1. Disabled members are skipped and do NOT require registered providers or credentials.
        2. Local providers (e.g. Ollama) require no API credentials.
        3. Cloud providers require credentials present in CredentialStore or manual configuration.
        4. Zero paid inference requests are made during pre-flight validation.
        5. Zero secret leakage: Error diagnostics never print API keys or secret values.
        """
        config.validate()

        for role, member in config.members.items():
            if not member.enabled:
                continue

            provider = self.provider_manager.get_provider(member.provider_id)
            if not provider:
                raise TeamConfigurationError(
                    f"Team '{config.team_id}' member '{role.value}' specifies provider '{member.provider_id}', which is not registered.",
                    team_id=config.team_id,
                )

            if not member.model_id or not member.model_id.strip():
                raise TeamConfigurationError(
                    f"Team '{config.team_id}' member '{role.value}' has an empty model_id.",
                    team_id=config.team_id,
                )

            # Credential readiness check for non-local providers
            is_local = getattr(provider, "is_local", False)
            if not is_local:
                has_cred = False
                if hasattr(provider, "has_credentials") and callable(provider.has_credentials):
                    try:
                        has_cred = provider.has_credentials(self.provider_manager.credential_store)
                    except TypeError:
                        has_cred = provider.has_credentials()
                if not has_cred and getattr(provider, "_manual_api_key", None):
                    has_cred = bool(str(provider._manual_api_key).strip())
                if not has_cred and hasattr(provider, "credential_store") and provider.credential_store and hasattr(provider.credential_store, "has_api_key"):
                    has_cred = bool(provider.credential_store.has_api_key(member.provider_id))
                if not has_cred and self.provider_manager.credential_store and hasattr(self.provider_manager.credential_store, "has_api_key"):
                    has_cred = bool(self.provider_manager.credential_store.has_api_key(member.provider_id))

                if not has_cred:
                    raise TeamConfigurationError(
                        f"Team '{config.team_id}' member '{role.value}' specifies provider '{member.provider_id}', which is missing required API credentials in CredentialStore.",
                        team_id=config.team_id,
                    )

    def instantiate_and_register_agents(
        self,
        config: TeamConfig,
        overwrite: bool = True,
    ) -> List[AgentBase]:
        """Instantiate and register ONLY enabled agents into the registry with team-namespaced IDs."""
        self.validate_team_readiness(config)
        registered_agents: List[AgentBase] = []

        agent_classes = {
            AgentRole.PLANNER: PlannerAgent,
            AgentRole.RESEARCHER: ResearcherAgent,
            AgentRole.CREATOR: CreatorAgent,
            AgentRole.CRITIC: CriticAgent,
            AgentRole.FINALIZER: FinalizerAgent,
        }

        for role, member in config.members.items():
            if not member.enabled:
                continue

            cls = agent_classes.get(role)
            if not cls:
                continue

            agent_id = member.agent_id or f"{config.team_id}_{role.value}"
            member.agent_id = agent_id

            agent_cfg = AgentConfig(
                provider_id=member.provider_id,
                model_id=member.model_id,
                temperature=member.temperature if member.temperature is not None else 0.7,
                timeout_sec=member.timeout_sec,
                max_retries=member.max_retries,
                system_instruction=member.system_instruction or "",
                extra_params=dict(member.extra_params),
            )

            agent = cls(
                agent_id=agent_id,
                name=f"{config.name} ({role.value.title()})",
                config=agent_cfg,
                provider_manager=self.provider_manager,
            )

            self.registry.register(agent, overwrite=overwrite)
            registered_agents.append(agent)
            logger.debug("Registered team agent '%s' for team '%s'.", agent_id, config.team_id)

        return registered_agents

    def compile_workflow_config(self, config: TeamConfig) -> WorkflowConfig:
        """Compile a dynamic, validated WorkflowConfig for the active enabled members.
        
        Performs dynamic dependency rewriting and selective context input scoping.
        """
        config.validate()
        enabled_roles = {r for r, m in config.members.items() if m.enabled}

        steps: List[WorkflowStepConfig] = []
        active_step_ids: Set[str] = set()

        # Step 1: Planner (Optional)
        planner_active = AgentRole.PLANNER in enabled_roles
        if planner_active:
            m = config.members[AgentRole.PLANNER]
            step_id = f"{config.team_id}_step_plan"
            steps.append(
                WorkflowStepConfig(
                    step_id=step_id,
                    agent_id=m.agent_id or f"{config.team_id}_planner",
                    step_name=f"{config.name} - Plan",
                    failure_policy=m.failure_policy,
                    fallback_agent_id=m.fallback_agent_id,
                    max_retries=m.max_retries,
                    dependencies=[],
                    required_inputs=None,
                )
            )
            active_step_ids.add(step_id)

        # Step 2: Researcher (Optional)
        researcher_active = AgentRole.RESEARCHER in enabled_roles
        if researcher_active:
            m = config.members[AgentRole.RESEARCHER]
            step_id = f"{config.team_id}_step_research"
            deps = [f"{config.team_id}_step_plan"] if planner_active else []
            req_in = [f"{config.team_id}_step_plan"] if planner_active else None
            steps.append(
                WorkflowStepConfig(
                    step_id=step_id,
                    agent_id=m.agent_id or f"{config.team_id}_researcher",
                    step_name=f"{config.name} - Research",
                    failure_policy=m.failure_policy,
                    fallback_agent_id=m.fallback_agent_id,
                    max_retries=m.max_retries,
                    dependencies=deps,
                    required_inputs=req_in,
                )
            )
            active_step_ids.add(step_id)

        # Step 3: Creator (Mandatory)
        creator_active = AgentRole.CREATOR in enabled_roles
        if not creator_active:
            raise TeamConfigurationError("Cannot compile workflow: Creator agent must be enabled.", team_id=config.team_id)

        m_create = config.members[AgentRole.CREATOR]
        create_step_id = f"{config.team_id}_step_create"
        create_deps = [s for s in [f"{config.team_id}_step_plan", f"{config.team_id}_step_research"] if s in active_step_ids]
        create_req_in = create_deps if create_deps else None
        steps.append(
            WorkflowStepConfig(
                step_id=create_step_id,
                agent_id=m_create.agent_id or f"{config.team_id}_creator",
                step_name=f"{config.name} - Create",
                failure_policy=m_create.failure_policy,
                fallback_agent_id=m_create.fallback_agent_id,
                max_retries=m_create.max_retries,
                dependencies=create_deps,
                required_inputs=create_req_in,
            )
        )
        active_step_ids.add(create_step_id)

        # Step 4: Critic (Optional)
        critic_active = AgentRole.CRITIC in enabled_roles
        critic_step_id = None
        if critic_active:
            m_crit = config.members[AgentRole.CRITIC]
            critic_step_id = f"{config.team_id}_step_critic"
            crit_req = [s for s in [create_step_id, f"{config.team_id}_step_plan"] if s in active_step_ids]
            steps.append(
                WorkflowStepConfig(
                    step_id=critic_step_id,
                    agent_id=m_crit.agent_id or f"{config.team_id}_critic",
                    step_name=f"{config.name} - Critic Review",
                    failure_policy=m_crit.failure_policy,
                    fallback_agent_id=m_crit.fallback_agent_id,
                    max_retries=m_crit.max_retries,
                    dependencies=[create_step_id],
                    is_critic_step=True,
                    allowed_revision_targets=[create_step_id],
                    required_inputs=crit_req,
                )
            )
            active_step_ids.add(critic_step_id)

        # Step 5: Finalizer (Optional)
        finalizer_active = AgentRole.FINALIZER in enabled_roles
        if finalizer_active:
            m_fin = config.members[AgentRole.FINALIZER]
            fin_step_id = f"{config.team_id}_step_finalize"
            fin_deps = [critic_step_id] if critic_active else [create_step_id]
            fin_req = [s for s in [create_step_id, f"{config.team_id}_step_plan"] if s in active_step_ids]
            steps.append(
                WorkflowStepConfig(
                    step_id=fin_step_id,
                    agent_id=m_fin.agent_id or f"{config.team_id}_finalizer",
                    step_name=f"{config.name} - Finalize",
                    failure_policy=m_fin.failure_policy,
                    fallback_agent_id=m_fin.fallback_agent_id,
                    max_retries=m_fin.max_retries,
                    dependencies=fin_deps,
                    required_inputs=fin_req,
                )
            )
            active_step_ids.add(fin_step_id)

        return WorkflowConfig(
            workflow_id=f"wf_{config.team_id}",
            name=f"Workflow for {config.name}",
            description=config.description,
            steps=steps,
            timeout_sec=config.timeout_sec,
            max_revision_loops=config.max_revision_loops,
            max_steps=config.max_steps,
            token_budget=config.token_budget,
            cost_budget=config.cost_budget,
        )

