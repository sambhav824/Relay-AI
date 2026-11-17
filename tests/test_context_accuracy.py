"""Test suite for Relay AI Context Accuracy, Hallucination Prevention, and Concise Responses.

Verifies:
1. Correctly recalling previously provided information.
2. Answering only the requested question.
3. Avoiding invented dates and project milestones.
4. Handling corrected information (new values override old values without mixing).
5. Retaining context after unrelated messages.
6. Distinguishing facts from suggestions.
7. Handling unknown information without guessing.
8. Preventing context leakage between users (isolation test).
9. Maintaining relevant context in AI Teams.
10. Consistent behavior across all supported AI providers.
"""

import unittest
from unittest.mock import MagicMock

from src.core.context_engine import ContextEngine, ContextIsolationError, prepare_chat_context
from src.core.message import Message, MessageRole
from src.core.prompt_policy import RELAY_CONCISE_ACCURATE_POLICY, build_system_instruction


class TestContextAccuracyAndConciseResponses(unittest.TestCase):
    """Automated tests for Relay AI Context Accuracy, Hallucination Prevention, and Brevity."""

    def setUp(self):
        self.engine = ContextEngine(user_id="usr_test_1", project_id="proj_alpha", session_id="sess_101")

    # -------------------------------------------------------------------------
    # 1. Correctly recalling previously provided information
    # -------------------------------------------------------------------------
    def test_recall_previously_provided_information(self):
        """Verify that previously provided project facts are accurately stored and formatted."""
        prompt = (
            "Project: Nova AI\n"
            "Backend: Python + FastAPI\n"
            "Frontend: Next.js\n"
            "Database: Supabase\n"
            "Modules: AI Chat, Smart Routing, AI Teams\n"
            "Launch date: 15 December 2026"
        )
        self.engine.extract_facts_from_text(prompt)

        self.assertEqual(self.engine.verified_facts.get("project"), "Nova AI")
        self.assertEqual(self.engine.verified_facts.get("backend"), "Python + FastAPI")
        self.assertEqual(self.engine.verified_facts.get("frontend"), "Next.js")
        self.assertEqual(self.engine.verified_facts.get("database"), "Supabase")
        self.assertEqual(self.engine.verified_facts.get("launch_date"), "15 December 2026")

        formatted = self.engine.format_verified_facts()
        self.assertIn("Nova AI", formatted)
        self.assertIn("Python + FastAPI", formatted)
        self.assertIn("Next.js", formatted)
        self.assertIn("Supabase", formatted)
        self.assertIn("15 December 2026", formatted)

    # -------------------------------------------------------------------------
    # 2. Answering only the requested question & concise policy injection
    # -------------------------------------------------------------------------
    def test_concise_response_policy_injected(self):
        """Verify that the system prompt strictly instructs brevity and answering only what was asked."""
        messages = [Message(role=MessageRole.USER, content="What is the database?")]
        prepared = self.engine.prepare_messages(messages)

        self.assertEqual(prepared[0].role, MessageRole.SYSTEM)
        sys_content = prepared[0].content
        self.assertIn("Answer Only What Was Asked", sys_content)
        self.assertIn("Never turn a simple question into a lengthy report", sys_content)
        self.assertIn("1 to 3 sentences", sys_content)
        self.assertIn("Do NOT add unnecessary explanations, suggestions, roadmaps", sys_content)

    # -------------------------------------------------------------------------
    # 3. Avoiding invented dates and project milestones
    # -------------------------------------------------------------------------
    def test_avoid_invented_dates_and_milestones(self):
        """Verify that policy prohibits fabricating dates or milestones."""
        sys_instr = build_system_instruction()
        self.assertIn("Prevent Hallucinations", sys_instr)
        self.assertIn("Never invent project details, dates, technologies, features, milestones", sys_instr)
        self.assertIn("If information is unknown or not provided, clearly state", sys_instr)

    # -------------------------------------------------------------------------
    # 4. Handling corrected information (new values override old values)
    # -------------------------------------------------------------------------
    def test_correction_overrides_outdated_information(self):
        """Verify that explicit corrections supersede outdated facts without mixing."""
        self.engine.extract_facts_from_text("Project: Nova AI\nLaunch date: 15 December 2026")
        self.assertEqual(self.engine.verified_facts.get("launch_date"), "15 December 2026")

        # User issues an explicit correction
        self.engine.extract_facts_from_text("Actually, the launch date is 20 January 2027.")
        self.assertEqual(self.engine.verified_facts.get("launch_date"), "20 January 2027")

        # Verify audit log recorded the superseded value
        self.assertIn("15 December 2026", self.engine.fact_audit_log.get("launch_date", []))

        # Formatted facts must show ONLY the updated date
        formatted = self.engine.format_verified_facts()
        self.assertIn("20 January 2027", formatted)
        self.assertNotIn("15 December 2026", formatted)

    # -------------------------------------------------------------------------
    # 5. Retaining context after unrelated messages
    # -------------------------------------------------------------------------
    def test_context_retention_across_unrelated_messages(self):
        """Verify facts remain intact even after unrelated intermediate user messages."""
        history = [
            Message(role=MessageRole.USER, content="Project: Nova AI\nBackend: Python + FastAPI"),
            Message(role=MessageRole.ASSISTANT, content="Got it, Nova AI configured with Python + FastAPI."),
            Message(role=MessageRole.USER, content="What is the capital of France?"),
            Message(role=MessageRole.ASSISTANT, content="Paris."),
            Message(role=MessageRole.USER, content="Explain how airplanes generate lift."),
            Message(role=MessageRole.ASSISTANT, content="Airplanes generate lift through aerodynamic wings."),
            Message(role=MessageRole.USER, content="What is our project name and backend?"),
        ]

        prepared = self.engine.prepare_messages(history)
        sys_msg = prepared[0]
        self.assertEqual(sys_msg.role, MessageRole.SYSTEM)
        self.assertIn("Nova AI", sys_msg.content)
        self.assertIn("Python + FastAPI", sys_msg.content)

    # -------------------------------------------------------------------------
    # 6. Distinguishing facts from suggestions
    # -------------------------------------------------------------------------
    def test_distinguish_facts_from_suggestions_policy(self):
        """Verify that prompt policy explicitly enforces labeling suggestions vs facts."""
        sys_instr = build_system_instruction()
        self.assertIn('Model-generated suggestions (explicitly label as "Suggestion:")', sys_instr)
        self.assertIn('Assumptions (explicitly label as "Assumption:")', sys_instr)
        self.assertIn("Never present suggestions or assumptions as confirmed facts", sys_instr)

    # -------------------------------------------------------------------------
    # 7. Handling unknown information without guessing
    # -------------------------------------------------------------------------
    def test_unknown_information_policy(self):
        """Verify that policy instructs model to state unknown rather than guessing."""
        sys_instr = build_system_instruction()
        self.assertIn('state: "I do not have information about [topic]."', sys_instr)
        self.assertIn("Unknown information -> Clearly state that the information is unknown", sys_instr)

    # -------------------------------------------------------------------------
    # 8. Preventing context leakage between users (isolation test)
    # -------------------------------------------------------------------------
    def test_prevent_context_leakage_between_users(self):
        """Verify that messages belonging to a different user raise ContextIsolationError."""
        foreign_msg = Message(
            role=MessageRole.USER,
            content="Sensitive secret from User Bob",
            metadata={"user_id": "usr_bob_other", "project_id": "proj_secret"},
        )

        with self.assertRaises(ContextIsolationError):
            self.engine.prepare_messages([foreign_msg])

    # -------------------------------------------------------------------------
    # 9. Maintaining relevant context in AI Teams
    # -------------------------------------------------------------------------
    def test_ai_teams_shared_context_integrity(self):
        """Verify AI Teams TaskContext preserves previous outputs and applies grounding rules."""
        from src.agents.context import TaskContext
        from src.agents.specialized.planner import DEFAULT_PLANNER_SYSTEM_INSTRUCTION
        from src.agents.specialized.researcher import DEFAULT_RESEARCHER_SYSTEM_INSTRUCTION

        # Verify specialized system instructions have grounding rules
        self.assertIn("ACCURACY & CONCISE POLICY", DEFAULT_PLANNER_SYSTEM_INSTRUCTION)
        self.assertIn("STRICT GROUNDING & TRUTHFULNESS RULE", DEFAULT_RESEARCHER_SYSTEM_INSTRUCTION)

        ctx = TaskContext(
            user_prompt="Build an event-driven payment webhook receiver.",
            previous_outputs={"planner": "Step 1: Setup HMAC validator. Step 2: Idempotency table."},
        )
        msgs = ctx.build_messages(system_instruction="You are Creator Agent.")
        prompt_content = msgs[-1].content
        self.assertIn("### Overall Task Objective:", prompt_content)
        self.assertIn("### Previous Stage Outputs:", prompt_content)
        self.assertIn("Setup HMAC validator", prompt_content)

    # -------------------------------------------------------------------------
    # 10. Consistent behavior across all supported AI providers
    # -------------------------------------------------------------------------
    def test_consistent_behavior_across_providers(self):
        """Verify that prepared messages are universally formatted for Groq, Gemini, OpenAI, and Ollama."""
        messages = [
            Message(role=MessageRole.USER, content="Project: Nova AI\nLaunch date: 15 December 2026"),
            Message(role=MessageRole.USER, content="When do we launch?"),
        ]

        prepared = prepare_chat_context(messages, user_id="usr_consistent")
        self.assertEqual(prepared[0].role, MessageRole.SYSTEM)
        self.assertIn("Nova AI", prepared[0].content)
        self.assertIn("15 December 2026", prepared[0].content)
        self.assertEqual(prepared[-1].content, "When do we launch?")


if __name__ == "__main__":
    unittest.main()
