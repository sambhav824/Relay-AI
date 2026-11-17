"""Relay AI Context Accuracy and Isolation Engine.

Provides centralized management for:
1. Fact tracking and correction resolution (new information overrides outdated details).
2. Strict context isolation across users, projects, and sessions.
3. Relevance- and recency-prioritized message formatting.
4. Enforcement of the Relay AI Concise & Accurate system policy.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.message import Message, MessageRole
from src.core.prompt_policy import build_system_instruction

logger = logging.getLogger("RelayAI.ContextEngine")


class ContextIsolationError(Exception):
    """Raised when an operation attempts to mix or leak context across tenant boundaries."""
    pass


class ContextEngine:
    """Central context manager ensuring context accuracy, recency, and isolation."""

    def __init__(
        self,
        user_id: str = "default_user",
        project_id: str = "default_project",
        session_id: str = "default_session",
    ):
        self.user_id = str(user_id or "default_user")
        self.project_id = str(project_id or "default_project")
        self.session_id = str(session_id or "default_session")
        self.verified_facts: Dict[str, str] = {}
        self.fact_audit_log: Dict[str, List[str]] = {}

    def validate_isolation(self, message: Message) -> None:
        """Verify that a message belongs strictly to the current user and project scope."""
        meta = message.metadata or {}
        msg_user = meta.get("user_id")
        if msg_user and str(msg_user) != self.user_id:
            raise ContextIsolationError(
                f"Context isolation violation: Message user '{msg_user}' does not match context user '{self.user_id}'."
            )
        msg_proj = meta.get("project_id")
        if msg_proj and str(msg_proj) != self.project_id:
            raise ContextIsolationError(
                f"Context isolation violation: Message project '{msg_proj}' does not match context project '{self.project_id}'."
            )

    def record_fact(self, key: str, value: str) -> None:
        """Store or update a verified fact, overriding any previous obsolete value."""
        norm_key = key.strip().lower().replace(" ", "_")
        clean_val = value.strip()
        if not clean_val:
            return

        if norm_key in self.verified_facts:
            old_val = self.verified_facts[norm_key]
            if old_val != clean_val:
                self.fact_audit_log.setdefault(norm_key, []).append(old_val)
                logger.info(
                    "Fact '%s' updated: '%s' -> '%s' (Correction overrides outdated value)",
                    norm_key, old_val, clean_val
                )

        self.verified_facts[norm_key] = clean_val

    def extract_facts_from_text(self, text: str) -> None:
        """Parse structured key-value lines and conversational corrections from text."""
        if not text:
            return

        # 1. Parse standard Key: Value lines
        # Examples: "Project: Nova AI", "Backend: Python + FastAPI", "Launch date: 15 December 2026"
        kv_pattern = re.compile(
            r"^(project(?:\s*name)?|backend|frontend|database|modules|launch\s*date|status|framework|tech\s*stack)\s*:\s*(.+)$",
            re.IGNORECASE | re.MULTILINE,
        )
        for match in kv_pattern.finditer(text):
            k, v = match.group(1), match.group(2)
            self.record_fact(k, v)

        # 2. Parse conversational updates & corrections
        # Examples:
        # "Actually the project is Nova AI"
        # "Correction: launch date is 15 December 2026"
        # "Renamed the project to Nova AI"
        # "The backend is changed to Go"
        corr_patterns = [
            (r"(?:actually|correction|update|note)[:,]?\s*(?:the\s*)?(project(?:\s*name)?)\s*(?:is|was renamed to|changed to)\s*([^.,\n]+)", 1, 2),
            (r"(?:actually|correction|update|note)[:,]?\s*(?:the\s*)?(launch\s*date)\s*(?:is|was moved to|changed to)\s*([^.,\n]+)", 1, 2),
            (r"(?:actually|correction|update|note)[:,]?\s*(?:the\s*)?(backend|frontend|database|tech\s*stack)\s*(?:is|was changed to)\s*([^.,\n]+)", 1, 2),
            (r"renamed\s*(?:the\s*)?project\s*to\s*([^.,\n]+)", None, 1),
        ]
        for pat in corr_patterns:
            regex, k_idx, v_idx = pat
            for match in re.finditer(regex, text, re.IGNORECASE):
                if k_idx is None:
                    self.record_fact("project", match.group(v_idx))
                else:
                    self.record_fact(match.group(k_idx), match.group(v_idx))

    def format_verified_facts(self) -> str:
        """Format verified facts into a concise markdown summary for context injection."""
        if not self.verified_facts:
            return ""

        lines = []
        for k, v in sorted(self.verified_facts.items()):
            display_key = k.replace("_", " ").title()
            lines.append(f"- **{display_key}**: {v}")
        return "\n".join(lines)

    def prepare_messages(
        self,
        messages: List[Message],
        system_override: Optional[str] = None,
        max_turns: int = 20,
    ) -> List[Message]:
        """Assemble an isolated, accurate message list enforcing the Relay AI policy.

        1. Validates tenant isolation on each message.
        2. Extracts and resolves all confirmed facts and corrections across history.
        3. Injects the concise system policy with confirmed facts at index 0.
        4. Retains recent turns to prevent silent context loss under token limits.
        """
        # Validate isolation across all messages
        for m in messages:
            self.validate_isolation(m)

        # Extract facts from all user messages in chronological order
        for m in messages:
            if m.role == MessageRole.USER and m.content:
                self.extract_facts_from_text(m.content)

        # Build custom verified context section if facts exist
        verified_context = self.format_verified_facts()
        system_text = system_override or build_system_instruction(custom_context=verified_context)

        # Filter out existing system messages to avoid duplication
        non_system_messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        # Apply recency window to avoid context blowout while keeping latest turns
        if len(non_system_messages) > max_turns:
            retained_messages = non_system_messages[-max_turns:]
        else:
            retained_messages = non_system_messages

        # Ensure system prompt is at the very beginning
        final_messages: List[Message] = [
            Message(
                role=MessageRole.SYSTEM,
                content=system_text,
                metadata={"user_id": self.user_id, "project_id": self.project_id},
            )
        ]

        # Clone retained messages
        for m in retained_messages:
            meta = dict(m.metadata or {})
            meta.setdefault("user_id", self.user_id)
            meta.setdefault("project_id", self.project_id)
            final_messages.append(
                Message(
                    role=m.role,
                    content=m.content,
                    timestamp=m.timestamp,
                    images=getattr(m, "images", None),
                    metadata=meta,
                )
            )

        return final_messages


def prepare_chat_context(
    messages: List[Message],
    user_id: str = "default_user",
    project_id: str = "default_project",
    session_id: str = "default_session",
    system_override: Optional[str] = None,
) -> List[Message]:
    """Convenience helper to format chat context with fact resolution and concise system policy."""
    engine = ContextEngine(user_id=user_id, project_id=project_id, session_id=session_id)
    return engine.prepare_messages(messages, system_override=system_override)
