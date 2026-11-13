import copy
from dataclasses import dataclass, field
import threading
from typing import Any, Callable, Dict, List, Optional
import uuid

from src.core.message import Message, MessageRole


@dataclass
class TaskContext:
    """Isolated execution context passed to an individual AI agent.

    Protects against cross-agent state leakage by creating deep copies of mutable
    structures upon initialization.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_prompt: str = ""
    conversation_history: List[Message] = field(default_factory=list)
    instructions: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    previous_outputs: Dict[str, str] = field(default_factory=dict)
    shared_metadata: Dict[str, Any] = field(default_factory=dict)
    cancel_event: Optional[threading.Event] = None
    is_cancelled_callback: Optional[Callable[[], bool]] = None

    def __post_init__(self) -> None:
        # Defensive isolation: ensure mutations by an executing agent cannot corrupt
        # the caller's lists or shared caller structures.
        isolated_history: List[Message] = []
        for m in self.conversation_history:
            if isinstance(m, Message):
                isolated_history.append(
                    Message(
                        role=m.role,
                        content=m.content,
                        timestamp=m.timestamp,
                        metadata=dict(m.metadata) if m.metadata else {},
                    )
                )
            elif isinstance(m, dict):
                isolated_history.append(Message.from_dict(m))
            else:
                isolated_history.append(Message(role=MessageRole.USER, content=str(m)))
        self.conversation_history = isolated_history

        try:
            self.input_data = copy.deepcopy(self.input_data)
        except Exception:
            self.input_data = dict(self.input_data)

        self.previous_outputs = dict(self.previous_outputs)

        try:
            self.shared_metadata = copy.deepcopy(self.shared_metadata)
        except Exception:
            self.shared_metadata = dict(self.shared_metadata)

    def is_cancelled(self) -> bool:
        """Check whether cancellation has been requested via Event or callback."""
        if self.cancel_event is not None and self.cancel_event.is_set():
            return True
        if self.is_cancelled_callback is not None:
            try:
                return bool(self.is_cancelled_callback())
            except Exception:
                return False
        return False

    def get_input(self, key: str, default: Any = None) -> Any:
        """Safely retrieve a value from input_data."""
        return self.input_data.get(key, default)

    def get_previous_output(self, key: str) -> Optional[str]:
        """Safely retrieve the output from a previous agent or role."""
        return self.previous_outputs.get(key)

    def build_messages(self, system_instruction: str = "") -> List[Message]:
        """Assemble standardized Message payload for provider invocation.

        Prepares system instruction, prior conversation history, and a structured
        user task prompt without duplicating user messages.
        """
        messages: List[Message] = []

        if system_instruction:
            messages.append(Message(role=MessageRole.SYSTEM, content=system_instruction.strip()))

        # Append cloned conversation history
        for m in self.conversation_history:
            messages.append(Message(role=m.role, content=m.content))

        # Build composite task payload for this specific agent step
        sections: List[str] = []

        if self.user_prompt:
            sections.append(f"### Overall Task Objective:\n{self.user_prompt.strip()}")

        if self.previous_outputs:
            prev_sections = []
            for role_or_id, text in self.previous_outputs.items():
                prev_sections.append(f"**From {role_or_id.title()}:**\n{text.strip()}")
            sections.append(f"### Previous Stage Outputs:\n" + "\n\n".join(prev_sections))

        if self.instructions:
            sections.append(f"### Step Instructions:\n{self.instructions.strip()}")

        combined_prompt = "\n\n".join(sections).strip()
        if combined_prompt:
            messages.append(Message(role=MessageRole.USER, content=combined_prompt))

        return messages

