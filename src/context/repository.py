"""Relay AI Universal Context Engine — Local Context Repository.

Provides unified, transactional data access for projects, conversations,
messages, memories, and execution telemetry, automatically enqueuing
cloud-synchronization outbox operations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from typing import Any, Dict, List, Optional

from src.context.sync.queue import SyncQueueManager

logger = logging.getLogger(__name__)


class ContextRepository:
    """Manages local SQLite context entities with automatic cloud-sync outbox staging."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        sync_queue: Optional[SyncQueueManager] = None,
        default_user_id: Optional[str] = None,
    ) -> None:
        self.conn = connection
        self.sync_queue = sync_queue
        self.default_user_id = default_user_id

    def set_user_id(self, user_id: Optional[str]) -> None:
        """Update active authenticated user ID for sync staging."""
        self.default_user_id = user_id

    def _resolve_user_id(self, user_id: Optional[str] = None) -> Optional[str]:
        return user_id or self.default_user_id

    def _stage_sync(
        self,
        table_name: str,
        record_id: str,
        operation: str,
        payload: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        uid = self._resolve_user_id(user_id)
        if not self.sync_queue or not uid:
            return None
        try:
            return self.sync_queue.stage(
                user_id=uid,
                table_name=table_name,
                record_id=record_id,
                operation=operation,
                payload=payload,
            )
        except Exception as exc:
            logger.warning("Failed to stage sync outbox item for %s/%s: %s", table_name, record_id, exc)
            return None

    # ==================== Projects ====================

    def create_project(
        self,
        name: str,
        description: str = "",
        root_path: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        pid = project_id or str(uuid.uuid4())
        meta_json = json.dumps(metadata or {}, separators=(",", ":"), sort_keys=True)
        self.conn.execute(
            """INSERT INTO projects (id, name, description, root_path, metadata_json)
               VALUES (?, ?, ?, ?, ?)""",
            (pid, name, description, root_path, meta_json),
        )
        self.conn.commit()

        self._stage_sync(
            table_name="projects",
            record_id=pid,
            operation="INSERT",
            payload={"name": name, "description": description},
            user_id=user_id,
        )
        return pid

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT id, name, description, root_path, created_at, updated_at, metadata_json FROM projects WHERE id=?",
            (project_id,),
        ).fetchone()
        if not row:
            return None
        return dict(row)

    def list_projects(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, name, description, root_path, created_at, updated_at FROM projects ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_project(self, project_id: str, user_id: Optional[str] = None) -> bool:
        cursor = self.conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
        self.conn.commit()
        if cursor.rowcount > 0:
            self._stage_sync(
                table_name="projects",
                record_id=project_id,
                operation="DELETE",
                user_id=user_id,
            )
            return True
        return False

    # ==================== Conversations ====================

    def create_conversation(
        self,
        title: str = "New Conversation",
        project_id: str = "global",
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        cid = conversation_id or str(uuid.uuid4())
        meta_json = json.dumps(metadata or {}, separators=(",", ":"), sort_keys=True)
        self.conn.execute(
            """INSERT INTO conversations (id, project_id, title, model_id, system_prompt, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cid, project_id, title, model_id, system_prompt, meta_json),
        )
        self.conn.commit()

        sync_payload: Dict[str, Any] = {
            "title": title,
            "type": "chat",
            "model_id": model_id,
            "metadata": metadata or {},
        }
        if project_id and project_id != "global":
            sync_payload["project_id"] = project_id

        self._stage_sync(
            table_name="conversations",
            record_id=cid,
            operation="INSERT",
            payload=sync_payload,
            user_id=user_id,
        )
        return cid

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            """SELECT id, project_id, title, created_at, updated_at, model_id, system_prompt,
                      is_archived, metadata_json FROM conversations WHERE id=?""",
            (conversation_id,),
        ).fetchone()
        if not row:
            return None
        return dict(row)

    def list_conversations(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if project_id:
            rows = self.conn.execute(
                """SELECT id, project_id, title, created_at, updated_at, model_id, is_archived
                   FROM conversations WHERE project_id=? ORDER BY updated_at DESC""",
                (project_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT id, project_id, title, created_at, updated_at, model_id, is_archived
                   FROM conversations ORDER BY updated_at DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def update_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        model_id: Optional[str] = None,
        is_archived: Optional[bool] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        updates = []
        params = []
        payload: Dict[str, Any] = {}

        if title is not None:
            updates.append("title = ?")
            params.append(title)
            payload["title"] = title
        if model_id is not None:
            updates.append("model_id = ?")
            params.append(model_id)
            payload["model_id"] = model_id
        if is_archived is not None:
            updates.append("is_archived = ?")
            params.append(1 if is_archived else 0)

        if not updates:
            return False

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(conversation_id)

        sql = f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?"
        cursor = self.conn.execute(sql, params)
        self.conn.commit()

        if cursor.rowcount > 0:
            if payload:
                self._stage_sync(
                    table_name="conversations",
                    record_id=conversation_id,
                    operation="UPDATE",
                    payload=payload,
                    user_id=user_id,
                )
            return True
        return False

    def delete_conversation(self, conversation_id: str, user_id: Optional[str] = None) -> bool:
        cursor = self.conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        self.conn.commit()
        if cursor.rowcount > 0:
            self._stage_sync(
                table_name="conversations",
                record_id=conversation_id,
                operation="DELETE",
                user_id=user_id,
            )
            return True
        return False

    # ==================== Messages ====================

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        token_count: int = 0,
        model_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        mid = message_id or str(uuid.uuid4())
        meta_json = json.dumps(metadata or {}, separators=(",", ":"), sort_keys=True)
        self.conn.execute(
            """INSERT INTO messages (id, conversation_id, role, content, token_count, model_id, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (mid, conversation_id, role, content, token_count, model_id, meta_json),
        )
        self.conn.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,),
        )
        self.conn.commit()

        self._stage_sync(
            table_name="messages",
            record_id=mid,
            operation="INSERT",
            payload={
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "token_count": token_count,
                "metadata": metadata or {},
            },
            user_id=user_id,
        )
        return mid

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT id, conversation_id, role, content, created_at, token_count, model_id, metadata_json
               FROM messages WHERE conversation_id=? ORDER BY created_at ASC""",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_message(self, message_id: str, user_id: Optional[str] = None) -> bool:
        cursor = self.conn.execute("DELETE FROM messages WHERE id=?", (message_id,))
        self.conn.commit()
        if cursor.rowcount > 0:
            self._stage_sync(
                table_name="messages",
                record_id=message_id,
                operation="DELETE",
                user_id=user_id,
            )
            return True
        return False

    # ==================== Memories ====================

    def add_memory(
        self,
        category: str,
        content: str,
        project_id: str = "global",
        provenance: str = "USER_EXPLICIT",
        confidence: float = 1.0,
        source_attribution: Optional[str] = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        mid = memory_id or str(uuid.uuid4())
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.conn.execute(
            """INSERT INTO memories (id, project_id, category, content, content_hash,
                                     provenance, confidence, source_attribution)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (mid, project_id, category, content, content_hash, provenance, confidence, source_attribution),
        )
        self.conn.commit()

        sync_payload: Dict[str, Any] = {
            "category": category,
            "content": content,
            "content_hash": content_hash,
            "provenance": provenance,
            "confidence": confidence,
        }
        if project_id and project_id != "global":
            sync_payload["project_id"] = project_id

        self._stage_sync(
            table_name="memories",
            record_id=mid,
            operation="INSERT",
            payload=sync_payload,
            user_id=user_id,
        )
        return mid

    def get_memories(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if project_id:
            rows = self.conn.execute(
                """SELECT id, project_id, category, content, content_hash, provenance, confidence,
                          is_approved, source_attribution, created_at, updated_at
                   FROM memories WHERE project_id=? ORDER BY created_at DESC""",
                (project_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT id, project_id, category, content, content_hash, provenance, confidence,
                          is_approved, source_attribution, created_at, updated_at
                   FROM memories ORDER BY created_at DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_memory(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        cursor = self.conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self.conn.commit()
        if cursor.rowcount > 0:
            self._stage_sync(
                table_name="memories",
                record_id=memory_id,
                operation="DELETE",
                user_id=user_id,
            )
            return True
        return False

    # ==================== Smart Routing & Telemetry ====================

    def record_smart_routing(
        self,
        user_prompt: str,
        provider: str,
        model: str,
        conversation_id: Optional[str] = None,
        classified_intent: Optional[str] = None,
        task_complexity: Optional[str] = None,
        routing_rationale: Optional[str] = None,
        confidence_score: Optional[float] = None,
        log_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        lid = log_id or str(uuid.uuid4())
        self._stage_sync(
            table_name="smart_routing_logs",
            record_id=lid,
            operation="INSERT",
            payload={
                "user_prompt": user_prompt,
                "selected_provider": provider,
                "selected_model": model,
                "conversation_id": conversation_id,
                "classified_intent": classified_intent,
                "task_complexity": task_complexity,
                "routing_rationale": routing_rationale,
                "confidence_score": confidence_score,
            },
            user_id=user_id,
        )
        return lid

    # ==================== Model Comparisons ====================

    def record_model_comparison(
        self,
        prompt: str,
        slots: List[Dict[str, Any]],
        conversation_id: Optional[str] = None,
        comparison_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        cid = comparison_id or str(uuid.uuid4())
        self._stage_sync(
            table_name="model_comparisons",
            record_id=cid,
            operation="INSERT",
            payload={
                "prompt": prompt,
                "conversation_id": conversation_id,
            },
            user_id=user_id,
        )
        for idx, slot in enumerate(slots):
            sid = str(uuid.uuid4())
            self._stage_sync(
                table_name="comparison_slots",
                record_id=sid,
                operation="INSERT",
                payload={
                    "comparison_id": cid,
                    "slot_index": idx,
                    "provider_id": slot.get("provider_id", ""),
                    "model_id": slot.get("model_id", ""),
                    "response_text": slot.get("response_text", ""),
                    "duration_ms": slot.get("duration_ms"),
                    "token_count": slot.get("token_count"),
                },
                user_id=user_id,
            )
        return cid

    # ==================== AI Team Runs ====================

    def record_team_run(
        self,
        team_name: str,
        task_prompt: str,
        status: str = "COMPLETED",
        final_deliverable: Optional[str] = None,
        step_records: Optional[List[Dict[str, Any]]] = None,
        project_id: Optional[str] = None,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        rid = run_id or str(uuid.uuid4())
        self._stage_sync(
            table_name="team_runs",
            record_id=rid,
            operation="INSERT",
            payload={
                "team_name": team_name,
                "task_prompt": task_prompt,
                "status": status,
                "final_deliverable": final_deliverable,
                "project_id": project_id if project_id != "global" else None,
            },
            user_id=user_id,
        )
        for idx, step in enumerate(step_records or []):
            s_id = str(uuid.uuid4())
            self._stage_sync(
                table_name="team_step_records",
                record_id=s_id,
                operation="INSERT",
                payload={
                    "team_run_id": rid,
                    "step_id": step.get("step_id", f"step_{idx}"),
                    "agent_role": step.get("agent_role", "custom"),
                    "output_text": step.get("output_text", ""),
                    "revision_index": step.get("revision_index", 0),
                    "critic_feedback": step.get("critic_feedback"),
                },
                user_id=user_id,
            )
        return rid
