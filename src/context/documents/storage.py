"""Private Supabase Storage adapter for Relay AI documents."""

from __future__ import annotations

import mimetypes
import re
from pathlib import PurePosixPath
from typing import Optional
from urllib.parse import quote

import httpx


class StorageError(RuntimeError):
    """Raised when a private document storage operation fails."""


class SupabaseDocumentStorage:
    """Upload, download, and delete objects from the private relay-documents bucket.

    All requests use the current user's JWT and the public anon key. No service-role
    credentials are accepted or stored by this class.
    """

    BUCKET = "relay-documents"

    def __init__(
        self,
        supabase_url: str,
        anon_key: str,
        access_token: str,
        *,
        http_client: Optional[httpx.Client] = None,
        timeout: float = 60.0,
    ) -> None:
        if not supabase_url or not anon_key or not access_token:
            raise ValueError("Supabase URL, anon key, and user access token are required")
        self.base_url = supabase_url.rstrip("/")
        self.anon_key = anon_key
        self.access_token = access_token
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None

    @staticmethod
    def _safe_segment(value: str, label: str) -> str:
        if not value or value in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError(f"Invalid {label}")
        return value

    @classmethod
    def build_object_path(cls, user_id: str, document_id: str, file_name: str) -> str:
        owner = cls._safe_segment(user_id, "user id")
        doc_id = cls._safe_segment(document_id, "document id")
        # Keep only the basename and reject traversal/control characters.
        name = PurePosixPath(file_name.replace("\\", "/")).name
        if not name or name in {".", ".."} or any(ord(c) < 32 for c in name):
            raise ValueError("Invalid file name")
        return f"{owner}/{doc_id}/{name}"

    def _headers(self, *, content_type: Optional[str] = None) -> dict:
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "apikey": self.anon_key,
        }
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def upload(self, object_path: str, data: bytes, content_type: Optional[str] = None) -> str:
        if not data:
            raise ValueError("Cannot upload an empty document")
        safe_path = "/".join(quote(p, safe="._-") for p in object_path.split("/"))
        mime = content_type or mimetypes.guess_type(object_path)[0] or "application/octet-stream"
        response = self._client.post(
            f"{self.base_url}/storage/v1/object/{self.BUCKET}/{safe_path}",
            content=data,
            headers={**self._headers(content_type=mime), "x-upsert": "false"},
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise StorageError(f"Document upload failed (HTTP {exc.response.status_code}): {exc.response.text[:500]}") from exc
        return object_path

    def download(self, object_path: str) -> bytes:
        safe_path = "/".join(quote(p, safe="._-") for p in object_path.split("/"))
        response = self._client.get(
            f"{self.base_url}/storage/v1/object/{self.BUCKET}/{safe_path}",
            headers=self._headers(),
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise StorageError(f"Document download failed (HTTP {exc.response.status_code}): {exc.response.text[:500]}") from exc
        return response.content

    def delete(self, object_path: str) -> None:
        response = self._client.delete(
            f"{self.base_url}/storage/v1/object/{self.BUCKET}",
            json={"prefixes": [object_path]},
            headers=self._headers(),
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise StorageError(f"Document delete failed (HTTP {exc.response.status_code}): {exc.response.text[:500]}") from exc

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
