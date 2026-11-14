"""Relay AI Universal Context Engine — Authentication & Session Manager.

Manages active Supabase session in memory, securely persists refresh tokens
in Windows Credential Manager (keyring), and handles automatic token refreshing.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Callable, List, Optional
import keyring

from src.context.auth.client import AuthClient, AuthSession, AuthUser, SupabaseAuthError

logger = logging.getLogger(__name__)

KEYRING_SERVICE_NAME = "RelayAI_Supabase_Auth"
KEYRING_USERNAME_REFRESH = "refresh_token"
KEYRING_USERNAME_META = "session_meta"


class AuthManager:
    """Session lifecycle manager orchestrating Supabase authentication for Relay AI."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_anon_key: Optional[str] = None,
        client: Optional[AuthClient] = None,
        keyring_service: str = KEYRING_SERVICE_NAME,
    ) -> None:
        """Initialize AuthManager.

        Args:
            supabase_url: Base Supabase project URL.
            supabase_anon_key: Public anon key.
            client: Optional preconfigured AuthClient.
            keyring_service: Service identifier for secure keyring storage.
        """
        self._keyring_service = keyring_service
        self._lock = threading.RLock()
        self._session: Optional[AuthSession] = None
        self._listeners: List[Callable[[Optional[AuthSession]], None]] = []

        url = supabase_url or os.environ.get("SUPABASE_URL", "https://placeholder.supabase.co")
        key = supabase_anon_key or os.environ.get("SUPABASE_ANON_KEY", "placeholder_anon_key")

        self.client = client or AuthClient(supabase_url=url, supabase_anon_key=key)

    @property
    def session(self) -> Optional[AuthSession]:
        """Return the current active session."""
        with self._lock:
            return self._session

    @property
    def user(self) -> Optional[AuthUser]:
        """Return the current authenticated user, if any."""
        with self._lock:
            return self._session.user if self._session else None

    @property
    def is_authenticated(self) -> bool:
        """Return True if an active, valid session is established."""
        with self._lock:
            return self._session is not None and bool(self._session.access_token)

    def add_session_listener(self, listener: Callable[[Optional[AuthSession]], None]) -> None:
        """Register a callback invoked when authentication session changes."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_session_listener(self, listener: Callable[[Optional[AuthSession]], None]) -> None:
        """Remove a previously registered session listener."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _notify_listeners(self) -> None:
        """Notify all registered listeners of a session change."""
        current = self.session
        for listener in list(self._listeners):
            try:
                listener(current)
            except Exception as e:
                logger.error("Error in session listener callback: %s", e)

    def _persist_session_to_keyring(self, session: AuthSession) -> None:
        """Securely store refresh token and session metadata in system keyring."""
        try:
            keyring.set_password(self._keyring_service, KEYRING_USERNAME_REFRESH, session.refresh_token)
            meta = {
                "user_id": session.user.id if session.user else "",
                "email": session.user.email if session.user else "",
                "expires_at": session.expires_at,
            }
            keyring.set_password(self._keyring_service, KEYRING_USERNAME_META, json.dumps(meta))
        except Exception as exc:
            logger.warning("Failed to store credentials in keyring: %s", exc)

    def _clear_keyring_session(self) -> None:
        """Purge stored credentials from system keyring."""
        try:
            keyring.delete_password(self._keyring_service, KEYRING_USERNAME_REFRESH)
        except Exception:
            pass
        try:
            keyring.delete_password(self._keyring_service, KEYRING_USERNAME_META)
        except Exception:
            pass

    def restore_session(self) -> bool:
        """Attempt to restore session from stored refresh token on application launch."""
        with self._lock:
            try:
                stored_refresh = keyring.get_password(self._keyring_service, KEYRING_USERNAME_REFRESH)
                if not stored_refresh:
                    return False

                logger.info("Found stored session refresh token; restoring Supabase session...")
                new_session = self.client.refresh_session(stored_refresh)
                self._session = new_session
                self._persist_session_to_keyring(new_session)
                self._notify_listeners()
                return True
            except Exception as exc:
                logger.info("Session restoration failed: %s. Clearing invalid session.", exc)
                self._clear_keyring_session()
                self._session = None
                self._notify_listeners()
                return False

    def get_access_token(self) -> Optional[str]:
        """Return a valid access token JWT, refreshing it automatically if expired."""
        with self._lock:
            if not self._session:
                return None

            if self._session.is_expired(buffer_seconds=60):
                logger.debug("Access token is near expiration; refreshing session...")
                try:
                    new_session = self.client.refresh_session(self._session.refresh_token)
                    self._session = new_session
                    self._persist_session_to_keyring(new_session)
                    self._notify_listeners()
                except Exception as exc:
                    logger.error("Automatic token refresh failed: %s", exc)
                    return None

            return self._session.access_token

    def login(self, email: str, password: str) -> AuthSession:
        """Sign in with email and password and persist session."""
        with self._lock:
            session = self.client.sign_in_with_password(email, password)
            self._session = session
            self._persist_session_to_keyring(session)
            self._notify_listeners()
            return session

    def signup(
        self,
        email: str,
        password: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Register a new user account."""
        with self._lock:
            result = self.client.sign_up(email, password, metadata)
            if "session" in result and isinstance(result["session"], AuthSession):
                self._session = result["session"]
                self._persist_session_to_keyring(result["session"])
                self._notify_listeners()
            return result

    def send_magic_link(self, email: str) -> dict:
        """Request a one-time OTP or magic link code."""
        return self.client.sign_in_with_otp(email)

    def verify_otp(self, email: str, token: str, otp_type: str = "email") -> AuthSession:
        """Verify an OTP code and persist session."""
        with self._lock:
            session = self.client.verify_otp(email, token, otp_type)
            self._session = session
            self._persist_session_to_keyring(session)
            self._notify_listeners()
            return session

    def logout(self) -> bool:
        """Log out the current user, invalidate remote session, and clear credentials."""
        with self._lock:
            access_token = self._session.access_token if self._session else ""
            if access_token:
                try:
                    self.client.sign_out(access_token)
                except Exception as exc:
                    logger.debug("Remote sign out error ignored during local logout: %s", exc)

            self._clear_keyring_session()
            self._session = None
            self._notify_listeners()
            return True
