"""Relay AI Universal Context Engine — Supabase GoTrue Auth Client.

Provides native HTTP-based authentication interacting directly with Supabase Auth
using standard synchronous httpx calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, Optional
import httpx

logger = logging.getLogger(__name__)


class SupabaseAuthError(Exception):
    """Exception raised when a Supabase Auth operation fails."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code

    def __str__(self) -> str:
        code_str = f" [{self.error_code}]" if self.error_code else ""
        status_str = f" (Status {self.status_code})" if self.status_code else ""
        return f"{self.message}{code_str}{status_str}"


@dataclass
class AuthUser:
    """Represents an authenticated Supabase user."""

    id: str
    email: str
    created_at: str = ""
    user_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuthUser:
        """Construct an AuthUser from a Supabase JSON dictionary."""
        return cls(
            id=str(data.get("id", "")),
            email=str(data.get("email", "")),
            created_at=str(data.get("created_at", "")),
            user_metadata=data.get("user_metadata", {}) or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "email": self.email,
            "created_at": self.created_at,
            "user_metadata": self.user_metadata,
        }


@dataclass
class AuthSession:
    """Represents an active Supabase authentication session."""

    access_token: str
    refresh_token: str
    expires_in: int
    expires_at: float
    token_type: str = "bearer"
    user: Optional[AuthUser] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuthSession:
        """Construct an AuthSession from a Supabase token response."""
        now = time.time()
        expires_in = int(data.get("expires_in", 3600))
        expires_at = float(data.get("expires_at", now + expires_in))
        user_data = data.get("user")
        user = AuthUser.from_dict(user_data) if user_data else None

        return cls(
            access_token=str(data.get("access_token", "")),
            refresh_token=str(data.get("refresh_token", "")),
            expires_in=expires_in,
            expires_at=expires_at,
            token_type=str(data.get("token_type", "bearer")),
            user=user,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_in": self.expires_in,
            "expires_at": self.expires_at,
            "token_type": self.token_type,
            "user": self.user.to_dict() if self.user else None,
        }

    def is_expired(self, buffer_seconds: int = 60) -> bool:
        """Return True if the access token has expired or will expire within buffer_seconds."""
        return time.time() >= (self.expires_at - buffer_seconds)


class AuthClient:
    """Synchronous HTTP client for Supabase GoTrue Auth API."""

    def __init__(
        self,
        supabase_url: str,
        supabase_anon_key: str,
        timeout: float = 10.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        """Initialize Supabase Auth client.

        Args:
            supabase_url: Base URL of Supabase project (e.g. https://xyz.supabase.co).
            supabase_anon_key: Public anon key for API access.
            timeout: HTTP request timeout in seconds.
            http_client: Optional injected httpx.Client for testing.
        """
        self.supabase_url = supabase_url.rstrip("/")
        self.anon_key = supabase_anon_key
        self.timeout = timeout
        self._auth_base = f"{self.supabase_url}/auth/v1"
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None

    def close(self) -> None:
        """Close the underlying HTTP client if owned."""
        if self._owns_client and hasattr(self._client, "close"):
            self._client.close()

    def _get_headers(self, access_token: Optional[str] = None) -> Dict[str, str]:
        """Prepare standard Supabase HTTP headers."""
        headers = {
            "apikey": self.anon_key,
            "Content-Type": "application/json",
            "User-Agent": "RelayAI-Desktop/1.0",
        }
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        else:
            headers["Authorization"] = f"Bearer {self.anon_key}"
        return headers

    def _handle_response_errors(self, response: httpx.Response) -> Dict[str, Any]:
        """Inspect response and raise SupabaseAuthError if status indicates error."""
        try:
            data = response.json()
        except Exception:
            data = {"message": response.text or f"HTTP Error {response.status_code}"}

        if response.is_error:
            msg = (
                data.get("msg")
                or data.get("error_description")
                or data.get("message")
                or f"Authentication failed with status {response.status_code}"
            )
            err_code = data.get("error_code") or data.get("code")
            logger.warning("Supabase Auth API error (%d): %s (code: %s)", response.status_code, msg, err_code)
            raise SupabaseAuthError(message=str(msg), status_code=response.status_code, error_code=err_code)

        return data

    def sign_in_with_password(self, email: str, password: str) -> AuthSession:
        """Sign in using email and password.

        Args:
            email: User's email address.
            password: User's password.

        Returns:
            AuthSession containing access and refresh tokens.
        """
        url = f"{self._auth_base}/token?grant_type=password"
        payload = {"email": email.strip(), "password": password}
        try:
            resp = self._client.post(url, json=payload, headers=self._get_headers())
            data = self._handle_response_errors(resp)
            return AuthSession.from_dict(data)
        except httpx.RequestError as exc:
            logger.error("Network error during password sign in: %s", exc)
            raise SupabaseAuthError(f"Network error communicating with authentication server: {exc}")

    def sign_up(
        self,
        email: str,
        password: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a new user account.

        Args:
            email: User email.
            password: User password.
            metadata: Optional user profile metadata.

        Returns:
            Dictionary containing user details and optional session if email confirmation is disabled.
        """
        url = f"{self._auth_base}/signup"
        payload: Dict[str, Any] = {"email": email.strip(), "password": password}
        if metadata:
            payload["data"] = metadata

        try:
            resp = self._client.post(url, json=payload, headers=self._get_headers())
            data = self._handle_response_errors(resp)
            # If Supabase auto-confirms or returns tokens directly, parse session
            if "access_token" in data:
                data["session"] = AuthSession.from_dict(data)
            return data
        except httpx.RequestError as exc:
            logger.error("Network error during sign up: %s", exc)
            raise SupabaseAuthError(f"Network error communicating with authentication server: {exc}")

    def sign_in_with_otp(self, email: str) -> Dict[str, Any]:
        """Request a one-time magic link / OTP token sent to the user's email."""
        url = f"{self._auth_base}/otp"
        payload = {"email": email.strip(), "create_user": False}
        try:
            resp = self._client.post(url, json=payload, headers=self._get_headers())
            return self._handle_response_errors(resp)
        except httpx.RequestError as exc:
            logger.error("Network error during OTP request: %s", exc)
            raise SupabaseAuthError(f"Network error requesting OTP: {exc}")

    def verify_otp(self, email: str, token: str, otp_type: str = "email") -> AuthSession:
        """Verify an email OTP code or magic link token.

        Args:
            email: User's email.
            token: OTP token string.
            otp_type: Type of verification (usually 'email' or 'magiclink').

        Returns:
            AuthSession on successful verification.
        """
        url = f"{self._auth_base}/verify"
        payload = {"email": email.strip(), "token": token.strip(), "type": otp_type}
        try:
            resp = self._client.post(url, json=payload, headers=self._get_headers())
            data = self._handle_response_errors(resp)
            return AuthSession.from_dict(data)
        except httpx.RequestError as exc:
            logger.error("Network error during OTP verification: %s", exc)
            raise SupabaseAuthError(f"Network error verifying OTP: {exc}")

    def refresh_session(self, refresh_token: str) -> AuthSession:
        """Exchange a refresh token for a fresh access token session.

        Args:
            refresh_token: The persistent refresh token.

        Returns:
            Updated AuthSession.
        """
        url = f"{self._auth_base}/token?grant_type=refresh_token"
        payload = {"refresh_token": refresh_token.strip()}
        try:
            resp = self._client.post(url, json=payload, headers=self._get_headers())
            data = self._handle_response_errors(resp)
            return AuthSession.from_dict(data)
        except httpx.RequestError as exc:
            logger.error("Network error during session refresh: %s", exc)
            raise SupabaseAuthError(f"Network error refreshing authentication session: {exc}")

    def sign_out(self, access_token: str) -> bool:
        """Invalidate the session on Supabase Auth."""
        url = f"{self._auth_base}/logout"
        try:
            resp = self._client.post(url, headers=self._get_headers(access_token))
            self._handle_response_errors(resp)
            return True
        except Exception as exc:
            logger.warning("Error during remote sign out (proceeding locally): %s", exc)
            return False

    def get_user(self, access_token: str) -> AuthUser:
        """Fetch current user profile using active JWT."""
        url = f"{self._auth_base}/user"
        try:
            resp = self._client.get(url, headers=self._get_headers(access_token))
            data = self._handle_response_errors(resp)
            return AuthUser.from_dict(data)
        except httpx.RequestError as exc:
            logger.error("Network error fetching user: %s", exc)
            raise SupabaseAuthError(f"Network error fetching user profile: {exc}")
