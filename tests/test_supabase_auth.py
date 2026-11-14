"""Unit tests for Supabase Authentication Layer (Phase 10.1)."""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

import httpx
from PySide6.QtWidgets import QApplication

from src.context.auth.client import (
    AuthClient,
    AuthSession,
    AuthUser,
    SupabaseAuthError,
)
from src.context.auth.manager import (
    AuthManager,
    KEYRING_SERVICE_NAME,
    KEYRING_USERNAME_META,
    KEYRING_USERNAME_REFRESH,
)
from src.ui.auth.login_dialog import LoginDialog

# Ensure single QApplication instance for UI tests
app = QApplication.instance() or QApplication([])


class TestAuthModels(unittest.TestCase):
    """Test suite for AuthUser and AuthSession data contracts."""

    def test_auth_user_serialization(self) -> None:
        raw = {
            "id": "u-123",
            "email": "test@example.com",
            "created_at": "2026-11-14T00:00:00Z",
            "user_metadata": {"full_name": "Test User"},
        }
        user = AuthUser.from_dict(raw)
        self.assertEqual(user.id, "u-123")
        self.assertEqual(user.email, "test@example.com")
        self.assertEqual(user.user_metadata["full_name"], "Test User")

        out = user.to_dict()
        self.assertEqual(out["id"], "u-123")
        self.assertEqual(out["email"], "test@example.com")

    def test_auth_session_expiration(self) -> None:
        now = time.time()
        # Session expiring in 30 seconds should be treated as expired (buffer is 60s)
        session = AuthSession(
            access_token="tok_1",
            refresh_token="ref_1",
            expires_in=30,
            expires_at=now + 30,
            user=AuthUser(id="u1", email="a@b.com"),
        )
        self.assertTrue(session.is_expired(buffer_seconds=60))

        # Session expiring in 3600 seconds is not expired
        fresh_session = AuthSession(
            access_token="tok_2",
            refresh_token="ref_2",
            expires_in=3600,
            expires_at=now + 3600,
            user=AuthUser(id="u1", email="a@b.com"),
        )
        self.assertFalse(fresh_session.is_expired(buffer_seconds=60))


class TestAuthClient(unittest.TestCase):
    """Test suite for AuthClient HTTP calls and error handling."""

    def setUp(self) -> None:
        self.mock_http = MagicMock(spec=httpx.Client)
        self.client = AuthClient(
            supabase_url="https://test.supabase.co",
            supabase_anon_key="anon_test_key",
            http_client=self.mock_http,
        )

    def test_sign_in_password_success(self) -> None:
        resp = MagicMock(spec=httpx.Response)
        resp.is_error = False
        resp.status_code = 200
        resp.json.return_value = {
            "access_token": "acc_jwt",
            "refresh_token": "ref_jwt",
            "expires_in": 3600,
            "user": {"id": "u-1", "email": "test@example.com"},
        }
        self.mock_http.post.return_value = resp

        session = self.client.sign_in_with_password("test@example.com", "secret123")
        self.assertEqual(session.access_token, "acc_jwt")
        self.assertEqual(session.refresh_token, "ref_jwt")
        self.assertIsNotNone(session.user)
        self.assertEqual(session.user.id, "u-1")

    def test_sign_in_password_invalid_credentials(self) -> None:
        resp = MagicMock(spec=httpx.Response)
        resp.is_error = True
        resp.status_code = 400
        resp.json.return_value = {
            "error_code": "invalid_credentials",
            "message": "Invalid login credentials",
        }
        self.mock_http.post.return_value = resp

        with self.assertRaises(SupabaseAuthError) as ctx:
            self.client.sign_in_with_password("bad@example.com", "wrong")
        self.assertIn("Invalid login credentials", str(ctx.exception))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_refresh_session_success(self) -> None:
        resp = MagicMock(spec=httpx.Response)
        resp.is_error = False
        resp.status_code = 200
        resp.json.return_value = {
            "access_token": "acc_jwt_new",
            "refresh_token": "ref_jwt_new",
            "expires_in": 3600,
            "user": {"id": "u-1", "email": "test@example.com"},
        }
        self.mock_http.post.return_value = resp

        session = self.client.refresh_session("old_ref")
        self.assertEqual(session.access_token, "acc_jwt_new")
        self.assertEqual(session.refresh_token, "ref_jwt_new")

    def test_network_request_error(self) -> None:
        self.mock_http.post.side_effect = httpx.ConnectError("Connection refused")

        with self.assertRaises(SupabaseAuthError) as ctx:
            self.client.sign_in_with_password("test@example.com", "pw")
        self.assertIn("Network error communicating with authentication server", str(ctx.exception))

    def test_get_user_success(self) -> None:
        resp = MagicMock(spec=httpx.Response)
        resp.is_error = False
        resp.status_code = 200
        resp.json.return_value = {
            "id": "u-user",
            "email": "user@test.com",
            "created_at": "2026-11-14T00:00:00Z",
            "user_metadata": {"role": "developer"},
        }
        self.mock_http.get.return_value = resp

        user = self.client.get_user("jwt_token")
        self.assertEqual(user.id, "u-user")
        self.assertEqual(user.email, "user@test.com")
        self.assertEqual(user.user_metadata.get("role"), "developer")


class TestAuthManager(unittest.TestCase):
    """Test suite for session lifecycle, keyring storage, and auto-refresh."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=AuthClient)
        self.keyring_storage = {}

        def mock_set_pwd(service, username, password):
            self.keyring_storage[(service, username)] = password

        def mock_get_pwd(service, username):
            return self.keyring_storage.get((service, username))

        def mock_del_pwd(service, username):
            self.keyring_storage.pop((service, username), None)

        self.patch_set = patch("keyring.set_password", side_effect=mock_set_pwd)
        self.patch_get = patch("keyring.get_password", side_effect=mock_get_pwd)
        self.patch_del = patch("keyring.delete_password", side_effect=mock_del_pwd)

        self.patch_set.start()
        self.patch_get.start()
        self.patch_del.start()

        self.manager = AuthManager(
            supabase_url="https://test.supabase.co",
            supabase_anon_key="anon_test",
            client=self.mock_client,
            keyring_service="TestService",
        )

    def tearDown(self) -> None:
        self.patch_set.stop()
        self.patch_get.stop()
        self.patch_del.stop()

    def test_login_persists_to_keyring(self) -> None:
        session = AuthSession(
            access_token="jwt_123",
            refresh_token="ref_123",
            expires_in=3600,
            expires_at=time.time() + 3600,
            user=AuthUser(id="u-1", email="user@test.com"),
        )
        self.mock_client.sign_in_with_password.return_value = session

        logged_session = self.manager.login("user@test.com", "pass")
        self.assertEqual(logged_session.access_token, "jwt_123")
        self.assertTrue(self.manager.is_authenticated)

        # Verify keyring contains refresh token
        self.assertEqual(self.keyring_storage.get(("TestService", KEYRING_USERNAME_REFRESH)), "ref_123")

    def test_restore_session_success(self) -> None:
        self.keyring_storage[("TestService", KEYRING_USERNAME_REFRESH)] = "stored_ref"
        new_session = AuthSession(
            access_token="restored_jwt",
            refresh_token="new_ref",
            expires_in=3600,
            expires_at=time.time() + 3600,
            user=AuthUser(id="u-1", email="user@test.com"),
        )
        self.mock_client.refresh_session.return_value = new_session

        restored = self.manager.restore_session()
        self.assertTrue(restored)
        self.assertTrue(self.manager.is_authenticated)
        self.assertEqual(self.manager.get_access_token(), "restored_jwt")

    def test_auto_refresh_on_expired_token(self) -> None:
        # Create an expired session
        expired_session = AuthSession(
            access_token="old_jwt",
            refresh_token="ref_to_refresh",
            expires_in=10,
            expires_at=time.time() - 10,
            user=AuthUser(id="u-1", email="user@test.com"),
        )
        self.manager._session = expired_session

        refreshed_session = AuthSession(
            access_token="refreshed_jwt",
            refresh_token="new_ref_jwt",
            expires_in=3600,
            expires_at=time.time() + 3600,
            user=AuthUser(id="u-1", email="user@test.com"),
        )
        self.mock_client.refresh_session.return_value = refreshed_session

        token = self.manager.get_access_token()
        self.assertEqual(token, "refreshed_jwt")
        self.mock_client.refresh_session.assert_called_once_with("ref_to_refresh")

    def test_logout_clears_keyring_and_session(self) -> None:
        session = AuthSession(
            access_token="jwt_123",
            refresh_token="ref_123",
            expires_in=3600,
            expires_at=time.time() + 3600,
            user=AuthUser(id="u-1", email="user@test.com"),
        )
        self.manager._session = session
        self.keyring_storage[("TestService", KEYRING_USERNAME_REFRESH)] = "ref_123"

        self.manager.logout()
        self.assertFalse(self.manager.is_authenticated)
        self.assertIsNone(self.manager.session)
        self.assertNotIn(("TestService", KEYRING_USERNAME_REFRESH), self.keyring_storage)


class TestLoginDialog(unittest.TestCase):
    """Test suite for PySide6 LoginDialog."""

    def setUp(self) -> None:
        self.mock_manager = MagicMock(spec=AuthManager)
        self.dialog = LoginDialog(auth_manager=self.mock_manager, is_dark=True)
        self.dialog.show()
        QApplication.processEvents()

    def tearDown(self) -> None:
        self.dialog.close()
        QApplication.processEvents()

    def test_dialog_components_exist(self) -> None:
        self.assertIsNotNone(self.dialog.signin_email_edit)
        self.assertIsNotNone(self.dialog.signin_pass_edit)
        self.assertIsNotNone(self.dialog.signin_btn)
        self.assertIsNotNone(self.dialog.signup_email_edit)
        self.assertIsNotNone(self.dialog.signup_pass_edit)
        self.assertIsNotNone(self.dialog.signup_btn)

    def test_empty_email_password_validation(self) -> None:
        self.dialog.signin_email_edit.setText("")
        self.dialog.signin_pass_edit.setText("")
        self.dialog.signin_btn.click()
        QApplication.processEvents()

        self.assertTrue(self.dialog.error_lbl.isVisible())
        self.assertIn("Please provide both email and password", self.dialog.error_lbl.text())

    def test_password_mismatch_signup_validation(self) -> None:
        self.dialog.signup_email_edit.setText("test@test.com")
        self.dialog.signup_pass_edit.setText("password123")
        self.dialog.signup_confirm_edit.setText("mismatch456")
        self.dialog.signup_btn.click()
        QApplication.processEvents()

        self.assertTrue(self.dialog.error_lbl.isVisible())
        self.assertIn("Passwords do not match", self.dialog.error_lbl.text())


if __name__ == "__main__":
    unittest.main()
