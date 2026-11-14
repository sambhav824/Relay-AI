"""Relay AI Native Authentication Dialog.

Provides native PySide6 login, registration, and OTP verification dialog
connecting directly to Supabase Auth off the UI thread.
"""

from __future__ import annotations

import logging
from typing import Optional
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QWidget,
    QProgressBar,
)

from src.context.auth.client import AuthSession, SupabaseAuthError
from src.context.auth.manager import AuthManager

logger = logging.getLogger(__name__)


class AuthWorker(QThread):
    """Background worker executing Supabase Auth network calls."""

    success = Signal(object)  # Emits AuthSession or result dict
    failure = Signal(str)     # Emits error message

    def __init__(self, task_fn, *args, **kwargs) -> None:
        super().__init__()
        self._task_fn = task_fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._task_fn(*self._args, **self._kwargs)
            self.success.emit(result)
        except SupabaseAuthError as err:
            self.failure.emit(str(err))
        except Exception as exc:
            logger.error("Unhandled auth exception: %s", exc)
            self.failure.emit(f"Authentication error: {exc}")


class LoginDialog(QDialog):
    """Native authentication dialog for Relay AI supporting password, signup, and OTP."""

    authenticated = Signal(object)  # Emits AuthSession

    def __init__(
        self,
        auth_manager: AuthManager,
        parent: Optional[QWidget] = None,
        is_dark: bool = True,
    ) -> None:
        super().__init__(parent)
        self.auth_manager = auth_manager
        self.is_dark = is_dark
        self._active_worker: Optional[AuthWorker] = None

        self.setWindowTitle("Relay AI — Cloud Authentication")
        self.setMinimumWidth(420)
        self.setModal(True)

        self._setup_ui()
        self._apply_theme()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header Title
        title_lbl = QLabel("Relay AI Cloud Sync")
        title_lbl.setObjectName("authHeaderTitle")
        title_lbl.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title_lbl)

        subtitle_lbl = QLabel("Connect to your Supabase account to sync history, projects, and context.")
        subtitle_lbl.setWordWrap(True)
        subtitle_lbl.setStyleSheet("color: #94A3B8; font-size: 13px;")
        layout.addWidget(subtitle_lbl)

        # Error banner (hidden by default)
        self.error_lbl = QLabel()
        self.error_lbl.setObjectName("authErrorLabel")
        self.error_lbl.setWordWrap(True)
        self.error_lbl.setStyleSheet("color: #EF4444; background: rgba(239, 68, 68, 0.1); padding: 8px; border-radius: 6px; font-size: 12px;")
        self.error_lbl.hide()
        layout.addWidget(self.error_lbl)

        # Tab widget for auth modes
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._create_signin_tab()
        self._create_signup_tab()
        self._create_otp_tab()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Status text
        self.status_lbl = QLabel()
        self.status_lbl.setStyleSheet("color: #64748B; font-size: 12px;")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.hide()
        layout.addWidget(self.status_lbl)

    def _create_signin_tab(self) -> None:
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.setSpacing(12)
        vbox.setContentsMargins(0, 16, 0, 0)

        vbox.addWidget(QLabel("Email Address"))
        self.signin_email_edit = QLineEdit()
        self.signin_email_edit.setPlaceholderText("user@example.com")
        vbox.addWidget(self.signin_email_edit)

        vbox.addWidget(QLabel("Password"))
        self.signin_pass_edit = QLineEdit()
        self.signin_pass_edit.setEchoMode(QLineEdit.Password)
        self.signin_pass_edit.setPlaceholderText("••••••••")
        vbox.addWidget(self.signin_pass_edit)

        self.signin_btn = QPushButton("Sign In")
        self.signin_btn.setFixedHeight(36)
        self.signin_btn.setStyleSheet("font-weight: bold;")
        self.signin_btn.clicked.connect(self._on_signin_clicked)
        vbox.addWidget(self.signin_btn)

        self.tabs.addTab(tab, "Sign In")

    def _create_signup_tab(self) -> None:
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.setSpacing(12)
        vbox.setContentsMargins(0, 16, 0, 0)

        vbox.addWidget(QLabel("Email Address"))
        self.signup_email_edit = QLineEdit()
        self.signup_email_edit.setPlaceholderText("user@example.com")
        vbox.addWidget(self.signup_email_edit)

        vbox.addWidget(QLabel("Password (min 6 characters)"))
        self.signup_pass_edit = QLineEdit()
        self.signup_pass_edit.setEchoMode(QLineEdit.Password)
        self.signup_pass_edit.setPlaceholderText("••••••••")
        vbox.addWidget(self.signup_pass_edit)

        vbox.addWidget(QLabel("Confirm Password"))
        self.signup_confirm_edit = QLineEdit()
        self.signup_confirm_edit.setEchoMode(QLineEdit.Password)
        self.signup_confirm_edit.setPlaceholderText("••••••••")
        vbox.addWidget(self.signup_confirm_edit)

        self.signup_btn = QPushButton("Create Account")
        self.signup_btn.setFixedHeight(36)
        self.signup_btn.setStyleSheet("font-weight: bold;")
        self.signup_btn.clicked.connect(self._on_signup_clicked)
        vbox.addWidget(self.signup_btn)

        self.tabs.addTab(tab, "Create Account")

    def _create_otp_tab(self) -> None:
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.setSpacing(12)
        vbox.setContentsMargins(0, 16, 0, 0)

        vbox.addWidget(QLabel("Email Address"))
        self.otp_email_edit = QLineEdit()
        self.otp_email_edit.setPlaceholderText("user@example.com")
        vbox.addWidget(self.otp_email_edit)

        send_code_btn = QPushButton("Send One-Time Code")
        send_code_btn.clicked.connect(self._on_send_otp_clicked)
        vbox.addWidget(send_code_btn)

        vbox.addWidget(QLabel("Verification Code"))
        self.otp_code_edit = QLineEdit()
        self.otp_code_edit.setPlaceholderText("Enter 6-digit code")
        vbox.addWidget(self.otp_code_edit)

        self.verify_otp_btn = QPushButton("Verify & Sign In")
        self.verify_otp_btn.setFixedHeight(36)
        self.verify_otp_btn.setStyleSheet("font-weight: bold;")
        self.verify_otp_btn.clicked.connect(self._on_verify_otp_clicked)
        vbox.addWidget(self.verify_otp_btn)

        self.tabs.addTab(tab, "One-Time Code (OTP)")

    def _set_loading(self, is_loading: bool, status_msg: str = "") -> None:
        self.tabs.setEnabled(not is_loading)
        if is_loading:
            self.error_lbl.hide()
            self.progress_bar.show()
            self.status_lbl.setText(status_msg)
            self.status_lbl.show()
        else:
            self.progress_bar.hide()
            self.status_lbl.hide()

    def _show_error(self, message: str) -> None:
        self.error_lbl.setText(message)
        self.error_lbl.show()

    def _on_signin_clicked(self) -> None:
        email = self.signin_email_edit.text().strip()
        password = self.signin_pass_edit.text()
        if not email or not password:
            self._show_error("Please provide both email and password.")
            return

        self._set_loading(True, "Authenticating with Supabase...")
        self._start_worker(self.auth_manager.login, email, password)

    def _on_signup_clicked(self) -> None:
        email = self.signup_email_edit.text().strip()
        pwd = self.signup_pass_edit.text()
        confirm = self.signup_confirm_edit.text()

        if not email or not pwd:
            self._show_error("Please provide both email and password.")
            return
        if len(pwd) < 6:
            self._show_error("Password must be at least 6 characters.")
            return
        if pwd != confirm:
            self._show_error("Passwords do not match.")
            return

        self._set_loading(True, "Creating account...")
        self._start_worker(self.auth_manager.signup, email, pwd)

    def _on_send_otp_clicked(self) -> None:
        email = self.otp_email_edit.text().strip()
        if not email:
            self._show_error("Please enter an email address to receive OTP.")
            return

        self._set_loading(True, "Sending verification code...")
        self._start_worker(self.auth_manager.send_magic_link, email)

    def _on_verify_otp_clicked(self) -> None:
        email = self.otp_email_edit.text().strip()
        code = self.otp_code_edit.text().strip()
        if not email or not code:
            self._show_error("Please enter email and the verification code received.")
            return

        self._set_loading(True, "Verifying code...")
        self._start_worker(self.auth_manager.verify_otp, email, code)

    def _start_worker(self, fn, *args, **kwargs) -> None:
        self._active_worker = AuthWorker(fn, *args, **kwargs)
        self._active_worker.success.connect(self._on_worker_success)
        self._active_worker.failure.connect(self._on_worker_failure)
        self._active_worker.finished.connect(self._active_worker.deleteLater)
        self._active_worker.start()

    def _on_worker_success(self, result: object) -> None:
        self._set_loading(False)
        if isinstance(result, AuthSession):
            self.authenticated.emit(result)
            self.accept()
        elif isinstance(result, dict) and "session" in result and isinstance(result["session"], AuthSession):
            self.authenticated.emit(result["session"])
            self.accept()
        else:
            # e.g. OTP sent confirmation
            self.status_lbl.setText("Verification code sent! Please check your email.")
            self.status_lbl.show()

    def _on_worker_failure(self, error_msg: str) -> None:
        self._set_loading(False)
        self._show_error(error_msg)

    def _apply_theme(self) -> None:
        if self.is_dark:
            self.setStyleSheet(
                """
                QDialog {
                    background-color: #0F172A;
                    color: #F8FAFC;
                }
                QLabel {
                    color: #F8FAFC;
                }
                QLineEdit {
                    background-color: #1E293B;
                    color: #F8FAFC;
                    border: 1px solid #334155;
                    border-radius: 6px;
                    padding: 8px;
                    font-size: 13px;
                }
                QLineEdit:focus {
                    border: 1px solid #3B82F6;
                }
                QPushButton {
                    background-color: #2563EB;
                    color: #FFFFFF;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #1D4ED8;
                }
                QPushButton:disabled {
                    background-color: #334155;
                    color: #64748B;
                }
                QTabWidget::pane {
                    border: none;
                }
                QTabBar::tab {
                    background-color: #1E293B;
                    color: #94A3B8;
                    padding: 8px 16px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    margin-right: 4px;
                }
                QTabBar::tab:selected {
                    background-color: #2563EB;
                    color: #FFFFFF;
                    font-weight: bold;
                }
                QProgressBar {
                    background-color: #1E293B;
                    border-radius: 2px;
                }
                QProgressBar::chunk {
                    background-color: #3B82F6;
                    border-radius: 2px;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QDialog {
                    background-color: #FFFFFF;
                    color: #0F172A;
                }
                QLabel {
                    color: #0F172A;
                }
                QLineEdit {
                    background-color: #F8FAFC;
                    color: #0F172A;
                    border: 1px solid #CBD5E1;
                    border-radius: 6px;
                    padding: 8px;
                    font-size: 13px;
                }
                QLineEdit:focus {
                    border: 1px solid #2563EB;
                }
                QPushButton {
                    background-color: #2563EB;
                    color: #FFFFFF;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #1D4ED8;
                }
                QPushButton:disabled {
                    background-color: #E2E8F0;
                    color: #94A3B8;
                }
                QTabWidget::pane {
                    border: none;
                }
                QTabBar::tab {
                    background-color: #F1F5F9;
                    color: #64748B;
                    padding: 8px 16px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    margin-right: 4px;
                }
                QTabBar::tab:selected {
                    background-color: #2563EB;
                    color: #FFFFFF;
                    font-weight: bold;
                }
                """
            )
