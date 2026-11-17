"""Voice Assistant integration for Relay AI.

Provides:
- SpeechSynthesizer: Text-to-Speech using PySide6 QtTextToSpeech (WinRT/SAPI).
- SpeechRecognizerWorker: Speech-to-Text dictation using native Windows Speech Recognition or Whisper.
- VoiceAssistantController: State-driven conversational voice assistant (Listen -> Query -> Speak).
- WaveformWidget: Real-time animated audio visualizer.
- VoiceOverlayPanel: Modern interactive voice UI matching the design system.
"""

from __future__ import annotations

import base64
import math
import os
import re
import subprocess
import sys
from typing import Optional, List
from enum import Enum

from PySide6.QtCore import Qt, QObject, QThread, Signal, QTimer, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import (
    QWidget,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
)

from src.ui.theme import get_theme, badge_style, RAVEN, CHARCOAL, WHITE, GRAY


class VoiceState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


def clean_markdown_for_speech(text: str) -> str:
    """Strip markdown formatting, URLs, code blocks, and math for natural speech."""
    if not text:
        return ""
    # Remove code blocks
    s = re.sub(r"```[\s\S]*?```", " Code block omitted. ", text)
    # Remove inline code
    s = re.sub(r"`([^`]+)`", r"\1", s)
    # Remove markdown links: [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", s)
    # Remove raw URLs
    s = re.sub(r"https?://\S+", "", s)
    # Remove HTML tags
    s = re.sub(r"<[^>]+>", " ", s)
    # Remove headers: # Title -> Title
    s = re.sub(r"^[#\s]+", "", s, flags=re.MULTILINE)
    # Remove bold/italic: **bold**, *italic*
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    # Remove list bullets
    s = re.sub(r"^[-*•]\s+", "", s, flags=re.MULTILINE)
    # Normalize whitespaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


class SpeechSynthesizer(QObject):
    """Native Text-to-Speech wrapper using QtTextToSpeech."""

    started = Signal()
    finished = Signal()
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._tts = None
        self._is_available = False
        self._init_tts()

    def _init_tts(self) -> None:
        try:
            from PySide6.QtTextToSpeech import QTextToSpeech
            self._tts = QTextToSpeech(self)
            self._tts.stateChanged.connect(self._on_state_changed)
            self._is_available = True
        except Exception as exc:
            self._tts = None
            self._is_available = False

    def is_available(self) -> bool:
        return self._is_available and self._tts is not None

    def speak(self, text: str) -> None:
        """Speak given text aloud after cleaning markdown syntax."""
        if not self.is_available():
            self.finished.emit()
            return
        cleaned = clean_markdown_for_speech(text)
        if not cleaned:
            self.finished.emit()
            return

        try:
            from PySide6.QtTextToSpeech import QTextToSpeech
            if self._tts.state() == QTextToSpeech.State.Speaking:
                self._tts.stop()
            self._tts.say(cleaned)
            self.started.emit()
        except Exception as exc:
            self.error.emit(str(exc))
            self.finished.emit()

    def stop(self) -> None:
        """Stop speaking immediately."""
        if self.is_available():
            try:
                self._tts.stop()
            except Exception:
                pass
            self.finished.emit()

    def pause(self) -> None:
        if self.is_available():
            try:
                self._tts.pause()
            except Exception:
                pass

    def resume(self) -> None:
        if self.is_available():
            try:
                self._tts.resume()
            except Exception:
                pass

    def is_speaking(self) -> bool:
        if not self.is_available():
            return False
        try:
            from PySide6.QtTextToSpeech import QTextToSpeech
            return self._tts.state() == QTextToSpeech.State.Speaking
        except Exception:
            return False

    def _on_state_changed(self, state) -> None:
        try:
            from PySide6.QtTextToSpeech import QTextToSpeech
            if state == QTextToSpeech.State.Ready:
                self.finished.emit()
        except Exception:
            pass


class SpeechTranscriberWorker(QThread):
    """Background worker that transcribes recorded audio using Groq Whisper, OpenAI, or SAPI."""

    recognized = Signal(str)
    error = Signal(str)
    finished_processing = Signal()

    def __init__(self, audio_file_path: str, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.audio_file_path = audio_file_path
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        if not self.audio_file_path or not os.path.exists(self.audio_file_path):
            self.error.emit("No audio recorded.")
            self.finished_processing.emit()
            return

        size = os.path.getsize(self.audio_file_path)
        if size < 4000:
            self.error.emit("No speech detected.")
            self._cleanup()
            self.finished_processing.emit()
            return

        from src.core.security import CredentialStore
        store = CredentialStore()

        # 1. Tier 1: Groq Whisper Large v3 (Fastest, ~0.3s)
        groq_key = store.get_api_key("groq")
        if groq_key and not self._is_cancelled:
            try:
                import groq
                client = groq.Groq(api_key=groq_key.strip())
                with open(self.audio_file_path, "rb") as f:
                    res = client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=("recording.wav", f.read()),
                    )
                text = (res.text or "").strip()
                # Filter out pure Whisper hallucination phrases on silence
                filtered_hallucinations = {"you", "you.", "thank you", "thank you.", "bye", "bye.", "you're welcome", "."}
                if text and text.lower() not in filtered_hallucinations:
                    self.recognized.emit(text)
                    self._cleanup()
                    self.finished_processing.emit()
                    return
            except Exception:
                pass

        # 2. Tier 2: OpenAI Whisper (whisper-1)
        openai_key = store.get_api_key("openai")
        if openai_key and not self._is_cancelled:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key.strip())
                with open(self.audio_file_path, "rb") as f:
                    res = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=("recording.wav", f.read()),
                    )
                text = (res.text or "").strip()
                if text:
                    self.recognized.emit(text)
                    self._cleanup()
                    self.finished_processing.emit()
                    return
            except Exception:
                pass

        # 3. Tier 3: Windows SAPI native audio file transcription fallback
        if not self._is_cancelled:
            sapi_text = self._transcribe_windows_sapi(self.audio_file_path)
            if sapi_text:
                self.recognized.emit(sapi_text)
                self._cleanup()
                self.finished_processing.emit()
                return

        self.error.emit("Could not transcribe speech.")
        self._cleanup()
        self.finished_processing.emit()

    def _cleanup(self) -> None:
        try:
            if self.audio_file_path and os.path.exists(self.audio_file_path):
                os.unlink(self.audio_file_path)
        except Exception:
            pass

    def _transcribe_windows_sapi(self, wav_path: str) -> Optional[str]:
        ps_script = f"""
Add-Type -AssemblyName System.Speech
try {{
    $engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine
    $engine.SetInputToWaveFile('{wav_path.replace(os.sep, "/")}')
    $grammar = New-Object System.Speech.Recognition.DictationGrammar
    $engine.LoadGrammar($grammar)
    $res = $engine.Recognize()
    if ($res -and $res.Text) {{ Write-Output "RESULT:$($res.Text)" }}
}} catch {{}}
"""
        encoded = base64.b64encode(ps_script.encode("utf-16le")).decode("ascii")
        try:
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0

            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                capture_output=True,
                text=True,
                timeout=12,
                startupinfo=startupinfo,
            )
            out = proc.stdout.strip()
            if "RESULT:" in out:
                return out.split("RESULT:", 1)[1].strip()
        except Exception:
            pass
        return None


class SpeechRecognizerWorker(QThread):
    """Worker keeping compatibility with existing test suites."""

    recognized = Signal(str)
    started_listening = Signal()
    stopped_listening = Signal()
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._is_cancelled = False

    def run(self) -> None:
        self._is_cancelled = False
        self.started_listening.emit()

    def cancel(self) -> None:
        self._is_cancelled = True


class WaveformWidget(QWidget):
    """Dynamic animated waveform widget for voice feedback."""

    def __init__(self, bars_count: int = 5, dark: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.bars_count = bars_count
        self.dark = dark
        self.is_active = False
        self._phase = 0.0

        self.setMinimumSize(48, 28)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self._timer = QTimer(self)
        self._timer.setInterval(40)  # ~25 FPS
        self._timer.timeout.connect(self._update_wave)

    def set_active(self, active: bool) -> None:
        self.is_active = active
        if active:
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()
            self.update()

    def _update_wave(self) -> None:
        self._phase += 0.25
        if self._phase > math.pi * 2:
            self._phase -= math.pi * 2
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        t = get_theme(self.dark)
        bar_color = QColor(WHITE if self.dark else RAVEN)
        if not self.is_active:
            bar_color = QColor(GRAY)

        w = self.width()
        h = self.height()
        gap = 4
        bar_w = max(3, (w - (self.bars_count - 1) * gap) // self.bars_count)
        start_x = (w - (self.bars_count * bar_w + (self.bars_count - 1) * gap)) // 2

        painter.setBrush(QBrush(bar_color))
        painter.setPen(Qt.NoPen)

        for i in range(self.bars_count):
            if self.is_active:
                offset = (i * 0.8) + self._phase
                factor = 0.3 + 0.7 * abs(math.sin(offset))
                bar_h = max(4, int(h * 0.85 * factor))
            else:
                bar_h = 4  # Static idle dot/dash

            x = start_x + i * (bar_w + gap)
            y = (h - bar_h) // 2
            painter.drawRoundedRect(x, y, bar_w, bar_h, bar_w // 2, bar_w // 2)

        painter.end()


class VoiceOverlayPanel(QFrame):
    """Floating/docked interactive voice assistant control card."""

    stop_requested = Signal()
    close_requested = Signal()

    def __init__(self, dark: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.dark = dark
        self.setObjectName("Panel")

        self._build_ui()
        self.apply_theme(self.dark)

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        # 1. Waveform visualizer
        self.waveform = WaveformWidget(bars_count=5, dark=self.dark, parent=self)
        layout.addWidget(self.waveform, 0)

        # 2. Status & Transcript text box
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        self.status_label = QLabel("Voice Assistant")
        self.status_label.setStyleSheet("font-weight: 700; font-size: 13px;")

        self.subtitle_label = QLabel("Listening for your voice…")
        self.subtitle_label.setObjectName("Muted")
        self.subtitle_label.setStyleSheet("font-size: 11px;")

        text_layout.addWidget(self.status_label)
        text_layout.addWidget(self.subtitle_label)
        layout.addLayout(text_layout, 1)

        # 3. Action controls
        self.interrupt_btn = QPushButton("⏹ Mute / Interrupt")
        self.interrupt_btn.setObjectName("Secondary")
        self.interrupt_btn.setFixedHeight(30)
        self.interrupt_btn.clicked.connect(self.stop_requested.emit)
        layout.addWidget(self.interrupt_btn, 0)

        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("Secondary")
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.setToolTip("Exit Voice Mode")
        self.close_btn.clicked.connect(self.close_requested.emit)
        layout.addWidget(self.close_btn, 0)

    def set_state(self, state: VoiceState, detail: str = "") -> None:
        t = get_theme(self.dark)
        if state == VoiceState.LISTENING:
            self.waveform.set_active(True)
            self.status_label.setText("🎙 Listening…")
            self.subtitle_label.setText(detail or "Speak clearly into your microphone")
            self.interrupt_btn.setText("Cancel Listening")
        elif state == VoiceState.THINKING:
            self.waveform.set_active(True)
            self.status_label.setText("✦ Thinking…")
            self.subtitle_label.setText(detail or "Synthesizing AI response")
            self.interrupt_btn.setText("Cancel Query")
        elif state == VoiceState.SPEAKING:
            self.waveform.set_active(True)
            self.status_label.setText("🔊 Relay AI Speaking…")
            self.subtitle_label.setText(detail or "Tap interrupt to stop speech")
            self.interrupt_btn.setText("⏹ Stop Speaking")
        else:  # IDLE
            self.waveform.set_active(False)
            self.status_label.setText("Voice Assistant Ready")
            self.subtitle_label.setText("Click the mic or voice button to speak")
            self.interrupt_btn.setText("Start Listening")

    def apply_theme(self, dark: bool) -> None:
        self.dark = dark
        t = get_theme(dark)
        self.waveform.dark = dark
        self.setStyleSheet(
            f"QFrame#Panel {{"
            f"  background-color: {t.bg_card};"
            f"  border: 1px solid {t.border_strong};"
            f"  border-radius: 12px;"
            f"}}"
        )


class VoiceAssistantController(QObject):
    """Master controller connecting Voice UI, SpeechSynthesizer, and SpeechRecognizer."""

    state_changed = Signal(str, str)  # (state_name, detail)
    transcription_ready = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.state = VoiceState.IDLE
        self.is_voice_mode_active = False

        self.synthesizer = SpeechSynthesizer(self)
        self.synthesizer.finished.connect(self._on_speech_finished)

        self.recognizer = SpeechRecognizerWorker(self)
        self.recognizer.recognized.connect(self._on_recognized)
        self.recognizer.error.connect(self._on_recognition_error)
        self.recognizer.stopped_listening.connect(self._on_stopped_listening)

        self._capture_session = None
        self._audio_input = None
        self._recorder = None
        self._current_audio_file = None
        self._auto_stop_timer = None
        self._transcriber = None
        self._discard_recording = False
        self._init_recorder()

    def _init_recorder(self) -> None:
        try:
            from PySide6.QtMultimedia import QMediaCaptureSession, QAudioInput, QMediaRecorder, QMediaFormat
            self._capture_session = QMediaCaptureSession(self)
            self._audio_input = QAudioInput(self)
            self._capture_session.setAudioInput(self._audio_input)
            self._recorder = QMediaRecorder(self)
            self._capture_session.setRecorder(self._recorder)

            media_format = QMediaFormat()
            media_format.setFileFormat(QMediaFormat.FileFormat.Wave)
            media_format.setAudioCodec(QMediaFormat.AudioCodec.Wave)
            self._recorder.setMediaFormat(media_format)
            self._recorder.recorderStateChanged.connect(self._on_recorder_state_changed)
        except Exception:
            self._recorder = None

    def toggle_voice_mode(self) -> bool:
        """Toggle live conversational voice assistant mode."""
        self.is_voice_mode_active = not self.is_voice_mode_active
        if self.is_voice_mode_active:
            self.start_listening("Live Voice Mode Active")
        else:
            self.stop_all()
        return self.is_voice_mode_active

    def start_listening(self, detail: str = "") -> None:
        """Start capturing microphone audio."""
        self.stop_speech()
        self._discard_recording = False
        self._set_state(VoiceState.LISTENING, detail or "Listening for speech…")

        if self._recorder is not None:
            try:
                import tempfile
                from PySide6.QtCore import QUrl, QTimer
                temp_dir = tempfile.gettempdir()
                self._current_audio_file = os.path.join(temp_dir, f"relay_stt_{os.getpid()}_{os.urandom(4).hex()}.wav")
                self._recorder.setOutputLocation(QUrl.fromLocalFile(self._current_audio_file))
                self._recorder.record()

                if self._auto_stop_timer is None:
                    self._auto_stop_timer = QTimer(self)
                    self._auto_stop_timer.setSingleShot(True)
                    self._auto_stop_timer.timeout.connect(self.stop_listening)
                self._auto_stop_timer.start(20000)
            except Exception:
                pass

        if hasattr(self.recognizer, "start") and callable(self.recognizer.start):
            try:
                if not self.recognizer.isRunning():
                    self.recognizer.start()
            except Exception:
                pass

    def stop_speech(self) -> None:
        if self.synthesizer.is_speaking():
            self.synthesizer.stop()

    def stop_listening(self) -> None:
        """Stop microphone recording and trigger transcription."""
        if self._auto_stop_timer and self._auto_stop_timer.isActive():
            self._auto_stop_timer.stop()

        if self.recognizer.isRunning():
            self.recognizer.cancel()

        try:
            from PySide6.QtMultimedia import QMediaRecorder
            if self._recorder is not None and self._recorder.recorderState() == QMediaRecorder.RecorderState.RecordingState:
                self._set_state(VoiceState.THINKING, "Transcribing speech…")
                self._recorder.stop()
                return
        except Exception:
            pass

        self._set_state(VoiceState.IDLE, "Ready")

    def stop_all(self) -> None:
        """Stop listening and speaking, cancel pending transcription, and reset to IDLE."""
        self.stop_speech()
        self.is_voice_mode_active = False
        self._discard_recording = True

        if self._auto_stop_timer and self._auto_stop_timer.isActive():
            self._auto_stop_timer.stop()

        try:
            from PySide6.QtMultimedia import QMediaRecorder
            if self._recorder is not None and self._recorder.recorderState() == QMediaRecorder.RecorderState.RecordingState:
                self._recorder.stop()
        except Exception:
            pass

        if self.recognizer.isRunning():
            self.recognizer.cancel()
            self.recognizer.wait(2000)

        self._set_state(VoiceState.IDLE, "Ready")

    def _on_recorder_state_changed(self, state) -> None:
        try:
            from PySide6.QtMultimedia import QMediaRecorder
            if state == QMediaRecorder.RecorderState.StoppedState:
                if self._discard_recording:
                    if self._current_audio_file and os.path.exists(self._current_audio_file):
                        try:
                            os.unlink(self._current_audio_file)
                        except Exception:
                            pass
                    self._set_state(VoiceState.IDLE, "Ready")
                    return

                audio_path = self._current_audio_file
                if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                    self._set_state(VoiceState.THINKING, "Transcribing…")
                    self._transcriber = SpeechTranscriberWorker(audio_path, parent=self)
                    self._transcriber.recognized.connect(self._on_recognized)
                    self._transcriber.error.connect(self._on_recognition_error)
                    self._transcriber.finished_processing.connect(lambda: self._set_state(VoiceState.IDLE, "Ready"))
                    self._transcriber.start()
                else:
                    self._set_state(VoiceState.IDLE, "Ready")
        except Exception:
            self._set_state(VoiceState.IDLE, "Ready")

    def on_prompt_sent(self) -> None:
        """Notify voice assistant that a prompt has been submitted to AI."""
        if self.is_voice_mode_active:
            self._set_state(VoiceState.THINKING, "Consulting model…")

    def on_response_ready(self, response_text: str) -> None:
        """Speak the AI answer if voice mode is active."""
        if self.is_voice_mode_active and response_text:
            self._set_state(VoiceState.SPEAKING, "Reading answer aloud…")
            self.synthesizer.speak(response_text)
        elif self.state == VoiceState.THINKING:
            self._set_state(VoiceState.IDLE, "Done")

    def _on_recognized(self, text: str) -> None:
        self.transcription_ready.emit(text)
        if self.is_voice_mode_active:
            self._set_state(VoiceState.THINKING, f'Query: "{text[:40]}…"' if len(text) > 40 else f'Query: "{text}"')
        else:
            self._set_state(VoiceState.IDLE, "Transcription complete")

    def _on_recognition_error(self, err_msg: str) -> None:
        if self.is_voice_mode_active:
            # If in live voice mode, stay ready to listen again
            self._set_state(VoiceState.IDLE, f"Notice: {err_msg}")
        else:
            self._set_state(VoiceState.IDLE, err_msg)

    def _on_stopped_listening(self) -> None:
        if self.state == VoiceState.LISTENING:
            self._set_state(VoiceState.IDLE, "Finished listening")

    def _on_speech_finished(self) -> None:
        if self.is_voice_mode_active:
            # Re-arm listening in continuous conversation mode
            self.start_listening("Listening for follow-up…")
        else:
            self._set_state(VoiceState.IDLE, "Ready")

    def _set_state(self, new_state: VoiceState, detail: str = "") -> None:
        self.state = new_state
        self.state_changed.emit(new_state.value, detail)
