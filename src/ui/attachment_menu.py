"""Relay AI - Attachment Picker Menu & Camera Dialog.

Provides an attachment popup menu matching the modern capsule design
(Camera, Photos, Files) and a native camera capture dialog.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, QByteArray, QSize, QPoint
from PySide6.QtGui import QIcon, QPixmap, QPainter, QImage, QColor, QFont
from PySide6.QtWidgets import (
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
)

# -----------------------------------------------------------------------------
# SVG Icons (Monochromatic Outline Style)
# -----------------------------------------------------------------------------
CAMERA_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
  <circle cx="12" cy="13" r="4"/>
</svg>"""

PHOTOS_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <rect x="3" y="3" width="18" height="18" rx="3" ry="3"/>
  <circle cx="8.5" cy="8.5" r="1.5"/>
  <polyline points="21 15 16 10 5 21"/>
</svg>"""

FILES_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
</svg>"""

MIC_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
  <line x1="12" y1="19" x2="12" y2="23"/>
  <line x1="8" y1="23" x2="16" y2="23"/>
</svg>"""

MIC_RECORDING_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" fill="#FFFFFF"/>
  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
  <line x1="12" y1="19" x2="12" y2="23"/>
  <line x1="8" y1="23" x2="16" y2="23"/>
</svg>"""


def render_svg_pixmap(svg_template: str, color: str = "#FFFFFF", size: int = 24) -> QPixmap:
    """Render an SVG template string with color replacement into a QPixmap."""
    from PySide6.QtSvg import QSvgRenderer
    svg_code = svg_template.format(color=color)
    renderer = QSvgRenderer(QByteArray(svg_code.encode("utf-8")))
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    renderer.render(painter)
    painter.end()
    return QPixmap.fromImage(img)


def render_svg_icon(svg_template: str, color: str = "#FFFFFF", size: int = 24) -> QIcon:
    """Render an SVG template string into a QIcon."""
    return QIcon(render_svg_pixmap(svg_template, color=color, size=size))


# -----------------------------------------------------------------------------
# Attachment Menu Item (Row with circular icon badge & label)
# -----------------------------------------------------------------------------
class AttachmentMenuItem(QFrame):
    """A row in the attachment popup with a circular icon badge and label."""

    clicked = Signal()

    def __init__(self, title: str, svg_template: str, dark: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.dark = dark
        self.setObjectName("AttachmentMenuItem")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(50)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 12, 4)
        layout.setSpacing(14)

        # Circular Icon Badge
        badge = QFrame()
        badge.setObjectName("AttachmentItemBadge")
        badge.setFixedSize(38, 38)
        badge_layout = QVBoxLayout(badge)
        badge_layout.setContentsMargins(0, 0, 0, 0)
        badge_layout.setAlignment(Qt.AlignCenter)

        icon_color = "#FFFFFF" if dark else "#202124"
        pixmap = render_svg_pixmap(svg_template, color=icon_color, size=20)
        icon_lbl = QLabel()
        icon_lbl.setPixmap(pixmap)
        icon_lbl.setAlignment(Qt.AlignCenter)
        badge_layout.addWidget(icon_lbl)
        layout.addWidget(badge, 0)

        # Title Label
        self.label = QLabel(title)
        self.label.setObjectName("AttachmentItemLabel")
        font = self.label.font()
        font.setPointSize(11)
        font.setWeight(QFont.Weight.Medium)
        self.label.setFont(font)
        layout.addWidget(self.label, 1)

        self._apply_style()

    def _apply_style(self):
        bg_hover = "rgba(255, 255, 255, 0.08)" if self.dark else "rgba(0, 0, 0, 0.05)"
        badge_bg = "#383A40" if self.dark else "#E8EAED"
        text_color = "#FFFFFF" if self.dark else "#202124"

        self.setStyleSheet(f"""
            QFrame#AttachmentMenuItem {{
                background-color: transparent;
                border-radius: 12px;
            }}
            QFrame#AttachmentMenuItem:hover {{
                background-color: {bg_hover};
            }}
            QFrame#AttachmentItemBadge {{
                background-color: {badge_bg};
                border-radius: 19px;
            }}
            QLabel#AttachmentItemLabel {{
                color: {text_color};
                background: transparent;
            }}
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)


# -----------------------------------------------------------------------------
# Attachment Menu Popup Widget (Camera, Photos, Files)
# -----------------------------------------------------------------------------
class AttachmentMenu(QWidget):
    """Floating attachment menu popup matching the reference design."""

    camera_selected = Signal()
    photos_selected = Signal()
    files_selected = Signal()

    def __init__(self, dark: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.dark = dark
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedWidth(200)

        # Outer root layout
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(0)

        # Elevated card container
        self.card = QFrame()
        self.card.setObjectName("AttachmentCardContainer")
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(6, 8, 6, 8)
        card_layout.setSpacing(4)

        # 1. Camera Item
        self.item_camera = AttachmentMenuItem("Camera", CAMERA_SVG, dark=self.dark, parent=self.card)
        self.item_camera.clicked.connect(self._on_camera)
        card_layout.addWidget(self.item_camera)

        # 2. Photos Item
        self.item_photos = AttachmentMenuItem("Photos", PHOTOS_SVG, dark=self.dark, parent=self.card)
        self.item_photos.clicked.connect(self._on_photos)
        card_layout.addWidget(self.item_photos)

        # 3. Files Item
        self.item_files = AttachmentMenuItem("Files", FILES_SVG, dark=self.dark, parent=self.card)
        self.item_files.clicked.connect(self._on_files)
        card_layout.addWidget(self.item_files)

        root_layout.addWidget(self.card)
        self._apply_style()

    def _apply_style(self):
        bg_card = "#26282B" if self.dark else "#FFFFFF"
        border_col = "#383A40" if self.dark else "#CBD0D5"
        self.card.setStyleSheet(f"""
            QFrame#AttachmentCardContainer {{
                background-color: {bg_card};
                border: 1px solid {border_col};
                border-radius: 18px;
            }}
        """)

    def _on_camera(self):
        self.close()
        self.camera_selected.emit()

    def _on_photos(self):
        self.close()
        self.photos_selected.emit()

    def _on_files(self):
        self.close()
        self.files_selected.emit()

    def show_above(self, anchor_widget: QWidget):
        """Display popup centered or aligned right above the anchor widget."""
        self.adjustSize()
        anchor_global = anchor_widget.mapToGlobal(QPoint(0, 0))
        h = self.sizeHint().height()
        target_x = anchor_global.x()
        target_y = anchor_global.y() - h - 6
        self.move(target_x, target_y)
        self.show()


# -----------------------------------------------------------------------------
# Native Camera Capture Dialog (Viewfinder + Snapshot)
# -----------------------------------------------------------------------------
class CameraCaptureDialog(QDialog):
    """Dialog for capturing photos via webcam with live preview."""

    photo_captured = Signal(str)

    def __init__(self, dark: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.dark = dark
        self.setWindowTitle("Camera — Relay AI")
        self.setMinimumSize(480, 400)
        self.resize(520, 440)

        self._temp_photo_path: Optional[str] = None
        self._camera = None
        self._session = None
        self._capture = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Viewfinder Container
        self.viewfinder_frame = QFrame()
        self.viewfinder_frame.setObjectName("ViewfinderFrame")
        self.viewfinder_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vf_layout = QVBoxLayout(self.viewfinder_frame)
        vf_layout.setContentsMargins(0, 0, 0, 0)

        # Check for camera hardware
        has_camera = self._setup_camera(vf_layout)
        layout.addWidget(self.viewfinder_frame, 1)

        # Status / Instructions
        self.status_lbl = QLabel("Point your camera and click Capture to attach photo." if has_camera else "No camera device detected.")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet(f"color: {'#B0B0B0' if dark else '#5F6368'}; font-size: 12px;")
        layout.addWidget(self.status_lbl, 0)

        # Controls Row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("Secondary")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        btn_row.addStretch()

        if not has_camera:
            self.btn_browse = QPushButton("Select Photo from Disk…")
            self.btn_browse.setObjectName("Secondary")
            self.btn_browse.clicked.connect(self._browse_fallback)
            btn_row.addWidget(self.btn_browse)

        self.btn_capture = QPushButton("📷 Capture Photo")
        self.btn_capture.setObjectName("Primary")
        self.btn_capture.setEnabled(has_camera)
        self.btn_capture.setFixedHeight(36)
        self.btn_capture.clicked.connect(self._on_capture_clicked)
        btn_row.addWidget(self.btn_capture)

        layout.addLayout(btn_row)
        self._apply_dialog_style()

    def _setup_camera(self, container_layout: QVBoxLayout) -> bool:
        """Initialize Qt multimedia camera and viewfinder."""
        try:
            from PySide6.QtMultimedia import QCamera, QMediaCaptureSession, QImageCapture, QMediaDevices
            from PySide6.QtMultimediaWidgets import QVideoWidget

            cameras = QMediaDevices.videoInputs()
            if not cameras:
                lbl = QLabel("No camera detected on this system.\nYou can select an image file directly.")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("font-size: 13px; color: #757575; padding: 40px;")
                container_layout.addWidget(lbl)
                return False

            self._camera = QCamera(cameras[0])
            self._session = QMediaCaptureSession()
            self._session.setCamera(self._camera)

            self._viewfinder = QVideoWidget()
            self._session.setVideoOutput(self._viewfinder)
            container_layout.addWidget(self._viewfinder)

            self._capture = QImageCapture()
            self._session.setImageCapture(self._capture)
            self._capture.imageSaved.connect(self._on_image_saved)
            self._capture.errorOccurred.connect(self._on_capture_error)

            self._camera.start()
            return True
        except Exception as e:
            lbl = QLabel(f"Camera initialization note:\n{e}\nYou can select an image file directly.")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size: 13px; color: #757575; padding: 40px;")
            container_layout.addWidget(lbl)
            return False

    def _apply_dialog_style(self):
        bg = "#202124" if self.dark else "#F8F9FA"
        card = "#282A2E" if self.dark else "#FFFFFF"
        border = "#383A40" if self.dark else "#CBD0D5"
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {bg};
            }}
            QFrame#ViewfinderFrame {{
                background-color: {card};
                border: 1px solid {border};
                border-radius: 12px;
                overflow: hidden;
            }}
        """)

    def _on_capture_clicked(self):
        if not self._capture:
            return
        self.btn_capture.setEnabled(False)
        self.btn_capture.setText("Capturing…")
        temp_dir = tempfile.gettempdir()
        target_path = os.path.join(temp_dir, f"relay_camera_{os.getpid()}_{os.urandom(4).hex()}.png")
        self._temp_photo_path = target_path
        self._capture.captureToFile(target_path)

    def _on_image_saved(self, request_id: int, file_path: str):
        path_to_use = file_path or self._temp_photo_path
        if path_to_use and os.path.exists(path_to_use):
            self.photo_captured.emit(path_to_use)
            self.accept()
        else:
            self.btn_capture.setEnabled(True)
            self.btn_capture.setText("📷 Capture Photo")

    def _on_capture_error(self, request_id: int, error, message: str):
        self.status_lbl.setText(f"Capture error: {message}")
        self.btn_capture.setEnabled(True)
        self.btn_capture.setText("📷 Capture Photo")

    def _browse_fallback(self):
        """Allow user to pick photo file if camera is unavailable."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Photo to Attach",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif);;All Files (*.*)"
        )
        if path:
            self.photo_captured.emit(path)
            self.accept()

    def closeEvent(self, event):
        if self._camera:
            try:
                self._camera.stop()
            except Exception:
                pass
        super().closeEvent(event)
