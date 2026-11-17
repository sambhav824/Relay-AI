"""Relay AI - Multimodal Image Utilities.

Handles reading, base64 encoding, mime-type detection, and data URI formatting
for passing vision inputs to OpenAI, Groq, Gemini, Ollama, and OpenRouter.
"""

import base64
import mimetypes
import os
from pathlib import Path
from typing import Optional, Tuple


def read_image_bytes_and_mime(file_path: str) -> Tuple[Optional[bytes], str]:
    """Read binary data from local image file and guess mime type."""
    p = Path(file_path)
    if not p.is_file():
        return None, "image/jpeg"
    try:
        data = p.read_bytes()
        mime, _ = mimetypes.guess_type(p.name)
        if not mime or not mime.startswith("image/"):
            ext = p.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif ext == ".png":
                mime = "image/png"
            elif ext == ".webp":
                mime = "image/webp"
            elif ext == ".gif":
                mime = "image/gif"
            elif ext == ".bmp":
                mime = "image/bmp"
            else:
                mime = "image/jpeg"
        return data, mime
    except Exception:
        return None, "image/jpeg"


def encode_image_to_base64_data_uri(file_path_or_uri: str) -> Optional[str]:
    """Encode an image file into a standard data URI: 'data:{mime};base64,{data}'."""
    if not file_path_or_uri:
        return None
    if file_path_or_uri.startswith("data:image/"):
        return file_path_or_uri

    data, mime = read_image_bytes_and_mime(file_path_or_uri)
    if not data:
        return None
    b64_str = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64_str}"


def encode_image_to_raw_base64(file_path_or_uri: str) -> Optional[str]:
    """Extract or encode raw base64 string without data URI prefix (for Ollama)."""
    if not file_path_or_uri:
        return None
    if file_path_or_uri.startswith("data:image/"):
        # Strip data:image/...;base64, prefix
        parts = file_path_or_uri.split(",", 1)
        return parts[1] if len(parts) > 1 else parts[0]

    data, _ = read_image_bytes_and_mime(file_path_or_uri)
    if not data:
        return None
    return base64.b64encode(data).decode("utf-8")
