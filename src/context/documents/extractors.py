"""Safe text extraction for PDF, DOCX, and plain-text documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import zipfile
import xml.etree.ElementTree as ET


class ExtractionError(RuntimeError):
    """Raised when a supported document cannot be converted to text."""


@dataclass
class ExtractedDocument:
    text: str
    page_count: int = 1
    pages: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


class DocumentExtractor:
    """Extract readable text while preserving page boundaries when available."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".log"}

    def extract(self, file_path: str | Path) -> ExtractedDocument:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ExtractionError(f"Unsupported document type: {suffix or 'no extension'}")
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            if suffix == ".pdf":
                return self._extract_pdf(path)
            if suffix == ".docx":
                return self._extract_docx(path)
            return self._extract_text(path)
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"Could not extract {path.name}: {exc}") from exc

    @staticmethod
    def _extract_pdf(path: Path) -> ExtractedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ExtractionError("PDF support requires pypdf. Install it with: pip install pypdf") from exc
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        text = "\n\n".join(page for page in pages if page)
        if not text.strip():
            raise ExtractionError("PDF contains no extractable text; OCR is not available in this milestone")
        return ExtractedDocument(text=text, page_count=len(reader.pages), pages=pages,
                                 metadata={"format": "pdf", "title": str(reader.metadata.title or "") if reader.metadata else ""})

    @staticmethod
    def _extract_docx(path: Path) -> ExtractedDocument:
        with zipfile.ZipFile(path) as archive:
            xml_data = archive.read("word/document.xml")
        root = ET.fromstring(xml_data)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for paragraph in root.findall(".//w:p", ns):
            value = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns))
            if value.strip():
                paragraphs.append(value)
        text = "\n".join(paragraphs)
        if not text.strip():
            raise ExtractionError("DOCX contains no extractable text")
        return ExtractedDocument(text=text, page_count=1, pages=[text], metadata={"format": "docx"})

    @staticmethod
    def _extract_text(path: Path) -> ExtractedDocument:
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-16", "cp1252"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ExtractionError("Text file encoding is not supported")
        if not text.strip():
            raise ExtractionError("Document is empty")
        return ExtractedDocument(text=text, page_count=1, pages=[text], metadata={"format": path.suffix.lower().lstrip(".")})
