"""Deterministic text chunking with page and source metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Chunk:
    index: int
    content: str
    token_estimate: int
    page_number: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Split text into bounded, overlapping chunks without dropping source offsets.

    Token estimates use a conservative character heuristic (about four characters
    per token); this avoids requiring a model-specific tokenizer at ingestion time.
    """

    def __init__(self, chunk_size: int = 1200, overlap: int = 180) -> None:
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    @staticmethod
    def _token_estimate(text: str) -> int:
        return max(1, (len(text) + 3) // 4) if text else 0

    def chunk(self, text: str, *, pages: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
        source_pages = pages if pages and any(p.strip() for p in pages) else [text]
        output: List[Chunk] = []
        for page_number, page_text in enumerate(source_pages, start=1):
            page = page_text.strip()
            if not page:
                continue
            start = 0
            while start < len(page):
                end = min(start + self.chunk_size, len(page))
                if end < len(page):
                    boundary = max(page.rfind("\n", start, end), page.rfind(" ", start, end))
                    if boundary > start + self.chunk_size // 2:
                        end = boundary
                piece = page[start:end].strip()
                if piece:
                    output.append(Chunk(
                        index=len(output),
                        content=piece,
                        token_estimate=self._token_estimate(piece),
                        page_number=page_number,
                        metadata={**(metadata or {}), "char_start": start, "char_end": end},
                    ))
                if end >= len(page):
                    break
                start = max(start + 1, end - self.overlap)
        return output
