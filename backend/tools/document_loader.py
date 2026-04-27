"""
MCP Tool 1 — Document Loader
Reads PDF files and returns raw text content.
"""
import os
from pathlib import Path

import PyPDF2

from observability import get_logger, timed

logger = get_logger(__name__)


class DocumentLoader:
    """Load and extract text from PDF documents."""

    @timed(logger)
    def load(self, file_path: str) -> str:
        """
        Extract all text from a PDF file.

        Args:
            file_path: Absolute or relative path to the PDF.

        Returns:
            Concatenated text from all pages.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a PDF or is empty.
        """
        path = Path(file_path)

        if not path.exists():
            logger.error("File not found: %s", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() != ".pdf":
            logger.error("Invalid file type: %s", path.suffix)
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        logger.info("📄 Loading document: %s", path.name)

        text_parts: list[str] = []
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            total_pages = len(reader.pages)
            logger.info("   Pages found: %d", total_pages)

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                logger.debug("   Page %d/%d extracted (%d chars)", i + 1, total_pages, len(page_text))

        full_text = "\n".join(text_parts).strip()

        if not full_text:
            raise ValueError("PDF appears to be empty or contains only images (no extractable text).")

        logger.info("✅ Document loaded — total chars: %d", len(full_text))
        return full_text

    def load_metadata(self, file_path: str) -> dict:
        """Return basic metadata about the PDF."""
        path = Path(file_path)
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            meta = reader.metadata or {}
            return {
                "filename": path.name,
                "pages": len(reader.pages),
                "title": meta.get("/Title", ""),
                "author": meta.get("/Author", ""),
                "size_bytes": os.path.getsize(path),
            }
