"""
MCP Tool 2 — Text Chunker
Splits raw document text into overlapping chunks suitable for embedding.
"""
from observability import get_logger, timed
from config import settings

logger = get_logger(__name__)


class TextChunker:
    """Split text into fixed-size overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            "TextChunker initialised — size=%d, overlap=%d",
            chunk_size,
            chunk_overlap,
        )

    @timed(logger)
    def chunk(self, text: str) -> list[str]:
        """
        Split *text* into overlapping chunks.

        Strategy: word-boundary splitting to avoid cutting mid-word.

        Args:
            text: Raw document text.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to chunker — returning empty list")
            return []

        words = text.split()
        chunks: list[str] = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            logger.debug("  Chunk %d: words %d–%d (%d chars)", len(chunks), start, end, len(chunk))
            # Advance by (chunk_size - overlap) so consecutive chunks share context
            start += self.chunk_size - self.chunk_overlap

        logger.info("✅ Chunking complete — %d chunks produced", len(chunks))
        return chunks
