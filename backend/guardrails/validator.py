"""
Guardrails — input validation and safety layer.

Validates:
  • Query length and content
  • Harmful / irrelevant content patterns
  • File upload type and size
"""
import re
from pathlib import Path

from config import settings
from observability import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when a guardrail check fails."""
    def __init__(self, message: str, code: str = "VALIDATION_ERROR"):
        super().__init__(message)
        self.code = code
        self.message = message


# ── Harmful content patterns ──────────────────────────────────────────────────
_HARMFUL_PATTERNS = re.compile(
    r"\b("
    r"hack|exploit|malware|virus|ransomware|phishing|ddos|sql\s*injection|"
    r"bomb|weapon|kill|murder|suicide|self.harm|drug|cocaine|heroin|"
    r"porn|xxx|nude|naked|sex\s*tape|"
    r"password\s*crack|bypass\s*auth|jailbreak"
    r")\b",
    re.IGNORECASE,
)

# ── Off-topic / irrelevant patterns ──────────────────────────────────────────
_IRRELEVANT_PATTERNS = re.compile(
    r"\b("
    r"lottery|casino|gambling|bet\s*on|stock\s*tip|crypto\s*pump|"
    r"make\s*money\s*fast|get\s*rich|mlm|pyramid\s*scheme"
    r")\b",
    re.IGNORECASE,
)


class QueryValidator:
    """Validate user queries before processing."""

    def validate(self, query: str) -> str:
        """
        Validate and sanitize a query string.

        Args:
            query: Raw user input.

        Returns:
            Sanitized query string.

        Raises:
            ValidationError: If the query fails any guardrail check.
        """
        # 1. Null / empty check
        if not query or not query.strip():
            logger.warning("🛡️  Rejected: empty query")
            raise ValidationError("Query cannot be empty.", code="EMPTY_QUERY")

        query = query.strip()

        # 2. Minimum length
        if len(query) < settings.MIN_QUERY_LENGTH:
            logger.warning("🛡️  Rejected: query too short (%d chars)", len(query))
            raise ValidationError(
                f"Query is too short. Please provide at least {settings.MIN_QUERY_LENGTH} characters.",
                code="QUERY_TOO_SHORT",
            )

        # 3. Maximum length
        if len(query) > settings.MAX_QUERY_LENGTH:
            logger.warning("🛡️  Rejected: query too long (%d chars)", len(query))
            raise ValidationError(
                f"Query exceeds the maximum length of {settings.MAX_QUERY_LENGTH} characters.",
                code="QUERY_TOO_LONG",
            )

        # 4. Harmful content
        if _HARMFUL_PATTERNS.search(query):
            logger.warning("🛡️  Rejected: harmful content detected in query")
            raise ValidationError(
                "Your query contains content that cannot be processed. "
                "Please ask academic or educational questions only.",
                code="HARMFUL_CONTENT",
            )

        # 5. Irrelevant content
        if _IRRELEVANT_PATTERNS.search(query):
            logger.warning("🛡️  Rejected: irrelevant content detected in query")
            raise ValidationError(
                "This system is designed for academic queries only. "
                "Please ask questions related to your uploaded document.",
                code="IRRELEVANT_CONTENT",
            )

        logger.debug("✅ Query passed all guardrail checks")
        return query


class UploadValidator:
    """Validate file uploads before processing."""

    ALLOWED_EXTENSIONS = {".pdf"}

    def validate(self, filename: str, file_size: int) -> None:
        """
        Validate an uploaded file.

        Args:
            filename: Original filename from the upload.
            file_size: File size in bytes.

        Raises:
            ValidationError: If the file fails any guardrail check.
        """
        # 1. Filename present
        if not filename or not filename.strip():
            raise ValidationError("No filename provided.", code="NO_FILENAME")

        # 2. Extension check
        ext = Path(filename).suffix.lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            logger.warning("🛡️  Rejected upload: invalid extension '%s'", ext)
            raise ValidationError(
                f"Only PDF files are accepted. Received: '{ext}'",
                code="INVALID_FILE_TYPE",
            )

        # 3. Size check
        if file_size <= 0:
            raise ValidationError("Uploaded file is empty.", code="EMPTY_FILE")

        if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            logger.warning("🛡️  Rejected upload: file too large (%.2f MB)", size_mb)
            raise ValidationError(
                f"File size ({size_mb:.1f} MB) exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB limit.",
                code="FILE_TOO_LARGE",
            )

        logger.debug("✅ Upload passed all guardrail checks: %s", filename)
