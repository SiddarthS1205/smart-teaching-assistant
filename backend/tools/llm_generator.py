"""
MCP Tool 6 — LLM Generator
Generates answers from retrieved context.

Strategy (auto-detected at startup):
  1. OpenAI gpt-4o-mini with strict extraction prompt (best quality)
  2. Rule-based precise value extractor (free fallback — no API key needed)

OUTPUT FORMAT (mandatory):
  <Label> : <Value>
  e.g.  Student ID : 26013526
        Parent Name : SIVAKOZUNDU
        Course : BTECH - Artificial Intelligence and Data Science
"""
import re
from config import settings
from observability import get_logger, timed

logger = get_logger(__name__)

# ── System prompt (used for OpenAI) ──────────────────────────────────────────
EXTRACTION_PROMPT = """You are a document QA extraction assistant.

STRICT RULES:
- Answer ONLY the asked question.
- Extract ONLY the relevant field from the document.
- DO NOT return full sentence, chunk, or extra fields.
- DO NOT include unrelated data.
- DO NOT repeat entire document content.

OUTPUT FORMAT (MANDATORY):
<Label> : <Value>

RULES:
- Label must match the field name in the document.
- Value must be exact (no extra words).
- If value contains "/ null", return only the valid value.
- If not found → return: NOT FOUND

EXAMPLES:
Q: What is the parent name?
A: Parent Name : SIVAKOZUNDU

Q: What is the student ID?
A: Student ID : 26013526

Q: What is the course?
A: Course : BTECH - Artificial Intelligence and Data Science

Q: What is the payment mode?
A: Mode : APP

Q: Is the document computer-generated?
A: Computer Generated : YES

BAD OUTPUT (DO NOT DO):
: 26013526 Course : BTECH ... : 6 Parent Name : SIVAKOZUNDU / null Mode : APP"""

# ── Stop words ────────────────────────────────────────────────────────────────
_STOP = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with","is","are",
    "was","were","be","been","being","have","has","had","do","does","did","will","would",
    "could","should","may","might","this","that","these","those","it","its","by","from",
    "as","into","through","during","before","after","above","below","between","each",
    "all","both","few","more","most","i","you","he","she","we","they","what","which",
    "who","how","when","where","why","not","no","so","if","then","than","there","here",
    "can","just","about","up","out","also","any","some","such","only","same","other",
    "me","my","your","our","their","am","get","got","give","tell","show","find",
    "please","want","need","like","make","let","know","see","use","go","come","take",
}

# ── Extraction rules: (intent regex, label, [value capture patterns]) ─────────
# Each entry: (query_pattern, display_label, [value_regex, ...])
_RULES: list[tuple[re.Pattern, str, list[re.Pattern]]] = [

    # ── Student / Academic document fields ────────────────────────────────────
    (re.compile(r"\bstudent\s*(id|number|no|#)\b", re.I),
     "Student ID",
     [re.compile(r"student\s*(?:id|no|number)[:\s#]+([A-Z0-9\-]+)", re.I),
      re.compile(r"\b(\d{7,12})\b")]),

    (re.compile(r"\bstudent\s*name\b", re.I),
     "Student Name",
     [re.compile(r"student\s*name[:\s]+([A-Za-z\s\.]+?)(?:\n|$|student|id|course)", re.I),
      re.compile(r"name[:\s]+([A-Za-z\s\.]+?)(?:\n|$|id|roll)", re.I)]),

    (re.compile(r"\b(parent|father|mother|guardian)\s*(name)?\b", re.I),
     "Parent Name",
     [re.compile(r"(?:parent|father|mother|guardian)\s*name[:\s]+([A-Za-z\s\.]+?)(?:\s*/\s*null|\n|$|phone|mobile)", re.I),
      re.compile(r"(?:parent|father|mother|guardian)[:\s]+([A-Za-z\s\.]+?)(?:\s*/\s*null|\n|$)", re.I)]),

    (re.compile(r"\b(course|program|branch|department)\b", re.I),
     "Course",
     [re.compile(r"course[:\s]+([A-Za-z0-9\s\-&]+?)(?:\n|$|semester|year|batch)", re.I),
      re.compile(r"program[:\s]+([A-Za-z0-9\s\-&]+?)(?:\n|$)", re.I),
      re.compile(r"branch[:\s]+([A-Za-z0-9\s\-&]+?)(?:\n|$)", re.I),
      re.compile(r"\b(B\.?TECH[A-Za-z0-9\s\-&]*|B\.?E[A-Za-z0-9\s\-&]*|M\.?TECH[A-Za-z0-9\s\-&]*|MBA[A-Za-z0-9\s\-&]*|BCA[A-Za-z0-9\s\-&]*|MCA[A-Za-z0-9\s\-&]*)\b", re.I)]),

    (re.compile(r"\b(roll\s*(no|number)|enrollment\s*(no|number))\b", re.I),
     "Roll No",
     [re.compile(r"roll\s*(?:no|number)[:\s]+([A-Z0-9\-]+)", re.I),
      re.compile(r"enrollment\s*(?:no|number)[:\s]+([A-Z0-9\-]+)", re.I)]),

    (re.compile(r"\b(semester|sem)\b", re.I),
     "Semester",
     [re.compile(r"semester[:\s]+(\d+|[IVX]+)", re.I),
      re.compile(r"sem[:\s]+(\d+|[IVX]+)", re.I)]),

    (re.compile(r"\b(academic\s*year|year)\b", re.I),
     "Academic Year",
     [re.compile(r"academic\s*year[:\s]+(\d{4}\s*[-–]\s*\d{2,4})", re.I),
      re.compile(r"year[:\s]+(\d{4}\s*[-–]\s*\d{2,4})", re.I)]),

    (re.compile(r"\b(batch|admission\s*year)\b", re.I),
     "Batch",
     [re.compile(r"batch[:\s]+(\d{4}\s*[-–]\s*\d{2,4})", re.I),
      re.compile(r"admission\s*year[:\s]+(\d{4})", re.I)]),

    (re.compile(r"\b(section|division|class)\b", re.I),
     "Section",
     [re.compile(r"section[:\s]+([A-Z0-9]+)", re.I),
      re.compile(r"division[:\s]+([A-Z0-9]+)", re.I)]),

    (re.compile(r"\b(college|institution|university|school)\s*(name)?\b", re.I),
     "Institution",
     [re.compile(r"(?:college|institution|university|school)\s*(?:name)?[:\s]+([A-Za-z0-9\s,\.]+?)(?:\n|$)", re.I)]),

    (re.compile(r"\b(payment\s*mode|mode\s*of\s*payment|pay\s*mode)\b", re.I),
     "Mode",
     [re.compile(r"(?:payment\s*)?mode[:\s]+([A-Za-z0-9\s]+?)(?:\n|$|amount|date)", re.I),
      re.compile(r"\b(CASH|ONLINE|APP|UPI|NEFT|RTGS|DD|CHEQUE|CARD|NET\s*BANKING)\b", re.I)]),

    (re.compile(r"\b(amount|fee|fees|total\s*amount|paid)\b", re.I),
     "Amount",
     [re.compile(r"amount[:\s]*(?:rs\.?|inr|₹)?\s*([\d,]+\.?\d*)", re.I),
      re.compile(r"(?:rs\.?|inr|₹)\s*([\d,]+\.?\d*)", re.I),
      re.compile(r"total[:\s]*(?:rs\.?|inr|₹)?\s*([\d,]+\.?\d*)", re.I)]),

    (re.compile(r"\b(receipt\s*(no|number)|receipt\s*id)\b", re.I),
     "Receipt No",
     [re.compile(r"receipt\s*(?:no|number|id)[:\s]+([A-Z0-9\-/]+)", re.I)]),

    (re.compile(r"\b(date\s*of\s*payment|payment\s*date|paid\s*on)\b", re.I),
     "Payment Date",
     [re.compile(r"(?:payment\s*)?date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", re.I),
      re.compile(r"(?:payment\s*)?date[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})", re.I)]),

    (re.compile(r"\b(computer[\s-]*generated|auto[\s-]*generated)\b", re.I),
     "Computer Generated",
     [re.compile(r"computer[\s-]*generated[:\s]+(YES|NO|TRUE|FALSE)", re.I),
      re.compile(r"\b(computer[\s-]*generated)\b", re.I)]),

    (re.compile(r"\b(phone|mobile|contact)\s*(number|no)?\b", re.I),
     "Phone",
     [re.compile(r"(?:phone|mobile|contact)[:\s]+(\+?[\d\s\-]{10,15})", re.I)]),

    (re.compile(r"\b(email|e-mail)\b", re.I),
     "Email",
     [re.compile(r"(?:email|e-mail)[:\s]+([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})", re.I)]),

    (re.compile(r"\b(address)\b", re.I),
     "Address",
     [re.compile(r"address[:\s]+([A-Za-z0-9\s,\.\-#]+?)(?:\n\n|$|phone|email)", re.I)]),

    (re.compile(r"\b(gender|sex)\b", re.I),
     "Gender",
     [re.compile(r"gender[:\s]+(Male|Female|Other|M|F)", re.I),
      re.compile(r"\b(Male|Female)\b", re.I)]),

    (re.compile(r"\b(dob|date\s*of\s*birth|birth\s*date)\b", re.I),
     "Date of Birth",
     [re.compile(r"(?:dob|date\s*of\s*birth)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", re.I),
      re.compile(r"(?:dob|date\s*of\s*birth)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})", re.I)]),

    # ── Invoice / Finance fields ───────────────────────────────────────────────
    (re.compile(r"\bdue\s*date\b", re.I),
     "Due Date",
     [re.compile(r"due\s*date[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", re.I)]),

    (re.compile(r"\b(invoice\s*date|issue\s*date)\b", re.I),
     "Invoice Date",
     [re.compile(r"(?:invoice\s*)?date[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", re.I)]),

    (re.compile(r"\b(invoice\s*(number|no|#|id))\b", re.I),
     "Invoice #",
     [re.compile(r"Invoice\s*#[:\s]+([A-Z0-9\-]+)", re.I),
      re.compile(r"\b(INV-\d{4}-\d{4,})\b")]),

    (re.compile(r"\b(total\s*(due|amount)?|amount\s*due|grand\s*total)\b", re.I),
     "Total Due",
     [re.compile(r"total\s*due[:\s]*\$?([\d,]+\.?\d*)", re.I),
      re.compile(r"total[:\s]*\$?([\d,]+\.?\d*)", re.I)]),

    (re.compile(r"\bsubtotal\b", re.I),
     "Subtotal",
     [re.compile(r"subtotal[:\s]*\$?([\d,]+\.?\d*)", re.I)]),

    (re.compile(r"\b(invoice\s*status|approval\s*status|status)\b", re.I),
     "Status",
     [re.compile(r"status[:\s]+(APPROVED|REJECTED|PENDING|PAID|OVERDUE|CANCELLED)", re.I),
      re.compile(r"\b(APPROVED|REJECTED|PENDING|PAID|OVERDUE|CANCELLED)\b")]),

    (re.compile(r"\brisk\s*level\b", re.I),
     "Risk Level",
     [re.compile(r"risk\s*level[:\s]+(LOW|MEDIUM|HIGH|CRITICAL)", re.I)]),

    (re.compile(r"\b(vendor|bill\s*from|seller)\b", re.I),
     "Vendor",
     [re.compile(r"bill\s*from[:\s]+([A-Za-z0-9\s,\.]+?)(?:\n|\d{3}|$)", re.I),
      re.compile(r"vendor[:\s]+([A-Za-z0-9\s,\.]+?)(?:\n|$)", re.I)]),

    (re.compile(r"\b(client|customer|bill\s*to|buyer)\b", re.I),
     "Client",
     [re.compile(r"bill\s*to[:\s]+([A-Za-z0-9\s,\.]+?)(?:\n|\d{3}|$)", re.I)]),

    (re.compile(r"\b(payment\s*terms?|terms)\b", re.I),
     "Terms",
     [re.compile(r"terms?[:\s]+(Net\s*\d+|COD|Prepaid|Due\s*on\s*receipt)", re.I)]),

    (re.compile(r"\bcurrenc(y|ies)\b", re.I),
     "Currency",
     [re.compile(r"currency[:\s]+([A-Z]{3})", re.I)]),

    (re.compile(r"\b(checks?\s*passed|validations?\s*passed)\b", re.I),
     "Checks Passed",
     [re.compile(r"(\d+)\s*/\s*\d+\s*checks?\s*passed", re.I),
      re.compile(r"checks?\s*passed[:\s]+(\d+)", re.I)]),

    (re.compile(r"\b(score|validation\s*score)\b", re.I),
     "Score",
     [re.compile(r"score[:\s]+(\d+)(?:/\d+)?", re.I)]),

    (re.compile(r"\b(val(idation)?\s*(id|#))\b", re.I),
     "Val ID",
     [re.compile(r"val\s*id[:\s]+([A-Z0-9\-]+)", re.I),
      re.compile(r"\b(VAL-[A-Z0-9\-]+)\b")]),

    (re.compile(r"\b(bank|account\s*(number|#)|routing)\b", re.I),
     "Bank Account",
     [re.compile(r"account\s*#?[:\s]+([A-Za-z0-9\s]+?)(?:\n|routing|$)", re.I),
      re.compile(r"bank[:\s]+([A-Za-z0-9\s]+?)(?:\n|account|$)", re.I)]),

    (re.compile(r"\b(items?\s*(with|that\s*have|having)\s*tax|taxable\s*items?|tax\s*applied)\b", re.I),
     "Taxable Items",
     [re.compile(r"([A-Za-z][A-Za-z\s\(\)]+?)\s+\d+\s+\$?[\d,.]+\s+8%", re.I)]),
]


def _clean_null(value: str) -> str:
    """Remove '/ null' or '/ NULL' suffixes from extracted values."""
    return re.sub(r"\s*/\s*null\b.*", "", value, flags=re.I).strip().rstrip(".,;")


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP and len(t) > 1]


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 15]


def _score_sentence(sent_tokens: list[str], query_tokens: list[str],
                    query_bigrams: set) -> float:
    if not sent_tokens or not query_tokens:
        return 0.0
    sent_set = set(sent_tokens)
    query_set = set(query_tokens)
    overlap = len(sent_set & query_set) / max(len(query_set), 1)
    sent_bigrams = set(zip(sent_tokens, sent_tokens[1:]))
    bigram_bonus = len(sent_bigrams & query_bigrams) * 0.5
    return overlap + bigram_bonus


def _try_pattern_extraction(query: str, all_text: str) -> tuple[str, str] | None:
    """
    Returns (label, value) if a pattern matched, else None.
    """
    for intent_re, label, value_patterns in _RULES:
        if intent_re.search(query):
            for vp in value_patterns:
                m = vp.search(all_text)
                if m:
                    value = _clean_null(m.group(1).strip())
                    if value:
                        logger.debug("Pattern matched: label=%r value=%r", label, value[:60])
                        return label, value
    return None


def _format_answer(label: str, value: str) -> str:
    """Format as: Label : Value"""
    return f"{label} : {value}"


# ── Detect OpenAI availability ────────────────────────────────────────────────
_OPENAI_CLIENT = None
try:
    from openai import OpenAI
    _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    if settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("your_"):
        _OPENAI_CLIENT = _client
        logger.info("LLMGenerator: OpenAI backend available")
    else:
        logger.warning("LLMGenerator: No valid OpenAI key — using precise free extractor")
except Exception:
    logger.warning("LLMGenerator: OpenAI unavailable — using precise free extractor")


class LLMGenerator:
    """Generate precise Label : Value answers using OpenAI or rule-based fallback."""

    def __init__(self):
        self._openai = _OPENAI_CLIENT
        backend = "OpenAI gpt-4o-mini" if self._openai else "rule-based extractor"
        logger.info("LLMGenerator initialised — backend: %s", backend)

    @timed(logger)
    def generate(self, query: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            logger.warning("No context provided")
            return "NOT FOUND"

        if self._openai:
            try:
                return self._generate_openai(query, context_chunks)
            except Exception as exc:
                logger.warning("OpenAI call failed (%s) — falling back to extractor", exc)

        return self._generate_precise(query, context_chunks)

    # ── OpenAI path ───────────────────────────────────────────────────────────

    def _generate_openai(self, query: str, context_chunks: list[str]) -> str:
        from openai import RateLimitError, AuthenticationError
        context_text = "\n\n".join(context_chunks)
        try:
            response = self._openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"},
                ],
                temperature=0.0,
                max_tokens=128,
            )
            answer = response.choices[0].message.content.strip()
            logger.info("OpenAI answer: %r", answer[:80])
            return answer
        except (RateLimitError, AuthenticationError) as exc:
            logger.warning("OpenAI quota/auth error — switching to free extractor: %s", exc)
            self._openai = None
            raise

    # ── Free precise extraction path ──────────────────────────────────────────

    def _generate_precise(self, query: str, context_chunks: list[str]) -> str:
        """
        Stage 1 — Intent-matched regex → returns Label : Value
        Stage 2 — Sentence scoring + colon extraction → returns Label : Value
        Stage 3 — NOT FOUND
        """
        logger.info("Precise extractor for: %s", query[:60])
        all_text = " ".join(context_chunks)

        # Stage 1: pattern-based
        result = _try_pattern_extraction(query, all_text)
        if result:
            label, value = result
            logger.info("Stage-1 match: %s : %s", label, value[:60])
            return _format_answer(label, value)

        # Stage 2: sentence scoring
        query_tokens = _tokenize(query)
        query_bigrams = set(zip(query_tokens, query_tokens[1:]))

        scored: list[tuple[float, str]] = []
        for chunk in context_chunks:
            for sent in _split_sentences(chunk):
                st = _tokenize(sent)
                s = _score_sentence(st, query_tokens, query_bigrams)
                if s > 0:
                    scored.append((s, sent))

        if not scored:
            return "NOT FOUND"

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]

        # Try to extract label:value from the best sentence
        # Pattern: "Some Label : Some Value" or "Some Label: Some Value"
        kv_match = re.search(r"([A-Za-z][A-Za-z\s]{2,30}?)\s*:\s*(.+?)(?:\s{2,}|$)", best)
        if kv_match:
            label = kv_match.group(1).strip().title()
            value = _clean_null(kv_match.group(2).strip())
            if value and len(value) < 120:
                logger.info("Stage-2 kv match: %s : %s", label, value[:60])
                return _format_answer(label, value)

        # Colon split fallback
        colon = re.search(r":\s*(.+)$", best)
        if colon:
            value = _clean_null(colon.group(1).strip())
            if value and len(value) < 120:
                # Derive label from query
                label = _label_from_query(query)
                return _format_answer(label, value)

        return "NOT FOUND"


def _label_from_query(query: str) -> str:
    """
    Derive a clean label from the query string.
    e.g. "What is the student ID?" → "Student ID"
         "Who is the parent?"      → "Parent"
    """
    # Strip question words
    q = re.sub(r"^(what\s+is\s+(the\s+)?|who\s+is\s+(the\s+)?|"
               r"how\s+many\s+|when\s+is\s+(the\s+)?|"
               r"which\s+|tell\s+me\s+(the\s+)?|give\s+me\s+(the\s+)?)",
               "", query, flags=re.I).strip()
    q = re.sub(r"\?$", "", q).strip()
    return q.title()
