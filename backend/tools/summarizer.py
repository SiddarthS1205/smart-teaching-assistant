"""
MCP Tool 7 — Summarizer
Produces a structured academic summary of the entire document.

Strategy (auto-detected at startup):
  1. OpenAI gpt-4o-mini (best quality, requires API key + credits)
  2. Free extractive summarizer (no API key needed)
"""
from config import settings
from observability import get_logger, timed

logger = get_logger(__name__)

SUMMARIZER_PROMPT = """You are an expert academic summarizer.
Given the following document excerpts, produce a comprehensive summary that includes:
1. **Main Topic** — What is this document about?
2. **Key Concepts** — List the most important ideas or terms.
3. **Core Arguments / Findings** — What does the document argue or demonstrate?
4. **Conclusions** — What conclusions are drawn?
5. **Relevance** — Who would benefit from reading this document?

Be thorough but concise. Use bullet points where appropriate."""

# ── Detect OpenAI availability ────────────────────────────────────────────────
_OPENAI_CLIENT = None
try:
    from openai import OpenAI
    if settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("your_"):
        _OPENAI_CLIENT = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("Summarizer: OpenAI backend available")
    else:
        logger.warning("Summarizer: No valid OpenAI key — using free fallback")
except Exception:
    logger.warning("Summarizer: OpenAI unavailable — using free fallback")


class Summarizer:
    """Summarize document content using OpenAI or a free extractive fallback."""

    def __init__(self):
        self._openai = _OPENAI_CLIENT
        backend = "OpenAI gpt-4o-mini" if self._openai else "free extractive summarizer"
        logger.info("Summarizer initialised — backend: %s", backend)

    @timed(logger)
    def summarize(self, chunks: list[str]) -> str:
        if not chunks:
            return "No document content available to summarize."

        if self._openai:
            try:
                return self._summarize_openai(chunks)
            except Exception as exc:
                logger.warning("OpenAI summarize failed (%s) — falling back to free", exc)

        return self._summarize_free(chunks)

    def _summarize_openai(self, chunks: list[str]) -> str:
        from openai import RateLimitError, AuthenticationError
        step = max(1, len(chunks) // 20)
        sampled = chunks[::step][:20]
        combined = "\n\n---\n\n".join(sampled)
        logger.info("📝 OpenAI summarizing — using %d/%d chunks", len(sampled), len(chunks))
        try:
            response = self._openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SUMMARIZER_PROMPT},
                    {"role": "user", "content": f"Document excerpts:\n\n{combined}"},
                ],
                temperature=0.4,
                max_tokens=1500,
            )
            summary = response.choices[0].message.content.strip()
            logger.info("✅ OpenAI summary generated (%d chars)", len(summary))
            return summary
        except (RateLimitError, AuthenticationError) as exc:
            logger.warning("OpenAI quota/auth error — switching to free fallback: %s", exc)
            self._openai = None
            raise

    def _summarize_free(self, chunks: list[str]) -> str:
        """
        Free extractive summarizer:
        - Picks the first chunk (intro), a middle chunk, and the last chunk (conclusion)
        - Extracts unique sentences scored by word frequency
        """
        logger.info("💡 Using free extractive summarizer")

        # Sample representative chunks
        n = len(chunks)
        sampled_indices = list({0, n // 4, n // 2, 3 * n // 4, n - 1})
        sampled_indices = sorted([i for i in sampled_indices if i < n])
        sampled = [chunks[i] for i in sampled_indices]

        # Word frequency scoring
        all_text = " ".join(sampled)
        words = all_text.lower().split()
        stop_words = {
            "the","a","an","and","or","but","in","on","at","to","for","of","with",
            "is","are","was","were","be","been","being","have","has","had","do","does",
            "did","will","would","could","should","may","might","this","that","these",
            "those","it","its","by","from","as","into","through","during","before",
            "after","above","below","between","each","all","both","few","more","most",
        }
        freq: dict[str, int] = {}
        for w in words:
            w = w.strip(".,!?;:\"'()[]")
            if w and w not in stop_words and len(w) > 2:
                freq[w] = freq.get(w, 0) + 1

        # Score sentences
        sentences = []
        for chunk in sampled:
            for sent in chunk.replace("\n", " ").split(". "):
                sent = sent.strip()
                if len(sent) > 40:
                    score = sum(freq.get(w.lower().strip(".,!?;:"), 0) for w in sent.split())
                    sentences.append((score, sent))

        sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for _, s in sentences[:8]]

        # Build structured output
        full_text = " ".join(all_text.split())
        word_count = len(full_text.split())

        # Extract likely key terms (high-frequency non-stop words)
        key_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
        key_terms_str = ", ".join(f"**{w}**" for w, _ in key_terms)

        summary = (
            f"## 📄 Document Summary\n\n"
            f"**Document length:** ~{word_count:,} words across {n} chunks\n\n"
            f"---\n\n"
            f"### 🔑 Key Terms\n{key_terms_str}\n\n"
            f"### 📝 Key Extracted Passages\n\n"
            + "\n\n".join(f"• {s}." for s in top_sentences[:5])
            + f"\n\n---\n"
            f"*Note: This is a free extractive summary. "
            f"For AI-generated summaries, add OpenAI API credits at "
            f"https://platform.openai.com/settings/billing*"
        )
        logger.info("✅ Free summary generated (%d chars)", len(summary))
        return summary
