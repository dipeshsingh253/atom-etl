"""Citation extraction from tool artifacts.

Pipeline:  ToolMessages → per-tool builders → deduplicate & prioritise → return top 5.

Each tool uses ``response_format="content_and_artifact"`` so ToolMessages carry
structured metadata in ``.artifact`` that never reaches the LLM.  This module
reads those artifacts to build citation dicts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import ToolMessage
from loguru import logger

from src.modules.agent.tools.sql_tool import get_db_session
from src.modules.documents.service import get_table_page_number

# Tools whose artifacts are math calculations
_MATH_TOOLS = {
    "calculate_cagr",
    "calculate_percentage",
    "calculate_percentage_change",
    "calculate_arithmetic",
}

# ── Retrieval filter thresholds ──────────────────────────────────────────────

_MIN_VECTOR_SCORE = 0.35       # minimum cosine similarity from Qdrant
_MIN_KEYWORD_SCORE = 0.15      # minimum normalised keyword overlap (0-1)
_MIN_KEYWORD_HITS = 4          # minimum distinct answer-keyword matches
_MIN_COMBINED_SCORE = 0.28     # minimum (0.4 × vector + 0.6 × keyword)

# Words too generic to be useful for keyword matching
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will "
    "would shall should may might can could of in to for on with at by from "
    "as into through during before after above below between out off over "
    "under again further then once here there when where why how all each "
    "every both few more most other some such no nor not only own same so "
    "than too very and but or if while that this it its he she they them "
    "their what which who whom whose".split()
)


# ── AnswerMatcher — precomputes answer keywords once ─────────────────────────


@dataclass(frozen=True)
class AnswerMatcher:
    """Tokenises the final answer once and exposes scoring helpers.

    Both the relevance gate and the snippet extractor need to compare text
    against the answer.  This class avoids recomputing the answer keywords
    on every call.
    """

    words: frozenset[str] = field(default_factory=frozenset)
    stems: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_text(cls, text: str) -> AnswerMatcher:
        normalised = cls._normalise(text)
        words = frozenset(
            w for w in re.findall(r"[a-z0-9]+", normalised)
            if w not in _STOPWORDS and len(w) > 1
        )
        stems = frozenset(cls._stem(w) for w in words)
        return cls(words=words, stems=stems)

    # -- public helpers -------------------------------------------------------

    def keyword_relevance(self, text: str) -> tuple[float, int]:
        """Return ``(score_0_to_1, distinct_hit_count)`` for *text* vs the answer."""
        if not self.words:
            return 0.0, 0
        text_words = set(re.findall(r"[a-z0-9]+", self._normalise(text)))
        exact = text_words & self.words
        remaining = {self._stem(w) for w in text_words if w not in exact}
        stem_hits = remaining & self.stems
        score = (len(exact) + 0.7 * len(stem_hits)) / len(self.words)
        return score, len(exact) + len(stem_hits)

    def best_snippet(self, content: str, max_len: int = 300) -> str:
        """Pick the most answer-relevant sentence (or window) from *content*.

        Strategy depends on how the chunk is structured:

        Multiple segments
            ``_split_segments()`` splits first on newlines (PDF text uses
            newlines far more than sentence-ending punctuation), then on
            ``.!?`` + uppercase boundary.  Each segment is scored by
            ``_score_sentence()`` and the winner is returned.  If the winner
            is very short (< 60 chars) it is merged with its neighbour to
            provide context.

        Single segment (blob)
            Common when PDF extraction produces one long line with no
            punctuation.  ``_best_window()`` slides a ``max_len``-wide window
            across the text in steps of ``max_len // 5`` and picks the
            position with the highest keyword density.

        No keywords / no segments
            Falls back to a plain truncation from the start.
        """
        segments = self._split_segments(content)
        if not segments or not self.words:
            return self._truncate(content, max_len)

        if len(segments) == 1:
            # Single block (common in PDF table dumps) – use sliding window
            return self._best_window(content, max_len)

        best_idx, _ = max(
            enumerate(self._score_sentence(s) for s in segments),
            key=lambda pair: pair[1],
        )
        best = segments[best_idx]

        # Merge with neighbour if the winning sentence is very short
        if len(best) < 60 and len(segments) > 1:
            if best_idx + 1 < len(segments):
                best = best + " " + segments[best_idx + 1]
            elif best_idx - 1 >= 0:
                best = segments[best_idx - 1] + " " + best

        return self._truncate(best, max_len)

    # -- private helpers ------------------------------------------------------

    def _score_sentence(self, sentence: str) -> float:
        """Score a sentence against the answer keywords.

        Keyword weights
        ---------------
        10.0 — numeric token with 3+ digits (e.g. ``7351``, ``17333``).
               Specific multi-digit numbers are the strongest signal that a
               sentence contains the exact claim being cited.  The answer
               normalises ``7,351`` → ``7351`` so the match is reliable.
               Example: answer has "7,351" → word "7351"; sentence "c. 7,351 staff" →
               token "7351" → isdigit + len≥3 → +10.0
        3.0  — numeric token with 1–2 digits (e.g. ``10``, ``22``).
               Year fragments or small counts — meaningful but much more
               common so weighted lower than long numbers.
               Example: answer "10% growth" → word "10"; sentence "grew by 10%" →
               token "10" → isdigit + len<3 → +3.0
        1.0  — regular word exact match.
               Example: answer token "security"; sentence has "security" → +1.0
        0.7  — stem match (word not in exact set but stems to an answer stem).
               Lower weight because stemming is fuzzy: two unrelated words can
               occasionally collapse to the same root.
               Example: answer has "Ireland" → stem "ireland"; sentence has
               "Ireland's" → token "irelands" → stem "ireland" → match → +0.7

        Length normalisation
        --------------------
        Score is divided by ``√(word_count)`` so long sentences are penalised
        mildly.  Using the full word count (instead of √) would be too
        aggressive — a 50-word sentence with 5 hits would score the same as a
        5-word sentence with 1 hit.  Square root keeps the penalty gentle while
        still preventing keyword-sparse walls of text from outscoring tight,
        precise sentences.
        Example: 4 hits in a 9-word sentence → 4/√9 = 1.33
                 4 hits in a 36-word sentence → 4/√36 = 0.67  (same hits, lower score)
        """
        words = set(re.findall(r"[a-z0-9]+", self._normalise(sentence)))
        exact = words & self.words
        score = sum(
            10.0 if (w.isdigit() and len(w) >= 3) else   # key claim numbers
            3.0  if w.isdigit() else                      # short numbers
            1.0  for w in exact
        )
        remaining = {self._stem(w) for w in words if w not in exact}
        score += 0.7 * len(remaining & self.stems)
        norm = (len(words) ** 0.5) if words else 1.0
        return score / norm

    @staticmethod
    def _split_segments(text: str) -> list[str]:
        """Split into meaningful segments, handling PDF formatting."""
        parts: list[str] = []
        for line in re.split(r"\n+", text):
            line = line.strip()
            if not line:
                continue
            # Further split on sentence boundaries
            subs = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line)
            parts.extend(s.strip() for s in subs if s.strip())
        return parts

    def _best_window(self, text: str, max_len: int) -> str:
        """Sliding-window extraction when the chunk is a single block."""
        if len(text) <= max_len:
            return text
        normalised = self._normalise(text)
        best_start, best_score = 0, -1.0
        step = max(1, max_len // 5)
        end = max(1, len(text) - max_len + 1)
        for start in range(0, end, step):
            window = normalised[start : start + max_len]
            words = set(re.findall(r"[a-z0-9]+", window))
            hits = words & self.words
            score = sum(
                10.0 if (w.isdigit() and len(w) >= 3) else
                3.0  if w.isdigit() else
                1.0  for w in hits
            )
            if score > best_score:
                best_score = score
                best_start = start
        # Snap to word boundary
        while best_start > 0 and text[best_start - 1].isalnum():
            best_start -= 1
        return self._truncate(text[best_start : best_start + max_len], max_len)

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase + strip thousands-separator commas."""
        out = text.lower()
        while True:
            replaced = re.sub(r"(\d),(\d{3})\b", r"\1\2", out)
            if replaced == out:
                return out
            out = replaced

    @staticmethod
    def _stem(word: str) -> str:
        """Conservative suffix-stripping stemmer."""
        if len(word) <= 4:
            return word
        for sfx in (
            "ation", "tion", "sion", "ment", "ness",
            "ying", "ies", "ing", "ous", "ive", "ful",
            "ers", "est", "ity", "ble", "ant", "ent",
            "ate", "ize", "ise", "ely", "ily",
            "ees", "ed", "es", "er", "ly", "al", "en",
        ):
            if word.endswith(sfx) and len(word) - len(sfx) >= 3:
                return word[: -len(sfx)]
        if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
            return word[:-1]
        return word

    @staticmethod
    def _truncate(text: str, max_len: int = 250) -> str:
        if len(text) <= max_len:
            return text
        cut = text[:max_len]
        for sep in (". ", ".\n", "? ", "! "):
            idx = cut.rfind(sep)
            if idx > max_len // 3:
                return cut[: idx + 1].strip()
        space = cut.rfind(" ")
        if space > max_len // 3:
            return cut[:space].strip() + "..."
        return cut.strip() + "..."


# ── Per-tool citation builders ───────────────────────────────────────────────


def _cite_retrieval(artifacts: list[dict], matcher: AnswerMatcher) -> list[dict]:
    """Build citations from ``retrieve_documents`` artifacts.

    Each chunk returned by the vector DB passes through 3 sequential gates.
    A chunk must clear ALL three to become a citation — failure at any gate
    drops it immediately without evaluating the rest.

    Gate 1 — Vector similarity (``_MIN_VECTOR_SCORE = 0.35``)
        Qdrant returns a cosine similarity score in [0, 1].  Chunks below 0.35
        are semantically distant from the *query* and almost never relevant to
        the *answer*.  This gate is fast (no text processing) and kills the
        most obvious noise.
        Example: chunk about "data privacy legislation" scored 0.28 against a
        query about "cyber security employment" → dropped here.

    Gate 2 — Keyword overlap (``_MIN_KEYWORD_SCORE = 0.15``, ``_MIN_KEYWORD_HITS = 4``)
        ``AnswerMatcher.keyword_relevance()`` counts how many of the answer's
        keywords appear in the chunk (exact + stemmed).  The score is
        normalised by answer-keyword count so it stays in [0, 1] regardless
        of answer length.

        Two thresholds are required *simultaneously*:
        - ``score ≥ 0.15``: at least 15 % of the answer's keywords are present
          (prevents passing on a single lucky hit in a big answer).
          Example: answer has 20 keywords, chunk matches 2 → score=0.10 → fail.
        - ``hits ≥ 4``: at least 4 distinct keyword matches (prevents a chunk
          with one very-weighted number from passing on score alone).
          Example: chunk contains only "7351" → score=10/20=0.50 but hits=1 → fail.

    Gate 3 — Blended score (``_MIN_COMBINED_SCORE = 0.28``)
        ``combined = vector * 0.4 + keyword * 0.6``

        Keyword overlap is weighted 60 % because a chunk can be semantically
        close to the *query* but not actually back up the *answer* (e.g. a
        neighbouring paragraph on the same topic).  Keyword overlap directly
        measures agreement with the answer text, so it deserves more weight.
        The 40/60 split was calibrated empirically on the Cyber Ireland report.
        Example: vector=0.55, keyword=0.08 → combined=0.55*0.4 + 0.08*0.6=0.27 → fail.
                 vector=0.40, keyword=0.35 → combined=0.40*0.4 + 0.35*0.6=0.37 → pass.
    """
    citations: list[dict] = []
    for item in artifacts:
        page = item.get("page_number", 0)
        content = item.get("content", "")
        vector_score = item.get("score", 0.0)
        if not content or len(content) < 10:
            continue

        # Gate 1 — vector similarity
        if vector_score < _MIN_VECTOR_SCORE:
            logger.debug(f"cite_retrieval SKIP p{page}: vector={vector_score:.3f}")
            continue

        # Gate 2 — keyword overlap with the answer
        kw_score, kw_hits = matcher.keyword_relevance(content)
        if kw_score < _MIN_KEYWORD_SCORE or kw_hits < _MIN_KEYWORD_HITS:
            logger.debug(f"cite_retrieval SKIP p{page}: kw={kw_score:.3f} hits={kw_hits}")
            continue

        # Gate 3 — blended score
        combined = vector_score * 0.4 + kw_score * 0.6
        if combined < _MIN_COMBINED_SCORE:
            logger.debug(f"cite_retrieval SKIP p{page}: combined={combined:.3f}")
            continue

        logger.debug(
            f"cite_retrieval KEEP p{page}: vec={vector_score:.3f} "
            f"kw={kw_score:.3f}({kw_hits}) comb={combined:.3f}"
        )
        citations.append({
            "page": page,
            "text": matcher.best_snippet(content),
            "source": "document_text",
            "_score": combined,
        })
    # Sort descending so higher-quality chunks win page-level dedup
    citations.sort(key=lambda c: -c["_score"])
    return citations


async def _cite_table(artifact: dict) -> list[dict]:
    """Build a citation from a ``run_sql_query`` artifact.

    Reads ``table_name``, ``table_description``, ``page_number``, and
    ``is_discovery`` — all pre-resolved by the SQL tool.
    If ``page_number`` is missing, falls back to querying ``document_tables``.
    """
    if artifact.get("is_discovery"):
        return []

    table_name = artifact.get("table_name")
    page = artifact.get("page_number")
    row_count = artifact.get("row_count", 0)

    if not table_name and not page:
        return []

    # Fallback: fetch page_number from document_tables if not in artifact
    if not page and table_name:
        try:
            db = get_db_session()
            page = await get_table_page_number(db, table_name)
            logger.debug(f"Looked up page for table '{table_name}': {page}")
        except Exception as e:
            logger.warning(f"Failed to look up page for table '{table_name}': {e}")

    desc = f"Table: {table_name}" if table_name else "SQL query result"
    if row_count:
        desc += f" ({row_count} rows)"
    table_desc = artifact.get("table_description")
    if table_desc:
        desc += f"\n{table_desc[:200]}"

    return [{
        "page": page or 0,
        "text": desc,
        "source": "table",
        "table_name": table_name,
    }]


def _cite_calculation(tool_name: str, artifact: dict, content: str) -> dict | None:
    """Build a citation from a math tool artifact."""
    if "error" in content.lower()[:50]:
        return None

    if tool_name == "calculate_cagr":
        desc = (
            f"CAGR calculation: start={artifact.get('start_value')}, "
            f"end={artifact.get('end_value')}, periods={artifact.get('years')}"
        )
    elif tool_name == "calculate_percentage":
        desc = f"Percentage: {artifact.get('value')} of {artifact.get('total')}"
    elif tool_name == "calculate_percentage_change":
        desc = f"Change: from {artifact.get('old_value')} to {artifact.get('new_value')}"
    elif tool_name == "calculate_arithmetic":
        desc = f"Arithmetic: {artifact.get('expression', 'calculation')}"
    else:
        desc = f"Calculation: {tool_name}"

    desc += f"\nResult: {content.strip()[:200]}"
    return {"page": 0, "text": desc, "source": "calculation"}


# ── Main entry point ─────────────────────────────────────────────────────────


async def extract_citations(messages: list, final_answer: str) -> list[dict]:
    """Extract up to 5 citations from tool-message artifacts.

    Steps:
        1. Tokenise the answer once (``AnswerMatcher``).
        2. Walk every ``ToolMessage`` and dispatch to the right builder.
        3. Deduplicate, prioritise pages mentioned in the answer, cap at 5.

    Returns:
        ``[{"page": int, "text": str, "source": str, "table_name"?: str}, …]``
    """
    matcher = AnswerMatcher.from_text(final_answer)

    # Pages the LLM explicitly referenced (e.g. "Page 27")
    answer_pages: set[int] = {
        int(m.group(1)) for m in re.finditer(r"[Pp]age\s+(\d+)", final_answer)
    }

    # Collect raw citations from every tool call
    evidence: list[dict] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        artifact = getattr(msg, "artifact", None)
        content = str(msg.content) if msg.content else ""

        if msg.name == "retrieve_documents" and isinstance(artifact, list):
            evidence.extend(_cite_retrieval(artifact, matcher))
        elif msg.name == "run_sql_query" and isinstance(artifact, dict):
            evidence.extend(await _cite_table(artifact))
        elif msg.name in _MATH_TOOLS and isinstance(artifact, dict):
            cit = _cite_calculation(msg.name, artifact, content)
            if cit:
                evidence.append(cit)

    # ── Deduplicate & prioritise ─────────────────────────────────────────
    seen: set[str] = set()
    prioritised: list[dict] = []    # pages cited in the answer text
    calculations: list[dict] = []   # math results (page 0)
    secondary: list[dict] = []      # everything else

    for ev in evidence:
        key = f"{ev['page']}:{ev['source']}:{ev.get('table_name', '')}"
        if key in seen:
            continue
        seen.add(key)

        page = ev["page"]
        if page in answer_pages:
            prioritised.append(ev)
        elif page == 0 and ev["source"] in ("calculation", "table"):
            calculations.append(ev)
        elif page > 0:
            secondary.append(ev)

    sort_key = lambda c: (-c.get("_score", 0), c["page"])  # noqa: E731
    prioritised.sort(key=sort_key)
    calculations.sort(key=sort_key)
    secondary.sort(key=sort_key)

    final = (prioritised + calculations + secondary)[:5]
    for c in final:
        c.pop("_score", None)
    return final
