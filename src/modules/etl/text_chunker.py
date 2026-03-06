"""Split extracted text into overlapping chunks suitable for vector search."""

import re

import tiktoken
from loguru import logger


_ENCODER = tiktoken.get_encoding("cl100k_base")

# TODO: Make these configurable via .env 
DEFAULT_CHUNK_SIZE = 800  # tokens
DEFAULT_CHUNK_OVERLAP = 150  # tokens


def chunk_pages(
    pages: list[dict],
    document_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Split page texts into overlapping token-based chunks.

    Args:
        pages: List of dicts with keys: content, page_number.
        document_id: Document ID for metadata.
        chunk_size: Maximum chunk size in tokens.
        chunk_overlap: Overlap between consecutive chunks in tokens.

    Returns:
        List of chunk dicts with keys: content, page_number, section, document_id.
    """
    chunks: list[dict] = []

    for page in pages:
        content = page["content"]
        page_number = page["page_number"]
        section = _detect_section(content)

        tokens = _ENCODER.encode(content)

        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = _ENCODER.decode(chunk_tokens)

            chunks.append(
                {
                    "content": chunk_text,
                    "page_number": page_number,
                    "section": section,
                    "document_id": document_id,
                }
            )

            if end >= len(tokens):
                break
            start += chunk_size - chunk_overlap

    logger.info(
        f"Created {len(chunks)} chunks from {len(pages)} pages "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks

# TODO: This is a very basic implementation as of now. We can improve it by looking for specific patterns.
def _detect_section(text: str) -> str:
    """Attempt to detect a section heading from the beginning of the text.

    Looks for common heading patterns (numbered sections, all-caps lines, etc.)
    Returns the detected section title or empty string.
    """
    lines = text.strip().split("\n")
    if not lines:
        return ""

    first_line = lines[0].strip()

    # Numbered heading: "1. Introduction", "2.3 Methodology"
    if re.match(r"^\d+(\.\d+)*\.?\s+\w", first_line) and len(first_line) < 100:
        return first_line

    # All-caps heading
    if first_line.isupper() and len(first_line) < 100 and len(first_line) > 3:
        return first_line

    # Title-case short line
    if first_line.istitle() and len(first_line) < 80:
        return first_line

    return ""
