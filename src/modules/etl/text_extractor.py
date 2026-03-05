"""Extract text content from PDF documents page-by-page using PyMuPDF."""

import re

import fitz  # PyMuPDF
from loguru import logger


def extract_text(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF document.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts with keys: content, page_number.
    """
    pages: list[dict] = []

    try:
        doc = fitz.open(pdf_path)
        logger.info(f"Opened PDF: {pdf_path} ({doc.page_count} pages)")

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text("text")  # type: ignore[arg-type]

            # Clean the extracted text
            cleaned = _clean_text(text)

            if cleaned.strip():
                pages.append(
                    {
                        "content": cleaned,
                        "page_number": page_num + 1,  # 1-indexed
                    }
                )

        doc.close()
        logger.info(f"Extracted text from {len(pages)} pages")

    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        raise

    return pages


def _clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove excessive newlines but keep paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def get_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF."""
    doc = fitz.open(pdf_path)
    count = doc.page_count
    doc.close()
    return count
