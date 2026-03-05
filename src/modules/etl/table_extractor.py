"""Extract tables from PDF documents using pdfplumber."""

from typing import Any

import pdfplumber
from loguru import logger


def extract_tables(pdf_path: str) -> list[dict[str, Any]]:
    """Extract all tables from a PDF document.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts, each representing a table with keys:
            - table_name: str (auto-generated or inferred)
            - page_number: int
            - headers: list[str]
            - rows: list[dict[str, str]] (each row is a dict of column_name -> value)
    """
    tables: list[dict[str, Any]] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_tables = page.extract_tables()

                if not page_tables:
                    continue

                for table_idx, raw_table in enumerate(page_tables):
                    if not raw_table or len(raw_table) < 2:
                        # Skip tables with no data rows
                        continue

                    parsed = _parse_table(raw_table, page_num, table_idx)
                    if parsed:
                        tables.append(parsed)

        logger.info(f"Extracted {len(tables)} tables from {pdf_path}")

    except Exception as e:
        logger.error(f"Failed to extract tables from {pdf_path}: {e}")
        raise

    return tables


def _parse_table(
    raw_table: list[list], page_number: int, table_index: int
) -> dict[str, Any] | None:
    """Parse a raw pdfplumber table into structured format.

    Args:
        raw_table: 2D list from pdfplumber.extract_tables().
        page_number: Source page number.
        table_index: Index of the table on the page.

    Returns:
        Structured table dict or None if table is invalid.
    """
    # First row is typically the header
    raw_headers = raw_table[0]
    headers = _clean_headers(raw_headers)

    if not headers:
        return None

    # Parse data rows
    rows: list[dict[str, str]] = []
    for row_idx, raw_row in enumerate(raw_table[1:]):
        row_dict: dict[str, str] = {}
        for col_idx, cell in enumerate(raw_row):
            if col_idx < len(headers):
                col_name = headers[col_idx]
                value = _clean_cell(cell)
                row_dict[col_name] = value
        if any(v for v in row_dict.values()):  # Skip fully empty rows
            rows.append(row_dict)

    if not rows:
        return None

    table_name = _infer_table_name(headers, page_number, table_index)

    return {
        "table_name": table_name,
        "page_number": page_number,
        "headers": headers,
        "rows": rows,
    }


def _clean_headers(raw_headers: list) -> list[str]:
    """Clean and normalize table headers."""
    headers: list[str] = []
    for i, h in enumerate(raw_headers):
        if h is None or str(h).strip() == "":
            headers.append(f"column_{i}")
        else:
            cleaned = str(h).strip().replace("\n", " ")
            headers.append(cleaned)
    return headers


def _clean_cell(cell) -> str:
    """Clean a single table cell value."""
    if cell is None:
        return ""
    return str(cell).strip().replace("\n", " ")

# TODO: The table name inference is very basic. We can improve it by looking for specific patterns in the headers or even the surrounding text on the page. Also if required we can use LLM to generate a more descriptive name based on the content of the table and surrounding text.
def _infer_table_name(
    headers: list[str], page_number: int, table_index: int
) -> str:
    """Generate a descriptive table name from headers."""
    meaningful_headers = [h for h in headers if not h.startswith("column_")]
    if meaningful_headers:
        name_parts = meaningful_headers[:3]
        return " | ".join(name_parts)
    return f"Table (page {page_number}, #{table_index + 1})"
