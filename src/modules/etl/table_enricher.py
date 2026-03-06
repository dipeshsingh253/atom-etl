"""Enrich extracted tables with LLM-generated names, column labels, and descriptions.

Sends each table's raw headers and a few sample rows to gpt-4o-mini to produce
human-readable metadata that improves both agent SQL accuracy and citation quality.
"""

import json

from loguru import logger
from openai import AsyncOpenAI

from src.core.config import get_settings

_ENRICHMENT_PROMPT = """\
You are a data-labelling assistant. Given a table extracted from a PDF, produce:

1. **table_name** — a short, descriptive name (3-8 words, title case).
2. **columns** — a JSON object mapping each original column header to a clean,
   descriptive column name (keep it concise, use snake_case).
3. **description** — one sentence describing what the table contains.

Rules:
- Do NOT invent data. Base everything on the headers and sample rows provided.
- If a header is already clean and descriptive, keep it unchanged in the mapping.
- Placeholder headers like "column_0" should be renamed based on the data.
- Return ONLY valid JSON, no markdown fences, no extra text.

Response format:
{"table_name": "...", "columns": {"original_header": "clean_name", ...}, "description": "..."}
"""


async def enrich_tables(tables: list[dict]) -> list[dict]:
    """Enrich a batch of extracted tables with LLM-generated metadata.

    For each table, sends headers + up to 3 sample rows to gpt-4o-mini.
    Overwrites ``table_name``, ``table_description``, ``headers``, and
    remaps row dict keys to the cleaned column names.

    Falls back to original values on any per-table LLM failure.

    Args:
        tables: List of table dicts from ``extract_tables``.

    Returns:
        The same list, mutated in place with enriched metadata.
    """
    if not tables:
        return tables

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    model = settings.openai_mini_model

    for table in tables:
        try:
            enrichment = await _enrich_single(client, model, table)
            _apply_enrichment(table, enrichment)
        except Exception as e:
            logger.warning(
                f"Table enrichment failed for '{table.get('table_name')}' "
                f"on page {table.get('page_number')}: {e}"
            )
            # Keep original values — ingestion continues
            if "table_description" not in table:
                table["table_description"] = f"Headers: {', '.join(table['headers'])}"

    return tables


async def _enrich_single(
    client: AsyncOpenAI, model: str, table: dict,
) -> dict:
    """Call gpt-4o-mini for a single table and return parsed enrichment JSON."""
    headers = table["headers"]
    sample_rows = table["rows"][:3]

    user_message = (
        f"Page: {table['page_number']}\n"
        f"Headers: {json.dumps(headers)}\n"
        f"Sample rows ({len(sample_rows)}):\n"
    )
    for row in sample_rows:
        user_message += json.dumps(row, default=str) + "\n"

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _ENRICHMENT_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=512,
    )

    raw = response.choices[0].message.content or ""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    return json.loads(text)


def _apply_enrichment(table: dict, enrichment: dict) -> None:
    """Apply LLM enrichment to a table dict in place."""
    # Update table name
    new_name = enrichment.get("table_name", "").strip()
    if new_name:
        table["table_name"] = new_name

    # Update description
    desc = enrichment.get("description", "").strip()
    table["table_description"] = desc if desc else f"Headers: {', '.join(table['headers'])}"

    # Remap column names
    col_mapping = enrichment.get("columns", {})
    if not col_mapping:
        return

    old_headers = table["headers"]
    new_headers = [col_mapping.get(h, h) for h in old_headers]
    table["headers"] = new_headers

    # Remap row dict keys
    new_rows: list[dict] = []
    for row in table["rows"]:
        new_row = {}
        for old_key, value in row.items():
            new_key = col_mapping.get(old_key, old_key)
            new_row[new_key] = value
        new_rows.append(new_row)
    table["rows"] = new_rows

    logger.debug(
        f"Enriched table: '{table['table_name']}' — "
        f"{len(new_headers)} columns remapped"
    )
