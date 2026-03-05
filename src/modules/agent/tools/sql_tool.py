"""SQL tool — queries structured data from PostgreSQL (tables and visual data)."""

from langchain_core.tools import tool
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Module-level session holder — set by the graph before invocation
_db_session: AsyncSession | None = None

# SQL schema description for the LLM
SQL_SCHEMA_DESCRIPTION = """
Available SQL tables for querying structured data extracted from PDF documents:

1. document_tables — Metadata about extracted tables
   Columns: id, document_id, table_name, page_number, table_description, created_at

2. table_rows — Individual cell values from tables (Entity-Attribute-Value format)
   Columns: id, table_id, row_index, column_name, value, page_number
   Note: Each row in the original table is split into multiple records, one per column.
   To reconstruct a table row, GROUP BY table_id and row_index.

3. document_visuals — Metadata about extracted charts/graphs/heatmaps
   Columns: id, document_id, visual_type, title, page_number, created_at

4. visual_data — Data points extracted from charts/graphs
   Columns: id, visual_id, label, value, extra_metadata, page_number

Example queries:
- Find all values in a specific table:
  SELECT tr.row_index, tr.column_name, tr.value
  FROM table_rows tr
  JOIN document_tables dt ON tr.table_id = dt.id
  WHERE dt.table_name LIKE '%keyword%'
  ORDER BY tr.row_index, tr.column_name

- Find chart data:
  SELECT vd.label, vd.value
  FROM visual_data vd
  JOIN document_visuals dv ON vd.visual_id = dv.id
  WHERE dv.title LIKE '%keyword%'

IMPORTANT: Only SELECT queries are allowed. No INSERT, UPDATE, DELETE, DROP, or ALTER.
"""


def set_db_session(session: AsyncSession) -> None:
    """Set the database session for SQL tool execution."""
    global _db_session
    _db_session = session


def get_db_session() -> AsyncSession:
    """Get the current database session."""
    if _db_session is None:
        raise RuntimeError("Database session not set. Call set_db_session() first.")
    return _db_session


@tool
async def run_sql_query(sql_query: str) -> str:
    """Execute a read-only SQL query against the structured data in PostgreSQL.

    Use this tool when you need to retrieve numerical values, table data, chart data,
    or any structured information extracted from the document.

    The database contains:
    - document_tables + table_rows: Data from extracted PDF tables
    - document_visuals + visual_data: Data from extracted charts/graphs/heatmaps

    Args:
        sql_query: A SELECT SQL query. Only read-only queries are allowed.

    Returns:
        Query results formatted as text, or an error message.
    """
    # Safety check — reject non-SELECT queries
    normalized = sql_query.strip().upper()
    if not normalized.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed. Please provide a read-only query."

    dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
    for keyword in dangerous_keywords:
        if keyword in normalized:
            return f"Error: '{keyword}' operations are not allowed. Only SELECT queries are permitted."

    db = get_db_session()

    try:
        result = await db.execute(text(sql_query))
        rows = result.fetchall()
        columns = list(result.keys())

        if not rows:
            return "Query returned no results."

        # Format as readable text
        lines: list[str] = []
        lines.append(f"Columns: {', '.join(columns)}")
        lines.append(f"Rows returned: {len(rows)}")
        lines.append("")

        for row in rows[:50]:  # Limit to 50 rows
            row_data = {col: str(val) for col, val in zip(columns, row)}
            lines.append(str(row_data))

        if len(rows) > 50:
            lines.append(f"... and {len(rows) - 50} more rows")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"SQL query failed: {e}")
        return f"SQL query error: {str(e)}"
