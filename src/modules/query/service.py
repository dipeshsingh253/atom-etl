from typing import Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.agent.graph import run_agent


async def process_query(
    db: AsyncSession,
    query: str,
    document_id: Optional[str] = None,
) -> dict:
    """Process a user query through the LangGraph agent.

    Args:
        db: Async database session.
        query: The user's question.
        document_id: Optional document ID to scope the query.

    Returns:
        Dict with 'answer' (str) and 'citations' (list[dict]).
    """
    logger.info(f"Processing query: {query[:100]}...")

    result = await run_agent(
        query=query,
        db=db,
        document_id=document_id,
    )

    logger.info(
        f"Query processed. Answer length: {len(result.get('answer', ''))}, "
        f"Citations: {len(result.get('citations', []))}"
    )

    return result
