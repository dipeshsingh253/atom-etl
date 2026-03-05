from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.utils import success_response
from src.db.session import get_db
from src.modules.query.evaluator import compute_confidence
from src.modules.query.schemas import Citation, QueryRequest, QueryResponse
from src.modules.query.service import process_query

router = APIRouter(prefix="/query", tags=["query"])


@router.post("")
async def query_knowledge_base(
    request: Request,
    body: QueryRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Ask a question about ingested documents.

    The agent uses a combination of:
    - Vector search (for narrative text)
    - SQL queries (for structured table/chart data)
    - Math tools (for calculations)

    to generate a grounded answer with citations.
    """
    result = await process_query(
        db=db,
        query=body.query,
        document_id=body.document_id,
    )

    confidence, confidence_reason = await compute_confidence(
        result=result,
        query=body.query,
    )

    response_data = QueryResponse(
        answer=result["answer"],
        citations=[
            Citation(
                page=c["page"],
                text=c["text"],
                source=c.get("source", "document_text"),
                table_name=c.get("table_name"),
            )
            for c in result.get("citations", [])
        ],
        confidence=confidence,
        confidence_reason=confidence_reason,
        execution_trace=result.get("execution_trace", []),
    )

    return success_response(
        data=response_data.model_dump(),
        message="Query processed successfully",
        request=request,
    )
