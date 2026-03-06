from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.utils import success_response
from src.db.session import get_db
from src.modules.documents import service as documents_service
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
    document = await documents_service.get_document(db, body.document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    result = await process_query(
        db=db,
        query=body.query,
        document_id=body.document_id,
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
        langsmith_run_id=result.get("langsmith_run_id"),
    )

    return success_response(
        data=response_data.model_dump(),
        message="Query processed successfully",
        request=request,
    )
