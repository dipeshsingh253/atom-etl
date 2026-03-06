from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for querying the knowledge base."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask about the ingested documents",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional document ID to scope the query to a specific document",
    )


class Citation(BaseModel):
    """A source citation from the document."""

    page: int = Field(..., description="Page number in the source document (0 for calculations)")
    text: str = Field(
        ...,
        description="Searchable text passage, table reference, or calculation description",
    )
    source: str = Field(
        default="document_text",
        description="Citation source type: document_text, table, or calculation",
    )
    table_name: Optional[str] = Field(
        default=None,
        description="Table name when source is 'table'",
    )


class QueryResponse(BaseModel):
    """Response containing the answer, citations, and execution trace."""

    answer: str = Field(..., description="The generated answer to the query")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Source citations from the document",
    )
    langsmith_run_id: Optional[str] = Field(
        default=None,
        description="LangSmith run ID — use this to find the trace in LangSmith",
    )
