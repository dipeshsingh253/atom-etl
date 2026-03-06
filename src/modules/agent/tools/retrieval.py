"""Retrieval tool — searches the vector database for relevant text passages."""

from typing import Optional

from langchain_core.tools import tool

from src.vectorstore import qdrant


@tool(response_format="content_and_artifact")
async def retrieve_documents(
    query: str, document_id: Optional[str] = None,
) -> tuple[str, list[dict]]:
    """Search the knowledge base for relevant text passages from ingested documents.

    Use this tool when you need to find narrative text, explanations, methodology
    descriptions, or contextual information from the document.

    Args:
        query: The search query describing what information you need.
        document_id: Optional document ID to restrict search to a specific document.

    Returns:
        Formatted text with relevant passages, page numbers, and relevance scores.
    """
    results = await qdrant.search(
        query=query,
        document_id=document_id,
        top_k=5,
    )

    if not results:
        return "No relevant passages found in the knowledge base.", []

    formatted_parts: list[str] = []
    artifacts: list[dict] = []

    for i, result in enumerate(results, 1):
        page = result.get("page_number", 0)
        section = result.get("section", "")
        score = result.get("score", 0)
        content = result.get("content", "")

        section_info = f" (Section: {section})" if section else ""
        formatted_parts.append(
            f"[Result {i}] Page {page}{section_info} (relevance: {score:.3f}):\n{content}"
        )
        artifacts.append({
            "page_number": page if isinstance(page, int) else 0,
            "content": content,
            "score": score,
            "section": section,
        })

    return "\n\n---\n\n".join(formatted_parts), artifacts
