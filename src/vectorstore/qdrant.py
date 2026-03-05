import uuid
from typing import Optional

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.core.config import get_settings
from src.providers.factory import get_embedding_provider


def _get_client() -> AsyncQdrantClient:
    settings = get_settings()
    return AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


async def init_collection() -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    settings = get_settings()
    client = _get_client()
    embedding_provider = get_embedding_provider()
    collection_name = settings.qdrant_collection_name

    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]

    if collection_name not in existing_names:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_provider.get_dimensions(),
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            f"Created Qdrant collection '{collection_name}' "
            f"with {embedding_provider.get_dimensions()} dimensions"
        )
    else:
        logger.info(f"Qdrant collection '{collection_name}' already exists")

    await client.close()


async def store_chunks(
    chunks: list[dict],
    document_id: str,
) -> int:
    """Embed and store text chunks in Qdrant.

    Args:
        chunks: List of dicts with keys: content, page_number, section.
        document_id: The source document ID.

    Returns:
        Number of points stored.
    """
    if not chunks:
        return 0

    settings = get_settings()
    embedding_provider = get_embedding_provider()
    client = _get_client()

    texts = [chunk["content"] for chunk in chunks]
    embeddings = await embedding_provider.embed_batch(texts)

    points = []
    for chunk, embedding in zip(chunks, embeddings):
        point_id = str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": chunk["content"],
                    "page_number": chunk.get("page_number"),
                    "section": chunk.get("section", ""),
                    "document_id": document_id,
                },
            )
        )

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        await client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=batch,
        )

    await client.close()
    logger.info(f"Stored {len(points)} chunks for document {document_id}")
    return len(points)


async def search(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 5,
) -> list[dict]:
    """Search for similar text chunks in Qdrant.

    Args:
        query: The search query text.
        document_id: Optional filter to a specific document.
        top_k: Number of results to return.

    Returns:
        List of dicts with content, page_number, section, score.
    """
    settings = get_settings()
    embedding_provider = get_embedding_provider()
    client = _get_client()

    query_embedding = await embedding_provider.embed_text(query)

    query_filter = None
    if document_id:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

    results = await client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=query_embedding,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    await client.close()

    return [
        {
            "content": (point.payload or {}).get("content", ""),
            "page_number": (point.payload or {}).get("page_number"),
            "section": (point.payload or {}).get("section", ""),
            "document_id": (point.payload or {}).get("document_id", ""),
            "score": point.score,
        }
        for point in results.points
    ]


async def delete_document(document_id: str) -> None:
    """Delete all points associated with a document."""
    settings = get_settings()
    client = _get_client()

    await client.delete(
        collection_name=settings.qdrant_collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        ),
    )

    await client.close()
    logger.info(f"Deleted all chunks for document {document_id}")
