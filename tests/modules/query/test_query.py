"""Tests for the query endpoint."""

from unittest.mock import AsyncMock, patch

import pytest


class TestQueryEndpoint:
    """Test cases for POST /api/v1/query."""

    @staticmethod
    async def _create_document(client) -> str:
        pdf_content = b"%PDF-1.4 minimal test content"
        files = {"file": ("query_test.pdf", pdf_content, "application/pdf")}
        response = await client.post("/api/v1/documents/ingest", files=files)
        assert response.status_code == 201
        return response.json()["data"]["document_id"]

    @pytest.mark.asyncio
    async def test_query_success(self, client):
        """Test that a valid query returns a response with answer and citations."""
        document_id = await self._create_document(client)

        mock_result = {
            "answer": "Ireland's cyber security sector generated approximately €1.1bn in GVA in 2021.",
            "citations": [
                {"page": 36, "text": "Page 36: cyber security sector generated approximately €1.1bn"}
            ],
        }

        with patch(
            "src.modules.query.service.run_agent",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = await client.post(
                "/api/v1/query",
                json={
                    "query": "What is the total GVA of the cyber security sector?",
                    "document_id": document_id,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "answer" in data["data"]
        assert "citations" in data["data"]
        assert len(data["data"]["citations"]) > 0
        assert data["data"]["citations"][0]["page"] == 36

    @pytest.mark.asyncio
    async def test_query_with_document_id(self, client):
        """Test query scoped to a specific document."""
        document_id = await self._create_document(client)

        mock_result = {
            "answer": "Test answer.",
            "citations": [],
        }

        with patch(
            "src.modules.query.service.run_agent",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = await client.post(
                "/api/v1/query",
                json={
                    "query": "Test question",
                    "document_id": document_id,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_query_empty_rejected(self, client):
        """Test that an empty query is rejected."""
        document_id = await self._create_document(client)

        response = await client.post(
            "/api/v1/query",
            json={"query": "", "document_id": document_id},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_document_not_found(self, client):
        """Test query fails fast when document does not exist."""
        response = await client.post(
            "/api/v1/query",
            json={
                "query": "Test question",
                "document_id": "missing-document-id",
            },
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_query_missing_body_rejected(self, client):
        """Test that missing request body is rejected."""
        response = await client.post("/api/v1/query")
        assert response.status_code == 422
