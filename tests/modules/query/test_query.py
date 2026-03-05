"""Tests for the query endpoint."""

from unittest.mock import AsyncMock, patch

import pytest


class TestQueryEndpoint:
    """Test cases for POST /api/v1/query."""

    @pytest.mark.asyncio
    async def test_query_success(self, client):
        """Test that a valid query returns a response with answer and citations."""
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
                json={"query": "What is the total GVA of the cyber security sector?"},
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
                    "document_id": "some-doc-id",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_query_empty_rejected(self, client):
        """Test that an empty query is rejected."""
        response = await client.post(
            "/api/v1/query",
            json={"query": ""},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_missing_body_rejected(self, client):
        """Test that missing request body is rejected."""
        response = await client.post("/api/v1/query")
        assert response.status_code == 422
