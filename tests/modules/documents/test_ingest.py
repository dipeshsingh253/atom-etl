"""Tests for the document ingestion endpoints."""

import io

import pytest


class TestDocumentIngest:
    """Test cases for POST /api/v1/documents/ingest."""

    @pytest.mark.asyncio
    async def test_ingest_pdf_success(self, client):
        """Test successful PDF upload returns 201 with document_id."""
        # Create a minimal PDF-like file
        pdf_content = b"%PDF-1.4 minimal test content"
        files = {"file": ("test_document.pdf", io.BytesIO(pdf_content), "application/pdf")}

        response = await client.post("/api/v1/documents/ingest", files=files)

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["status"] == "ingestion_started"
        assert "document_id" in data["data"]

    @pytest.mark.asyncio
    async def test_ingest_non_pdf_rejected(self, client):
        """Test that non-PDF files are rejected with 400."""
        files = {"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")}

        response = await client.post("/api/v1/documents/ingest", files=files)

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_ingest_no_file_rejected(self, client):
        """Test that request without file is rejected."""
        response = await client.post("/api/v1/documents/ingest")
        assert response.status_code == 422


class TestDocumentStatus:
    """Test cases for GET /api/v1/documents/{document_id}/status."""

    @pytest.mark.asyncio
    async def test_get_status_after_ingest(self, client):
        """Test that we can retrieve the status after ingestion."""
        pdf_content = b"%PDF-1.4 minimal test content"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}

        # Ingest first
        ingest_response = await client.post("/api/v1/documents/ingest", files=files)
        assert ingest_response.status_code == 201
        document_id = ingest_response.json()["data"]["document_id"]

        # Check status
        status_response = await client.get(f"/api/v1/documents/{document_id}/status")
        assert status_response.status_code == 200
        data = status_response.json()
        assert data["success"] is True
        assert data["data"]["document_id"] == document_id
        assert data["data"]["status"] == "pending"
        assert data["data"]["filename"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_get_status_not_found(self, client):
        """Test 404 for non-existent document."""
        response = await client.get("/api/v1/documents/nonexistent-id/status")
        assert response.status_code == 404
