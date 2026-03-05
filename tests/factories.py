"""Test data factories for generating test fixtures."""

from typing import Any, Dict

from faker import Faker

fake = Faker()


class DocumentFactory:
    """Factory for generating document test data."""

    @staticmethod
    def create_document_payload(**overrides: Any) -> Dict[str, Any]:
        return {
            "filename": f"{fake.word()}_{fake.random_int()}.pdf",
            **overrides,
        }


class ChunkFactory:
    """Factory for generating text chunk test data."""

    @staticmethod
    def create_chunk(document_id: str = "test-doc-id", **overrides: Any) -> Dict[str, Any]:
        return {
            "content": fake.paragraph(nb_sentences=5),
            "page_number": fake.random_int(min=1, max=100),
            "section": fake.sentence(nb_words=4),
            "document_id": document_id,
            **overrides,
        }

    @staticmethod
    def create_chunks(
        count: int = 5, document_id: str = "test-doc-id"
    ) -> list[Dict[str, Any]]:
        return [ChunkFactory.create_chunk(document_id=document_id) for _ in range(count)]


class TableRowFactory:
    """Factory for generating table row test data."""

    @staticmethod
    def create_table_data(**overrides: Any) -> Dict[str, Any]:
        return {
            "table_name": fake.sentence(nb_words=3),
            "page_number": fake.random_int(min=1, max=100),
            "headers": ["Region", "Count", "Percentage"],
            "rows": [
                {"Region": fake.city(), "Count": str(fake.random_int()), "Percentage": f"{fake.random_int(max=100)}%"}
                for _ in range(5)
            ],
            **overrides,
        }
