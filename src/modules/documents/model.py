from sqlalchemy import ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import BaseModel


class Document(BaseModel):
    """Represents an uploaded PDF document."""

    __tablename__ = "documents"

    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending", index=True
    )
    total_pages: Mapped[int] = mapped_column(Integer, nullable=True)
    source_document: Mapped[str] = mapped_column(String(500), nullable=True)

    # Relationships
    tables: Mapped[list["DocumentTable"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    visuals: Mapped[list["DocumentVisual"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"


class DocumentTable(BaseModel):
    """Metadata for a table extracted from a document."""

    __tablename__ = "document_tables"

    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    table_name: Mapped[str] = mapped_column(String(500), nullable=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    table_description: Mapped[str] = mapped_column(Text, nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="tables")
    rows: Mapped[list["TableRow"]] = relationship(
        back_populates="table", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<DocumentTable(id={self.id}, name={self.table_name}, page={self.page_number})>"


class TableRow(BaseModel):
    """A single cell value from an extracted table (EAV format)."""

    __tablename__ = "table_rows"

    table_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document_tables.id", ondelete="CASCADE"), nullable=False, index=True
    )
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    column_name: Mapped[str] = mapped_column(String(500), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=True)

    # Relationships
    table: Mapped["DocumentTable"] = relationship(back_populates="rows")

    def __repr__(self) -> str:
        return (
            f"<TableRow(table_id={self.table_id}, row={self.row_index}, "
            f"col={self.column_name}, val={self.value})>"
        )


class DocumentVisual(BaseModel):
    """Metadata for a chart, graph, or heatmap extracted from a document."""

    __tablename__ = "document_visuals"

    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    visual_type: Mapped[str] = mapped_column(String(100), nullable=True)
    title: Mapped[str] = mapped_column(String(500), nullable=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="visuals")
    data_points: Mapped[list["VisualData"]] = relationship(
        back_populates="visual", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<DocumentVisual(id={self.id}, type={self.visual_type}, page={self.page_number})>"


class VisualData(BaseModel):
    """A single data point extracted from a visual element."""

    __tablename__ = "visual_data"

    visual_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document_visuals.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(500), nullable=True)
    value: Mapped[str] = mapped_column(Text, nullable=True)
    extra_metadata: Mapped[dict] = mapped_column(JSON, nullable=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=True)

    # Relationships
    visual: Mapped["DocumentVisual"] = relationship(back_populates="data_points")

    def __repr__(self) -> str:
        return f"<VisualData(visual_id={self.visual_id}, label={self.label}, value={self.value})>"
