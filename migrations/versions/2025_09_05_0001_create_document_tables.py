"""create document tables

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2025-09-05 00:01:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Documents table
    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, index=True),
        sa.Column("total_pages", sa.Integer(), nullable=True),
        sa.Column("source_document", sa.String(500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Document tables (extracted table metadata)
    op.create_table(
        "document_tables",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("table_name", sa.String(500), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("table_description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Table rows (EAV format for extracted table data)
    op.create_table(
        "table_rows",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "table_id",
            sa.String(36),
            sa.ForeignKey("document_tables.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("row_index", sa.Integer(), nullable=False),
        sa.Column("column_name", sa.String(500), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Document visuals (extracted images/charts)
    op.create_table(
        "document_visuals",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("visual_type", sa.String(100), nullable=True),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Visual data (structured data extracted from visuals)
    op.create_table(
        "visual_data",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "visual_id",
            sa.String(36),
            sa.ForeignKey("document_visuals.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("label", sa.String(500), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("extra_metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("visual_data")
    op.drop_table("document_visuals")
    op.drop_table("table_rows")
    op.drop_table("document_tables")
    op.drop_table("documents")
