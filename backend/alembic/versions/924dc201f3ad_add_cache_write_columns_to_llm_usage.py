"""add_cache_write_columns_to_llm_usage

Revision ID: 924dc201f3ad
Revises: 41b59b0bcbb5
Create Date: 2026-10-14 10:08:12.782098

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '924dc201f3ad'
down_revision: Union[str, Sequence[str], None] = '41b59b0bcbb5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add cache_write_tokens and cache_write_cost columns to llm_usage table."""
    op.add_column(
        'llm_usage',
        sa.Column(
            'cache_write_tokens',
            sa.Integer(),
            nullable=True,
            comment='Number of cache-write prompt tokens reported by the API.',
        ),
    )
    op.add_column(
        'llm_usage',
        sa.Column(
            'cache_write_cost',
            sa.Numeric(precision=12, scale=6),
            nullable=True,
            comment='Calculated cache-write cost in USD.',
        ),
    )


def downgrade() -> None:
    """Drop cache_write_tokens and cache_write_cost columns from llm_usage table."""
    op.drop_column('llm_usage', 'cache_write_cost')
    op.drop_column('llm_usage', 'cache_write_tokens')

