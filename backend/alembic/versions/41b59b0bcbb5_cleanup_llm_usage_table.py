"""cleanup_llm_usage_table

Revision ID: 41b59b0bcbb5
Revises: 87fa73a56b06
Create Date: 2026-10-13 12:11:20.610306

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '41b59b0bcbb5'
down_revision: Union[str, Sequence[str], None] = '87fa73a56b06'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Alter llm_usage table: drop unused columns and add cached token tracking."""
    # Drop unnecessary columns
    op.drop_column('llm_usage', 'pricing_metadata')
    op.drop_column('llm_usage', 'pricing_cached_rate')
    op.drop_column('llm_usage', 'pricing_output_rate')
    op.drop_column('llm_usage', 'pricing_input_rate')
    op.drop_column('llm_usage', 'error_message')
    op.drop_column('llm_usage', 'status')
    op.drop_column('llm_usage', 'total_tokens')

    # Add cached-token tracking columns
    op.add_column(
        'llm_usage',
        sa.Column(
            'cached_input_tokens',
            sa.Integer(),
            nullable=True,
            comment='Number of cached input tokens reported by the API.',
        ),
    )
    op.add_column(
        'llm_usage',
        sa.Column(
            'cached_input_cost',
            sa.Numeric(precision=12, scale=6),
            nullable=True,
            comment='Calculated cached-input cost in USD.',
        ),
    )


def downgrade() -> None:
    """Downgrade schema: drop cached token columns and restore removed columns."""
    # Drop new columns
    op.drop_column('llm_usage', 'cached_input_cost')
    op.drop_column('llm_usage', 'cached_input_tokens')

    # Restore removed columns with original constraints
    op.add_column(
        'llm_usage',
        sa.Column(
            'total_tokens',
            sa.Integer(),
            nullable=True,
            comment='Total tokens consumed.',
        ),
    )
    op.add_column(
        'llm_usage',
        sa.Column(
            'status',
            sa.String(length=50),
            server_default='success',
            nullable=False,
            comment='Execution status. Expected: success, failed.',
        ),
    )
    op.add_column(
        'llm_usage',
        sa.Column(
            'error_message',
            sa.Text(),
            nullable=True,
            comment='Error message if the LLM call failed.',
        ),
    )
    op.add_column(
        'llm_usage',
        sa.Column(
            'pricing_input_rate',
            sa.Numeric(precision=12, scale=6),
            nullable=True,
            comment='Price per 1M input tokens in USD.',
        ),
    )
    op.add_column(
        'llm_usage',
        sa.Column(
            'pricing_output_rate',
            sa.Numeric(precision=12, scale=6),
            nullable=True,
            comment='Price per 1M output tokens in USD.',
        ),
    )
    op.add_column(
        'llm_usage',
        sa.Column(
            'pricing_cached_rate',
            sa.Numeric(precision=12, scale=6),
            nullable=True,
            comment='Price per 1M cached tokens in USD.',
        ),
    )
    op.add_column(
        'llm_usage',
        sa.Column(
            'pricing_metadata',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment='Raw token pricing rate metadata snapshot.',
        ),
    )
