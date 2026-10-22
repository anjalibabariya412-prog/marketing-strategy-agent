"""add_status_and_attempt_tracking_to_llm_usage

Revision ID: 7c49a33be608
Revises: c0be158afac5
Create Date: 2026-10-21 10:07:34.767475

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7c49a33be608'
down_revision: Union[str, Sequence[str], None] = 'c0be158afac5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add status, attempt_number, and error_message columns to llm_usage table."""
    op.add_column(
        'llm_usage',
        sa.Column(
            'status',
            sa.String(length=50),
            server_default='success',
            nullable=False,
            comment='Execution outcome status (e.g. success, failed).',
        ),
    )
    op.create_index(op.f('ix_llm_usage_status'), 'llm_usage', ['status'], unique=False)
    op.add_column(
        'llm_usage',
        sa.Column(
            'attempt_number',
            sa.Integer(),
            server_default='1',
            nullable=False,
            comment='Attempt sequence number for this LLM invocation.',
        ),
    )
    op.add_column(
        'llm_usage',
        sa.Column(
            'error_message',
            sa.Text(),
            nullable=True,
            comment='Safely sanitized error description or category if the call or attempt failed.',
        ),
    )


def downgrade() -> None:
    """Drop error_message, attempt_number, and status columns from llm_usage table."""
    op.drop_column('llm_usage', 'error_message')
    op.drop_column('llm_usage', 'attempt_number')
    op.drop_index(op.f('ix_llm_usage_status'), table_name='llm_usage')
    op.drop_column('llm_usage', 'status')
