"""create llm_usage table

Revision ID: a3dfa9d85a15
Revises: 6eb3b9045c09
Create Date: 2026-10-12 10:35:12.606048

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a3dfa9d85a15'
down_revision: Union[str, Sequence[str], None] = '6eb3b9045c09'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to create llm_usage table and indexes."""
    op.create_table(
        'llm_usage',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('strategy_request_id', sa.UUID(), nullable=True, comment='Optional foreign key to strategy_requests.id.'),
        sa.Column('conversation_id', sa.UUID(), nullable=True, comment='Optional foreign key to conversations.id (thread_id).'),
        sa.Column('operation', sa.String(length=100), nullable=False, comment='Operation name (e.g. relevance_analysis, question_generation, strategy_generation).'),
        sa.Column('provider', sa.String(length=50), server_default='openai', nullable=False, comment='LLM provider name (e.g. openai, groq, anthropic).'),
        sa.Column('model', sa.String(length=100), nullable=False, comment='Name/identifier of LLM model used (e.g. gpt-4o-mini).'),
        sa.Column('input_tokens', sa.Integer(), nullable=True, comment='Number of prompt/input tokens.'),
        sa.Column('output_tokens', sa.Integer(), nullable=True, comment='Number of completion/output tokens.'),
        sa.Column('total_tokens', sa.Integer(), nullable=True, comment='Total tokens consumed.'),
        sa.Column('status', sa.String(length=50), server_default='success', nullable=False, comment='Execution status. Expected: success, failed.'),
        sa.Column('duration_ms', sa.Integer(), nullable=True, comment='Call execution duration in milliseconds.'),
        sa.Column('error_message', sa.Text(), nullable=True, comment='Error message if the LLM call failed.'),
        sa.Column('pricing_input_rate', sa.Numeric(precision=12, scale=6), nullable=True, comment='Price per 1M input tokens in USD.'),
        sa.Column('pricing_output_rate', sa.Numeric(precision=12, scale=6), nullable=True, comment='Price per 1M output tokens in USD.'),
        sa.Column('pricing_cached_rate', sa.Numeric(precision=12, scale=6), nullable=True, comment='Price per 1M cached tokens in USD.'),
        sa.Column('pricing_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment='Raw token pricing rate metadata snapshot.'),
        sa.Column('input_cost', sa.Numeric(precision=12, scale=6), nullable=True, comment='Calculated input cost in USD.'),
        sa.Column('output_cost', sa.Numeric(precision=12, scale=6), nullable=True, comment='Calculated output cost in USD.'),
        sa.Column('total_cost', sa.Numeric(precision=12, scale=6), nullable=True, comment='Total calculated call cost in USD.'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['strategy_request_id'], ['strategy_requests.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_llm_usage_strategy_request_id'), 'llm_usage', ['strategy_request_id'], unique=False)
    op.create_index(op.f('ix_llm_usage_conversation_id'), 'llm_usage', ['conversation_id'], unique=False)
    op.create_index(op.f('ix_llm_usage_operation'), 'llm_usage', ['operation'], unique=False)
    op.create_index(op.f('ix_llm_usage_created_at'), 'llm_usage', ['created_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema to drop llm_usage table and indexes."""
    op.drop_index(op.f('ix_llm_usage_created_at'), table_name='llm_usage')
    op.drop_index(op.f('ix_llm_usage_operation'), table_name='llm_usage')
    op.drop_index(op.f('ix_llm_usage_conversation_id'), table_name='llm_usage')
    op.drop_index(op.f('ix_llm_usage_strategy_request_id'), table_name='llm_usage')
    op.drop_table('llm_usage')
