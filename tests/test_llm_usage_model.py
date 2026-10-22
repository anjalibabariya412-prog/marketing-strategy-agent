import uuid
from decimal import Decimal
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import UUID

from backend.app.db.base import Base
from backend.app.models.llm_usage import LLMUsage


def test_llm_usage_model_registration():
    """Verify LLMUsage table is registered with Base.metadata."""
    assert "llm_usage" in Base.metadata.tables
    table = Base.metadata.tables["llm_usage"]
    assert table.name == "llm_usage"


def test_llm_usage_columns_and_types():
    """Verify exact 20 columns, types, and nullability on LLMUsage."""
    table = Base.metadata.tables["llm_usage"]
    cols = table.columns

    # Verify exactly 20 columns exist
    expected_20_columns = [
        "id",
        "strategy_request_id",
        "conversation_id",
        "operation",
        "provider",
        "model",
        "status",
        "attempt_number",
        "error_message",
        "input_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "output_tokens",
        "duration_ms",
        "input_cost",
        "cached_input_cost",
        "cache_write_cost",
        "output_cost",
        "total_cost",
        "created_at",
    ]
    assert len(cols) == 20, f"Expected 20 columns, got {len(cols)}: {list(cols.keys())}"
    assert set(cols.keys()) == set(expected_20_columns)

    # Verify legacy removed columns are NOT present
    removed_columns = [
        "total_tokens",
        "pricing_input_rate",
        "pricing_output_rate",
        "pricing_cached_rate",
        "pricing_metadata",
    ]
    for col in removed_columns:
        assert col not in cols, f"Removed column '{col}' should not be in llm_usage"

    # Primary key
    assert cols["id"].primary_key is True

    # Foreign keys (nullable)
    assert cols["strategy_request_id"].nullable is True
    assert cols["conversation_id"].nullable is True

    # Invocation & model metadata
    assert cols["operation"].nullable is False
    assert cols["provider"].nullable is False
    assert cols["model"].nullable is False

    # Attempt and status tracking
    assert cols["status"].nullable is False
    assert cols["attempt_number"].nullable is False
    assert cols["error_message"].nullable is True

    # Token counts (all nullable)
    assert cols["input_tokens"].nullable is True
    assert cols["cached_input_tokens"].nullable is True
    assert cols["cache_write_tokens"].nullable is True
    assert cols["output_tokens"].nullable is True

    # Duration (nullable)
    assert cols["duration_ms"].nullable is True

    # Monetary values (Numeric(12, 6), all nullable)
    for cost_col in ["input_cost", "cached_input_cost", "cache_write_cost", "output_cost", "total_cost"]:
        assert cols[cost_col].nullable is True
        assert cols[cost_col].type.precision == 12
        assert cols[cost_col].type.scale == 6

    # Timestamps
    assert cols["created_at"].nullable is False


def test_llm_usage_indexes():
    """Verify indexes on strategy_request_id, conversation_id, operation, status, created_at."""
    table = Base.metadata.tables["llm_usage"]
    index_columns = {idx.name: [c.name for c in idx.columns] for idx in table.indexes}

    assert any("strategy_request_id" in cols for cols in index_columns.values())
    assert any("conversation_id" in cols for cols in index_columns.values())
    assert any("operation" in cols for cols in index_columns.values())
    assert any("status" in cols for cols in index_columns.values())
    assert any("created_at" in cols for cols in index_columns.values())


def test_llm_usage_instance_creation():
    """Verify model instance can be instantiated with 20-column schema."""
    usage_id = uuid.uuid4()
    req_id = uuid.uuid4()
    conv_id = uuid.uuid4()

    record = LLMUsage(
        id=usage_id,
        strategy_request_id=req_id,
        conversation_id=conv_id,
        operation="relevance_analysis",
        provider="openai",
        model="gpt-4o-mini",
        status="success",
        attempt_number=1,
        error_message=None,
        input_tokens=100,
        cached_input_tokens=40,
        cache_write_tokens=10,
        output_tokens=25,
        duration_ms=450,
        input_cost=Decimal("0.000015"),
        cached_input_cost=Decimal("0.000005"),
        cache_write_cost=Decimal("0.000001"),
        output_cost=Decimal("0.000015"),
        total_cost=Decimal("0.000036"),
    )

    assert record.id == usage_id
    assert record.strategy_request_id == req_id
    assert record.conversation_id == conv_id
    assert record.operation == "relevance_analysis"
    assert record.provider == "openai"
    assert record.model == "gpt-4o-mini"
    assert record.status == "success"
    assert record.attempt_number == 1
    assert record.error_message is None
    assert record.input_tokens == 100
    assert record.cached_input_tokens == 40
    assert record.cache_write_tokens == 10
    assert record.output_tokens == 25
    assert record.duration_ms == 450
    assert record.input_cost == Decimal("0.000015")
    assert record.cached_input_cost == Decimal("0.000005")
    assert record.cache_write_cost == Decimal("0.000001")
    assert record.output_cost == Decimal("0.000015")
    assert record.total_cost == Decimal("0.000036")
    assert "gpt-4o-mini" in repr(record)
    assert "status='success'" in repr(record)
