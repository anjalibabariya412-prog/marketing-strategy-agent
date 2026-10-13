import uuid
from decimal import Decimal
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import JSONB, UUID

from backend.app.db.base import Base
from backend.app.db.orm.llm_usage import LLMUsage


def test_llm_usage_model_registration():
    """Verify LLMUsage table is registered with Base.metadata."""
    assert "llm_usage" in Base.metadata.tables
    table = Base.metadata.tables["llm_usage"]
    assert table.name == "llm_usage"


def test_llm_usage_columns_and_types():
    """Verify columns, types, and nullability on LLMUsage."""
    table = Base.metadata.tables["llm_usage"]
    cols = table.columns

    # Primary key
    assert "id" in cols
    assert cols["id"].primary_key is True

    # Foreign keys (nullable)
    assert cols["strategy_request_id"].nullable is True
    assert cols["conversation_id"].nullable is True

    # Invocation & model metadata
    assert cols["operation"].nullable is False
    assert cols["provider"].nullable is False
    assert cols["model"].nullable is False

    # Token counts
    assert cols["input_tokens"].nullable is True
    assert cols["output_tokens"].nullable is True
    assert cols["total_tokens"].nullable is True

    # Status & duration
    assert cols["status"].nullable is False
    assert cols["duration_ms"].nullable is True
    assert cols["error_message"].nullable is True

    # Pricing & costs
    assert cols["pricing_input_rate"].nullable is True
    assert cols["pricing_output_rate"].nullable is True
    assert cols["pricing_cached_rate"].nullable is True
    assert cols["pricing_metadata"].nullable is True
    assert cols["input_cost"].nullable is True
    assert cols["output_cost"].nullable is True
    assert cols["total_cost"].nullable is True

    # Precision for monetary values
    assert cols["input_cost"].type.precision == 12
    assert cols["input_cost"].type.scale == 6
    assert cols["total_cost"].type.precision == 12
    assert cols["total_cost"].type.scale == 6

    # Timestamps
    assert cols["created_at"].nullable is False


def test_llm_usage_indexes():
    """Verify indexes on strategy_request_id, conversation_id, operation, created_at."""
    table = Base.metadata.tables["llm_usage"]
    index_columns = {idx.name: [c.name for c in idx.columns] for idx in table.indexes}

    assert any("strategy_request_id" in cols for cols in index_columns.values())
    assert any("conversation_id" in cols for cols in index_columns.values())
    assert any("operation" in cols for cols in index_columns.values())
    assert any("created_at" in cols for cols in index_columns.values())


def test_llm_usage_instance_creation():
    """Verify model instance can be instantiated with required & optional fields."""
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
        input_tokens=100,
        output_tokens=25,
        total_tokens=125,
        duration_ms=450,
        pricing_input_rate=Decimal("0.150000"),
        pricing_output_rate=Decimal("0.600000"),
        pricing_cached_rate=Decimal("0.075000"),
        pricing_metadata={"cache_read_input_tokens": 0},
        input_cost=Decimal("0.000015"),
        output_cost=Decimal("0.000015"),
        total_cost=Decimal("0.000030"),
        status="success",
    )

    assert record.id == usage_id
    assert record.strategy_request_id == req_id
    assert record.conversation_id == conv_id
    assert record.operation == "relevance_analysis"
    assert record.provider == "openai"
    assert record.model == "gpt-4o-mini"
    assert record.input_tokens == 100
    assert record.output_tokens == 25
    assert record.total_tokens == 125
    assert record.total_cost == Decimal("0.000030")
    assert "gpt-4o-mini" in repr(record)
