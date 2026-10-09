from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional, Union
import uuid

from backend.app.db.orm.strategy import Strategy
from backend.app.db.session import SessionLocal
from backend.app.models.marketing_strategy import MarketingStrategy

logger = logging.getLogger(__name__)


def save_strategy(
    conversation_id: uuid.UUID,
    strategy: Union[MarketingStrategy, Dict[str, Any]],
    version: int = 1,
) -> Optional[uuid.UUID]:
    """
    Persists the final generated MarketingStrategy into the strategies database table
    for the specified conversation_id and version (default 1).
    Idempotent: updates strategy_json if a record for (conversation_id, version) already exists.
    Returns the Strategy record ID if successful, or None on failure.
    """
    if not conversation_id or not strategy:
        return None

    # Serialize Pydantic model or dict to JSON-compatible dictionary
    if isinstance(strategy, MarketingStrategy):
        strategy_dict = strategy.model_dump(mode="json")
    elif isinstance(strategy, dict):
        strategy_dict = strategy
    else:
        logger.error(f"Unsupported strategy object type for persistence: {type(strategy)}")
        return None

    try:
        with SessionLocal() as db:
            existing_strategy = (
                db.query(Strategy)
                .filter(
                    Strategy.conversation_id == conversation_id,
                    Strategy.version == version,
                )
                .first()
            )

            if existing_strategy:
                existing_strategy.strategy_json = strategy_dict
                db.commit()
                db.refresh(existing_strategy)
                logger.info(
                    f"Successfully updated Strategy id='{existing_strategy.id}' (version={version}) "
                    f"for Conversation '{conversation_id}'."
                )
                return existing_strategy.id
            else:
                new_strategy = Strategy(
                    conversation_id=conversation_id,
                    version=version,
                    strategy_json=strategy_dict,
                )
                db.add(new_strategy)
                db.commit()
                db.refresh(new_strategy)
                logger.info(
                    f"Successfully persisted new Strategy id='{new_strategy.id}' (version={version}) "
                    f"for Conversation '{conversation_id}'."
                )
                return new_strategy.id

    except Exception as db_err:
        logger.error(
            f"Failed to persist Strategy for Conversation '{conversation_id}': {db_err}"
        )
        return None
