from datetime import datetime, timezone
import logging
from typing import Optional
import uuid

from backend.app.db.orm.chat_message import ChatMessage
from backend.app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def save_chat_message(
    conversation_id: uuid.UUID,
    sender: str,
    message_type: str,
    content: str,
    requirement_id: Optional[str] = None,
) -> Optional[int]:
    """
    Persists a single chat message (agent question or user answer) into the
    chat_messages database table.
    Returns the generated message ID if successful, or None on failure.
    """
    if not conversation_id or not sender or not content:
        return None

    try:
        with SessionLocal() as db:
            msg = ChatMessage(
                conversation_id=conversation_id,
                sender=sender,
                message_type=message_type,
                requirement_id=requirement_id,
                content=content,
            )
            db.add(msg)
            db.commit()
            db.refresh(msg)
            logger.info(
                f"Successfully persisted ChatMessage id={msg.id} (sender='{sender}', "
                f"type='{message_type}', req_id='{requirement_id}') for Conversation '{conversation_id}'."
            )
            return msg.id
    except Exception as db_err:
        logger.error(
            f"Failed to persist ChatMessage (sender='{sender}', type='{message_type}') "
            f"for Conversation '{conversation_id}': {db_err}"
        )
        return None
