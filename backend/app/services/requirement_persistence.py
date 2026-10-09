from datetime import datetime, timezone
import logging
from typing import List, Optional
import uuid

from backend.app.db.orm.information_requirement import InformationRequirement as InformationRequirementORM
from backend.app.db.session import SessionLocal
from backend.app.models.information_requirement import (
    InformationRequirement as InformationRequirementModel,
    RequirementStatus,
)

logger = logging.getLogger(__name__)


def save_information_requirements(
    conversation_id: uuid.UUID,
    requirements: List[InformationRequirementModel],
) -> None:
    """
    Persists or updates in-memory InformationRequirement models into the
    information_requirements database table for a given conversation_id.
    Uses composite primary key (conversation_id, requirement_id) for upsert semantics.
    """
    if not conversation_id or not requirements:
        return

    try:
        with SessionLocal() as db:
            now_utc = datetime.now(timezone.utc)
            for req in requirements:
                req_status_str = (
                    req.status.value if isinstance(req.status, RequirementStatus) else str(req.status)
                )

                # Query existing requirement row by composite PK
                existing_row = (
                    db.query(InformationRequirementORM)
                    .filter(
                        InformationRequirementORM.conversation_id == conversation_id,
                        InformationRequirementORM.requirement_id == req.id,
                    )
                    .first()
                )

                # Determine whether status is resolved
                is_resolved = req_status_str in (
                    RequirementStatus.KNOWN.value,
                    RequirementStatus.UNAVAILABLE.value,
                    RequirementStatus.NOT_RELEVANT.value,
                )

                if existing_row:
                    existing_row.title = req.title
                    existing_row.description = req.description
                    existing_row.status = req_status_str
                    existing_row.value = req.value
                    existing_row.is_must_have = req.is_must_have
                    existing_row.is_custom = req.is_custom
                    if is_resolved and existing_row.resolved_at is None:
                        existing_row.resolved_at = now_utc
                    elif not is_resolved:
                        existing_row.resolved_at = None
                else:
                    new_row = InformationRequirementORM(
                        conversation_id=conversation_id,
                        requirement_id=req.id,
                        title=req.title,
                        description=req.description,
                        status=req_status_str,
                        value=req.value,
                        is_must_have=req.is_must_have,
                        is_custom=req.is_custom,
                        resolved_at=now_utc if is_resolved else None,
                    )
                    db.add(new_row)

            db.commit()
            logger.info(
                f"Successfully persisted {len(requirements)} InformationRequirement record(s) "
                f"for Conversation '{conversation_id}'."
            )
    except Exception as db_err:
        logger.error(
            f"Failed to persist InformationRequirement records for Conversation '{conversation_id}': {db_err}"
        )
