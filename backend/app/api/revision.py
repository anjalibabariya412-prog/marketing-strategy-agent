"""
API endpoints for strategy revision conversations and structured revision state.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.orm.revision_message import RevisionMessage
from backend.app.db.session import SessionLocal
from backend.app.schemas.revision import (
    RevisionApplyResponse,
    RevisionHistoryResponse,
    RevisionMessageItem,
    RevisionMessageRequest,
    RevisionMessageResponse,
    RevisionStateResponse,
)
from backend.app.services.revision_service import (
    apply_confirmed_revision,
    load_revision_state,
    process_revision_message,
    resolve_strategy_request,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategy-requests", tags=["Strategy Revisions"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post(
    "/{strategy_request_id}/revisions/messages",
    response_model=RevisionMessageResponse,
    status_code=status.HTTP_200_OK,
)
def send_revision_message(
    strategy_request_id: str,
    request: RevisionMessageRequest,
    db: Session = Depends(get_db),
):
    """
    Submits a revision message for a generated marketing strategy.
    Understands the requested change in the context of the business and existing strategy,
    identifies affected sections, updates structured revision state, and returns an assistant
    proposal or clarifying question.
    """
    if not strategy_request_id or not strategy_request_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Strategy request ID is required.",
        )

    strat_req = resolve_strategy_request(db, strategy_request_id)
    if not strat_req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy request '{strategy_request_id}' not found.",
        )

    try:
        reply_text, assistant_record, updated_state = process_revision_message(
            db=db,
            strategy_request=strat_req,
            user_message=request.message,
        )
        return RevisionMessageResponse(
            reply=reply_text,
            status="success",
            message_id=assistant_record.id,
            created_at=assistant_record.created_at,
            revision_state=updated_state,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except RuntimeError as run_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(run_err),
        )
    except Exception as e:
        logger.error(
            "Unexpected error processing revision message for request %s: %s",
            strategy_request_id,
            e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your revision request. Please try again.",
        )


@router.get(
    "/{strategy_request_id}/revisions/messages",
    response_model=RevisionHistoryResponse,
    status_code=status.HTTP_200_OK,
)
def get_revision_history(
    strategy_request_id: str,
    db: Session = Depends(get_db),
):
    """
    Retrieves the complete revision conversation history (user and assistant messages)
    along with the latest structured revision state for a given strategy request.
    """
    if not strategy_request_id or not strategy_request_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Strategy request ID is required.",
        )

    strat_req = resolve_strategy_request(db, strategy_request_id)
    if not strat_req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy request '{strategy_request_id}' not found.",
        )

    # Only retrieve chat dialogue messages for display, ignoring internal state records
    messages = (
        db.query(RevisionMessage)
        .filter(
            RevisionMessage.strategy_request_id == strat_req.id,
            RevisionMessage.sender.in_(["user", "assistant"]),
        )
        .order_by(RevisionMessage.id.asc())
        .all()
    )

    items = [
        RevisionMessageItem(
            id=msg.id,
            sender=msg.sender,
            content=msg.content,
            created_at=msg.created_at,
        )
        for msg in messages
    ]

    latest_state = load_revision_state(db, strat_req)

    return RevisionHistoryResponse(
        strategy_request_id=str(strat_req.id),
        messages=items,
        revision_state=latest_state,
    )


@router.get(
    "/{strategy_request_id}/revisions/state",
    response_model=RevisionStateResponse,
    status_code=status.HTTP_200_OK,
)
def get_revision_state(
    strategy_request_id: str,
    db: Session = Depends(get_db),
):
    """
    Retrieves the current structured revision state for a given strategy request.
    """
    if not strategy_request_id or not strategy_request_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Strategy request ID is required.",
        )

    strat_req = resolve_strategy_request(db, strategy_request_id)
    if not strat_req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy request '{strategy_request_id}' not found.",
        )

    current_state = load_revision_state(db, strat_req)

    return RevisionStateResponse(
        strategy_request_id=str(strat_req.id),
        revision_state=current_state,
    )


@router.post(
    "/{strategy_request_id}/revisions/apply",
    response_model=RevisionApplyResponse,
    status_code=status.HTTP_200_OK,
)
def apply_revision(
    strategy_request_id: str,
    db: Session = Depends(get_db),
):
    """
    Applies an explicitly confirmed revision proposal to the saved marketing strategy.
    Validates that a confirmed proposal exists, updates only affected sections, validates
    the resulting strategy against MarketingStrategy, and transactionally persists the update.
    """
    if not strategy_request_id or not strategy_request_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Strategy request ID is required.",
        )

    strat_req = resolve_strategy_request(db, strategy_request_id)
    if not strat_req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy request '{strategy_request_id}' not found.",
        )

    try:
        updated_strategy, updated_state, checkpoint_synced, already_applied = apply_confirmed_revision(
            db=db,
            strategy_request=strat_req,
        )
        msg = (
            "Strategy revisions have already been applied."
            if already_applied
            else "Strategy revisions successfully applied and persisted."
        )
        return RevisionApplyResponse(
            status="success",
            message=msg,
            strategy_request_id=str(strat_req.id),
            strategy=updated_strategy,
            application_status="already_applied" if already_applied else updated_state.application_status,
            revision_state=updated_state,
            checkpoint_synced=checkpoint_synced,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except RuntimeError as run_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(run_err),
        )
    except Exception as e:
        logger.error(
            "Unexpected error applying revisions for request %s: %s",
            strategy_request_id,
            e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while applying strategy revisions. Please try again.",
        )

