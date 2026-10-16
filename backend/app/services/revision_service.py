"""
Service layer for managing marketing strategy revision conversations,
intelligent intent analysis, section identification, and structured revision state.
"""

from datetime import datetime, timezone
import copy
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from backend.app.models.business import Business
from backend.app.models.conversation import Conversation
from backend.app.models.online_presence_source import OnlinePresenceSource
from backend.app.models.revision_message import RevisionMessage
from backend.app.models.strategy import Strategy
from backend.app.models.strategy_request import StrategyRequest
from backend.app.schemas.agent.agent_state import MarketingAgentState
from backend.app.schemas.agent.marketing_strategy import MarketingStrategy
from backend.app.schemas.agent.revision_state import (
    LLMRevisionAnalysisResult,
    ProposedSectionChange,
    RevisionState,
)
from backend.app.services.llm_service import LLMServiceError, get_llm_response

logger = logging.getLogger(__name__)

STANDARD_SECTIONS = [
    ("business_overview", "1. Business Overview"),
    ("target_audience_insights", "2. Target Audience & Insights"),
    ("competitive_positioning", "3. Competitive Positioning"),
    ("value_proposition", "4. Value Proposition & Positioning"),
    ("marketing_channels_and_tactics", "5. Marketing Channels & Tactics"),
    ("customer_acquisition_approach", "6. Customer Acquisition Approach"),
    ("budget_considerations", "7. Budget Considerations & Allocation"),
    ("kpis", "8. KPIs & Success Metrics"),
    ("action_plan", "9. Action Plan & Timeline"),
]

VALID_SECTION_KEYS = {key for key, _ in STANDARD_SECTIONS}
SECTION_KEY_TO_NAME = dict(STANDARD_SECTIONS)


def resolve_strategy_request(db: Session, identifier: str) -> Optional[StrategyRequest]:
    """
    Resolves a StrategyRequest by its primary UUID or by an associated Conversation thread ID.
    Returns None if not found or if the identifier is not a valid UUID string.
    """
    if not identifier:
        return None

    try:
        req_uuid = uuid.UUID(identifier.strip())
    except (ValueError, AttributeError):
        return None

    # 1. Direct match on strategy_requests.id
    strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == req_uuid).first()
    if strat_req:
        return strat_req

    # 2. Match via conversation.id (thread_id)
    conv = db.query(Conversation).filter(Conversation.id == req_uuid).first()
    if conv and conv.strategy_request_id:
        return db.query(StrategyRequest).filter(StrategyRequest.id == conv.strategy_request_id).first()

    return None


def get_latest_strategy_data(
    db: Session, strategy_request: StrategyRequest
) -> Tuple[Optional[Dict], Optional[uuid.UUID]]:
    """
    Loads the latest generated marketing strategy and conversation ID for a StrategyRequest.
    Checks the database Strategy records first, with fallback to the LangGraph checkpoint state.
    """
    conv = strategy_request.conversation
    conv_id = conv.id if conv else None

    if conv_id:
        # Check DB strategies table
        latest_db_strategy = (
            db.query(Strategy)
            .filter(Strategy.conversation_id == conv_id)
            .order_by(Strategy.version.desc(), Strategy.created_at.desc())
            .first()
        )
        if latest_db_strategy and latest_db_strategy.strategy_json:
            return latest_db_strategy.strategy_json, conv_id

        # Fallback to LangGraph checkpoint state
        try:
            from backend.app.agent.graph import graph

            config = {"configurable": {"thread_id": str(conv_id)}}
            snapshot = graph.get_state(config)
            if snapshot and snapshot.values:
                state = MarketingAgentState.model_validate(snapshot.values)
                if state.final_strategy:
                    return state.final_strategy.model_dump(), conv_id
        except Exception as state_err:
            logger.warning(
                "Could not retrieve strategy from LangGraph checkpoint for conv_id %s: %s",
                conv_id,
                state_err,
            )

    return None, conv_id


def format_business_context_summary(biz: Optional[Business]) -> str:
    """
    Formats the business context into a clean text summary for the revision assistant.
    """
    if not biz:
        return "No explicit business profile found."

    budget_str = "Not specified"
    if biz.budget_amount is not None:
        budget_str = f"{biz.budget_amount} {biz.budget_currency or ''}".strip()

    return (
        f"- Company / Name: {biz.company_name or 'N/A'}\n"
        f"- Core Product / Service: {biz.product_or_service or 'N/A'}\n"
        f"- Primary Marketing Goal: {biz.marketing_goal or 'N/A'}\n"
        f"- Target Audience: {biz.target_audience or 'N/A'}\n"
        f"- Stated Budget: {budget_str}\n"
        f"- Current Channels: {biz.current_marketing_channels or 'None specified'}\n"
        f"- Website / Social Links: {biz.website_social_links or 'None provided'}"
    )


def format_online_presence_summary(
    sources: List[OnlinePresenceSource],
    biz: Optional[Business] = None,
) -> str:
    """
    Formats the online presence and website scraping findings from stored
    OnlinePresenceSource records and Business profile links for LLM prompt context.
    """
    if not sources and not (biz and biz.website_social_links):
        return "- No website or online presence sources provided by the user."

    lines = []
    website_source_found = False

    for src in sources:
        platform = (src.platform or "unknown").lower()
        if platform == "website":
            website_source_found = True

        status = src.scrape_status or "pending"
        url = src.normalized_url or src.original_url or "Unknown URL"

        lines.append(f"### Platform: {platform.capitalize()} ({url})")
        lines.append(f"- URL: {url}")
        lines.append(f"- Scrape Status: {status}")

        if status == "success" and src.summary and src.summary.strip():
            lines.append(f"- Stored Summary / Findings: {src.summary.strip()}")
            if src.details and isinstance(src.details, dict):
                evidence = src.details.get("evidence")
                items = src.details.get("items")
                if evidence and isinstance(evidence, list):
                    ev_texts = [
                        str(e.get("fact", e)) if isinstance(e, dict) else str(e)
                        for e in evidence[:5]
                    ]
                    lines.append(f"- Key Evidence / Highlights: {'; '.join(ev_texts)}")
                elif items and isinstance(items, list):
                    page_urls = [
                        item.get("url")
                        for item in items
                        if isinstance(item, dict) and item.get("url")
                    ]
                    if page_urls:
                        lines.append(f"- Scraped Page URLs: {', '.join(page_urls[:5])}")
        elif status in ("failed", "skipped"):
            err = src.error_message or "No content could be extracted."
            lines.append(f"- Status Detail: Scrape failed or skipped ({err}). No useful page content was extracted.")
        else:
            lines.append("- Status Detail: No useful summary or page content available in stored records.")
        lines.append("")

    if biz and biz.website_social_links:
        raw_links = biz.website_social_links.strip()
        if not website_source_found and any(
            kw in raw_links.lower() for kw in ("http://", "https://", "www.", ".com", ".in", ".org", ".io", ".co")
        ):
            lines.append(f"### Provided Website / Social Links: {raw_links}")
            lines.append("- Scrape Status: Not scraped / No website analysis available in stored system records.")
            lines.append("- Status Detail: The user previously provided this link, but no website content has been scraped or analyzed for it.")
            lines.append("")

    return "\n".join(lines).strip() or "- No website or online presence sources provided by the user."


def format_strategy_summary(strategy_data: Optional[Dict]) -> str:
    """
    Formats the 9 fixed sections of the marketing strategy for LLM prompt context.
    """
    if not strategy_data:
        return "No generated strategy available."

    lines = []
    for key, label in STANDARD_SECTIONS:
        content = strategy_data.get(key)
        lines.append(f"### {label}")
        if not content:
            lines.append("*(Section empty or not generated)*\n")
            continue

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Structured items: BudgetItem or KPIItem
                    if "label" in item and "amount" in item:
                        pct = f" ({item['percentage']})" if item.get("percentage") else ""
                        lines.append(f"- **{item['label']}**: {item['amount']}{pct} — {item.get('description', '')}")
                    elif "metric" in item and "target" in item:
                        lines.append(f"- **{item['metric']}**: Target {item['target']} — {item.get('description', '')}")
                    else:
                        lines.append(f"- {json.dumps(item)}")
                else:
                    lines.append(f"- {str(item)}")
        else:
            lines.append(f"- {str(content)}")
        lines.append("")

    return "\n".join(lines)


def load_revision_state(db: Session, strategy_request: StrategyRequest) -> RevisionState:
    """
    Retrieves the latest structured RevisionState persisted for this strategy request.
    If no state has been saved yet, returns a default initial state.
    """
    latest_state_msg = (
        db.query(RevisionMessage)
        .filter(
            RevisionMessage.strategy_request_id == strategy_request.id,
            RevisionMessage.sender == "revision_state",
        )
        .order_by(RevisionMessage.id.desc())
        .first()
    )

    if latest_state_msg and latest_state_msg.content:
        try:
            return RevisionState.model_validate_json(latest_state_msg.content)
        except Exception as err:
            logger.warning("Failed to deserialize RevisionState from DB: %s", err)

    return RevisionState(
        status="in_progress",
        confirmation_status="pending",
        is_ready_for_confirmation=False,
    )


def save_revision_state(
    db: Session, strategy_request: StrategyRequest, state: RevisionState
) -> RevisionMessage:
    """
    Durably saves a snapshot of the structured RevisionState to the revision_messages table.
    """
    state.last_updated_at = datetime.now(timezone.utc)
    state_record = RevisionMessage(
        strategy_request_id=strategy_request.id,
        sender="revision_state",
        content=state.model_dump_json(),
    )
    db.add(state_record)
    return state_record


from backend.app.prompts.revision import (
    format_out_of_context_response,
    build_revision_system_prompt,
    build_revision_editor_system_prompt,
    build_revision_editor_prompt,
)

# Alias for backwards compatibility
build_system_prompt = build_revision_system_prompt



def parse_llm_analysis(
    content: str, current_state: RevisionState, user_message: str
) -> LLMRevisionAnalysisResult:
    """
    Parses and validates the JSON output from the LLM, with fallback handling
    in case of formatting or decoding issues.
    """
    cleaned = content.strip()
    try:
        data = json.loads(cleaned)
        # Validate through Pydantic
        analysis = LLMRevisionAnalysisResult.model_validate(data)

        # Sanitize affected_sections to only contain valid keys
        sanitized_sections = [
            sec for sec in analysis.affected_sections if sec in VALID_SECTION_KEYS
        ]
        analysis.affected_sections = sanitized_sections

        # Ensure section_name is populated on proposed_changes
        for change in analysis.proposed_changes:
            if not change.section_name and change.section_id in SECTION_KEY_TO_NAME:
                change.section_name = SECTION_KEY_TO_NAME[change.section_id]

        return analysis
    except Exception as parse_err:
        logger.warning(
            "Failed to parse LLM structured revision output (%s). Falling back to conversational response.",
            parse_err,
        )

    # Fallback heuristic if LLM produced plain text or invalid JSON
    lower_msg = user_message.strip().lower()
    is_confirm = (
        current_state.is_ready_for_confirmation
        and lower_msg in ("yes", "confirm", "confirmed", "looks good", "approved", "go ahead", "please apply")
    )
    is_reject = (
        current_state.is_ready_for_confirmation
        and lower_msg in ("no", "cancel", "discard", "reject", "don't do that", "never mind")
    )

    is_out_of_context_content = (
        "specifically to help you refine your" in content.lower()
        or "not able to help with that" in content.lower()
    )

    if is_confirm:
        status = "confirmed"
        conf_status = "confirmed"
        ready = False
        intent_type = "confirmation"
    elif is_reject:
        status = "rejected"
        conf_status = "rejected"
        ready = False
        intent_type = "rejection"
    elif is_out_of_context_content:
        status = current_state.status or "in_progress"
        conf_status = current_state.confirmation_status or "pending"
        ready = current_state.is_ready_for_confirmation
        intent_type = "out_of_context"
    else:
        status = current_state.status or "in_progress"
        conf_status = "pending"
        ready = current_state.is_ready_for_confirmation
        intent_type = "inquiry"

    return LLMRevisionAnalysisResult(
        reply=content,
        intent_type=intent_type,
        status=status,
        normalized_intent=current_state.normalized_intent or user_message[:100],
        affected_sections=current_state.affected_sections,
        proposed_changes=current_state.proposed_changes,
        unchanged_facts=current_state.unchanged_facts,
        missing_information=[],
        current_clarification_question=None,
        is_ready_for_confirmation=ready,
        confirmation_status=conf_status,
        assumptions_and_constraints=current_state.assumptions_and_constraints,
    )


def process_revision_message(
    db: Session,
    strategy_request: StrategyRequest,
    user_message: str,
) -> Tuple[str, RevisionMessage, RevisionState]:
    """
    Processes a user's revision message:
    1. Loads business and strategy context.
    2. Loads current durable RevisionState.
    3. Retrieves prior chat history (user & assistant messages only).
    4. Calls LLM with full context, prior state, and conversation history.
    5. Parses structured analysis and updates RevisionState.
    6. Persists user message, assistant reply, and RevisionState snapshot.
    7. Returns reply text, assistant message record, and updated RevisionState.
    """
    strategy_data, conv_id = get_latest_strategy_data(db, strategy_request)
    if not strategy_data:
        raise ValueError("Strategy has not been generated yet for this request.")

    biz = strategy_request.business
    biz_name = biz.company_name if biz else None
    biz_summary = format_business_context_summary(biz)
    strat_summary = format_strategy_summary(strategy_data)

    # Load online presence sources for website and social context
    online_presence_sources = (
        db.query(OnlinePresenceSource)
        .filter(OnlinePresenceSource.strategy_request_id == strategy_request.id)
        .all()
    )
    online_presence_summary = format_online_presence_summary(online_presence_sources, biz)

    # 1. Load prior revision state
    current_state = load_revision_state(db, strategy_request)
    if current_state.application_status == "applied" or current_state.status in ("applied", "rejected"):
        current_state = RevisionState(
            status="in_progress",
            confirmation_status="pending",
            is_ready_for_confirmation=False,
            application_status="not_applied",
            original_request=user_message,
        )

    # 2. Build system instructions with context and state
    system_prompt = build_system_prompt(
        biz_summary,
        strat_summary,
        current_state,
        business_name=biz_name,
        online_presence_summary=online_presence_summary,
    )

    # 3. Fetch prior chat messages (excluding state records)
    prior_messages = (
        db.query(RevisionMessage)
        .filter(
            RevisionMessage.strategy_request_id == strategy_request.id,
            RevisionMessage.sender.in_(["user", "assistant"]),
        )
        .order_by(RevisionMessage.id.asc())
        .all()
    )

    llm_messages: List[Dict[str, str]] = []
    for msg in prior_messages:
        role = "assistant" if msg.sender == "assistant" else "user"
        llm_messages.append({"role": role, "content": msg.content})

    llm_messages.append({"role": "user", "content": user_message})

    # 4. Call centralized LLM service requesting JSON output
    try:
        raw_response = get_llm_response(
            messages=llm_messages,
            system_prompt=system_prompt,
            response_format={"type": "json_object"},
            operation="strategy_revision",
            strategy_request_id=strategy_request.id,
            conversation_id=conv_id,
        )
    except LLMServiceError as llm_err:
        logger.error(
            "LLM service error during strategy revision for request %s: %s",
            strategy_request.id,
            llm_err,
        )
        raise RuntimeError("The AI service is temporarily unavailable. Please try again shortly.") from llm_err
    except Exception as e:
        logger.error(
            "Unexpected error during strategy revision LLM call for request %s: %s",
            strategy_request.id,
            e,
        )
        raise RuntimeError("Failed to generate revision proposal. Please try again.") from e

    # 5. Parse analysis and update revision state
    analysis = parse_llm_analysis(raw_response, current_state, user_message)

    # Detect out-of-context intent
    is_out_of_context = (
        analysis.intent_type == "out_of_context"
        or (isinstance(analysis.reply, str) and "specifically to help you refine your" in analysis.reply.lower())
    )

    user_lower = user_message.lower()
    is_website_or_strategy_inquiry = any(
        kw in user_lower
        for kw in (
            "website",
            "web site",
            "url",
            "page",
            "site",
            "online",
            "link",
            "scraped",
            "finding",
            "find on",
        )
    )
    if is_website_or_strategy_inquiry and is_out_of_context:
        is_out_of_context = False
        analysis.intent_type = "inquiry"

    if is_out_of_context:
        exact_reply = format_out_of_context_response(biz_name)
        analysis.reply = exact_reply
        # Preserve existing state completely without modifying or resetting pending proposals
        updated_state = current_state.model_copy(deep=True)
        updated_state.last_updated_at = datetime.now(timezone.utc)
    else:
        # Track original request if this was the first request in the session
        original_req = current_state.original_request or user_message
        if analysis.intent_type == "new_request" and not current_state.original_request:
            original_req = user_message

        updated_state = RevisionState(
            status=analysis.status,
            original_request=original_req,
            normalized_intent=analysis.normalized_intent or current_state.normalized_intent,
            affected_sections=analysis.affected_sections or current_state.affected_sections,
            proposed_changes=analysis.proposed_changes if analysis.proposed_changes else current_state.proposed_changes,
            unchanged_facts=analysis.unchanged_facts or current_state.unchanged_facts,
            missing_information=analysis.missing_information,
            current_clarification_question=analysis.current_clarification_question,
            is_ready_for_confirmation=analysis.is_ready_for_confirmation,
            confirmation_status=analysis.confirmation_status,
            assumptions_and_constraints=analysis.assumptions_and_constraints
            or current_state.assumptions_and_constraints,
            last_updated_at=datetime.now(timezone.utc),
        )

    # 6. Database Transaction: Persist user message, assistant reply, and state snapshot
    user_record = RevisionMessage(
        strategy_request_id=strategy_request.id,
        sender="user",
        content=user_message,
    )
    db.add(user_record)
    db.flush()

    assistant_record = RevisionMessage(
        strategy_request_id=strategy_request.id,
        sender="assistant",
        content=analysis.reply,
    )
    db.add(assistant_record)
    db.flush()

    save_revision_state(db, strategy_request, updated_state)

    db.commit()
    db.refresh(assistant_record)

    return analysis.reply, assistant_record, updated_state


def generate_revised_section_content(
    biz_summary: str,
    strategy_data: Dict,
    affected_keys: List[str],
    proposed_changes: List[ProposedSectionChange],
    strategy_request_id: uuid.UUID,
    conv_id: Optional[uuid.UUID],
) -> Dict[str, Any]:
    """
    Invokes the centralized LLM to generate revised content for only the affected sections,
    strictly respecting the existing strategy schema and confirmed proposal changes.
    """
    sections_prompt_blocks = []
    for key in affected_keys:
        sec_name = SECTION_KEY_TO_NAME.get(key, key)
        current_content = strategy_data.get(key, [])
        matching_change = next((c for c in proposed_changes if c.section_id == key), None)
        change_desc = matching_change.proposed_change if matching_change else "Update as requested"
        unchanged = (
            matching_change.unchanged_aspects
            if matching_change and matching_change.unchanged_aspects
            else "Preserve all unmentioned items"
        )

        sections_prompt_blocks.append(
            f"### Section: {sec_name} (Key: \"{key}\")\n"
            f"- Current Content: {json.dumps(current_content, indent=2)}\n"
            f"- Confirmed Adjustment: {change_desc}\n"
            f"- Aspects to Preserve: {unchanged}\n"
        )

    sections_info = "\n".join(sections_prompt_blocks)

    system_prompt = build_revision_editor_system_prompt(
        biz_summary=biz_summary,
        sections_info=sections_info,
        affected_keys=affected_keys,
    )

    try:
        raw_response = get_llm_response(
            messages=[
                {"role": "user", "content": build_revision_editor_prompt(affected_keys)}
            ],
            system_prompt=system_prompt,
            response_format={"type": "json_object"},
            operation="strategy_revision_apply",
            strategy_request_id=strategy_request_id,
            conversation_id=conv_id,
        )
    except LLMServiceError as llm_err:
        logger.error("LLM service error generating revised sections: %s", llm_err)
        raise RuntimeError("The AI service is temporarily unavailable. Please try again shortly.") from llm_err
    except Exception as err:
        logger.error("Unexpected error in LLM call for revised sections: %s", err)
        raise RuntimeError("Failed to generate revised strategy sections. Please try again.") from err

    try:
        data = json.loads(raw_response.strip())
        if not isinstance(data, dict):
            raise ValueError("LLM returned a non-dictionary JSON payload.")

        revised_sections: Dict[str, Any] = {}
        for sec in affected_keys:
            if sec in data:
                revised_sections[sec] = data[sec]
            else:
                raise ValueError(f"LLM output missing requested section: '{sec}'")
        return revised_sections
    except Exception as parse_err:
        logger.error("Failed to parse LLM revised section output: %s", parse_err)
        raise ValueError(f"Failed to generate structured section updates: {parse_err}") from parse_err


def apply_confirmed_revision(
    db: Session,
    strategy_request: StrategyRequest,
) -> Tuple[MarketingStrategy, RevisionState, bool, bool]:
    """
    Applies an explicitly confirmed revision proposal to the saved marketing strategy.

    Workflow:
    1. Loads existing strategy data and conversation ID.
    2. Loads latest persisted RevisionState.
    3. Verifies proposal existence, rejection, confirmation, and idempotency.
    4. Determines affected sections and generates/extracts updated section contents.
    5. Preserves unaffected sections with 100% fidelity.
    6. Validates the merged strategy with MarketingStrategy.
    7. Atomically persists the updated strategy and state transition in PostgreSQL.
    8. Synchronizes LangGraph checkpoint if available.
    9. Returns (validated_strategy, updated_state, checkpoint_synced, already_applied).
    """
    existing_strategy_data, conv_id = get_latest_strategy_data(db, strategy_request)
    if not existing_strategy_data:
        raise ValueError("No generated marketing strategy found for this strategy request.")

    if not conv_id:
        conv = strategy_request.conversation
        conv_id = conv.id if conv else None
        if not conv_id:
            raise ValueError("No associated conversation found for strategy request.")

    state = load_revision_state(db, strategy_request)

    # 1. Proposal existence check
    if not state.proposed_changes and not state.affected_sections:
        raise ValueError("No revision proposal found for this strategy request.")

    # 2. Rejection check
    if state.status == "rejected" or state.confirmation_status == "rejected":
        raise ValueError("Cannot apply a rejected revision proposal.")

    # 3. Confirmation check
    if state.confirmation_status != "confirmed":
        raise ValueError("Revision proposal has not been confirmed. Please confirm the proposal before applying.")

    # 4. Idempotency check: Already applied
    if state.application_status == "applied" or state.status == "applied":
        logger.info("Revision proposal for %s has already been applied.", strategy_request.id)
        validated_existing = MarketingStrategy.model_validate(existing_strategy_data)
        return validated_existing, state, bool(state.checkpoint_synced), True

    # 5. Identify affected section keys
    affected_keys = [sec for sec in state.affected_sections if sec in VALID_SECTION_KEYS]
    if not affected_keys:
        affected_keys = [p.section_id for p in state.proposed_changes if p.section_id in VALID_SECTION_KEYS]
    if not affected_keys:
        raise ValueError("No valid affected strategy sections identified in the confirmed proposal.")

    # 6. Gather updated section content (reuse replacement_content if present, else LLM)
    new_section_contents: Dict[str, Any] = {}
    sections_needing_llm: List[str] = []

    for sec_id in affected_keys:
        matching_change = next((c for c in state.proposed_changes if c.section_id == sec_id), None)
        if matching_change and matching_change.replacement_content is not None:
            new_section_contents[sec_id] = matching_change.replacement_content
        else:
            sections_needing_llm.append(sec_id)

    if sections_needing_llm:
        biz = strategy_request.business
        biz_summary = format_business_context_summary(biz)
        llm_sections = generate_revised_section_content(
            biz_summary=biz_summary,
            strategy_data=existing_strategy_data,
            affected_keys=sections_needing_llm,
            proposed_changes=state.proposed_changes,
            strategy_request_id=strategy_request.id,
            conv_id=conv_id,
        )
        new_section_contents.update(llm_sections)

    # 7. Merge into existing strategy preserving unaffected sections
    updated_dict = copy.deepcopy(existing_strategy_data)
    for sec_id, content in new_section_contents.items():
        updated_dict[sec_id] = content

    # 8. Validate full updated strategy with MarketingStrategy
    try:
        validated_strategy = MarketingStrategy.model_validate(updated_dict)
    except Exception as val_err:
        logger.error("Revised strategy failed Pydantic validation: %s", val_err)
        state.application_status = "failed"
        state.application_error = f"Validation failed: {val_err}"
        try:
            save_revision_state(db, strategy_request, state)
            db.commit()
        except Exception:
            pass
        raise ValueError(f"Revised strategy failed validation: {val_err}") from val_err

    # 9. Transactional persistence in PostgreSQL
    try:
        latest_strat_row = (
            db.query(Strategy)
            .filter(Strategy.conversation_id == conv_id)
            .order_by(Strategy.version.desc(), Strategy.created_at.desc())
            .first()
        )
        if latest_strat_row:
            latest_strat_row.strategy_json = validated_strategy.model_dump(mode="json")
            target_version = latest_strat_row.version
        else:
            target_version = 1
            new_strat = Strategy(
                conversation_id=conv_id,
                version=target_version,
                strategy_json=validated_strategy.model_dump(mode="json"),
            )
            db.add(new_strat)

        state.status = "applied"
        state.application_status = "applied"
        state.applied_at = datetime.now(timezone.utc)
        state.applied_version = target_version
        state.application_error = None
        state.is_ready_for_confirmation = False

        applied_msg = RevisionMessage(
            strategy_request_id=strategy_request.id,
            sender="assistant",
            content=(
                f"The confirmed revisions have been successfully applied to your marketing strategy. "
                f"Updated sections: {', '.join(SECTION_KEY_TO_NAME.get(k, k) for k in affected_keys)}."
            ),
        )
        db.add(applied_msg)

        save_revision_state(db, strategy_request, state)
        db.commit()
    except Exception as db_err:
        db.rollback()
        logger.error("Database transaction error applying revisions: %s", db_err)
        try:
            state.application_status = "failed"
            state.application_error = f"Database error: {db_err}"
            save_revision_state(db, strategy_request, state)
            db.commit()
        except Exception:
            pass
        raise RuntimeError(f"Database transaction failed while applying strategy revisions: {db_err}") from db_err

    # 10. LangGraph checkpoint synchronization
    checkpoint_synced = False
    try:
        from backend.app.agent.graph import graph
        config = {"configurable": {"thread_id": str(conv_id)}}
        graph.update_state(config, {"final_strategy": validated_strategy})
        checkpoint_synced = True
        logger.info("Successfully synchronized LangGraph checkpoint for conv_id %s", conv_id)
    except Exception as cp_err:
        logger.warning("Could not synchronize LangGraph checkpoint for conv_id %s: %s", conv_id, cp_err)
        checkpoint_synced = False

    try:
        state.checkpoint_synced = checkpoint_synced
        save_revision_state(db, strategy_request, state)
        db.commit()
    except Exception:
        pass

    return validated_strategy, state, checkpoint_synced, False

