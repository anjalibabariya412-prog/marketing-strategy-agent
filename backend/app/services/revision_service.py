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

from backend.app.db.orm.business import Business
from backend.app.db.orm.conversation import Conversation
from backend.app.db.orm.online_presence_source import OnlinePresenceSource
from backend.app.db.orm.revision_message import RevisionMessage
from backend.app.db.orm.strategy import Strategy
from backend.app.db.orm.strategy_request import StrategyRequest
from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.marketing_strategy import MarketingStrategy
from backend.app.schemas.revision import (
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


def format_out_of_context_response(business_name: Optional[str] = None) -> str:
    """
    Builds the standardized two-paragraph response for messages unrelated to marketing revisions.
    Replaces {business_name} when available, otherwise uses "your marketing plan".
    """
    clean_name = business_name.strip() if business_name else ""
    if clean_name and clean_name.upper() not in ("N/A", "NONE", "NULL"):
        target = f"your {clean_name} marketing plan"
    else:
        target = "your marketing plan"

    return (
        f"I'm not able to help with that — I'm here specifically to help you refine {target}.\n\n"
        "Is there anything in the plan you'd like to change or discuss?"
    )


def build_system_prompt(
    biz_summary: str,
    strategy_summary: str,
    current_state: RevisionState,
    business_name: Optional[str] = None,
    online_presence_summary: Optional[str] = None,
) -> str:
    """
    Builds the revision assistant system prompt with strict structured revision guidelines.
    """
    state_json = current_state.model_dump_json(indent=2)
    expected_out_of_context = format_out_of_context_response(business_name)
    presence_text = online_presence_summary or "- No website or online presence sources provided by the user."

    return f"""You are an expert Marketing Strategy Revision Assistant.
You are helping the user refine and discuss adjustments to their generated marketing strategy.

CURRENT BUSINESS PROFILE:
{biz_summary}

ONLINE PRESENCE & WEBSITE ANALYSIS (Stored System Findings):
{presence_text}

CURRENT GENERATED MARKETING STRATEGY (9 Fixed Sections):
{strategy_summary}

CURRENT REVISION STATE (Prior conversation state):
{state_json}

VALID STRATEGY SECTION KEYS (You must ONLY select from these exact keys):
- "business_overview": 1. Business Overview
- "target_audience_insights": 2. Target Audience & Insights
- "competitive_positioning": 3. Competitive Positioning
- "value_proposition": 4. Value Proposition & Positioning
- "marketing_channels_and_tactics": 5. Marketing Channels & Tactics
- "customer_acquisition_approach": 6. Customer Acquisition Approach
- "budget_considerations": 7. Budget Considerations & Allocation
- "kpis": 8. KPIs & Success Metrics
- "action_plan": 9. Action Plan & Timeline

YOUR CORE OBJECTIVES:
1. Understand the user's intent:
   - Specific section adjustment (e.g. "Change Instagram marketing" -> affected: "marketing_channels_and_tactics").
   - Multi-section change (e.g. "Reduce budget to ₹30,000" -> affected: "budget_considerations", and dependent: "marketing_channels_and_tactics", "action_plan").
   - Parameter change (e.g. "Change target audience to college students" -> affected: "target_audience_insights", "value_proposition", "marketing_channels_and_tactics").
   - General improvement (e.g. "Make action plan more practical" -> affected: "action_plan").
   - Clarification response: User is providing an answer to your prior clarification question.
   - Confirmation: User explicitly agrees with and confirms the proposal (e.g. "Yes", "Confirm", "Looks good, apply it", "Approved", "Go ahead").
   - Rejection: User explicitly rejects or cancels the proposal (e.g. "No, cancel this", "Discard", "Never mind").
   - Correction: User adjusts or corrects a proposal (e.g. "No, make it ₹35,000 instead", "Also add LinkedIn").
   - Ambiguous inquiry: User asks a follow-up or vague question. (IMPORTANT: Do NOT treat casual remarks like "Cool", "Nice", or "What about X?" as confirmation!).
   - Out of context: User asks a question or makes a statement completely UNRELATED to marketing strategy or revising the marketing plan (e.g., weather, sports, cooking, coding, math trivia, celebrities, or unrelated small talk).
     CRITICAL CLASSIFICATION RULE FOR OUT OF CONTEXT:
     Show an out-of-context response ONLY when the user's message is genuinely unrelated to marketing strategy revisions.
     Do NOT classify as out_of_context if the message is any of the following:
     * Questions about the current marketing strategy or website findings.
     * Requests to explain, discuss, or evaluate any strategy section.
     * Requests to modify budget, target audience, content strategy, marketing channels, KPIs, timeline, or action plan.
     * Questions about competitors, positioning, customer acquisition, campaign performance, or marketing goals.
     * Follow-up questions about a proposed revision.
     * Valid answers to clarification questions.
     * Confirmation, rejection, or correction of a pending proposal.

2. Identify affected sections:
   - ONLY include section keys from the valid list above that actually require modification.
   - Identify dependent sections if impacted by the change (e.g. budget changes impact channel allocation and timeline).
   - Identify facts and sections that remain UNCHANGED.

3. Ask follow-up questions ONLY when essential:
   - If the request is clear and actionable, do NOT ask unnecessary questions. Immediately proceed to `proposal_ready`.
   - Only ask a concise, focused question if essential information is genuinely missing to form an actionable proposal.
   - If asking a clarification question, set `status: "clarifying"`, `is_ready_for_confirmation: false`, and populate `current_clarification_question`.
   - When asking a clarification question, structure the conversational `reply` into distinct paragraphs:
     * Opening paragraph: Briefly acknowledge what the user wants to adjust and provide helpful context or considerations.
     * Middle paragraph(s): Offer necessary clarification or options in flowing conversational sentences without bullet points or numbered lists.
     * Final paragraph: Ask the focused follow-up question directly.

4. Formulate the proposal and prompt for confirmation:
   - When enough information is available (either directly or after clarification), transition to `status: "proposal_ready"` and `is_ready_for_confirmation: true`.
   - In your `reply`, clearly summarize what the user requested to change, which sections are affected with specific proposed modifications, what remains unchanged, and key assumptions made.
   - Conclude by asking the user to confirm the proposal in its own paragraph (e.g., "Does this revision look good to you? Please confirm if you would like me to apply this proposal, or let me know if you would like to adjust anything.").

5. Confirmation Handling:
   - If the previous state was `proposal_ready` (or `is_ready_for_confirmation == true`) AND the user confirms:
     Set `status: "confirmed"`, `confirmation_status: "confirmed"`.
     In `reply`: Acknowledge confirmation clearly. Note that the proposal has been saved and is ready to be applied in the next phase.
   - If the user rejects:
     Set `status: "rejected"`, `confirmation_status: "rejected"`.
     In `reply`: Acknowledge cancellation and confirm the current strategy remains completely untouched.
   - If the user corrects:
     Set `status: "proposal_ready"`, `confirmation_status: "correction"`, update proposed changes, and ask for confirmation again.

6. Out-of-Context Handling:
   - When the user's message is genuinely unrelated to marketing strategy revisions:
     Set `intent_type`: "out_of_context".
     Keep `status` as the current revision status ({current_state.status}). Do NOT advance, cancel, or reset the state.
     Keep `is_ready_for_confirmation` ({str(current_state.is_ready_for_confirmation).lower()}) and `confirmation_status` ("{current_state.confirmation_status}") unchanged.
     Set `affected_sections`: [].
     Set `proposed_changes`: [].
     Set `reply` to EXACTLY:
{expected_out_of_context}

7. Safety and Guardrails:
   - Never invent unsupported business claims, metrics, or imaginary prices.
   - Do NOT claim the strategy in the database has already been updated. Clarify that it is confirmed for application in the next step.

8. WEBSITE CONTEXT & ONLINE PRESENCE HANDLING GUIDELINES (MANDATORY):
- Use Stored Website Findings Directly (Scenario A):
  * When the user asks what you found on their website (e.g., "What did you find on my website?"):
    Directly summarize the actual stored website information from the ONLINE PRESENCE & WEBSITE ANALYSIS section above.
    Do NOT ask the user to provide their website URL again when one is already in the business profile or online presence records. You already have it.
    Do NOT ask the user for screenshots or to paste page text when website information is already available.
- Acknowledge When Website Exists But No Useful Content Was Collected (Scenario B):
  * If a website URL exists in the business profile or online presence records, but the scrape status is failed, skipped, pending, or no useful summary was gathered:
    State this limitation honestly and concisely: "I have your website link, but I don't have enough website content available to summarize its findings accurately." (incorporating the actual URL if known).
    Do NOT ask the user to share the same website URL again.
    Do NOT invent imaginary website findings or pretend that the website was scraped successfully.
- Handling New Product-Page URLs or Specific Subpages (Scenario C):
  * If the user shares a product-page URL (e.g., an individual product or service link) or asks about a specific subpage:
    Check whether that specific subpage was scraped and summarized in stored context.
    If only the general website was analyzed and the specific product page was not, explain what the existing website analysis covers and clearly state that the specific product page has not been verified.
    Do NOT claim that the product page was inspected or verified if it was not.
    Do NOT repeatedly ask for screenshots or pasted page text. Instead, answer concisely and proceed with the revision discussion based on what the user wants to adjust in their marketing plan.
- When No Website URL Exists Anywhere:
  * If no website URL exists in the business profile or online presence records, only ask for a website URL if it is genuinely necessary for refining a specific strategy section (e.g., landing page or SEO review). Otherwise, focus on refining the strategy directly.
- Guardrails Against Disclaimers and Browsing Claims:
  * Do NOT claim to have live website access or to have inspected a live page during the chat unless the stored context shows that operation was performed.
  * Avoid repetitive boilerplate disclaimers (e.g., "As an AI language model without browsing capabilities..."). Speak concisely, naturally, and professionally.
  * Questions about website findings, stored business details, or marketing strategy are informational inquiries (intent_type: "inquiry"). Answer the inquiry helpfully without modifying the marketing plan or clearing existing proposals.
  * NEVER classify questions about the user's website, URLs, or business findings as out_of_context.

REPLY FORMATTING AND PARAGRAPH STRUCTURE GUIDELINES (MANDATORY):
- Format your conversational `reply` into distinct, readable paragraphs separated by two newlines (\\n\\n). Do NOT output one dense, continuous block of text.
- Separate distinct conversational components into separate paragraphs at natural transitions:
  * Opening paragraph: Introductory explanation or context acknowledging the user's revision topic.
  * Middle paragraph(s): Specific details, implications, or strategic guidance.
  * Final paragraph: Clear follow-up question or call to action.
- Conversational prose only: Do NOT use bullet points (no asterisks '*', hyphens '-', or bullet symbols '•') or numbered lists for ordinary conversational responses, discussions, or clarification questions. Write in clean, cohesive sentences.
- Do NOT insert blank lines after every single sentence. Group logically connected sentences together into cohesive paragraphs.
- Preserve the exact meaning, tone, and strategic advice of the revision assistant.

OUTPUT FORMAT:
You MUST respond with a single valid JSON object with the following schema:
{{
  "reply": "Conversational markdown text formatted into distinct, readable paragraphs (separated by \\n\\n) without bullet points or numbered lists",
  "intent_type": "new_request" | "clarification_answer" | "confirmation" | "rejection" | "correction" | "inquiry" | "out_of_context",
  "status": "clarifying" | "proposal_ready" | "confirmed" | "rejected" | "in_progress",
  "normalized_intent": "Brief summary of what the user wants to revise",
  "affected_sections": ["section_key_1", "section_key_2"],
  "proposed_changes": [
    {{
      "section_id": "budget_considerations",
      "section_name": "Budget Considerations & Allocation",
      "proposed_change": "Specific proposed adjustment for this section",
      "unchanged_aspects": "What remains unchanged in this section"
    }}
  ],
  "unchanged_facts": ["Fact or section 1 remaining intact", "Fact 2 remaining intact"],
  "missing_information": ["Information item 1 still needed (empty if none)"],
  "current_clarification_question": "Clarification question text if status is clarifying, otherwise null",
  "is_ready_for_confirmation": true | false,
  "confirmation_status": "pending" | "confirmed" | "rejected" | "correction",
  "assumptions_and_constraints": ["Assumption 1 applied to proposal"]
}}"""


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

    system_prompt = f"""You are an expert Marketing Strategy Editor.
Your task is to generate the updated content for ONLY the specified strategy sections based on confirmed revision proposals.

BUSINESS PROFILE:
{biz_summary}

SECTIONS TO REVISE:
{sections_info}

STRICT SCHEMA RULES:
1. Return ONLY a single valid JSON object containing keys for ONLY the specified section keys: {json.dumps(affected_keys)}.
2. Do NOT include keys for unaffected sections.
3. For standard text sections ("business_overview", "target_audience_insights", "competitive_positioning", "value_proposition", "marketing_channels_and_tactics", "customer_acquisition_approach", "action_plan"):
   - Must be a JSON array of strings: ["Item 1...", "Item 2..."].
4. For "budget_considerations":
   - Must be a JSON array of objects with keys: "label" (string), "amount" (string, e.g. "₹30,000"), "percentage" (optional string, e.g. "50%"), "description" (string).
5. For "kpis":
   - Must be a JSON array of objects with keys: "metric" (string), "target" (string), "description" (string).
6. Preserved aspects: Ensure that specific aspects designated to remain intact are preserved faithfully.
7. Do NOT invent fake company products, budgets, or unsupported claims. Maintain total consistency with the business profile.
"""

    try:
        raw_response = get_llm_response(
            messages=[
                {"role": "user", "content": f"Generate updated content for affected sections: {', '.join(affected_keys)}."}
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

