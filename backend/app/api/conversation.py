from datetime import datetime, timezone
import asyncio
import logging
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, status
from langgraph.types import Command

from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.business_context import BusinessContext
from backend.app.core.requirements_library import load_requirements_into_state
from backend.app.agent.graph import graph
from backend.app.schemas.conversation import (
    StartRequest,
    StartResponse,
    ReplyRequest,
    ReplyResponse,
    StrategyResponse
)
from backend.app.services.url_parser import parse_website_social_links
from backend.app.services.apify_service import scrape_parsed_urls
from backend.app.services.online_presence_processor import (
    process_online_presence,
    save_online_presence_sources,
)
from backend.app.services.requirement_persistence import save_information_requirements
from backend.app.services.chat_message_persistence import save_chat_message
from backend.app.services.strategy_persistence import save_strategy
from backend.app.db.session import SessionLocal
from backend.app.db.orm.strategy_request import StrategyRequest
from backend.app.db.orm.business import Business
from backend.app.db.orm.conversation import Conversation

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Conversation"])


def determine_session_intro(
    parsed_urls: Optional[list] = None,
    scraped_results: Optional[list] = None,
    past_marketing_doc: Optional[str] = None,
) -> str:
    """
    Determines the appropriate session introduction sentence based strictly on successfully available context.

    Cases:
    - CASE A: Website successfully analyzed (no social media)
      -> "A few short questions based on what we found on your website."
    - CASE B: Website + Social Media successfully analyzed (or multiple online presence sources)
      -> "A few short questions based on what we found about your online presence."
    - CASE C: Only Social Media successfully analyzed (no website)
      -> "A few short questions based on what we found on your social media presence."
    - CASE D: Past marketing document / PDF provided (and no website/social succeeded)
      -> "A few short questions based on the information you've shared about your business."
    - CASE E & F: No website/social succeeded (or failed) and no PDF provided
      -> "A few short questions to better understand your business and create a personalized marketing strategy."
    """
    website_succeeded = False
    social_succeeded_count = 0

    if parsed_urls and scraped_results:
        for parsed_url, res in zip(parsed_urls, scraped_results):
            if res and getattr(res, "success", False) and getattr(res, "data", None):
                platform = parsed_url.platform if parsed_url and getattr(parsed_url, "platform", None) else "unknown"
                if platform == "website":
                    website_succeeded = True
                elif platform in ("instagram", "facebook", "linkedin", "social"):
                    social_succeeded_count += 1

    if website_succeeded and social_succeeded_count > 0:
        return "A few short questions based on what we found about your online presence."
    elif website_succeeded:
        return "A few short questions based on what we found on your website."
    elif social_succeeded_count > 0:
        return "A few short questions based on what we found on your social media presence."
    elif past_marketing_doc and past_marketing_doc.strip():
        return "A few short questions based on the information you've shared about your business."
    else:
        return "A few short questions to better understand your business and create a personalized marketing strategy."


@router.post("/start", response_model=StartResponse, status_code=status.HTTP_200_OK)
def start_conversation(request: StartRequest):
    """
    Starts a new marketing strategy conversation using structured initial business context,
    persists StrategyRequest, Business, and Conversation records, initializes state, and invokes the LangGraph workflow.
    """
    try:
        thread_id = str(uuid.uuid4())
        conv_id = uuid.UUID(thread_id)
        logger.info(f"Starting new conversation with thread_id: '{thread_id}'")

        # 1. Persist initial StrategyRequest, Business, and Conversation records
        strategy_request_id = None
        try:
            with SessionLocal() as db:
                strategy_req = StrategyRequest(
                    status="in_progress",
                    current_stage="initial_form",
                )
                db.add(strategy_req)
                db.flush()

                budget_amt = request.budget_resources.amount if request.budget_resources else None
                budget_curr = request.budget_resources.currency if request.budget_resources else None

                business_rec = Business(
                    strategy_request_id=strategy_req.id,
                    company_name=request.company_name,
                    product_or_service=request.product_or_service,
                    marketing_goal=request.marketing_goal,
                    target_audience=request.target_audience,
                    budget_amount=budget_amt,
                    budget_currency=budget_curr,
                    current_marketing_channels=request.current_marketing_channels,
                    website_social_links=request.website_social_links,
                )
                db.add(business_rec)

                conversation_rec = Conversation(
                    id=conv_id,
                    strategy_request_id=strategy_req.id,
                    status="waiting_for_reply",
                    current_requirement_id=None,
                    is_sufficient=False,
                )
                db.add(conversation_rec)

                db.commit()
                strategy_request_id = strategy_req.id
                logger.info(
                    f"Persisted StrategyRequest '{strategy_request_id}', Business, and Conversation '{conv_id}' records successfully."
                )
        except Exception as db_err:
            logger.error(f"Database insertion failed for StrategyRequest/Business/Conversation: {db_err}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred while starting the conversation."
            )

        # 2. Build initial context from structured request fields
        context = BusinessContext(
            company_name=request.company_name,
            product_or_service=request.product_or_service,
            marketing_goal=request.marketing_goal,
            target_audience=request.target_audience,
            budget_resources=request.budget_resources,
            current_marketing_channels=request.current_marketing_channels,
            website_social_links=request.website_social_links,
            past_marketing_document=request.past_marketing_document
        )

        # Process online presence if website/social links are provided
        parsed_urls = None
        scraped_results = None
        online_presence_ctx = None
        if request.website_social_links:
            try:
                parsed_urls = parse_website_social_links(request.website_social_links)
                if parsed_urls:
                    scraped_results = asyncio.run(scrape_parsed_urls(parsed_urls))
                    online_presence_ctx = process_online_presence(scraped_results, context)

                    # Persist online presence sources to database
                    if strategy_request_id:
                        try:
                            save_online_presence_sources(
                                strategy_request_id=strategy_request_id,
                                parsed_urls=parsed_urls,
                                scraped_results=scraped_results,
                                online_presence_ctx=online_presence_ctx,
                            )
                        except Exception as save_err:
                            logger.error(f"Error persisting online presence sources: {save_err}")
            except Exception as e:
                logger.error(f"Failed to process online presence for website_social_links: {e}")

        # Determine dynamic session intro text
        session_intro_text = determine_session_intro(
            parsed_urls=parsed_urls,
            scraped_results=scraped_results,
            past_marketing_doc=request.past_marketing_document,
        )

        # 3. Create state and load requirements
        state = MarketingAgentState(
            business_context=context,
            online_presence_context=online_presence_ctx,
            thread_id=thread_id
        )
        load_requirements_into_state(state)

        # 4. Invoke graph with thread_id config
        try:
            config = {"configurable": {"thread_id": thread_id}}
            output = graph.invoke(state, config=config)
        except Exception as graph_err:
            logger.error(f"LangGraph invocation failed in /start: {graph_err}")
            if strategy_request_id:
                try:
                    with SessionLocal() as db:
                        sr = db.query(StrategyRequest).filter(StrategyRequest.id == strategy_request_id).first()
                        if sr:
                            sr.status = "failed"
                        conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
                        if conv:
                            conv.status = "failed"
                        db.commit()
                except Exception as fail_err:
                    logger.error(f"Failed to update failure status in DB: {fail_err}")
            raise

        # 5. Persist InformationRequirements to database
        try:
            snapshot = graph.get_state(config)
            latest_reqs = state.requirements
            if snapshot and snapshot.values:
                latest_state = MarketingAgentState.model_validate(snapshot.values)
                latest_reqs = latest_state.requirements
            save_information_requirements(conv_id, latest_reqs)
        except Exception as req_err:
            logger.error(f"Failed to persist information requirements in /start: {req_err}")

        # 6. Update StrategyRequest stage / Conversation record and handle response
        if isinstance(output, dict) and "__interrupt__" in output:
            interrupt_payload = output["__interrupt__"][0].value
            question_text = interrupt_payload.get("question") if isinstance(interrupt_payload, dict) else str(interrupt_payload)
            req_id = interrupt_payload.get("requirement_id") if isinstance(interrupt_payload, dict) else None

            # Persist first agent question to chat_messages
            try:
                save_chat_message(
                    conversation_id=conv_id,
                    sender="agent",
                    message_type="question",
                    requirement_id=req_id,
                    content=question_text,
                )
            except Exception as msg_err:
                logger.error(f"Error persisting first agent question in /start: {msg_err}")

            try:
                with SessionLocal() as db:
                    if strategy_request_id:
                        sr = db.query(StrategyRequest).filter(StrategyRequest.id == strategy_request_id).first()
                        if sr:
                            sr.current_stage = "dynamic_questions"
                    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
                    if conv:
                        conv.status = "waiting_for_reply"
                        conv.current_requirement_id = req_id
                        conv.is_sufficient = False
                    db.commit()
            except Exception as stage_err:
                logger.error(f"Failed to update StrategyRequest/Conversation stage in /start: {stage_err}")

            return StartResponse(
                thread_id=thread_id,
                status="waiting_for_reply",
                question=question_text,
                requirement_id=req_id,
                session_intro=session_intro_text,
            )

        # Persist final strategy if generated in /start
        try:
            snapshot = graph.get_state(config)
            if snapshot and snapshot.values:
                latest_state = MarketingAgentState.model_validate(snapshot.values)
                if latest_state.final_strategy:
                    save_strategy(conv_id, latest_state.final_strategy)
        except Exception as strat_err:
            logger.error(f"Failed to persist strategy in /start: {strat_err}")

        try:
            with SessionLocal() as db:
                if strategy_request_id:
                    sr = db.query(StrategyRequest).filter(StrategyRequest.id == strategy_request_id).first()
                    if sr:
                        sr.status = "completed"
                        sr.current_stage = "completed"
                        sr.completed_at = datetime.now(timezone.utc)
                conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
                if conv:
                    conv.status = "completed"
                    conv.current_requirement_id = None
                    conv.is_sufficient = True
                db.commit()
        except Exception as stage_err:
            logger.error(f"Failed to update StrategyRequest/Conversation completion status in /start: {stage_err}")

        return StartResponse(
            thread_id=thread_id,
            status="completed",
            question=None,
            requirement_id=None,
            session_intro=session_intro_text,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in POST /start: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while starting the conversation."
        )


@router.post("/reply", response_model=ReplyResponse, status_code=status.HTTP_200_OK)
def reply_to_question(request: ReplyRequest):
    """
    Submits a user answer to the active clarifying question and resumes the graph for the given thread_id.
    """
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        snapshot = graph.get_state(config)

        # 1. Validate thread existence
        if not snapshot or not snapshot.values:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation thread '{request.thread_id}' not found."
            )

        # 2. Validate that graph is expecting a reply (currently interrupted)
        if not snapshot.next:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Conversation thread is not currently waiting for a reply."
            )

        conv_id = None
        try:
            conv_id = uuid.UUID(request.thread_id)
        except ValueError:
            pass

        # 3. Retrieve active requirement_id before resuming graph
        active_req_id = None
        if conv_id:
            try:
                with SessionLocal() as db:
                    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
                    if conv:
                        active_req_id = conv.current_requirement_id
            except Exception as db_err:
                logger.error(f"Failed to fetch active requirement_id from DB in /reply: {db_err}")

        if not active_req_id and snapshot.values:
            active_req_id = snapshot.values.get("active_requirement_id")

        # 4. Persist user answer to chat_messages before resuming graph
        if conv_id:
            try:
                save_chat_message(
                    conversation_id=conv_id,
                    sender="user",
                    message_type="answer",
                    requirement_id=active_req_id,
                    content=request.message,
                )
            except Exception as msg_err:
                logger.error(f"Error persisting user answer in /reply: {msg_err}")

        # 5. Resume graph using Command(resume=message)
        logger.info(f"Resuming conversation '{request.thread_id}' with user reply.")
        try:
            output = graph.invoke(Command(resume=request.message), config=config)
        except Exception as graph_err:
            logger.error(f"LangGraph invocation failed in /reply: {graph_err}")
            if conv_id:
                try:
                    with SessionLocal() as db:
                        conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
                        if conv:
                            conv.status = "failed"
                            db.commit()
                except Exception as fail_err:
                    logger.error(f"Failed to update Conversation status to failed in /reply: {fail_err}")
            raise

        # 6. Persist updated InformationRequirements to database
        if conv_id:
            try:
                snapshot_after = graph.get_state(config)
                if snapshot_after and snapshot_after.values:
                    latest_state = MarketingAgentState.model_validate(snapshot_after.values)
                    save_information_requirements(conv_id, latest_state.requirements)
            except Exception as req_err:
                logger.error(f"Failed to persist information requirements in /reply: {req_err}")

        # 7. Handle next interrupt or completion and update Conversation record
        if isinstance(output, dict) and "__interrupt__" in output:
            interrupt_payload = output["__interrupt__"][0].value
            question_text = interrupt_payload.get("question") if isinstance(interrupt_payload, dict) else str(interrupt_payload)
            req_id = interrupt_payload.get("requirement_id") if isinstance(interrupt_payload, dict) else None

            # Persist next agent question to chat_messages
            if conv_id:
                try:
                    save_chat_message(
                        conversation_id=conv_id,
                        sender="agent",
                        message_type="question",
                        requirement_id=req_id,
                        content=question_text,
                    )
                except Exception as msg_err:
                    logger.error(f"Error persisting next agent question in /reply: {msg_err}")

                try:
                    with SessionLocal() as db:
                        conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
                        if conv:
                            conv.status = "waiting_for_reply"
                            conv.current_requirement_id = req_id
                            conv.is_sufficient = False
                            db.commit()
                except Exception as db_err:
                    logger.error(f"Failed to update Conversation record in /reply: {db_err}")

            return ReplyResponse(
                thread_id=request.thread_id,
                status="waiting_for_reply",
                question=question_text,
                requirement_id=req_id
            )

        # Completed flow
        if conv_id:
            try:
                snapshot_after = graph.get_state(config)
                if snapshot_after and snapshot_after.values:
                    latest_state = MarketingAgentState.model_validate(snapshot_after.values)
                    if latest_state.final_strategy:
                        save_strategy(conv_id, latest_state.final_strategy)
            except Exception as strat_err:
                logger.error(f"Failed to persist strategy in /reply: {strat_err}")

            try:
                with SessionLocal() as db:
                    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
                    if conv:
                        conv.status = "completed"
                        conv.current_requirement_id = None
                        conv.is_sufficient = True
                        db.commit()
            except Exception as db_err:
                logger.error(f"Failed to update Conversation completion in /reply: {db_err}")

        return ReplyResponse(
            thread_id=request.thread_id,
            status="completed",
            question=None,
            requirement_id=None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in POST /reply: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the reply."
        )


@router.get("/strategy", response_model=StrategyResponse, status_code=status.HTTP_200_OK)
def get_strategy(thread_id: str):
    """
    Retrieves the generated final marketing strategy for an existing conversation thread.
    """
    try:
        if not thread_id or not thread_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="thread_id query parameter is required."
            )

        config = {"configurable": {"thread_id": thread_id.strip()}}
        snapshot = graph.get_state(config)

        # 1. Validate thread existence
        if not snapshot or not snapshot.values:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation thread '{thread_id}' not found."
            )

        # 2. Parse state and check for final_strategy
        agent_state = MarketingAgentState.model_validate(snapshot.values)
        if not agent_state.final_strategy:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Strategy has not been generated yet for this conversation."
            )

        return StrategyResponse(
            thread_id=thread_id.strip(),
            status="completed",
            strategy=agent_state.final_strategy
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in GET /strategy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the strategy."
        )
