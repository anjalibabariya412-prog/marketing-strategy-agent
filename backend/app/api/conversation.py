from datetime import datetime, timezone
import asyncio
import json
import logging
import uuid
from typing import Callable, Optional
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from backend.app.core.progress_notifier import (
    emit_progress,
    set_progress_notifier,
    reset_progress_notifier,
)

from backend.app.schemas.agent.agent_state import MarketingAgentState
from backend.app.schemas.agent.business_context import BusinessContext
from backend.app.schemas.agent.document_context import HistoricalDocumentContext
from backend.app.schemas.agent.pdf_insights import PDFInsights
from backend.app.core.requirements_library import load_requirements_into_state
from backend.app.agent.graph import graph
from backend.app.schemas.api.conversation import (
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
from backend.app.services.chat_message_persistence import save_chat_message
from backend.app.services.strategy_persistence import save_strategy
from backend.app.db.session import SessionLocal
from backend.app.models.strategy import Strategy
from backend.app.models.strategy_request import StrategyRequest
from backend.app.models.business import Business
from backend.app.models.conversation import Conversation
from backend.app.models.uploaded_document import UploadedDocument
from backend.app.schemas.agent.marketing_strategy import MarketingStrategy

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


def execute_start_flow(
    request: StartRequest,
    on_progress: Optional[Callable[[str, Optional[str]], None]] = None,
) -> StartResponse:
    """
    Executes the start workflow synchronously, emitting real-time progress notifications
    when a progress notifier is provided or registered in context.
    """
    token = set_progress_notifier(on_progress)
    try:
        thread_id = str(uuid.uuid4())
        conv_id = uuid.UUID(thread_id)
        logger.info(f"Starting new conversation with thread_id: '{thread_id}'")

        if request.uploaded_document_id:
            emit_progress("Processing your uploaded document...", "Analyzing uploaded document insights and context.")
        else:
            emit_progress("Analyzing your business...", "Reviewing your business profile and goals.")

        # 1. Persist initial StrategyRequest, Business, and Conversation records
        strategy_request_id = None
        doc_summary = None
        doc_context = None
        try:
            with SessionLocal() as db:
                strategy_req = StrategyRequest(
                    status="in_progress",
                    current_stage="initial_form",
                )
                db.add(strategy_req)
                db.flush()

                # Associate uploaded document if provided
                if request.uploaded_document_id:
                    uploaded_doc = db.query(UploadedDocument).filter(
                        UploadedDocument.id == request.uploaded_document_id
                    ).first()
                    if not uploaded_doc:
                        logger.warning(f"Uploaded document ID '{request.uploaded_document_id}' not found.")
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Uploaded document '{request.uploaded_document_id}' not found."
                        )
                    if uploaded_doc.strategy_request_id and uploaded_doc.strategy_request_id != strategy_req.id:
                        logger.warning(
                            f"Uploaded document '{request.uploaded_document_id}' is already linked to StrategyRequest '{uploaded_doc.strategy_request_id}'."
                        )
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="The specified document is already associated with another strategy request."
                        )
                    uploaded_doc.strategy_request_id = strategy_req.id
                    doc_summary = uploaded_doc.executive_summary

                    pdf_insights_obj = None
                    if uploaded_doc.pdf_insights and isinstance(uploaded_doc.pdf_insights, dict):
                        try:
                            pdf_insights_obj = PDFInsights.model_validate(uploaded_doc.pdf_insights)
                        except Exception as val_err:
                            logger.warning(f"Could not parse pdf_insights for document {uploaded_doc.id}: {val_err}")

                    doc_context = HistoricalDocumentContext(
                        filename=uploaded_doc.filename,
                        is_marketing_relevant=uploaded_doc.is_marketing_relevant,
                        document_type=uploaded_doc.document_type,
                        relevance_reason=uploaded_doc.relevance_reason,
                        executive_summary=uploaded_doc.executive_summary,
                        pdf_insights=pdf_insights_obj,
                    )
                    logger.info(f"Associated UploadedDocument '{uploaded_doc.id}' with StrategyRequest '{strategy_req.id}'.")

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
        except HTTPException:
            raise
        except Exception as db_err:
            logger.error(f"Database insertion failed for StrategyRequest/Business/Conversation: {db_err}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred while starting the conversation."
            )

        # 2. Build initial context from structured request fields
        past_doc_val = request.past_marketing_document or doc_summary
        context = BusinessContext(
            company_name=request.company_name,
            product_or_service=request.product_or_service,
            marketing_goal=request.marketing_goal,
            target_audience=request.target_audience,
            budget_resources=request.budget_resources,
            current_marketing_channels=request.current_marketing_channels,
            website_social_links=request.website_social_links,
            past_marketing_document=past_doc_val
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
                    emit_progress("Analyzing your online presence...", "Synthesizing online profile information.")
                    online_presence_ctx = process_online_presence(
                        scraped_results,
                        context,
                        strategy_request_id=strategy_request_id,
                        conversation_id=conv_id,
                    )

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
            past_marketing_doc=past_doc_val,
        )

        # 3. Create state and load requirements
        state = MarketingAgentState(
            business_context=context,
            online_presence_context=online_presence_ctx,
            document_context=doc_context,
            thread_id=thread_id,
            strategy_request_id=str(strategy_request_id) if strategy_request_id else None,
        )
        load_requirements_into_state(state)

        # 4. Invoke graph with thread_id config
        try:
            emit_progress("Analyzing your business...", "Evaluating your marketing requirements.")
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
                strategy_request_id=str(strategy_request_id) if strategy_request_id else None,
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

        emit_progress("Preparing your marketing strategy...", "Finalizing your personalized marketing strategy.")

        return StartResponse(
            thread_id=thread_id,
            status="completed",
            question=None,
            requirement_id=None,
            session_intro=session_intro_text,
            strategy_request_id=str(strategy_request_id) if strategy_request_id else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in execute_start_flow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while starting the conversation."
        )
    finally:
        reset_progress_notifier(token)


async def stream_start_conversation(request: StartRequest):
    """
    Asynchronous SSE generator streaming dynamic progress events during conversation start.
    Yields 'progress', 'complete', and 'error' SSE frames.
    """
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue = asyncio.Queue()

    def on_progress(message: str, detail: Optional[str] = None):
        payload = {"type": "progress", "message": message, "detail": detail}
        loop.call_soon_threadsafe(event_queue.put_nowait, payload)

    def worker():
        try:
            resp = execute_start_flow(request, on_progress=on_progress)
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {"type": "complete", "data": resp.model_dump()},
            )
        except HTTPException as http_exc:
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {"type": "error", "error": http_exc.detail},
            )
        except Exception as exc:
            logger.error(f"Error in stream_start_conversation worker: {exc}")
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {"type": "error", "error": "An unexpected error occurred while starting the conversation."},
            )
        finally:
            loop.call_soon_threadsafe(event_queue.put_nowait, None)

    worker_task = asyncio.create_task(asyncio.to_thread(worker))

    try:
        while True:
            item = await event_queue.get()
            if item is None:
                break
            if item["type"] == "progress":
                yield f"event: progress\ndata: {json.dumps(item)}\n\n"
            elif item["type"] == "complete":
                yield f"event: complete\ndata: {json.dumps(item['data'])}\n\n"
            elif item["type"] == "error":
                yield f"event: error\ndata: {json.dumps({'error': item['error']})}\n\n"
    finally:
        await worker_task


@router.post("/start", status_code=status.HTTP_200_OK)
def start_conversation(request: StartRequest, http_request: Request):
    """
    Starts a new marketing strategy conversation using structured initial business context,
    persists StrategyRequest, Business, and Conversation records, initializes state, and invokes the LangGraph workflow.
    Streams SSE updates when requested via Accept: text/event-stream or ?stream=true.
    """
    accept_header = http_request.headers.get("accept", "")
    stream_param = http_request.query_params.get("stream", "").lower()

    if "text/event-stream" in accept_header or stream_param in ("true", "1"):
        return StreamingResponse(
            stream_start_conversation(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return execute_start_flow(request)


@router.post("/start/stream")
def start_conversation_stream(request: StartRequest):
    """Explicit SSE streaming endpoint for conversation initialization."""
    return StreamingResponse(
        stream_start_conversation(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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

        # Check database Strategy records first
        try:
            thread_uuid = uuid.UUID(thread_id.strip())
            with SessionLocal() as db:
                conv = db.query(Conversation).filter(Conversation.id == thread_uuid).first()
                if not conv:
                    sr = db.query(StrategyRequest).filter(StrategyRequest.id == thread_uuid).first()
                    if sr and sr.conversation:
                        conv = sr.conversation

                if conv:
                    strategy_request_id_str = str(conv.strategy_request_id) if conv.strategy_request_id else None
                    latest_db_strategy = (
                        db.query(Strategy)
                        .filter(Strategy.conversation_id == conv.id)
                        .order_by(Strategy.version.desc(), Strategy.created_at.desc())
                        .first()
                    )
                    if latest_db_strategy and latest_db_strategy.strategy_json:
                        validated_strat = MarketingStrategy.model_validate(latest_db_strategy.strategy_json)
                        return StrategyResponse(
                            thread_id=thread_id.strip(),
                            status="completed",
                            strategy=validated_strat,
                            strategy_request_id=strategy_request_id_str,
                        )
        except Exception as db_lookup_err:
            logger.debug(f"Database strategy lookup error, falling back to graph checkpoint: {db_lookup_err}")

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

        strategy_request_id_str = None
        try:
            with SessionLocal() as db:
                conv = db.query(Conversation).filter(Conversation.id == uuid.UUID(thread_id.strip())).first()
                if conv and conv.strategy_request_id:
                    strategy_request_id_str = str(conv.strategy_request_id)
        except Exception as lookup_err:
            logger.debug(f"Could not resolve strategy_request_id in get_strategy: {lookup_err}")

        return StrategyResponse(
            thread_id=thread_id.strip(),
            status="completed",
            strategy=agent_state.final_strategy,
            strategy_request_id=strategy_request_id_str,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in GET /strategy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the strategy."
        )
