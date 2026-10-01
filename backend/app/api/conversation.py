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
from backend.app.services.online_presence_processor import process_online_presence

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Conversation"])


@router.post("/start", response_model=StartResponse, status_code=status.HTTP_200_OK)
def start_conversation(request: StartRequest):
    """
    Starts a new marketing strategy conversation using structured initial business context,
    initializes state, and invokes the LangGraph workflow until it pauses on the first question or completes.
    """
    try:
        thread_id = str(uuid.uuid4())
        logger.info(f"Starting new conversation with thread_id: '{thread_id}'")

        # 1. Build initial context from structured request fields
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
        online_presence_ctx = None
        if request.website_social_links:
            try:
                parsed_urls = parse_website_social_links(request.website_social_links)
                if parsed_urls:
                    scraped_results = asyncio.run(scrape_parsed_urls(parsed_urls))
                    online_presence_ctx = process_online_presence(scraped_results, context)
            except Exception as e:
                logger.error(f"Failed to process online presence for website_social_links: {e}")

        # 2. Create state and load requirements
        state = MarketingAgentState(
            business_context=context,
            online_presence_context=online_presence_ctx,
            thread_id=thread_id
        )
        load_requirements_into_state(state)

        # 3. Invoke graph with thread_id config
        config = {"configurable": {"thread_id": thread_id}}
        output = graph.invoke(state, config=config)

        # 4. Handle interrupt / response
        if isinstance(output, dict) and "__interrupt__" in output:
            interrupt_payload = output["__interrupt__"][0].value
            question_text = interrupt_payload.get("question") if isinstance(interrupt_payload, dict) else str(interrupt_payload)
            req_id = interrupt_payload.get("requirement_id") if isinstance(interrupt_payload, dict) else None

            return StartResponse(
                thread_id=thread_id,
                status="waiting_for_reply",
                question=question_text,
                requirement_id=req_id
            )

        return StartResponse(
            thread_id=thread_id,
            status="completed",
            question=None,
            requirement_id=None
        )

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

        # 3. Resume graph using Command(resume=message)
        logger.info(f"Resuming conversation '{request.thread_id}' with user reply.")
        output = graph.invoke(Command(resume=request.message), config=config)

        # 4. Handle next interrupt or completion
        if isinstance(output, dict) and "__interrupt__" in output:
            interrupt_payload = output["__interrupt__"][0].value
            question_text = interrupt_payload.get("question") if isinstance(interrupt_payload, dict) else str(interrupt_payload)
            req_id = interrupt_payload.get("requirement_id") if isinstance(interrupt_payload, dict) else None

            return ReplyResponse(
                thread_id=request.thread_id,
                status="waiting_for_reply",
                question=question_text,
                requirement_id=req_id
            )

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
