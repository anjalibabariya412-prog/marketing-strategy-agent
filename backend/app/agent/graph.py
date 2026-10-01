import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langgraph.types import interrupt

from backend.app.core.config import settings
from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.information_requirement import RequirementStatus
from backend.app.agent.gap_analysis import analyze_relevance
from backend.app.agent.question_selection import prepare_next_question
from backend.app.agent.answer_processing import process_answer_for_requirement, process_answer_and_plan_next
from backend.app.agent.sufficiency_check import is_sufficient
from backend.app.agent.strategy_generation import generate_strategy

logger = logging.getLogger(__name__)


def analyze_node(state: MarketingAgentState) -> MarketingAgentState:
    """
    Graph Node: Executes relevance gap analysis against missing requirements,
    updating status to NOT_RELEVANT for inapplicable requirements.
    Runs at most once per conversation session (retrying if analyze_relevance returns False).
    """
    if not state.analysis_done:
        logger.info("Graph Node [analyze]: Running analyze_relevance()")
        success = analyze_relevance(state)
        if success:
            state.analysis_done = True
    else:
        logger.info("Graph Node [analyze]: Skipping analyze_relevance() because analysis_done is True")
    return state



def route_after_analyze(state: MarketingAgentState) -> str:
    """
    Conditional Edge Router: Evaluates if sufficient information has been collected.
    Returns 'generate' if is_sufficient(state) is True, otherwise returns 'ask'.
    """
    sufficient = is_sufficient(state)
    state.is_sufficient = sufficient
    if sufficient:
        logger.info("Graph Router [route_after_analyze]: Sufficiency reached -> Routing to 'generate'")
        return "generate"

    logger.info("Graph Router [route_after_analyze]: Insufficient information -> Routing to 'ask'")
    return "ask"


def ask_node(state: MarketingAgentState) -> MarketingAgentState:
    """
    Graph Node: Generates question if needed (using pending question if available),
    pauses for user answer via interrupt(), and processes the user's answer upon resumption.
    """
    logger.info(
        f"Graph Node [ask] ENTRY | thread_id: '{state.thread_id}' | "
        f"active_req: '{state.active_requirement_id}' | current_question: '{state.current_question}' | "
        f"pending_req: '{state.pending_requirement_id}'"
    )

    # 1. Prepare next question if one is not already pending (e.g. first time or after answer processing)
    if not state.current_question or not state.active_requirement_id:
        if state.pending_question and state.pending_requirement_id:
            pending_req = state.get_requirement_by_id(state.pending_requirement_id)
            asked_ids = {turn.requirement_id for turn in state.qa_history if turn.requirement_id} if state.qa_history else set()
            if pending_req and pending_req.status == RequirementStatus.UNKNOWN and pending_req.id not in asked_ids:
                logger.info(f"Graph Node [ask]: Using pre-planned pending question for requirement '{state.pending_requirement_id}'")
                state.current_question = state.pending_question
                state.active_requirement_id = state.pending_requirement_id
                state.pending_question = None
                state.pending_requirement_id = None
            else:
                logger.info("Graph Node [ask]: Pending requirement is no longer valid/UNKNOWN -> Falling back to prepare_next_question()")
                state.pending_question = None
                state.pending_requirement_id = None
                prepare_next_question(state)
        else:
            logger.info("Graph Node [ask]: No pending question -> Calling prepare_next_question()")
            state.pending_question = None
            state.pending_requirement_id = None
            prepare_next_question(state)
    else:
        logger.info(
            f"Graph Node [ask]: Skipping question prep because current question already active -> "
            f"active_req: '{state.active_requirement_id}'"
        )

    if not state.current_question:
        logger.warning("Graph Node [ask]: No current question generated.")
        return state

    # 2. Interrupt graph execution and wait for user's answer
    logger.info(f"Graph Node [ask]: Interrupting for user answer on requirement '{state.active_requirement_id}'")
    user_answer = interrupt({
        "question": state.current_question,
        "requirement_id": state.active_requirement_id
    })

    # 3. Resumed: Process the user answer and optionally plan next question in one call
    if user_answer:
        if settings.merged_turn_call_enabled:
            logger.info(f"Graph Node [ask]: Resumed with answer -> Calling process_answer_and_plan_next()")
            process_answer_and_plan_next(state, str(user_answer))
        else:
            logger.info(f"Graph Node [ask]: Resumed with answer -> Calling process_answer_for_requirement()")
            process_answer_for_requirement(state, str(user_answer))

    return state


def generate_node(state: MarketingAgentState) -> MarketingAgentState:
    """
    Graph Node: Synthesizes business context and known facts to generate the final MarketingStrategy,
    storing the output in state.final_strategy.
    """
    logger.info("Graph Node [generate]: Running generate_strategy()")
    state.pending_question = None
    state.pending_requirement_id = None
    strategy = generate_strategy(state)
    state.final_strategy = strategy
    state.is_sufficient = True
    return state


# Build StateGraph using MarketingAgentState schema
builder = StateGraph(MarketingAgentState)

# Add real nodes
builder.add_node("analyze", analyze_node)
builder.add_node("ask", ask_node)
builder.add_node("generate", generate_node)

# Entry point: START -> analyze
builder.add_edge(START, "analyze")

# Conditional Edge: analyze -> route_after_analyze -> ask OR generate
builder.add_conditional_edges(
    "analyze",
    route_after_analyze,
    {
        "ask": "ask",
        "generate": "generate"
    }
)

# Loop back from ask to analyze to re-evaluate after user answer
builder.add_edge("ask", "analyze")
builder.add_edge("generate", END)

# Postgres checkpointer for persistent session state across interrupts and server restarts
pool = ConnectionPool(
    conninfo=settings.database_url,
    max_size=20,
    kwargs={"autocommit": True, "prepare_threshold": 0},
)
checkpointer = PostgresSaver(pool)
checkpointer.setup()

# Compile graph with PostgresSaver checkpointer
graph = builder.compile(checkpointer=checkpointer)
