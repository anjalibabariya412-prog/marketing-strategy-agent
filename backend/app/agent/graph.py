import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langgraph.types import interrupt

from backend.app.core.config import settings
from backend.app.models.agent_state import MarketingAgentState
from backend.app.agent.gap_analysis import analyze_relevance
from backend.app.agent.question_selection import prepare_next_question
from backend.app.agent.answer_processing import process_answer_for_requirement
from backend.app.agent.sufficiency_check import is_sufficient
from backend.app.agent.strategy_generation import generate_strategy

logger = logging.getLogger(__name__)


def analyze_node(state: MarketingAgentState) -> MarketingAgentState:
    """
    Graph Node: Executes relevance gap analysis against missing requirements,
    updating status to NOT_RELEVANT for inapplicable requirements.
    Runs at most once per conversation session.
    """
    if not state.analysis_done:
        logger.info("Graph Node [analyze]: Running analyze_relevance()")
        analyze_relevance(state)
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
    Graph Node: Generates question if needed, pauses for user answer via interrupt(),
    and processes the user's answer via process_answer_for_requirement upon resumption.
    """
    logger.info(
        f"Graph Node [ask] ENTRY | thread_id: '{state.thread_id}' | "
        f"active_req: '{state.active_requirement_id}' | current_question: '{state.current_question}'"
    )

    # 1. Prepare next question ONLY if one is not already pending (e.g. first time, not resume)
    if not state.current_question or not state.active_requirement_id:
        logger.info("Graph Node [ask]: Preparing next question (prepare_next_question CALLED)")
        prepare_next_question(state)
        logger.info(
            f"Graph Node [ask]: After prepare_next_question -> active_req: '{state.active_requirement_id}', "
            f"question: '{state.current_question}'"
        )
    else:
        logger.info(
            f"Graph Node [ask]: Skipping prepare_next_question() because question already pending -> "
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

    # 3. Resumed: Process the user answer using LLM classification and state update
    logger.info(f"Graph Node [ask]: Resumed with user answer: '{user_answer}' on active_req: '{state.active_requirement_id}'")
    if user_answer:
        process_answer_for_requirement(state, str(user_answer))

    return state


def generate_node(state: MarketingAgentState) -> MarketingAgentState:
    """
    Graph Node: Synthesizes business context and known facts to generate the final MarketingStrategy,
    storing the output in state.final_strategy.
    """
    logger.info("Graph Node [generate]: Running generate_strategy()")
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
