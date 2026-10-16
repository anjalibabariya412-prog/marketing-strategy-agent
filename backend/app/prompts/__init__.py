"""
LLM Prompt templates, system prompts, and prompt builders for the Marketing Strategy Agent backend.
"""

from backend.app.prompts.strategy_generation import (
    STRATEGY_GENERATION_SYSTEM_PROMPT,
    build_strategy_generation_prompt,
)
from backend.app.prompts.gap_analysis import (
    GAP_ANALYSIS_SYSTEM_PROMPT,
    build_gap_analysis_prompt,
)
from backend.app.prompts.question_selection import (
    QUESTION_SELECTION_SYSTEM_PROMPT,
    QUESTION_GEN_SYSTEM_PROMPT,
    FALLBACK_QUESTIONS,
    FIRST_MESSAGE_SYSTEM_PROMPT,
    build_question_selection_prompt,
    build_question_generation_prompt,
    build_first_message_prompt,
    build_grounded_business_summary,
)
from backend.app.prompts.answer_processing import (
    INITIAL_CONTEXT_SYSTEM_PROMPT,
    ANSWER_CLASSIFICATION_SYSTEM_PROMPT,
    MERGED_TURN_SYSTEM_PROMPT,
    COMPETITOR_EXTRACTION_SYSTEM_PROMPT,
    build_initial_context_prompt,
    build_competitor_extraction_prompt,
    build_answer_classification_prompt,
    build_merged_turn_prompt,
)
from backend.app.prompts.revision import (
    format_out_of_context_response,
    build_revision_system_prompt,
    build_system_prompt,
    build_revision_editor_system_prompt,
    build_revision_editor_prompt,
)
from backend.app.prompts.pdf_extraction import (
    PDF_INSIGHT_SYSTEM_PROMPT,
    build_pdf_extraction_prompt,
)
from backend.app.prompts.online_presence import (
    ONLINE_PRESENCE_SYSTEM_PROMPT,
    build_online_presence_prompt,
)

__all__ = [
    # Strategy Generation
    "STRATEGY_GENERATION_SYSTEM_PROMPT",
    "build_strategy_generation_prompt",
    # Gap Analysis
    "GAP_ANALYSIS_SYSTEM_PROMPT",
    "build_gap_analysis_prompt",
    # Question Selection
    "QUESTION_SELECTION_SYSTEM_PROMPT",
    "QUESTION_GEN_SYSTEM_PROMPT",
    "FALLBACK_QUESTIONS",
    "FIRST_MESSAGE_SYSTEM_PROMPT",
    "build_question_selection_prompt",
    "build_question_generation_prompt",
    "build_first_message_prompt",
    "build_grounded_business_summary",
    # Answer Processing
    "INITIAL_CONTEXT_SYSTEM_PROMPT",
    "ANSWER_CLASSIFICATION_SYSTEM_PROMPT",
    "MERGED_TURN_SYSTEM_PROMPT",
    "COMPETITOR_EXTRACTION_SYSTEM_PROMPT",
    "build_initial_context_prompt",
    "build_competitor_extraction_prompt",
    "build_answer_classification_prompt",
    "build_merged_turn_prompt",
    # Revision
    "format_out_of_context_response",
    "build_revision_system_prompt",
    "build_system_prompt",
    "build_revision_editor_system_prompt",
    "build_revision_editor_prompt",
    # PDF Extraction
    "PDF_INSIGHT_SYSTEM_PROMPT",
    "build_pdf_extraction_prompt",
    # Online Presence
    "ONLINE_PRESENCE_SYSTEM_PROMPT",
    "build_online_presence_prompt",
]
