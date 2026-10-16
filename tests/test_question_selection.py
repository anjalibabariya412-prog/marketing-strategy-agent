import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.schemas.agent.agent_state import MarketingAgentState
from backend.app.schemas.agent.business_context import BusinessContext
from backend.app.schemas.agent.information_requirement import RequirementStatus
from backend.app.core.requirements_library import load_requirements_into_state
from backend.app.agent.gap_analysis import analyze_relevance
from backend.app.agent.question_selection import (
    select_next_requirement,
    generate_question,
    prepare_next_question,
    _get_default_fallback
)


def test_question_selection():
    print("=== Testing Task 5.1 & 5.2: Requirement Selection, Question Generation & State Wiring ===")

    state = MarketingAgentState()
    load_requirements_into_state(state)
    state.business_context = BusinessContext(
        company_name="CleanSeas Ocean Cleanup",
        product_or_service="Free ocean plastic cleanup drives and community environmental workshops",
        marketing_goal="Recruit 500 volunteer cleanup captains across coastal cities",
        target_audience="Environmentally conscious college students aged 18-30"
    )

    analyze_relevance(state)
    selected_req = select_next_requirement(state)
    assert selected_req is not None, "Expected requirement selection to return a requirement."

    generated_q = generate_question(state, selected_req)
    assert generated_q is not None and len(generated_q) > 0, "Expected generated question string."

    wired_state = MarketingAgentState(business_context=state.business_context)
    load_requirements_into_state(wired_state)
    analyze_relevance(wired_state)

    q_text = prepare_next_question(wired_state)
    assert wired_state.active_requirement_id is not None
    assert wired_state.current_question is not None

    print("=== Question Selection Tests Passed Successfully! ===")


import json
from unittest.mock import patch
from backend.app.services.llm_service import LLMServiceError
from backend.app.agent.question_selection import generate_first_contextual_message
from backend.app.prompts.question_selection import FIRST_MESSAGE_SYSTEM_PROMPT

REMOVED_TRANSITIONS = [
    "with that in mind",
    "to help shape the strategy around what sets your business apart",
    "to tailor our strategy to your audience's decision-making process",
    "to position your business effectively against alternatives",
    "to focus our marketing efforts in the right areas",
    "to ensure our campaign recommendations align with your commercial model",
    "to design marketing touchpoints that lead to actual sales",
    "to help shape the strategy around your business",
]

REMOVED_QUESTIONS = [
    "what makes your product or service distinctly better or different from alternative solutions?",
    "what specific problem are your target customers trying to solve, and what triggers them to look for a solution?",
    "who are your main direct competitors, and how does your offering differ from theirs?",
    "what is the specific geographical or industry market you are focusing on?",
    "what are your main pricing tiers, and what does your core offer include?",
    "how do prospective customers typically find you and what are the main steps before they buy?",
]


def test_generate_first_contextual_message_dynamic_no_predefined_transition_or_questions():
    """
    Verifies that the first message is generated dynamically:
    - No predefined transitions (e.g., 'With that in mind', 'To help shape the strategy...')
    - No predefined question templates from the removed dictionary
    - Exactly one question is returned targeting a valid candidate requirement
    - No greeting or context summary duplication
    """
    state = MarketingAgentState()
    load_requirements_into_state(state)
    state.business_context = BusinessContext(
        company_name="CleanSeas Ocean Cleanup",
        product_or_service="Free ocean plastic cleanup drives and community environmental workshops",
        marketing_goal="Recruit 500 volunteer cleanup captains across coastal cities",
        target_audience="Environmentally conscious college students aged 18-30"
    )
    analyze_relevance(state)

    candidate_ids = {
        req.id for req in state.requirements
        if req.status == RequirementStatus.UNKNOWN
    }
    assert "usp_differentiation" in candidate_ids

    dynamic_response = {
        "context_summary": "CleanSeas runs coastal ocean plastic cleanups and workshops aiming to recruit 500 college student cleanup captains.",
        "selected_requirement_id": "usp_differentiation",
        "question": "What distinctive leadership perks or training do you offer to attract volunteer captains?",
        "reasoning": "Understanding the unique captain value proposition is critical for volunteer recruitment strategy."
    }

    with patch("backend.app.agent.question_selection.get_llm_response", return_value=json.dumps(dynamic_response)) as mock_llm:
        full_message = generate_first_contextual_message(state)

        assert state.first_message_generated is True
        assert state.active_requirement_id == "usp_differentiation"
        assert state.active_requirement_id in candidate_ids

        # Assert no predefined transitions are used
        lower_msg = full_message.lower()
        for phrase in REMOVED_TRANSITIONS:
            assert phrase not in lower_msg, f"Found removed transition phrase '{phrase}' in message: {full_message}"

        # Assert no hardcoded question templates are used
        for q in REMOVED_QUESTIONS:
            assert q not in lower_msg, f"Found removed hardcoded question template '{q}' in message: {full_message}"

        # Assert structure: greeting, context summary, dynamic question
        parts = full_message.split("\n\n")
        assert len(parts) == 3, f"Expected exactly 3 parts (greeting, summary, question), got {len(parts)}: {parts}"
        greeting, summary, question = parts[0], parts[1], parts[2]

        assert "initial summary" in greeting.lower()
        assert "cleanseas" in summary.lower()
        assert question == dynamic_response["question"]
        assert question.count("?") == 1, "Expected exactly one question mark in the dynamic question."

        # Verify no duplicate greeting in message
        assert full_message.count("Hi! Let's get started on your marketing plan") == 1


def test_first_message_fallback_generates_dynamic_question():
    """
    Verifies that when first message LLM generation encounters an error,
    the fallback dynamically calls question generation rather than using
    hardcoded question templates or predefined transitions.
    """
    state = MarketingAgentState()
    load_requirements_into_state(state)
    state.business_context = BusinessContext(
        company_name="CleanSeas Ocean Cleanup",
        product_or_service="Free ocean plastic cleanup drives",
        marketing_goal="Recruit 500 volunteer cleanup captains",
        target_audience="College students aged 18-30"
    )
    analyze_relevance(state)

    candidate_ids = {
        req.id for req in state.requirements
        if req.status == RequirementStatus.UNKNOWN
    }

    dynamic_fallback_q = "What motivates college students to choose CleanSeas for volunteering?"

    with patch("backend.app.agent.question_selection.get_llm_response", side_effect=LLMServiceError("Simulated LLM error")), \
         patch("backend.app.agent.question_selection.generate_question", return_value=dynamic_fallback_q) as mock_gen_q:
        
        full_message = generate_first_contextual_message(state)

        assert state.first_message_generated is True
        assert state.active_requirement_id in candidate_ids
        mock_gen_q.assert_called_once()

        lower_msg = full_message.lower()
        for phrase in REMOVED_TRANSITIONS:
            assert phrase not in lower_msg, f"Found removed transition phrase '{phrase}' in fallback message"

        for q in REMOVED_QUESTIONS:
            assert q not in lower_msg, f"Found removed question template '{q}' in fallback message"

        assert dynamic_fallback_q in full_message
        assert full_message.count("Hi! Let's get started on your marketing plan") == 1


def test_prompt_and_schema_has_no_transition_field():
    """
    Verifies that FIRST_MESSAGE_SYSTEM_PROMPT does not include a 'transition' field in its JSON schema
    and FIRST_MESSAGE_TRANSITIONS_AND_QUESTIONS does not exist in the module.
    """
    import backend.app.prompts.question_selection as qs_prompt_mod

    assert not hasattr(qs_prompt_mod, "FIRST_MESSAGE_TRANSITIONS_AND_QUESTIONS")
    assert not hasattr(qs_prompt_mod, "build_first_message_fallback_text")
    assert hasattr(qs_prompt_mod, "build_grounded_business_summary")

    # Verify transition is not in the system prompt schema
    assert '"transition"' not in FIRST_MESSAGE_SYSTEM_PROMPT
    assert "With that in mind" not in FIRST_MESSAGE_SYSTEM_PROMPT


if __name__ == "__main__":
    test_question_selection()
    test_generate_first_contextual_message_dynamic_no_predefined_transition_or_questions()
    test_first_message_fallback_generates_dynamic_question()
    test_prompt_and_schema_has_no_transition_field()

