import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.core.requirements_library import load_requirements_into_state
from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.business_context import BusinessContext
from backend.app.models.information_requirement import RequirementStatus
from backend.app.agent.answer_processing import (
    extract_competitors,
    update_state_competitors,
    process_answer_for_requirement,
)
from backend.app.agent.graph import analyze_node, generate_node


def test_1_extract_named_and_generic_competitors():
    print("--- TEST 1: Extract named and generic competitors ---")
    mock_llm_json = '{"competitors": ["Swiggy", "Zomato", "two local tiffin services"]}'

    with patch("backend.app.agent.answer_processing.get_llm_response", return_value=mock_llm_json):
        result = extract_competitors("Swiggy, Zomato and two local tiffin services")
        assert result == ["Swiggy", "Zomato", "two local tiffin services"]
        print(f"✓ TEST 1 PASSED: Extracted competitors: {result}")


def test_2_deduplication_and_case_insensitive_handling():
    print("--- TEST 2: Deduplication and case-insensitive handling ---")
    mock_llm_json = '{"competitors": ["Swiggy", "Zomato", "swiggy", "Zomato", "Local Tiffin"]}'

    with patch("backend.app.agent.answer_processing.get_llm_response", return_value=mock_llm_json):
        result = extract_competitors("Swiggy, Zomato, swiggy, Zomato and Local Tiffin")
        assert result == ["Swiggy", "Zomato", "Local Tiffin"]
        print(f"✓ TEST 2 PASSED: Deduplicated competitors: {result}")


def test_3_no_competitors_returns_empty_list():
    print("--- TEST 3: User states no competitors ---")
    mock_llm_json = '{"competitors": []}'

    with patch("backend.app.agent.answer_processing.get_llm_response", return_value=mock_llm_json):
        result = extract_competitors("We have no direct competitors in our city.")
        assert result == []
        print("✓ TEST 3 PASSED: Returned [] when user has no competitors (no invented competitors).")


def test_4_update_state_competitors_stores_in_langgraph_state():
    print("--- TEST 4: Store competitors in LangGraph state ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)

    update_state_competitors(state, ["Swiggy", "Zomato", "two local tiffin services"])
    assert state.business_context.competitors == ["Swiggy", "Zomato", "two local tiffin services"]

    # Subsequent update should append non-duplicates
    update_state_competitors(state, ["swiggy", "Uber Eats"])
    assert state.business_context.competitors == ["Swiggy", "Zomato", "two local tiffin services", "Uber Eats"]
    print(f"✓ TEST 4 PASSED: State competitors updated correctly: {state.business_context.competitors}")


def test_5_end_to_end_answer_processing_extracts_competitors():
    print("--- TEST 5: End-to-end answer processing extracts competitors ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)

    state.active_requirement_id = "competitive_landscape"
    state.current_question = "Who are your main competitors?"

    answer_mock_json = '{"active_requirement": {"status": "known", "value": "Swiggy, Zomato and two local tiffin services"}, "incidentally_known_requirements": []}'
    comp_mock_json = '{"competitors": ["Swiggy", "Zomato", "two local tiffin services"]}'

    def mock_llm_side_effect(prompt, **kwargs):
        if "ACTIVE REQUIREMENT TO CLARIFY" in prompt:
            return answer_mock_json
        return comp_mock_json

    with patch("backend.app.agent.answer_processing.get_llm_response", side_effect=mock_llm_side_effect):
        process_answer_for_requirement(state, "Swiggy, Zomato and two local tiffin services")

    req = state.get_requirement_by_id("competitive_landscape")
    assert req.status == RequirementStatus.KNOWN
    assert state.business_context.competitors == ["Swiggy", "Zomato", "two local tiffin services"]
    print("✓ TEST 5 PASSED: Competitors successfully stored in state during requirement answer processing.")


def test_6_answering_competitor_question_does_not_trigger_web_research():
    print("--- TEST 6: Answering competitor question does NOT trigger web research ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)

    state.active_requirement_id = "competitive_landscape"
    state.current_question = "Who are your main competitors?"

    answer_mock_json = '{"active_requirement": {"status": "known", "value": "Brand A and Brand B"}, "incidentally_known_requirements": []}'
    comp_mock_json = '{"competitors": ["Brand A", "Brand B"]}'

    def mock_llm_side_effect(prompt, **kwargs):
        if "ACTIVE REQUIREMENT TO CLARIFY" in prompt:
            return answer_mock_json
        return comp_mock_json

    with patch("backend.app.agent.answer_processing.get_llm_response", side_effect=mock_llm_side_effect):
        process_answer_for_requirement(state, "Brand A and Brand B")

    # Run analyze node to ensure graph execution passes without web research
    updated_state = analyze_node(state)

    # Verify state stores competitors and no web research is present or called
    assert updated_state.business_context.competitors == ["Brand A", "Brand B"]
    assert not hasattr(updated_state, "competitor_research")
    print("✓ TEST 6 PASSED: Answering competitor question stores user competitors and triggers NO web research.")


def run_all_tests():
    print("==================================================")
    print("RUNNING COMPETITOR EXTRACTION TEST SUITE")
    print("==================================================")
    test_1_extract_named_and_generic_competitors()
    test_2_deduplication_and_case_insensitive_handling()
    test_3_no_competitors_returns_empty_list()
    test_4_update_state_competitors_stores_in_langgraph_state()
    test_5_end_to_end_answer_processing_extracts_competitors()
    test_6_answering_competitor_question_does_not_trigger_web_research()
    print("==================================================")
    print("ALL COMPETITOR EXTRACTION TEST SCENARIOS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    run_all_tests()
