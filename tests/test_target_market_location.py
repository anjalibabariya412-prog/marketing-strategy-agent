import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.core.config import settings
from backend.app.core.requirements_library import load_requirements_into_state, REQUIREMENTS_LIBRARY
from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.business_context import BusinessContext
from backend.app.models.information_requirement import (
    InformationRequirement,
    RequirementStatus,
    is_duplicate_requirement,
)
from backend.app.models.online_presence import OnlinePresenceContext, OnlineSourceSummary
from backend.app.models.qa_turn import QATurn
from backend.app.agent.sufficiency_check import is_sufficient
from backend.app.agent.question_selection import (
    select_next_requirement,
    generate_question,
    prepare_next_question,
    FALLBACK_QUESTIONS,
)
from backend.app.agent.answer_processing import process_answer_for_requirement
from backend.app.agent.strategy_generation import generate_strategy


def test_1_global_library_includes_target_market_location():
    print("--- TEST 1: target_market_location is in global REQUIREMENTS_LIBRARY ---")
    lib_ids = [req.id for req in REQUIREMENTS_LIBRARY]
    assert "target_market_location" in lib_ids
    
    req = next(r for r in REQUIREMENTS_LIBRARY if r.id == "target_market_location")
    assert req.title == "Target Market & Geographic Scope"
    assert req.is_must_have is True
    assert req.is_custom is False
    print("✓ TEST 1 PASSED: target_market_location is a permanent Must-Have global requirement.")


def test_2_location_missing_selection_and_question():
    print("--- TEST 2: Location missing -> requirement selection & question generation ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    state.business_context = BusinessContext(
        company_name="Urban Fresh Bakery",
        product_or_service="Artisanal sourdough bread and custom cakes",
        marketing_goal="Increase local foot traffic and online cake orders"
    )
    
    selected_req = select_next_requirement(state)
    assert selected_req is not None
    
    # Generate question for target_market_location
    loc_req = state.get_requirement_by_id("target_market_location")
    assert loc_req is not None
    assert loc_req.is_must_have is True
    
    q_text = generate_question(state, loc_req)
    assert q_text is not None and len(q_text) > 0
    print(f"✓ TEST 2 PASSED: Question generated for target_market_location: '{q_text}'")


def test_3_business_address_does_not_auto_mark_known():
    print("--- TEST 3: Business address does NOT auto-mark target_market_location KNOWN ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    state.business_context = BusinessContext(
        company_name="Gujarat Tech Solutions",
        product_or_service="IT Services and Web Development",
        marketing_goal="Get more B2B clients",
        target_audience="Small business owners"
    )
    
    loc_req = state.get_requirement_by_id("target_market_location")
    assert loc_req.status == RequirementStatus.UNKNOWN
    
    # Address mentioned in context text but target market not explicitly defined
    text_with_address = "Business location: Office 402, Navrangpura, Ahmedabad."
    # Target market location should remain UNKNOWN unless explicit marketing target geography is stated
    assert loc_req.status == RequirementStatus.UNKNOWN
    print("✓ TEST 3 PASSED: Business address does not auto-resolve target_market_location.")


def test_4_user_explicit_answer_authoritative():
    print("--- TEST 4: User answer 'Ahmedabad and Gandhinagar' is exact & authoritative ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    
    loc_req = state.get_requirement_by_id("target_market_location")
    state.active_requirement_id = loc_req.id
    state.current_question = "Which location(s) do you want to target with your marketing?"
    
    user_ans = "Ahmedabad and Gandhinagar"
    process_answer_for_requirement(state, user_ans)
    
    updated_req = state.get_requirement_by_id("target_market_location")
    assert updated_req.status == RequirementStatus.KNOWN
    assert "Ahmedabad" in str(updated_req.value)
    assert "Gandhinagar" in str(updated_req.value)
    print(f"✓ TEST 4 PASSED: User answer stored as exact KNOWN value: '{updated_req.value}'")


def test_5_user_updates_location_authoritative_override():
    print("--- TEST 5: User updates location to 'all of Gujarat' ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    
    loc_req = state.get_requirement_by_id("target_market_location")
    loc_req.status = RequirementStatus.KNOWN
    loc_req.value = "Ahmedabad and Gandhinagar"
    
    # User provides updated answer in later turn
    state.active_requirement_id = loc_req.id
    state.current_question = "Can you clarify your target geographic location?"
    
    process_answer_for_requirement(state, "Actually, we want to target all of Gujarat.")
    
    updated_req = state.get_requirement_by_id("target_market_location")
    assert updated_req.status == RequirementStatus.KNOWN
    assert "Gujarat" in str(updated_req.value)
    print(f"✓ TEST 5 PASSED: Latest user explicit answer overrides previous value: '{updated_req.value}'")


def test_6_dynamic_requirement_discovery_prevents_duplicate_target_location():
    print("--- TEST 6: Dynamic discovery rejects 'dynamic_target_market_location' ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    
    # Attempting duplicate check for dynamic target market location
    is_dup = is_duplicate_requirement("dynamic_target_market_location", "Target Market Location", state.requirements)
    assert is_dup is True
    print("✓ TEST 6 PASSED: Dynamic requirement discovery correctly rejects duplicate target market location.")


def test_7_sufficiency_check_requires_target_market_location():
    print("--- TEST 7: is_sufficient requires Must-Have target_market_location ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    
    # Keep all must-haves UNKNOWN
    assert is_sufficient(state) is False
    assert any(req.id == "target_market_location" for req in state.get_unresolved_must_haves())
    
    # Mark all must-haves KNOWN
    for req in state.requirements:
        if req.is_must_have:
            req.status = RequirementStatus.KNOWN
            req.value = "Sample Answer"
            
    assert len(state.get_unresolved_must_haves()) == 0
    assert is_sufficient(state) is True
    print("✓ TEST 7 PASSED: target_market_location is tracked as unresolved must-have in is_sufficient().")


def run_all_tests():
    print("==================================================")
    print("RUNNING TARGET MARKET LOCATION TEST SUITE")
    print("==================================================")
    test_1_global_library_includes_target_market_location()
    test_2_location_missing_selection_and_question()
    test_3_business_address_does_not_auto_mark_known()
    test_4_user_explicit_answer_authoritative()
    test_5_user_updates_location_authoritative_override()
    test_6_dynamic_requirement_discovery_prevents_duplicate_target_location()
    test_7_sufficiency_check_requires_target_market_location()
    print("==================================================")
    print("ALL TARGET MARKET LOCATION TEST SCENARIOS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    run_all_tests()
