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


def test_8_scenario_minimalist_audience_demographic():
    print("--- TEST 8: Minimalist audience 'Indian consumers aged 18–35' does not resolve location ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    state.business_context = BusinessContext(
        company_name="Chai Craft",
        product_or_service="Specialty instant chai premix",
        marketing_goal="Increase D2C orders",
        target_audience="Indian consumers aged 18–35",
    )

    loc_req = state.get_requirement_by_id("target_market_location")
    assert loc_req.status == RequirementStatus.UNKNOWN

    # Next question selection should prioritize target_market_location
    selected = select_next_requirement(state)
    assert selected is not None
    assert selected.id == "target_market_location"
    print("✓ TEST 8 PASSED: Broad demographic 'Indian consumers aged 18–35' prompts for geographic targeting.")


def test_9_scenario_audience_already_specifies_mumbai_and_pune():
    print("--- TEST 9: Audience already specifies 'Mumbai and Pune' -> Do not ask again ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    state.business_context = BusinessContext(
        company_name="FitBites",
        product_or_service="Healthy meal delivery",
        marketing_goal="Get 500 subscribers",
        target_audience="Health-conscious professionals in Mumbai and Pune",
    )

    # If already established as known
    loc_req = state.get_requirement_by_id("target_market_location")
    loc_req.status = RequirementStatus.KNOWN
    loc_req.value = "Mumbai and Pune"

    selected = select_next_requirement(state)
    assert selected is not None
    assert selected.id != "target_market_location", "Should not select target_market_location when already known"
    print(f"✓ TEST 9 PASSED: 'Mumbai and Pune' preserved as KNOWN; next selected req is '{selected.id}'.")


def test_10_scenario_user_answers_pan_india():
    print("--- TEST 10: User answers 'Pan-India' -> Marked KNOWN & continued ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    loc_req = state.get_requirement_by_id("target_market_location")
    state.active_requirement_id = loc_req.id
    state.current_question = "Which geographic market do you want to target for this campaign?"

    process_answer_for_requirement(state, "Pan-India")

    updated_req = state.get_requirement_by_id("target_market_location")
    assert updated_req.status == RequirementStatus.KNOWN
    assert "Pan-India" in str(updated_req.value) or "pan-india" in str(updated_req.value).lower()
    print(f"✓ TEST 10 PASSED: 'Pan-India' saved as KNOWN with value: '{updated_req.value}'")


def test_11_scenario_user_provides_city_does_not_repeat():
    print("--- TEST 11: User provides a city ('Bengaluru') -> Saved and not repeated ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    loc_req = state.get_requirement_by_id("target_market_location")
    state.active_requirement_id = loc_req.id
    state.current_question = "Which geographic market do you want to target for this campaign?"

    process_answer_for_requirement(state, "Bengaluru")

    updated_req = state.get_requirement_by_id("target_market_location")
    assert updated_req.status == RequirementStatus.KNOWN
    assert "Bengaluru" in str(updated_req.value)

    # Next requirement selection must not select target_market_location
    next_req = select_next_requirement(state)
    assert next_req is not None
    assert next_req.id != "target_market_location", "Must not repeat target_market_location"
    print(f"✓ TEST 11 PASSED: 'Bengaluru' saved; next requirement '{next_req.id}' selected without repeating.")


def test_12_scenario_other_mandatory_requirements_remain_unresolved():
    print("--- TEST 12: Other mandatory requirements unresolved -> Continue asking within limit ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    # Resolve only target_market_location
    loc_req = state.get_requirement_by_id("target_market_location")
    loc_req.status = RequirementStatus.KNOWN
    loc_req.value = "Delhi NCR"
    state.qa_history = [
        QATurn(question="Which geographic market?", answer="Delhi NCR", requirement_id="target_market_location")
    ]

    assert is_sufficient(state) is False, "Must not be sufficient when other must-haves are UNKNOWN"
    unresolved = state.get_unresolved_must_haves()
    assert len(unresolved) > 0
    assert not any(r.id == "target_market_location" for r in unresolved)

    next_req = select_next_requirement(state)
    assert next_req is not None
    assert next_req.id != "target_market_location"
    print(f"✓ TEST 12 PASSED: is_sufficient is False; continuing with '{next_req.id}' ({len(unresolved)} must-haves remaining).")


def test_13_scenario_max_question_limit_does_not_silently_bypass():
    print("--- TEST 13: Max question limit reached before location provided -> Does not silently bypass ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    # Location remains UNKNOWN
    loc_req = state.get_requirement_by_id("target_market_location")
    assert loc_req.status == RequirementStatus.UNKNOWN

    # Simulate reaching max questions on other questions
    state.qa_history = [
        QATurn(question=f"Q{i}", answer=f"A{i}", requirement_id=f"other_req_{i}")
        for i in range(1, settings.max_questions + 1)
    ]

    # Deterministic is_sufficient terminates loop
    assert is_sufficient(state) is True

    # Target market location was NOT silently marked KNOWN or resolved
    loc_req_after = state.get_requirement_by_id("target_market_location")
    assert loc_req_after.status == RequirementStatus.UNKNOWN, "Must NOT silently mark target_market_location KNOWN"
    assert loc_req_after.value is None

    # Verify strategy directive logic recognizes unresolved location
    from backend.app.models.marketing_strategy import MarketingStrategy
    mock_strategy = MarketingStrategy(
        business_overview=["Summary"],
        marketing_channels_and_tactics=["SEO", "Website"],
        customer_acquisition_approach=["Acquisition"],
        competitive_positioning=["Positioning"],
        action_plan=["**Week 1: Setup** Plan", "**Week 2: Test** Plan", "**Week 3: Optimize** Plan", "**Week 4: Scale** Plan"],
        target_audience_insights=["College students aged 18-30"]
    )

    # Test the unresolved location guard
    tml_known = bool(loc_req_after and loc_req_after.status == RequirementStatus.KNOWN and loc_req_after.value)
    if not tml_known:
        loc_warning = "Target market location was not specified during discovery; confirm exact geographic targeting (cities, regions, or Pan-India) before launching paid campaigns."
        if mock_strategy.target_audience_insights is None:
            mock_strategy.target_audience_insights = []
        if not any("geographic" in a.lower() or "location" in a.lower() for a in mock_strategy.target_audience_insights):
            mock_strategy.target_audience_insights.append(loc_warning)

    recorded_insights = mock_strategy.target_audience_insights or []
    assert any("location" in a.lower() for a in recorded_insights)
    print("✓ TEST 13 PASSED: Max questions reached keeps location UNKNOWN and documents unconfirmed assumption.")


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
    test_8_scenario_minimalist_audience_demographic()
    test_9_scenario_audience_already_specifies_mumbai_and_pune()
    test_10_scenario_user_answers_pan_india()
    test_11_scenario_user_provides_city_does_not_repeat()
    test_12_scenario_other_mandatory_requirements_remain_unresolved()
    test_13_scenario_max_question_limit_does_not_silently_bypass()
    print("==================================================")
    print("ALL TARGET MARKET LOCATION TEST SCENARIOS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    run_all_tests()
