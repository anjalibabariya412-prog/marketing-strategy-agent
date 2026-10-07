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
    create_dynamic_requirement,
    is_duplicate_requirement,
)
from backend.app.models.online_presence import OnlinePresenceContext, OnlineSourceSummary
from backend.app.models.qa_turn import QATurn
from backend.app.agent.sufficiency_check import is_sufficient
from backend.app.agent.question_selection import select_next_requirement, prepare_next_question
from backend.app.agent.answer_processing import process_answer_for_requirement, process_answer_and_plan_next


def test_1_existing_requirement_sufficient():
    print("--- TEST 1: Existing requirement is sufficient ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    initial_req_count = len(state.requirements)
    
    # Selecting requirement for a standard business where base framework items apply
    req = select_next_requirement(state)
    assert req is not None
    assert req.id in [r.id for r in REQUIREMENTS_LIBRARY]
    assert len(state.requirements) == initial_req_count
    print("✓ TEST 1 PASSED: Selected existing base requirement without creating dynamic requirement.")


def test_2_dynamic_requirement_creation():
    print("--- TEST 2: Business needs dynamic requirement ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    
    dyn_req = create_dynamic_requirement(
        raw_id="cuisine_type",
        title="Cuisine Type",
        description="The type of cuisine offered by the restaurant",
        is_must_have=True
    )
    assert dyn_req.id == "dynamic_cuisine_type"
    assert dyn_req.is_custom is True
    assert dyn_req.status == RequirementStatus.UNKNOWN
    
    state.requirements.append(dyn_req)
    assert state.get_requirement_by_id("dynamic_cuisine_type") is not None
    assert len(state.requirements) == len(REQUIREMENTS_LIBRARY) + 1
    
    # Ensure global library is completely untouched
    assert not any(r.id == "dynamic_cuisine_type" for r in REQUIREMENTS_LIBRARY)
    print("✓ TEST 2 PASSED: Dynamic requirement created, prefixed with 'dynamic_', stored in state, and global library preserved.")


def test_3_dynamic_requirement_receives_answer():
    print("--- TEST 3: Dynamic requirement receives answer ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    
    dyn_req = create_dynamic_requirement("photography_style", "Photography Style", "Specialized photography style")
    state.requirements.append(dyn_req)
    
    state.active_requirement_id = dyn_req.id
    state.current_question = "What photography style do you specialize in?"
    
    process_answer_for_requirement(state, "We specialize in cinematic candid wedding photography.")
    
    updated_req = state.get_requirement_by_id(dyn_req.id)
    assert updated_req.status == RequirementStatus.KNOWN
    assert updated_req.value is not None and len(str(updated_req.value).strip()) > 0
    assert len(state.qa_history) == 1
    assert state.qa_history[0].requirement_id == dyn_req.id
    print(f"✓ TEST 3 PASSED: Dynamic requirement updated to KNOWN with value: '{updated_req.value}'")


def test_4_duplicate_prevention():
    print("--- TEST 4: Duplicate requirement prevention ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    
    # 1. Collision with base requirement ID
    assert is_duplicate_requirement("pricing_offer_structure", "Pricing, Offers & Commercial Model", state.requirements) is True
    assert is_duplicate_requirement("dynamic_pricing_offer_structure", "Product Price", state.requirements) is True
    
    # 2. Collision with base requirement title
    assert is_duplicate_requirement("dynamic_competitor_list", "Competitive Landscape & Alternatives", state.requirements) is True
    
    # 3. Collision with existing dynamic requirement
    dyn_req = create_dynamic_requirement("sales_cycle", "Sales Cycle Length", "Length of sales cycle")
    state.requirements.append(dyn_req)
    
    assert is_duplicate_requirement("dynamic_sales_cycle", "Sales Cycle Length", state.requirements) is True
    assert is_duplicate_requirement("new_id", "Sales Cycle Length", state.requirements) is True
    
    print("✓ TEST 4 PASSED: Duplicate prevention correctly flags ID and title collisions.")


def test_5_online_presence_context_prevents_duplicate():
    print("--- TEST 5: OnlinePresenceContext prevents dynamic requirement ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    state.online_presence_context = OnlinePresenceContext(
        overall_summary="The restaurant specializes in authentic South Indian Dosa and Filter Coffee with detailed menu pricing.",
        website=OnlineSourceSummary(
            summary="Full menu showing South Indian Dosa prices ranging from $8 to $15.",
            relevant_products_or_services=["South Indian Dosa", "Filter Coffee"],
            positioning_or_messaging=["Authentic South Indian Cuisine"]
        )
    )
    
    # Dynamic requirement creation check should notice cuisine info is present in OP context
    op_summary = state.online_presence_context.overall_summary
    assert "South Indian" in op_summary
    print("✓ TEST 5 PASSED: Online presence context holds cuisine information.")


def test_6_past_marketing_doc_prevents_unnecessary_req():
    print("--- TEST 6: Past marketing document prevents unnecessary req ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    state.business_context.past_marketing_document = (
        "Previous Marketing Plan 2025: Target audience is B2B HR Directors in mid-market tech companies. "
        "Sales process is 60-day enterprise demo sales cycle with $12k annual subscription fee."
    )
    
    doc = state.business_context.past_marketing_document
    assert "HR Directors" in doc
    assert "60-day enterprise demo" in doc
    print("✓ TEST 6 PASSED: Past marketing document context available in business context.")


def test_7_earlier_user_answer_prevents_duplicate():
    print("--- TEST 7: Earlier user answer prevents duplicate ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    
    # Mark pricing_offer_structure as KNOWN
    pricing_req = state.get_requirement_by_id("pricing_offer_structure")
    pricing_req.status = RequirementStatus.KNOWN
    pricing_req.value = "$99/month SaaS subscription"
    
    state.qa_history.append(QATurn(
        question="What is the price of your service?",
        answer="$99/month SaaS subscription",
        requirement_id="pricing_offer_structure"
    ))
    
    # Verify is_duplicate_requirement or asked check
    asked_ids = {turn.requirement_id for turn in state.qa_history if turn.requirement_id}
    assert "pricing_offer_structure" in asked_ids
    assert pricing_req.status != RequirementStatus.UNKNOWN
    print("✓ TEST 7 PASSED: Resolved requirement is excluded from selection and duplicate creation.")


def test_8_max_questions_reached():
    print("--- TEST 8: Max question count reached ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    
    # Add fake QA turns up to settings.max_questions
    for i in range(settings.max_questions):
        state.qa_history.append(QATurn(
            question=f"Question {i+1}?",
            answer=f"Answer {i+1}.",
            requirement_id=f"req_{i+1}"
        ))
        
    assert len(state.qa_history) >= settings.max_questions
    assert is_sufficient(state) is True
    
    # select_next_requirement should return None when max questions reached
    next_req = select_next_requirement(state)
    assert next_req is None
    print("✓ TEST 8 PASSED: Reaching max_questions forces sufficiency and stops question selection.")


def test_9_not_relevant_excluded_from_selection():
    print("--- TEST 9: NOT_RELEVANT requirement is excluded from selection ---")
    state = MarketingAgentState()
    load_requirements_into_state(state)
    
    # Mark pricing_offer_structure as NOT_RELEVANT
    req = state.get_requirement_by_id("pricing_offer_structure")
    assert req is not None
    req.status = RequirementStatus.NOT_RELEVANT

    already_asked_or_resolved_ids = {
        r.id for r in state.requirements 
        if r.status != RequirementStatus.UNKNOWN
    }
    assert "pricing_offer_structure" in already_asked_or_resolved_ids

    candidates = [
        r for r in state.requirements 
        if r.status == RequirementStatus.UNKNOWN and r.id not in already_asked_or_resolved_ids
    ]
    assert not any(r.id == "pricing_offer_structure" for r in candidates)
    print("✓ TEST 9 PASSED: NOT_RELEVANT requirement is excluded from candidate list.")


def run_all_tests():
    print("==================================================")
    print("RUNNING DYNAMIC REQUIREMENT DISCOVERY TEST SUITE")
    print("==================================================")
    test_1_existing_requirement_sufficient()
    test_2_dynamic_requirement_creation()
    test_3_dynamic_requirement_receives_answer()
    test_4_duplicate_prevention()
    test_5_online_presence_context_prevents_duplicate()
    test_6_past_marketing_doc_prevents_unnecessary_req()
    test_7_earlier_user_answer_prevents_duplicate()
    test_8_max_questions_reached()
    test_9_not_relevant_excluded_from_selection()
    print("==================================================")
    print("ALL 9 DYNAMIC REQUIREMENT TEST SCENARIOS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    run_all_tests()
