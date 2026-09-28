import os
import sys

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.information_requirement import RequirementStatus
from backend.app.models.qa_turn import QATurn
from backend.app.core.requirements_library import load_requirements_into_state
from backend.app.agent.sufficiency_check import is_sufficient
from backend.app.core.config import settings


def test_sufficiency_check():
    print("=== Testing Phase 6: Sufficiency Check ===")
    print(f"Configured max_questions setting: {settings.max_questions}\n")

    # Scenario A: All must-haves resolved, few questions asked (e.g., 2 turns) -> Expected True
    print("--- Scenario A: All must-haves resolved, 2 QA turns ---")
    state_a = MarketingAgentState()
    load_requirements_into_state(state_a)

    # Mark all must-have items as KNOWN
    for req in state_a.requirements:
        if req.is_must_have:
            req.status = RequirementStatus.KNOWN
            req.value = "Sample resolved value"

    # Add 2 QA turns
    state_a.qa_history = [
        QATurn(question="Question 1", answer="Answer 1", requirement_id="req1"),
        QATurn(question="Question 2", answer="Answer 2", requirement_id="req2"),
    ]

    result_a = is_sufficient(state_a)
    unresolved_a = len(state_a.get_unresolved_must_haves())
    turns_a = len(state_a.qa_history)
    print(f"Unresolved must-haves: {unresolved_a}, QA turns: {turns_a}")
    print(f"is_sufficient(state_a) -> {result_a} (Expected: True)\n")
    assert result_a is True, "Scenario A failed! Expected True."

    # Scenario B: Must-haves still unresolved, qa_history has 6 entries -> Expected True
    print("--- Scenario B: Must-haves unresolved, 6 QA turns (max reached) ---")
    state_b = MarketingAgentState()
    load_requirements_into_state(state_b)
    # Must-haves remain UNKNOWN

    # Add max_questions QA turns
    state_b.qa_history = [
        QATurn(question=f"Question {i}", answer=f"Answer {i}", requirement_id=f"req{i}")
        for i in range(1, settings.max_questions + 1)
    ]

    result_b = is_sufficient(state_b)
    unresolved_b = len(state_b.get_unresolved_must_haves())
    turns_b = len(state_b.qa_history)
    print(f"Unresolved must-haves: {unresolved_b}, QA turns: {turns_b}")
    print(f"is_sufficient(state_b) -> {result_b} (Expected: True)\n")
    assert result_b is True, "Scenario B failed! Expected True."

    # Scenario C: Must-haves unresolved AND qa_history has fewer than max_questions entries (e.g. 3 turns) -> Expected False
    print("--- Scenario C: Must-haves unresolved, 3 QA turns ---")
    state_c = MarketingAgentState()
    load_requirements_into_state(state_c)
    # Must-haves remain UNKNOWN

    # Add 3 QA turns
    state_c.qa_history = [
        QATurn(question=f"Question {i}", answer=f"Answer {i}", requirement_id=f"req{i}")
        for i in range(1, 4)
    ]

    result_c = is_sufficient(state_c)
    unresolved_c = len(state_c.get_unresolved_must_haves())
    turns_c = len(state_c.qa_history)
    print(f"Unresolved must-haves: {unresolved_c}, QA turns: {turns_c}")
    print(f"is_sufficient(state_c) -> {result_c} (Expected: False)\n")
    assert result_c is False, "Scenario C failed! Expected False."

    print("=== All Sufficiency Check Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_sufficiency_check()
