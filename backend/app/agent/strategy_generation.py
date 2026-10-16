import json
import logging
import math
import re
from typing import Any, List, Optional
from pydantic import ValidationError

from backend.app.schemas.agent.agent_state import MarketingAgentState
from backend.app.schemas.agent.document_context import format_document_context
from backend.app.schemas.agent.information_requirement import RequirementStatus
from backend.app.schemas.agent.marketing_strategy import BudgetItem, MarketingStrategy
from backend.app.services.llm_service import get_llm_response, LLMServiceError

logger = logging.getLogger(__name__)


class StrategyGenerationError(Exception):
    """Raised when marketing strategy generation or parsing fails."""
    pass


from backend.app.prompts.strategy_generation import (
    STRATEGY_GENERATION_SYSTEM_PROMPT,
    build_strategy_generation_prompt,
)

SYSTEM_PROMPT = STRATEGY_GENERATION_SYSTEM_PROMPT


def parse_amount_number(val: Any) -> Optional[float]:
    """Extract numeric monetary amount from a string or number."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if val >= 0 else 0.0
    if not isinstance(val, str):
        return None
    s = val.strip()
    if not s:
        return None
    if re.search(r"-\s*[\$₹€£]?\s*\d", s):
        return 0.0
    match = re.search(r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*([kKmM])?", s)
    if not match:
        return None
    num_str = match.group(1).replace(",", "")
    try:
        num = float(num_str)
        mult = match.group(2)
        if mult in ("k", "K"):
            num *= 1_000
        elif mult in ("m", "M"):
            num *= 1_000_000
        return max(0.0, num)
    except ValueError:
        return None


def parse_percentage_number(val: Any) -> Optional[float]:
    """Extract numeric percentage from a string or number."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if val >= 0 else 0.0
    if not isinstance(val, str):
        return None
    s = val.strip()
    if not s:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", s)
    if not match:
        match = re.search(r"^([0-9]+(?:\.[0-9]+)?)$", s)
    if not match:
        return None
    try:
        pct = float(match.group(1))
        return max(0.0, pct)
    except ValueError:
        return None


def sync_description_numbers(
    description: str,
    old_amt: Optional[float],
    new_amt: float,
    old_pct: Optional[float],
    new_pct_str: str,
    currency_symbol: str
) -> str:
    """Updates outdated amounts and percentages embedded in description strings."""
    if not description:
        return description
    updated = description

    if old_amt is not None and abs(old_amt - new_amt) > 0.01 and old_amt > 0:
        new_amt_str = f"{currency_symbol}{int(new_amt):,}" if float(new_amt).is_integer() else f"{currency_symbol}{new_amt:,.2f}"
        old_int = int(round(old_amt))
        pattern_with_curr = rf"(?:[₹$€£]|Rs\.?)\s*(?:{old_int:,}|{old_int})(?!\d)"
        updated = re.sub(pattern_with_curr, new_amt_str, updated)

    if old_pct is not None and old_pct > 0:
        old_pct_int = int(round(old_pct))
        new_pct_clean = new_pct_str.rstrip("%")
        if str(old_pct_int) != new_pct_clean:
            updated = re.sub(rf"\b{old_pct_int}%\b", new_pct_str, updated)

    return updated


def normalize_budget_considerations(
    strategy: MarketingStrategy,
    budget_resources: Any
) -> MarketingStrategy:
    """
    Deterministically validates and normalizes budget_considerations:
    1. Preserves the user's budget as authoritative.
    2. Enforces non-negative allocation amounts.
    3. Redistributes amounts proportionally within the user's budget if LLM-generated amounts
       differ from the stated budget (unless explicitly marked as intentionally unallocated).
    4. Calculates percentages correctly: percentage = (category allocation / total budget) * 100.
    5. Applies the Largest Remainder Method so rounded allocations sum exactly to the total budget,
       and displayed percentages sum to 100%.
    6. Syncs corrected numbers into item descriptions if old amounts were mentioned.
    """
    if not strategy.budget_considerations:
        return strategy

    if not budget_resources or not hasattr(budget_resources, "amount") or not budget_resources.amount:
        return strategy

    try:
        total_budget = float(budget_resources.amount)
        if total_budget <= 0:
            return strategy
    except (TypeError, ValueError):
        return strategy

    currency = getattr(budget_resources, "currency", "INR")
    curr_str = str(currency.value if hasattr(currency, "value") else currency)
    currency_symbol = "₹" if curr_str in ("INR", "CurrencyEnum.INR") else "$"

    items = strategy.budget_considerations
    num_items = len(items)
    if num_items == 0:
        return strategy

    # Parse raw amounts and percentages
    parsed_amts: List[Optional[float]] = []
    parsed_pcts: List[Optional[float]] = []
    for item in items:
        parsed_amts.append(parse_amount_number(item.amount))
        parsed_pcts.append(parse_percentage_number(item.percentage))

    # Check for intentional unallocation
    is_intentionally_unallocated = False
    valid_amts = [a for a in parsed_amts if a is not None]
    if valid_amts and sum(valid_amts) < total_budget:
        unallocated_keywords = [
            "intentionally unallocated",
            "unallocated budget",
            "held in reserve",
            "remaining budget is unallocated"
        ]
        for item in items:
            desc_lower = (item.description or "").lower()
            label_lower = (item.label or "").lower()
            if any(k in desc_lower or k in label_lower for k in unallocated_keywords):
                is_intentionally_unallocated = True
                break

    # Determine allocation weights preserving relative priorities
    sum_parsed = sum(a for a in parsed_amts if a is not None and a > 0)
    if sum_parsed > 0:
        weights = [(a if a is not None and a > 0 else 0.0) / sum_parsed for a in parsed_amts]
    else:
        sum_pct = sum(p for p in parsed_pcts if p is not None and p > 0)
        if sum_pct > 0:
            weights = [(p if p is not None and p > 0 else 0.0) / sum_pct for p in parsed_pcts]
        else:
            weights = [1.0 / num_items] * num_items

    # Allocate amounts using Largest Remainder Method
    if is_intentionally_unallocated:
        corrected_amts = [a if a is not None and a >= 0 else 0.0 for a in parsed_amts]
    else:
        if float(total_budget).is_integer():
            target_int = int(total_budget)
            raw_floats = [total_budget * w for w in weights]
            floor_ints = [int(math.floor(x)) for x in raw_floats]
            deficit = target_int - sum(floor_ints)
            remainders = sorted(
                [(raw_floats[i] - floor_ints[i], raw_floats[i], i) for i in range(num_items)],
                reverse=True
            )
            for i in range(deficit):
                floor_ints[remainders[i][2]] += 1
            corrected_amts = [float(x) for x in floor_ints]
        else:
            target_cents = int(round(total_budget * 100))
            raw_cents = [target_cents * w for w in weights]
            floor_cents = [int(math.floor(x)) for x in raw_cents]
            deficit = target_cents - sum(floor_cents)
            remainders = sorted(
                [(raw_cents[i] - floor_cents[i], raw_cents[i], i) for i in range(num_items)],
                reverse=True
            )
            for i in range(deficit):
                floor_cents[remainders[i][2]] += 1
            corrected_amts = [c / 100.0 for c in floor_cents]

    # Calculate percentages: percentage = (category allocation / total marketing budget) * 100
    raw_pcts = [(amt / total_budget) * 100.0 for amt in corrected_amts]

    if not is_intentionally_unallocated:
        if all(abs(p - round(p)) < 0.05 for p in raw_pcts):
            target_pct = 100
            raw_floats = raw_pcts
            floor_ints = [int(round(x)) for x in raw_floats]
            deficit = target_pct - sum(floor_ints)
            remainders = sorted(
                [(raw_floats[i] - floor_ints[i], raw_floats[i], i) for i in range(num_items)],
                reverse=True
            )
            for i in range(deficit):
                floor_ints[remainders[i][2]] += 1
            pct_strings = [f"{p}%" for p in floor_ints]
        else:
            target_tenths = 1000
            raw_tenths = [p * 10.0 for p in raw_pcts]
            floor_tenths = [int(math.floor(x)) for x in raw_tenths]
            deficit = target_tenths - sum(floor_tenths)
            remainders = sorted(
                [(raw_tenths[i] - floor_tenths[i], raw_tenths[i], i) for i in range(num_items)],
                reverse=True
            )
            for i in range(deficit):
                floor_tenths[remainders[i][2]] += 1
            pct_strings = [
                f"{t / 10:.0f}%" if (t / 10).is_integer() else f"{t / 10:.1f}%"
                for t in floor_tenths
            ]
    else:
        pct_strings = [
            f"{round(p)}%" if abs(p - round(p)) < 0.05 else f"{p:.1f}%"
            for p in raw_pcts
        ]

    normalized_items: List[BudgetItem] = []
    for i, item in enumerate(items):
        new_amt = corrected_amts[i]
        new_amt_str = (
            f"{currency_symbol}{int(new_amt):,}"
            if float(new_amt).is_integer()
            else f"{currency_symbol}{new_amt:,.2f}"
        )
        new_pct_str = pct_strings[i]
        updated_desc = sync_description_numbers(
            item.description,
            parsed_amts[i],
            new_amt,
            parsed_pcts[i],
            new_pct_str,
            currency_symbol
        )

        normalized_items.append(
            BudgetItem(
                label=item.label,
                amount=new_amt_str,
                percentage=new_pct_str,
                description=updated_desc
            )
        )

    strategy.budget_considerations = normalized_items
    logger.info(
        f"Budget considerations normalized to total budget {currency_symbol}{total_budget:,.2f}: "
        f"amounts={[it.amount for it in normalized_items]}, pcts={[it.percentage for it in normalized_items]}"
    )
    return strategy


def generate_strategy(state: MarketingAgentState) -> MarketingStrategy:
    """
    Generates a structured MarketingStrategy object based on the state's business context,
    KNOWN requirement values, UNAVAILABLE requirement statuses, and OnlinePresenceContext.

    Does NOT modify state directly.
    Raises StrategyGenerationError if generation or schema parsing fails.
    """
    tml_req = state.get_requirement_by_id("target_market_location")
    tml_known = bool(tml_req and tml_req.status == RequirementStatus.KNOWN and tml_req.value)
    user_prompt = build_strategy_generation_prompt(state)

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            operation="strategy_generation",
            conversation_id=state.thread_id,
        )

        data = json.loads(raw_response)
        strategy = MarketingStrategy.model_validate(data)
        strategy = normalize_budget_considerations(strategy, state.business_context.budget_resources)
        if not tml_known:
            loc_warning = "Target market location was not specified during discovery; confirm exact geographic targeting (cities, regions, or Pan-India) before launching paid campaigns."
            if strategy.target_audience_insights is None:
                strategy.target_audience_insights = []
            if not any("geographic" in a.lower() or "location" in a.lower() for a in strategy.target_audience_insights):
                strategy.target_audience_insights.append(loc_warning)
        logger.info("Successfully generated, normalized, and validated MarketingStrategy.")
        return strategy

    except (LLMServiceError, json.JSONDecodeError, ValidationError, Exception) as e:
        logger.error(f"Failed to generate strategy: {e}")
        raise StrategyGenerationError(f"Failed to generate marketing strategy: {e}") from e
