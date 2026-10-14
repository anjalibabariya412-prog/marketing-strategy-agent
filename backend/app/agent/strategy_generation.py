import json
import logging
import math
import re
from typing import Any, List, Optional
from pydantic import ValidationError

from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.document_context import format_document_context
from backend.app.models.information_requirement import RequirementStatus
from backend.app.models.marketing_strategy import BudgetItem, MarketingStrategy
from backend.app.services.llm_service import get_llm_response, LLMServiceError

logger = logging.getLogger(__name__)


class StrategyGenerationError(Exception):
    """Raised when marketing strategy generation or parsing fails."""
    pass


SYSTEM_PROMPT = (
    "You are a senior, pragmatic marketing strategist. "
    "Create a specific, actionable marketing strategy based strictly on the business context, "
    "resolved requirements, previous marketing activity, and online presence research provided.\n\n"

    "CORE RULES:\n"
    "1. Be specific and business-specific. Avoid generic advice such as 'use social media' or "
    "'increase brand awareness'. Recommend concrete channels, tactics, messaging, priorities, and actions.\n\n"

    "2. NEVER invent business facts. Do not invent product features, capabilities, customers, "
    "pricing, locations, channels, performance, infrastructure, testimonials, reviews, or processes. "
    "If information is unknown, keep it unknown or make a clearly phrased recommendation.\n\n"

    "3. Distinguish facts from recommendations naturally. "
    "Existing information may be stated directly. New ideas must use language such as "
    "'should', 'could', 'consider', 'recommend', or 'test'. Never present a recommendation as an existing practice.\n\n"

    "4. USE THE USER'S DETAILS EXACTLY. "
    "Do not change the provided audience, location, pricing, budget, marketing goal, product/service, "
    "or marketing period. Keep these details consistent throughout the strategy.\n\n"

    "5. STRATEGIC COHERENCE. "
    "Build one connected strategy: business situation → target audience → customer needs and buying behavior → "
    "positioning → value proposition → channels and tactics → customer acquisition → budget → KPIs → action plan.\n\n"

    "6. CUSTOMER ANALYSIS. "
    "Use provided customer needs, motivations, preferences, decision factors, buying behavior, occasions, "
    "and objections when available. Do not invent customer pain points or behavior.\n\n"

    "7. TARGET MARKET. "
    "If target_market_location is provided, use the exact location. Do not broaden it unless the user requested it. "
    "Location-specific tactics may be recommended when relevant.\n\n"

    "8. COMPETITOR INFORMATION. "
    "Use only the competitor information provided directly by the user (or extracted from user responses) to inform competitive_positioning, value_proposition, "
    "marketing_channels_and_tactics, and customer_acquisition_approach. "
    "Do not invent competitor names or competitor details. Do not perform or assume web research for competitors. "
    "If no competitors were provided or if the user states there are no direct competitors, clearly work with the available information.\n\n"

    "9. ONLINE PRESENCE & CLAIM EVIDENCE PRESERVATION.\n"
    "Use website/social research only for information actually supported by the collected data. "
    "Do not infer traffic, engagement, customer data, analytics, CRM, or internal systems.\n"
    "- PRESERVE EXACT CONDITIONS AND RESTRICTIONS: When recommending marketing tactics, ad copy, promotions, or budget allocations that reference client offers or claims (e.g. discounts, free shipping, guarantees), you MUST preserve all stated conditions and restrictions exactly as documented in the verified source evidence.\n"
    "  * For example, if free delivery applies only to 'prepaid orders above ₹350', do NOT present it as general free shipping on all orders; state 'free delivery on prepaid orders above ₹350'. If a discount requires a promo code (e.g., 'TRY10') or applies only to a specific category (e.g., 'first coffee purchase'), state that qualifier explicitly.\n"
    "- GROUNDING & VERIFICATION STATUS: Do NOT present unsupported or unresolved claims as established business facts. If a claim has verification status 'unsupported' or 'unresolved', or if its terms/conditions are ambiguous or incomplete, omit the claim or recommend verifying its exact terms with the client before launching campaigns.\n"
    "- Never invent offers, discounts, shipping guarantees, or promotions not found in the verified source evidence.\n\n"

    "10. HISTORICAL MARKETING EVIDENCE & PREVIOUS ACTIVITY.\n"
    "Use historical document evidence (from uploaded marketing PDF) and previous marketing information as supporting evidence:\n"
    "- 10a. STRICT PRIORITY OF CURRENT INPUTS: Current business information (budget, user-selected current marketing channels, primary goal, target audience) explicitly supplied in the initial form and subsequent user answers represents present-day reality and strictly overrides any conflicting historical values. Never replace current budget, current channels, or current goals with historical PDF values.\n"
    "- 10b. BUILD ON PROVEN SUCCESSES: Identify marketing channels, tactics, campaigns, and creative messaging that performed well in past activities and build upon them when directly relevant to the current marketing goal.\n"
    "- 10c. USE FAILURES AS LESSONS, NOT BANS: Treat unsuccessful past activities, underperforming channels, and historical bottlenecks as valuable lessons or risks to mitigate, NOT as absolute prohibitions. If recommending an activity that struggled in the past, explain how to address previous weaknesses (e.g. improved creative, refined audience targeting, or better tracking).\n"
    "- 10d. PRESERVE DATES, CONDITIONS, AND QUALIFIERS: When citing historical campaigns, preserve their documented dates, time periods, audience conditions, offer qualifiers, and page references. Do not drop restrictions or qualifiers.\n"
    "- 10e. NO PERFORMANCE GUARANTEES: Do not present historical metrics as current baseline performance or guarantee that past conversion rates, ROAS, or results will recur in the new campaign.\n"
    "- 10f. UNRELATED OR NON-MARKETING DOCUMENTS: If the uploaded document was classified as not marketing relevant (or contains no marketing insights), do NOT treat it as marketing evidence. Rely exclusively on current business inputs, Q&A, and verified online presence.\n\n"
    "11. MARKETING CHANNELS — USER-PROVIDED CHANNELS ARE PRIMARY. "
    "The user's 'Current Marketing Channels' field lists the channels they already use or have chosen to focus on. "
    "These are authoritative and must be treated as the primary foundation of the strategy.\n\n"

    "11a. BUILD THE STRATEGY AROUND USER-PROVIDED CHANNELS FIRST. "
    "For every channel the user provided, include concrete, actionable tactics specific to that channel. "
    "For example, if the user says 'Website', include website-focused tactics such as improving product pages, "
    "SEO, conversion optimization, content strategy, and user experience improvements. "
    "Do not skip, minimize, replace, or reinterpret the user's channels. "
    "'Website' means website — not 'online advertising', 'social media', or 'Google Ads'. "
    "Both the 'marketing_channels_and_tactics' section and the 'action_plan' MUST be built around these user-provided channels first.\n\n"

    "11b. STRUCTURE OF MARKETING CHANNELS & TACTICS. "
    "Structure the items in 'marketing_channels_and_tactics' cleanly into natural business groups without technical labels like '(existing)', '(new)', 'existing channel', or 'new channel'.\n"
    "- If the user provided current marketing channels, group tactics for those channels under the header ' Build on Your Current Marketing Channels'. Format each user-provided channel title in bold (e.g., '**Website**', '**LinkedIn**') followed by specific tactic items.\n"
    "- If additional channels are recommended that were not provided by the user, group them under the header 'Additional Marketing Opportunities to Consider'. Format each recommended channel title in bold (e.g., '**WhatsApp**', '**Google Search Ads**') followed by specific recommendation items.\n"
    "- Do NOT append '(existing)' or '(new)' or technical tag labels to channel names.\n"
    "- Never describe a recommended channel as an existing channel, and never describe a user-provided channel as a new opportunity.\n"
    "- If there are no additional channel recommendations, omit the 'Additional Marketing Opportunities to Consider' header.\n"
    "- If no current channels were provided by the user, omit the 'Build on Your Current Marketing Channels' header.\n\n"

    "11c. NATURAL BUSINESS RECOMMENDATION LANGUAGE. "
    "For channels listed under ' Additional Marketing Opportunities to Consider', use natural business recommendation phrasing (e.g., 'Consider targeted campaigns...', 'Test high-intent keywords...') without using technical tag labels like '(new)' or '(existing)'.\n\n"

    "11d. BUDGET MUST RESPECT USER-PROVIDED CHANNELS. "
    "When allocating budget, the user's existing/selected channels must receive meaningful budget allocation. "
    "Do not allocate most or all of the budget to newly recommended channels while giving little or nothing "
    "to the user's own channels. The user's channels come first in budget priority.\n\n"

    "11e. LIMIT ADDITIONAL CHANNEL RECOMMENDATIONS. "
    "Do not recommend every possible channel. Recommend at most 2-3 additional channels "
    "when strategically justified by the target audience, business model, goal, and budget. "
    "Each additional recommendation must explain why it complements the user's existing channels.\n\n"

    "12. CUSTOMER ACQUISITION. "
    "Create a realistic path from awareness to consideration to enquiry/purchase and retention where relevant. "
    "If the current sales process is unknown, recommend what should be established instead of claiming it exists.\n\n"

    "13. BUDGET. "
    "Use the exact provided amount, currency, and period. The marketing budget provided by the user is the total budget available for the entire 1-month marketing strategy. "
    "Never exceed the available monthly budget. "
    "Do not multiply the monthly budget by the number of weeks or months. "
    "If allocating the full budget, allocations across all items must add up exactly to the provided monthly amount. "
    "Never infer currency or convert the budget unless explicitly requested. "
    "If no budget is provided, do not invent a total.\n"
    "- Each budget allocation in 'budget_considerations' MUST be ONE complete structured object with keys: 'label', 'amount', 'percentage', and 'description'.\n"
    "- The sum of all 'amount' values in 'budget_considerations' MUST equal the user's total stated marketing budget when the entire budget is allocated. Never generate allocations whose sum exceeds or falls below the total budget.\n"
    "- 'amount' MUST include the appropriate currency symbol (e.g. '₹10,000' or '$500') matching the user's currency.\n"
    "- 'percentage' MUST be mathematically consistent and calculated strictly as: (item allocation amount / total marketing budget) * 100, formatted with a '%' sign (e.g. '20%', '40%'). Never set arbitrary or identical percentages (e.g. setting 25% across items with unequal amounts). The sum of all item percentages must total 100% when the entire budget is allocated.\n"
    "- Any monetary amount mentioned inside the 'description' must strictly match the item's 'amount'.\n"
    "- Never split one budget allocation into multiple items.\n"
    "- 'label', 'amount', 'percentage', and 'description' belong to the SAME budget item.\n"
    "- The description must be a complete sentence. Never put a sentence fragment into a separate item.\n"
    "- Never split an item at 'and', commas, parentheses, colons, or other punctuation.\n"
    "- If one allocation mentions multiple activities, keep them inside the same item's description.\n\n"

    "14. KPIs. "
    "KPIs must directly support the stated marketing goal and be appropriate for the business model. "
    "Any numerical target not provided by the user is an estimate or planning assumption, not a guarantee. "
    "Keep all calculations mathematically consistent. Do not invent historical performance.\n"
    "- Each KPI in 'kpis' MUST be ONE complete structured object with keys: 'metric', 'target', and 'description'.\n"
    "- Never split a KPI metric and its target into separate items.\n"
    "- Never split an item at parentheses, colons, commas, or other punctuation.\n"
    "- The 'target' field must contain the complete target value (e.g., '150 leads').\n"
    "- The description must explain what the metric measures and how to track it.\n\n"
    "All KPI targets must represent the expected measurement period for this 1-month marketing strategy.\n"

    "15. ACTION PLAN AND TIMELINE — BUSINESS-SPECIFIC 4-WEEK ROADMAP.\n"
    "Create a practical, prioritized, and highly actionable execution roadmap for implementing the marketing strategy during exactly one month (four weeks).\n\n"
    "15a. BUILD DIRECTLY AROUND USER-SELECTED CHANNELS:\n"
    "- The initial form collects the marketing channels the user wants to focus on ('Current Marketing Channels').\n"
    "- Treat these user-selected channels as the primary channels for the entire marketing strategy and action plan.\n"
    "- Build the four-week plan around the selected channels instead of routinely recommending new ones.\n"
    "- Do not automatically introduce channels such as Google Ads, Google Shopping, influencer marketing, email marketing, or WhatsApp when the user has not selected them.\n"
    "- Focus on improving the effectiveness of the selected channels through appropriate content, campaigns, audience targeting, testing, optimization, and performance measurement.\n"
    "- Do not interpret the selected channels as a requirement to use every possible tactic available on those channels. Choose tactics according to the business, goal, audience, and budget.\n"
    "- Recommend an additional channel only when there is a clear, business-specific justification and a meaningful reason the selected channels alone may not adequately address the goal. Present it as an optional recommendation, not as an assumed part of the existing marketing setup.\n"
    "- Never replace the user's selected channels with different channels without justification.\n\n"
    "15b. FOUR DISTINCT WEEKLY PHASES WITH CONCRETE DETAILS:\n"
    "- 'action_plan' must contain exactly 4 comprehensive, distinct items corresponding to Week 1, Week 2, Week 3, and Week 4.\n"
    "- Format each item starting with a bold milestone header: '**Week X: [Milestone Title]**'.\n"
    "- For each week, define:\n"
    "  1. A clear strategic objective for that week.\n"
    "  2. Concrete execution steps explaining what should actually be done on the selected channels (e.g. preparing specific creative assets, drafting copy addressing target audience pain points, configuring audience targeting parameters, launching campaigns, fixing conversion friction). Do NOT use vague instructions such as 'improve engagement', 'optimize campaigns', or 'review performance' without specifying how.\n"
    "  3. Relevant measurement or review activities specifying what metrics to inspect and what evaluation criteria to apply.\n"
    "- Sequence activities logically:\n"
    "  * Week 1 (Preparation & Initial Setup): Asset creation, profile/landing page adjustments, prerequisite tracking setup, testing messaging, and initial campaign launch on selected channels.\n"
    "  * Week 2 (Execution & Audience Testing): Full rollout on selected channels, testing messaging variations across target audience segments, and monitoring early performance.\n"
    "  * Week 3 (Mid-Point Review & Optimization): Reviewing results, pausing underperforming creatives or ad sets, reallocating budget toward winning messages, and refining the user journey.\n"
    "  * Week 4 (Evaluation, Scaling & Ongoing Roadmap): Assessing overall performance against KPIs, scaling top-performing assets, and establishing an ongoing marketing cadence.\n"
    "- Ensure the four weeks have distinct purposes and do not repeat generic recommendations.\n\n"
    "15c. GROUNDED AND REALISTIC EXECUTION:\n"
    "- Fit all activities strictly within the user's actual marketing budget and a one-month timeline.\n"
    "- Use facts provided in the initial form, subsequent answers, and available online presence context. Preserve the user's provided information and do not contradict it.\n"
    "- Do not invent existing campaigns, customer counts, testimonials, product categories, sales figures, delivery policies, discounts, offers, or business capabilities.\n"
    "- Do not present unverified information as an established fact.\n"
    "- Avoid assuming the business has a marketing team, advertising infrastructure, analytics setup, customer database, or other resources unless supported by available information.\n"
    "- Include prerequisites when a proposed activity depends on setup or infrastructure that may not exist.\n"
    "- If suggesting a discount, promotion, retargeting campaign, or another activity that depends on an unavailable business capability, frame it as conditional where appropriate.\n"
    "- Do not invent historical performance or guarantee a particular percentage increase in sales, engagement, conversions, or revenue. If measurable targets are included, distinguish proposed targets from guaranteed outcomes.\n"
    "- Prefer a focused, achievable plan over an unnecessarily large list of activities.\n\n"

    "16. ASSUMPTIONS & PLANNING ESTIMATES. "
    "If assumptions or estimated numbers are necessary, integrate them directly into their relevant standard section "
    "(such as 'target_audience_insights', 'budget_considerations', 'kpis', or 'action_plan'). "
    "Do NOT create or return any custom or additional sections. All strategic insights must be placed strictly within the 9 fixed sections.\n\n"

    "17. FINAL QUALITY CHECK. "
    "Before responding, verify that all business facts, competitor facts, locations, pricing, budget, "
    "KPIs, and recommendations are consistent and grounded. Remove generic filler and unsupported claims.\n\n"

    "OUTPUT RULES:\n"
    "Return ONLY one valid JSON object. "
    "Do not return explanations, code fences, or commentary outside the JSON object. "
    "Do not change field names or field types.\n\n"

    "CONTENT RULES (for all list fields):\n"
    "Each list field must be a JSON array of plain-text strings or structured objects as specified in REQUIRED JSON STRUCTURE. "
    "Each item is one concise, self-contained strategy point or object. "
    "Do NOT use unnecessary Markdown syntax inside list items (except for structural section headers like '#### Build on Your Current Marketing Channels', '#### Additional Marketing Opportunities to Consider', and bold channel titles like '**Website**' where explicitly required). "
    "Write each point as a complete, meaningful sentence or short paragraph. "
    "Do not write single-word or empty items. "
    "Aim for 3-8 items per field depending on the depth needed.\n\n"

    "REQUIRED JSON STRUCTURE:\n"
    "All 9 sections below are mandatory and must be populated with business-tailored recommendations. Do not omit any section.\n"
    "{\n"
    '  "business_overview": ["string", ...],\n'
    '  "target_audience_insights": ["string", ...],\n'
    '  "competitive_positioning": ["string", ...],\n'
    '  "value_proposition": ["string", ...],\n'
    '  "marketing_channels_and_tactics": ["string", ...],\n'
    '  "customer_acquisition_approach": ["string", ...],\n'
    '  "budget_considerations": [\n'
    '    {"label": "string", "amount": "string", "percentage": "string or null", "description": "string"}\n'
    '  ],\n'
    '  "kpis": [\n'
    '    {"metric": "string", "target": "string", "description": "string"}\n'
    '  ],\n'
    '  "action_plan": ["string", ...]\n'
    "}\n\n"

    "18. MISSING OR UNKNOWN INFORMATION. "
    "If the user says 'I don't know', 'not sure', 'unknown', or provides no information for a requirement, "
    "treat that information as unknown. Never invent the missing fact. "
    "Continue generating the strategy using the reliable information that is available. "
    "When the missing information materially affects a recommendation, provide a practical test, assumption to verify, "
    "or discovery approach instead of pretending the information is known. "
    "Do not unnecessarily make the entire strategy vague simply because one requirement is unknown.\n\n"
)


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
    ctx = state.business_context
    budget_desc = "Not provided"
    if ctx.budget_resources:
        symbol = "₹" if str(ctx.budget_resources.currency) in ("INR", "CurrencyEnum.INR") else "$"
        budget_desc = f"amount = {ctx.budget_resources.amount}, currency = {ctx.budget_resources.currency.value if hasattr(ctx.budget_resources.currency, 'value') else ctx.budget_resources.currency} ({symbol}{ctx.budget_resources.amount:,.0f})"

    context_lines = [
        f"Company Name: {ctx.company_name or 'Not provided'}",
        f"Product/Service: {ctx.product_or_service or 'Not provided'}",
        f"Marketing Goal: {ctx.marketing_goal or 'Not provided'}",
        f"Target Audience: {ctx.target_audience or 'Not provided'}",
        f"Marketing Budget/Resources: {budget_desc}",
        f"User-Selected Marketing Channels (Primary Focus): {ctx.current_marketing_channels or 'Not provided'}",
        f"Competitors / Alternatives: {', '.join(ctx.competitors) if ctx.competitors else 'Not provided'}",
    ]

    # Collect facts from KNOWN requirements
    known_reqs = [
        req for req in state.requirements 
        if req.status == RequirementStatus.KNOWN and req.value is not None
    ]
    known_lines = []
    for req in known_reqs:
        known_lines.append(f"- {req.title} [{req.id}]: {req.value}")

    # Collect facts from UNAVAILABLE requirements
    unavailable_reqs = [
        req for req in state.requirements 
        if req.status == RequirementStatus.UNAVAILABLE
    ]
    unavailable_lines = []
    for req in unavailable_reqs:
        unavailable_lines.append(f"- {req.title} [{req.id}]: User indicated this information is currently unavailable/undecided.")

    # Assemble user prompt
    prompt_parts = [
        "BUSINESS CONTEXT:",
        "\n".join(context_lines)
    ]

    if known_lines:
        prompt_parts.extend([
            "\nCOLLECTED FACTS (KNOWN INFORMATION):",
            "\n".join(known_lines)
        ])

    if unavailable_lines:
        prompt_parts.extend([
            "\nUNAVAILABLE INFORMATION (CLIENT DID NOT HAVE DATA):",
            "\n".join(unavailable_lines)
        ])

    doc_str = format_document_context(state.document_context)
    if doc_str:
        prompt_parts.extend([
            "\nHISTORICAL MARKETING EVIDENCE (FROM UPLOADED MARKETING PDF):",
            doc_str,
            "\nHISTORICAL PDF EVIDENCE INSTRUCTIONS:\n"
            "- STRICT SEPARATION: The above document details PAST marketing efforts, campaigns, and historical metrics. It is historical supporting evidence, NOT the client's current business profile.\n"
            "- CURRENT VALUES OVERRIDE: The user's initial form values (current budget, user-selected current marketing channels, current goal, target audience) are authoritative and represent the present-day business. Never overwrite or replace current form values with historical PDF values (e.g. historical budgets, past goals, or past channel mixes).\n"
            "- BUILD ON SUCCESSES: Build on channels, activities, and creative concepts that performed well in the past when relevant to the current goal.\n"
            "- LEARN FROM FAILURES: Treat unsuccessful past activities and bottlenecks as valuable lessons or cautions to address, NOT as absolute prohibitions.\n"
            "- PRESERVE SOURCE INTEGRITY: Preserve dates, time periods, conditions, qualifiers, and page references when citing historical findings.\n"
            "- NO GUARANTEES: Never present historical metrics as current performance or guarantee that past results will recur.\n"
            "- NON-MARKETING PDFS: If the document is marked as not marketing relevant, do not treat it as marketing evidence."
        ])
    elif ctx.past_marketing_document:
        prompt_parts.extend([
            "\nPAST MARKETING DOCUMENT SUMMARY:",
            ctx.past_marketing_document
        ])

    if state.online_presence_context:
        op = state.online_presence_context
        op_lines = [f"Overall Summary: {op.overall_summary}"]

        for src_name, src_obj in [
            ("Website", op.website),
            ("Instagram", op.instagram),
            ("Facebook", op.facebook),
            ("LinkedIn", op.linkedin)
        ]:
            if src_obj:
                op_lines.append(f"{src_name} Summary: {src_obj.summary}")
                if src_obj.relevant_products_or_services:
                    op_lines.append(f"  {src_name} Products/Services: {', '.join(src_obj.relevant_products_or_services)}")
                if src_obj.positioning_or_messaging:
                    op_lines.append(f"  {src_name} Messaging: {', '.join(src_obj.positioning_or_messaging)}")
                if src_obj.marketing_content:
                    op_lines.append(f"  {src_name} Marketing Content: {', '.join(src_obj.marketing_content)}")
                if src_obj.other_strategy_relevant_information:
                    op_lines.append(f"  {src_name} Other Info: {', '.join(src_obj.other_strategy_relevant_information)}")
                if hasattr(src_obj, "evidence") and src_obj.evidence:
                    op_lines.append(f"  {src_name} Verified Claim Evidence & Offer Conditions:")
                    for ev in src_obj.evidence:
                        cond_str = f" [RESTRICTIONS/CONDITIONS: {ev.conditions}]" if ev.conditions else " [NO RESTRICTIONS STATED]"
                        op_lines.append(f"    - Claim: \"{ev.claim}\" (Status: {ev.verification_status.value}){cond_str}")
                        op_lines.append(f"      Source URL: {ev.source_url}")
                        op_lines.append(f"      Exact Supporting Text: \"{ev.supporting_text}\"")
                if src_name == "Instagram":
                    if src_obj.profile:
                        prof = src_obj.profile
                        prof_parts = []
                        if prof.get("username"):
                            prof_parts.append(f"@{prof.get('username')}")
                        if prof.get("fullName"):
                            prof_parts.append(f"Name: {prof.get('fullName')}")
                        if prof.get("followersCount") is not None:
                            prof_parts.append(f"Followers: {prof.get('followersCount')}")
                        if prof.get("postsCount") is not None:
                            prof_parts.append(f"Posts: {prof.get('postsCount')}")
                        if prof.get("biography"):
                            prof_parts.append(f"Bio: {prof.get('biography')}")
                        if prof_parts:
                            op_lines.append(f"  Instagram Profile: {' | '.join(prof_parts)}")
                    if src_obj.posts:
                        op_lines.append(f"  Instagram Posts ({len(src_obj.posts)} tracked)")
                elif src_name == "Facebook":
                    if src_obj.profile:
                        prof = src_obj.profile
                        prof_parts = []
                        if prof.get("pageName"):
                            prof_parts.append(f"Page: {prof.get('pageName')}")
                        if prof.get("username"):
                            prof_parts.append(f"@{prof.get('username')}")
                        if prof.get("followersCount") is not None:
                            prof_parts.append(f"Followers: {prof.get('followersCount')}")
                        if prof.get("likesCount") is not None:
                            prof_parts.append(f"Likes: {prof.get('likesCount')}")
                        if prof.get("about"):
                            prof_parts.append(f"About: {prof.get('about')}")
                        if prof_parts:
                            op_lines.append(f"  Facebook Page: {' | '.join(prof_parts)}")
                    if src_obj.posts:
                        op_lines.append(f"  Facebook Posts ({len(src_obj.posts)} tracked)")

        # Format any remaining evidence from top-level op.all_evidence
        printed_evidence = set()
        for _, s_obj in [("Website", op.website), ("Instagram", op.instagram), ("Facebook", op.facebook), ("LinkedIn", op.linkedin)]:
            if s_obj and hasattr(s_obj, "evidence") and s_obj.evidence:
                for ev in s_obj.evidence:
                    printed_evidence.add((ev.claim, ev.source_url))

        remaining_evidence = [ev for ev in op.all_evidence if (ev.claim, ev.source_url) not in printed_evidence]
        if remaining_evidence:
            op_lines.append("  Additional Verified Claim Evidence & Offer Conditions:")
            for ev in remaining_evidence:
                cond_str = f" [RESTRICTIONS/CONDITIONS: {ev.conditions}]" if ev.conditions else " [NO RESTRICTIONS STATED]"
                op_lines.append(f"    - Claim: \"{ev.claim}\" (Status: {ev.verification_status.value}){cond_str}")
                op_lines.append(f"      Source URL: {ev.source_url}")
                op_lines.append(f"      Exact Supporting Text: \"{ev.supporting_text}\"")

        prompt_parts.extend([
            "\nONLINE PRESENCE CONTEXT (SCRAPED FROM VERIFIED LINKS):",
            "\n".join(op_lines)
        ])

        if op.all_evidence:
            prompt_parts.append(
                "\nOFFER EVIDENCE RESTRICTIONS INSTRUCTION:\n"
                "When referencing or recommending client promotions, discounts, or delivery policies in marketing channels, tactics, budget, or action plans, "
                "strictly preserve all stated conditions and restrictions (e.g. prepaid order requirements, promo codes, minimum spend). "
                "Never drop qualifiers or present conditional perks as unconditional. "
                "If an offer or claim is marked 'unsupported' or 'unresolved', do not present it as an established business fact; omit it or recommend verifying it with the client before launch."
            )

    if ctx.budget_resources and ctx.budget_resources.amount and ctx.budget_resources.amount > 0:
        symbol = "₹" if str(ctx.budget_resources.currency) in ("INR", "CurrencyEnum.INR") else "$"
        tot_amt = ctx.budget_resources.amount
        amt_str = f"{symbol}{tot_amt:,.0f}" if float(tot_amt).is_integer() else f"{symbol}{tot_amt:,.2f}"
        curr_code = ctx.budget_resources.currency.value if hasattr(ctx.budget_resources.currency, 'value') else ctx.budget_resources.currency
        prompt_parts.append(
            f"\nBUDGET ALLOCATION DIRECTIVE:\n"
            f"- Authoritative Total Marketing Budget: {amt_str} ({curr_code}).\n"
            f"- In 'budget_considerations', the sum of all item amounts must add up EXACTLY to {amt_str}.\n"
            f"- For every item, calculate percentage strictly as: (item amount / {tot_amt}) * 100, formatted with '%' (e.g. '20%'). Ensure percentages accurately reflect amounts and sum to 100%.\n"
            f"- Allocate realistic proportions prioritizing the user's selected marketing channels."
        )

    prompt_parts.append(
        "\nACTION PLAN & TIMELINE DIRECTIVE:\n"
        "Generate 'action_plan' as exactly 4 distinct, business-specific milestone strings for Week 1, Week 2, Week 3, and Week 4.\n"
        "- Build the 4-week plan directly around the User-Selected Marketing Channels above. Do not routinely introduce unselected channels (e.g. Google Ads, WhatsApp, influencer marketing) unless strictly necessary, explicitly justified, and framed as optional tests.\n"
        "- Format each week starting with a bold header: '**Week 1: [Milestone Title]**', '**Week 2: [Milestone Title]**', '**Week 3: [Milestone Title]**', '**Week 4: [Milestone Title]**'.\n"
        "- Detail concrete execution steps for each week (content/assets to produce, messaging to test, targeting parameters, launch steps, optimization adjustments) and specific metrics to review. Avoid vague directives like 'improve engagement' or 'optimize campaigns' without concrete instructions.\n"
        "- Ground the timeline in the available budget and realistic 1-month capabilities without assuming nonexistent infrastructure."
    )

    tml_req = state.get_requirement_by_id("target_market_location")
    tml_known = bool(tml_req and tml_req.status == RequirementStatus.KNOWN and tml_req.value)
    if not tml_known:
        prompt_parts.append(
            "\nMANDATORY TARGET MARKET LOCATION UNRESOLVED DIRECTIVE:\n"
            "- Target market location was NOT provided or confirmed during discovery.\n"
            "- CRITICAL RULE: DO NOT fabricate, invent, or assume a specific target city, state, or region.\n"
            "- In 'target_audience_insights', you MUST explicitly document that the target geographic market is unconfirmed and must be specified before launching location-dependent campaigns or ad spend.\n"
            "- Formulate audience targeting and channel tactics as location-adaptive recommendations."
        )

    prompt_parts.append(
        "\nSynthesize all available context above into a high-quality, tailored marketing strategy JSON object."
    )

    user_prompt = "\n".join(prompt_parts)

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
