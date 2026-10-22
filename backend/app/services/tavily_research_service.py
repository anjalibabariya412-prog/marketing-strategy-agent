"""
Tavily Competitor Web Research Service.

Performs targeted web research on identifiable competitors using the official tavily-python SDK
and analyzes findings using the centralized LLM service to produce structured competitor intelligence.
Safe, robust, and non-blocking: network issues, missing keys, or timeouts never halt the workflow.
"""

from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from tavily import TavilyClient

from backend.app.core.config import settings
from backend.app.core.progress_notifier import emit_progress
from backend.app.schemas.agent.agent_state import MarketingAgentState
from backend.app.schemas.agent.competitor_analysis import (
    CompetitorAnalysisItem,
    CompetitorResearchReport,
    CompetitorSearchResult,
)
from backend.app.schemas.agent.information_requirement import RequirementStatus
from backend.app.prompts.competitor_research import (
    COMPETITOR_ANALYSIS_SYSTEM_PROMPT,
    build_competitor_analysis_prompt,
    COMPETITOR_INTENT_DETECTION_SYSTEM_PROMPT,
    build_competitor_intent_detection_prompt,
    COMPETITOR_DISCOVERY_SYSTEM_PROMPT,
    build_competitor_discovery_prompt,
)
from backend.app.services.llm_service import get_llm_response, LLMServiceError

logger = logging.getLogger(__name__)

# Maximum competitors to research per session
MAX_COMPETITORS_PER_RUN = 3
MAX_DISCOVERED_COMPETITORS = 3

def is_generic_substitute(name: str) -> bool:
    """
    Deprecated: Semantic substitute detection is now handled exclusively by the LLM classifier.
    Retained for interface compatibility without hardcoded vocabulary.
    """
    return False


def is_own_brand(name: str, business_context: Optional[Any]) -> bool:
    """
    Returns True if name matches or refers to the client's own company name or brand.
    """
    if not name or not business_context:
        return False
    company = getattr(business_context, "company_name", None) or ""
    company_clean = company.strip().lower()
    if not company_clean:
        return False
    clean = name.strip().lower()
    if clean == company_clean:
        return True
    # Strip common corporate suffixes
    comp_base = re.sub(r"\b(inc|llc|co|ltd|brand|company|enterprises|pvt)\b", "", company_clean).strip()
    clean_base = re.sub(r"\b(inc|llc|co|ltd|brand|company|enterprises|pvt)\b", "", clean).strip()
    if comp_base and clean_base and comp_base == clean_base:
        return True
    if len(clean) >= 4 and len(company_clean) >= 4:
        if clean in company_clean or company_clean in clean:
            return True
    return False


def extract_domain_or_clean_name(name_or_url: str) -> str:
    """
    If name_or_url is a website or URL (e.g. 'https://www.sleepyowl.co' or 'sleepyowl.co'),
    extracts the domain or brand name. Otherwise cleans the string.
    """
    if not name_or_url or not isinstance(name_or_url, str):
        return ""
    clean = name_or_url.strip()
    if clean.startswith("http://") or clean.startswith("https://") or re.search(r"^[a-zA-Z0-9-]+\.(?:com|in|co|org|io|net|ai)\b", clean):
        parsed = urlparse(clean if "://" in clean else f"https://{clean}")
        netloc = parsed.netloc or parsed.path
        domain_parts = netloc.replace("www.", "").split(".")
        if domain_parts and len(domain_parts[0]) > 1:
            return domain_parts[0].replace("-", " ").title()
    return clean


def is_identifiable_competitor(name: str) -> bool:
    """
    Performs basic technical validation on a candidate competitor name.
    Validates that the input is a non-empty string of meaningful length.
    Semantic classification (distinguishing identifiable brands from generic descriptions
    or non-commercial substitutes) is handled by the LLM classifier.
    """
    if not name or not isinstance(name, str):
        return False
    clean = name.strip()
    return len(clean) >= 2


def extract_user_stated_context_for_competitor(competitor_name: str, user_answer: Optional[str]) -> Optional[str]:
    """
    Extracts relevant sentence(s) or clauses from the user's natural language answer
    that specifically mention or discuss the given competitor.
    """
    if not user_answer or not competitor_name:
        return None
    sentences = re.split(r"[.\n;]", user_answer)
    comp_lower = competitor_name.strip().lower()
    matching = [
        s.strip() for s in sentences
        if comp_lower in s.lower() and len(s.strip()) > 3
    ]
    if matching:
        return "; ".join(matching)
    return None


def execute_tavily_search(
    query: str,
    api_key: Optional[str] = None,
    max_results: int = 5,
    timeout: float = 15.0,
) -> List[Dict[str, Any]]:
    """
    Executes a web search using the official tavily-python SDK.
    Deduplicates results by URL and returns clean structured result dictionaries.
    Catches all network and provider errors safely.
    """
    key = settings.tavily_api_key if api_key is None else api_key
    if not key or not key.strip():
        logger.warning("execute_tavily_search skipped: TAVILY_API_KEY is not configured.")
        return []

    try:
        client = TavilyClient(api_key=key.strip())
        logger.info(f"Calling Tavily search API with query: '{query}'")
        raw_res = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            timeout=timeout,
        )

        results = raw_res.get("results", []) if isinstance(raw_res, dict) else []
        seen_urls: Set[str] = set()
        deduped_results: List[Dict[str, Any]] = []

        now_str = datetime.now(timezone.utc).isoformat()
        for item in results:
            if not isinstance(item, dict):
                continue
            url = item.get("url", "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            deduped_results.append({
                "title": item.get("title", "No Title").strip(),
                "url": url,
                "content": item.get("content", "").strip(),
                "score": item.get("score"),
                "retrieval_timestamp": now_str,
            })

        logger.info(f"Tavily returned {len(deduped_results)} deduplicated result(s) for query: '{query}'")
        return deduped_results

    except Exception as e:
        logger.warning(f"Tavily search failed for query '{query}': {e}. Returning empty results.")
        return []


def analyze_competitor_with_llm(
    competitor_name: str,
    search_results: List[Dict[str, Any]],
    business_context_summary: str,
    user_stated_context: Optional[str] = None,
    conversation_id: Optional[str] = None,
    strategy_request_id: Optional[str] = None,
) -> Optional[CompetitorAnalysisItem]:
    """
    Analyzes retrieved Tavily search results using the centralized LLM service
    to generate a structured CompetitorAnalysisItem.
    Rejects irrelevant or unsupported entities.
    """
    clean_name = extract_domain_or_clean_name(competitor_name)
    user_prompt = build_competitor_analysis_prompt(
        competitor_name=clean_name,
        search_results=search_results,
        business_context_summary=business_context_summary,
        user_stated_context=user_stated_context,
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=COMPETITOR_ANALYSIS_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            operation="competitor_analysis",
            conversation_id=conversation_id,
            strategy_request_id=strategy_request_id,
        )

        data = json.loads(raw_response)
        if not isinstance(data, dict):
            return None

        # Check relevance: reject irrelevant or unsupported entities
        is_relevant = data.get("is_relevant_competitor")
        quality = str(data.get("evidence_quality", "")).lower()
        if is_relevant is False or quality == "irrelevant":
            logger.info(f"Competitor '{clean_name}' rejected as irrelevant or unsupported.")
            return None

        # Detect official website from URL input if available
        official_web = data.get("official_website")
        if not official_web:
            if competitor_name.startswith("http://") or competitor_name.startswith("https://"):
                official_web = competitor_name
            elif re.search(r"^[a-zA-Z0-9-]+\.(?:com|in|co|org|io|net)\b", competitor_name):
                official_web = f"https://{competitor_name}"

        sources = data.get("source_urls") or [r["url"] for r in search_results if "url" in r]
        if official_web and official_web not in sources:
            sources.insert(0, official_web)

        # Build analysis item with safety fallbacks
        return CompetitorAnalysisItem(
            competitor_name=data.get("competitor_name") or clean_name,
            user_stated_context=user_stated_context,
            official_website=official_web,
            products_and_services=data.get("products_and_services") or [],
            target_audience=data.get("target_audience"),
            positioning=data.get("positioning"),
            pricing_information=data.get("pricing_information"),
            differentiators=data.get("differentiators") or [],
            marketing_channels_and_campaigns=data.get("marketing_channels_and_campaigns") or [],
            strengths=data.get("strengths") or [],
            weaknesses=data.get("weaknesses") or [],
            source_urls=sources,
            confidence_score=data.get("confidence_score") or (0.75 if search_results else 0.3),
            evidence_quality=data.get("evidence_quality") or ("medium" if search_results else "incomplete"),
            summary=data.get("summary"),
        )

    except (LLMServiceError, json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to analyze competitor '{clean_name}' with LLM: {e}")
        # Return fallback with search URLs rather than failing
        if search_results:
            official_web = competitor_name if (competitor_name.startswith("http://") or competitor_name.startswith("https://")) else None
            sources = [r["url"] for r in search_results if "url" in r]
            if official_web and official_web not in sources:
                sources.insert(0, official_web)
            return CompetitorAnalysisItem(
                competitor_name=clean_name,
                official_website=official_web,
                user_stated_context=user_stated_context,
                source_urls=sources,
                evidence_quality="incomplete",
                summary=f"Public web information retrieved for {clean_name}.",
            )
        return None


def update_state_competitors(state: MarketingAgentState, competitor_list: List[str]) -> None:
    """
    Appends new non-duplicate competitor names to state.business_context.competitors,
    excluding the user's own brand. Technical validation ensures valid strings.
    """
    if not competitor_list:
        return
    if state.business_context.competitors is None:
        state.business_context.competitors = []

    existing_set = {c.lower() for c in state.business_context.competitors}
    for comp in competitor_list:
        if comp and isinstance(comp, str) and len(comp.strip()) >= 2:
            clean = comp.strip()
            # Exclude user's own brand
            if is_own_brand(clean, state.business_context):
                logger.info(f"update_state_competitors: Excluded client's own brand: '{clean}'")
                continue
            if clean.lower() not in existing_set:
                existing_set.add(clean.lower())
                state.business_context.competitors.append(clean)


def classify_competitor_response(
    user_answer: str,
    business_context_summary: Optional[str] = None,
    conversation_id: Optional[str] = None,
    strategy_request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classifies the semantic intent of a user's response regarding competitors into:
    - 'provided_specific': User provided concrete competitor/brand name(s) or websites.
    - 'request_discovery': User asked the agent to discover/identify competitors.
    - 'uncertain_or_unknown': User expressed uncertainty or doesn't know competitors.
    - 'generic_category': User provided a generic industry/product category or generic substitutes (e.g. 'coffee brands', 'making coffee at home').
    Uses the centralized LLM for semantic classification.
    Safe failure handling returns conservative uncertain_or_unknown intent without hardcoded fallback heuristics.
    """
    if not user_answer or not isinstance(user_answer, str) or not user_answer.strip():
        return {
            "intent": "uncertain_or_unknown",
            "competitor_names": [],
            "generic_substitutes": [],
            "generic_category": None,
            "reasoning": "Empty or non-string response",
        }

    user_prompt = build_competitor_intent_detection_prompt(
        user_answer=user_answer,
        business_context_summary=business_context_summary,
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=COMPETITOR_INTENT_DETECTION_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            operation="competitor_intent_classification",
            conversation_id=conversation_id,
            strategy_request_id=strategy_request_id,
        )

        data = json.loads(raw_response)
        if not isinstance(data, dict):
            raise ValueError(f"LLM did not return a JSON object: {type(data)}")

        raw_intent = str(data.get("intent", "")).strip().lower()
        valid_intents = {"provided_specific", "request_discovery", "uncertain_or_unknown", "generic_category"}
        # If model returned an unexpected or malformed intent, conservatively treat as uncertain_or_unknown
        intent = raw_intent if raw_intent in valid_intents else "uncertain_or_unknown"

        raw_names = data.get("competitor_names") or data.get("competitors") or []
        competitor_names: List[str] = []
        if isinstance(raw_names, list):
            seen = set()
            for n in raw_names:
                if isinstance(n, str) and len(n.strip()) >= 2:
                    clean_n = n.strip()
                    if clean_n.lower() not in seen:
                        seen.add(clean_n.lower())
                        competitor_names.append(clean_n)

        raw_subs = data.get("generic_substitutes") or []
        generic_substitutes: List[str] = []
        if isinstance(raw_subs, list):
            seen_subs = set()
            for s in raw_subs:
                if isinstance(s, str) and s.strip():
                    clean_s = s.strip()
                    if clean_s.lower() not in seen_subs:
                        seen_subs.add(clean_s.lower())
                        generic_substitutes.append(clean_s)

        generic_cat = data.get("generic_category")
        if isinstance(generic_cat, str) and generic_cat.strip():
            generic_cat = generic_cat.strip()
        else:
            generic_cat = None

        # Harmonize intent with extracted structured entities
        if competitor_names:
            intent = "provided_specific"
        elif intent == "provided_specific" and not competitor_names:
            if generic_cat or generic_substitutes:
                intent = "generic_category"
            else:
                intent = "uncertain_or_unknown"

        return {
            "intent": intent,
            "competitor_names": competitor_names,
            "generic_substitutes": generic_substitutes,
            "generic_category": generic_cat,
            "reasoning": str(data.get("reasoning", "")),
        }

    except (LLMServiceError, json.JSONDecodeError, Exception) as e:
        logger.warning(
            f"classify_competitor_response LLM call failed or unparseable: {e}. "
            "Returning safe uncertain_or_unknown fallback without hardcoded heuristics."
        )
        return {
            "intent": "uncertain_or_unknown",
            "competitor_names": [],
            "generic_substitutes": [],
            "generic_category": None,
            "reasoning": f"Safe fallback due to classification failure: {e}",
        }


def discover_competitors_with_tavily(
    state: MarketingAgentState,
    generic_category: Optional[str] = None,
    conversation_id: Optional[str] = None,
    strategy_request_id: Optional[str] = None,
) -> List[str]:
    """
    Discovers relevant competitors using business context and Tavily web research.
    Uses multiple focused search queries when appropriate.
    Limits to a maximum of 3 unique competitors per research run (Requirement 3 & 4).
    Returns fewer if sufficient credible evidence is unavailable.
    Does NOT restrict to nearby businesses unless geographical relevance is established.
    """
    ctx = state.business_context
    company_name = (ctx.company_name or "").strip()
    product_or_service = (ctx.product_or_service or "").strip()
    target_audience = (ctx.target_audience or "").strip()
    marketing_goal = (ctx.marketing_goal or "").strip()

    usp_req = state.get_requirement_by_id("usp_differentiation")
    positioning = usp_req.value.strip() if (usp_req and usp_req.status == RequirementStatus.KNOWN and usp_req.value) else None

    # Check established geographic relevance
    tml_req = state.get_requirement_by_id("target_market_location")
    location = None
    if tml_req and tml_req.status == RequirementStatus.KNOWN and tml_req.value:
        location = tml_req.value.strip()

    # Determine core search term
    if generic_category and generic_category.strip():
        search_term = generic_category.strip()
    elif product_or_service:
        search_term = product_or_service
    elif company_name:
        search_term = f"{company_name} industry"
    else:
        search_term = "commercial business"

    # Formulate multiple focused search queries to ensure credible discovery
    queries: List[str] = []
    if location:
        queries.append(f"top {search_term} competitors brands alternatives {location}")
        queries.append(f"best {search_term} brands companies market leaders {location}")
    else:
        queries.append(f"top {search_term} competitors brands alternatives market leaders")
        queries.append(f"best {search_term} brands companies")

    logger.info(f"discover_competitors_with_tavily: Searching with primary query '{queries[0]}'")
    search_results = execute_tavily_search(
        query=queries[0],
        api_key=settings.tavily_api_key,
        max_results=5,
        timeout=15.0,
    )

    # Use second query if primary query returned sparse results
    if len(search_results) < 3 and len(queries) > 1:
        logger.info(f"discover_competitors_with_tavily: Supplementing with second query '{queries[1]}'")
        second_res = execute_tavily_search(
            query=queries[1],
            api_key=settings.tavily_api_key,
            max_results=5,
            timeout=15.0,
        )
        seen_urls = {r.get("url") for r in search_results if r.get("url")}
        for item in second_res:
            u = item.get("url")
            if u and u not in seen_urls:
                seen_urls.add(u)
                search_results.append(item)

    if not search_results:
        logger.warning(f"Competitor discovery web search returned no results. Returning [].")
        return []

    # Build business context summary
    biz_lines = []
    if company_name:
        biz_lines.append(f"Company: {company_name}")
    if product_or_service:
        biz_lines.append(f"Product/Service: {product_or_service}")
    if target_audience:
        biz_lines.append(f"Target Audience: {target_audience}")
    if positioning:
        biz_lines.append(f"Positioning / USP: {positioning}")
    if marketing_goal:
        biz_lines.append(f"Marketing Goal: {marketing_goal}")
    if location:
        biz_lines.append(f"Target Market Location: {location}")

    biz_summary = "\n".join(biz_lines) if biz_lines else "General business profile"

    user_prompt = build_competitor_discovery_prompt(
        business_context_summary=biz_summary,
        search_results=search_results,
        generic_category=generic_category,
        client_company_name=company_name,
    )

    try:
        strat_id = strategy_request_id or (state.resolve_strategy_request_id() if hasattr(state, "resolve_strategy_request_id") else getattr(state, "strategy_request_id", None))
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=COMPETITOR_DISCOVERY_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            operation="competitor_discovery",
            conversation_id=conversation_id or state.thread_id,
            strategy_request_id=strat_id,
        )

        data = json.loads(raw_response)
        raw_discovered = data.get("discovered_competitors", [])
        if not isinstance(raw_discovered, list):
            return []

        # Deduplication, exclusion of client company name, validation, max 3 unique competitors
        existing_lower = {c.lower() for c in (state.business_context.competitors or [])}
        if state.competitor_research:
            existing_lower.update({r.lower() for r in state.competitor_research.researched_names})
        if company_name:
            existing_lower.add(company_name.lower())

        clean_discovered: List[str] = []
        seen: Set[str] = set()

        for item in raw_discovered:
            if not isinstance(item, str) or not item.strip():
                continue
            clean = extract_domain_or_clean_name(item)
            if not clean or len(clean.strip()) < 2:
                continue
            lower = clean.lower()
            if lower in seen or lower in existing_lower:
                continue
            if is_own_brand(clean, state.business_context):
                logger.info(f"discover_competitors_with_tavily: Excluded client's own brand '{clean}'")
                continue
            seen.add(lower)
            clean_discovered.append(clean)
            if len(clean_discovered) >= MAX_DISCOVERED_COMPETITORS:
                break

        logger.info(f"discover_competitors_with_tavily: Discovered {len(clean_discovered)} competitor(s): {clean_discovered}")
        return clean_discovered

    except (LLMServiceError, json.JSONDecodeError, Exception) as e:
        logger.error(f"discover_competitors_with_tavily failed during LLM analysis: {e}. Returning [].")
        return []


def conduct_competitor_research(
    state: MarketingAgentState,
    latest_user_answer: Optional[str] = None,
    strategy_request_id: Optional[str] = None,
) -> MarketingAgentState:
    """
    Orchestrates competitor research and discovery.
    - Does NOT depend on asking a dedicated competitor question: automatically discovers competitors
      from available business context when competitor names are missing.
    - Excludes client's own brand and generic substitutes (e.g. 'making coffee at home').
    - Discovers and researches a maximum of 3 unique, relevant competitors per strategy run.
    - Incorporates competitor names, websites, or details provided in ANY conversation turn.
    - Idempotent: avoids repeated searches if competitors have already been researched.
    - Safe, robust, and non-blocking: errors or search failures never break workflow.
    """
    try:
        strat_id = strategy_request_id or (state.resolve_strategy_request_id() if hasattr(state, "resolve_strategy_request_id") else getattr(state, "strategy_request_id", None))

        if not state.competitor_research:
            state.competitor_research = CompetitorResearchReport()

        researched_lower = {name.lower() for name in state.competitor_research.researched_names}

        # 1. Determine candidate answer or scan qa_history for competitor details
        candidate_answer = latest_user_answer
        if not candidate_answer and state.qa_history:
            for turn in reversed(state.qa_history):
                ans = str(turn.answer or "")
                if turn.requirement_id in ("competitive_landscape", "competitors") or "competitor" in str(turn.requirement_id or "").lower():
                    candidate_answer = ans
                    break
                elif any(kw in ans.lower() for kw in ["competitor", "compete", "rival", "vs", "versus"]) or "http" in ans:
                    candidate_answer = ans
                    break

        if not candidate_answer:
            comp_req = state.get_requirement_by_id("competitive_landscape")
            if comp_req and comp_req.value:
                candidate_answer = comp_req.value

        discovery_triggered = False
        generic_term = None

        if candidate_answer:
            biz_summary = f"{state.business_context.product_or_service or ''} {state.business_context.company_name or ''}".strip()
            classification = classify_competitor_response(
                user_answer=candidate_answer,
                business_context_summary=biz_summary,
                conversation_id=state.thread_id,
                strategy_request_id=strat_id,
            )
            intent = classification.get("intent")
            specific_names = classification.get("competitor_names", [])
            generic_term = classification.get("generic_category")

            # Prioritize user-provided competitor names
            if intent == "provided_specific" and specific_names:
                update_state_competitors(state, specific_names)
            elif intent in ("request_discovery", "uncertain_or_unknown"):
                discovery_triggered = True
            elif intent == "generic_category":
                discovery_triggered = True
                if not generic_term:
                    generic_term = candidate_answer.strip()

        # 2. Gather existing raw competitors from state and also check qa_history for any URLs or named entities
        raw_competitors = state.business_context.competitors or []

        # Filter candidates: technical validation and own-brand exclusion
        valid_candidates: List[str] = []
        seen: Set[str] = set()

        for comp in raw_competitors:
            if not comp or not isinstance(comp, str):
                continue
            clean = extract_domain_or_clean_name(comp)
            if not clean or len(clean.strip()) < 2:
                continue
            lower = clean.lower()
            if lower in seen or lower in researched_lower:
                continue
            if is_own_brand(clean, state.business_context):
                logger.info(f"conduct_competitor_research: Excluding client's own brand: '{clean}'")
                continue
            seen.add(lower)
            valid_candidates.append(clean)

        # 3. Automatic competitor discovery: trigger if no valid candidates exist, or if discovery requested
        should_discover = (
            (not valid_candidates and len(state.competitor_research.competitors) == 0 and not state.competitor_research.discovery_attempted)
            or (discovery_triggered and len(valid_candidates) < MAX_COMPETITORS_PER_RUN)
        )
        if should_discover:
            state.competitor_research.discovery_attempted = True
            logger.info("conduct_competitor_research: Competitor names missing or discovery requested -> Triggering automatic discovery via Tavily.")
            emit_progress("Discovering relevant competitors...", "Identifying market competitors using web intelligence.")
            discovered = discover_competitors_with_tavily(
                state=state,
                generic_category=generic_term,
                conversation_id=state.thread_id,
                strategy_request_id=strat_id,
            )
            for d in discovered:
                clean_d = extract_domain_or_clean_name(d)
                if not clean_d or len(clean_d.strip()) < 2:
                    continue
                d_lower = clean_d.lower()
                if d_lower not in seen and d_lower not in researched_lower:
                    if not is_own_brand(clean_d, state.business_context):
                        seen.add(d_lower)
                        valid_candidates.append(clean_d)
            if discovered:
                update_state_competitors(state, discovered)

        # If already researched or no candidates found even after discovery
        if not valid_candidates:
            if state.competitor_research.competitors:
                logger.info("conduct_competitor_research: All identifiable competitors already researched.")
            else:
                logger.info("conduct_competitor_research: No verifiable competitors found or discovered.")
                state.competitor_research.overall_summary = (
                    "No verifiable direct competitor web intelligence was found for this specific market. "
                    "Strategy will rely cleanly on the client's stated business context."
                )
            return state

        # 4. Limit to max allowed (3 total across the session)
        remaining_slots = max(0, MAX_COMPETITORS_PER_RUN - len(state.competitor_research.competitors))
        if remaining_slots == 0:
            logger.info("conduct_competitor_research: Maximum competitors (3) already researched.")
            return state

        to_research = valid_candidates[:remaining_slots]
        logger.info(f"conduct_competitor_research: Researching {len(to_research)} competitor(s): {to_research}")

        # Build context summary for analysis
        ctx = state.business_context
        biz_parts = []
        if ctx.company_name:
            biz_parts.append(f"Client Business: {ctx.company_name}")
        if ctx.product_or_service:
            biz_parts.append(f"Product/Service: {ctx.product_or_service}")
        if ctx.target_audience:
            biz_parts.append(f"Target Audience: {ctx.target_audience}")
        if ctx.marketing_goal:
            biz_parts.append(f"Marketing Goal: {ctx.marketing_goal}")

        tml_req = state.get_requirement_by_id("target_market_location")
        if tml_req and tml_req.status == RequirementStatus.KNOWN and tml_req.value:
            biz_parts.append(f"Geographic Target Market: {tml_req.value}")

        biz_summary = "\n".join(biz_parts) if biz_parts else "B2B / B2C business"
        industry_keyword = ctx.product_or_service or ctx.company_name or ""

        for comp_name in to_research:
            emit_progress(
                "Conducting competitor research...",
                f"Analyzing web presence and market positioning for {comp_name}."
            )

            user_notes = extract_user_stated_context_for_competitor(comp_name, candidate_answer)
            if not user_notes and state.qa_history:
                for turn in state.qa_history:
                    if comp_name.lower() in turn.answer.lower():
                        user_notes = extract_user_stated_context_for_competitor(comp_name, turn.answer)
                        if user_notes:
                            break

            # Targeted search query including positioning, reviews, and channel signals
            location_term = f" {tml_req.value}" if (tml_req and tml_req.status == RequirementStatus.KNOWN and tml_req.value) else ""
            query = f'"{comp_name}" {industry_keyword}{location_term} pricing positioning marketing channels reviews customers'.strip()

            search_results = execute_tavily_search(
                query=query,
                api_key=settings.tavily_api_key,
                max_results=5,
                timeout=15.0,
            )

            analysis_item = analyze_competitor_with_llm(
                competitor_name=comp_name,
                search_results=search_results,
                business_context_summary=biz_summary,
                user_stated_context=user_notes,
                conversation_id=state.thread_id,
                strategy_request_id=strat_id,
            )

            if analysis_item:
                state.competitor_research.competitors.append(analysis_item)
                logger.info(f"Successfully analyzed competitor '{comp_name}' with {len(analysis_item.source_urls)} source(s).")
            else:
                logger.warning(f"Could not verify or complete analysis for competitor '{comp_name}'.")

            state.competitor_research.researched_names.append(comp_name.strip().lower())

        if state.competitor_research.competitors:
            names = [c.competitor_name for c in state.competitor_research.competitors]
            limitation_note = " Note: Limited public competitor web evidence was available." if len(names) < MAX_COMPETITORS_PER_RUN else ""
            state.competitor_research.overall_summary = (
                f"Researched {len(names)} competitor(s): {', '.join(names)}.{limitation_note} "
                "Findings have been structured for positioning, differentiation, and channel planning."
            )

    except Exception as e:
        logger.error(f"conduct_competitor_research encountered unexpected error: {e}")

    return state
