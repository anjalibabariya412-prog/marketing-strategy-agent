"""
Unit and integration test suite for Competitor Web Research with Tavily and LLM.
"""

import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.core.requirements_library import load_requirements_into_state
from backend.app.schemas.agent.agent_state import MarketingAgentState
from backend.app.schemas.agent.business_context import BusinessContext
from backend.app.schemas.agent.competitor_analysis import (
    CompetitorAnalysisItem,
    CompetitorResearchReport,
)
from backend.app.schemas.agent.information_requirement import RequirementStatus
from backend.app.services.tavily_research_service import (
    is_identifiable_competitor,
    extract_user_stated_context_for_competitor,
    execute_tavily_search,
    analyze_competitor_with_llm,
    conduct_competitor_research,
    classify_competitor_response,
    discover_competitors_with_tavily,
    update_state_competitors,
)
from backend.app.agent.graph import competitor_research_node, generate_node
from backend.app.prompts.strategy_generation import build_strategy_generation_prompt


def test_1_is_identifiable_competitor():
    print("\n--- TEST 1: Identifiable vs Generic Competitor Detection ---")
    # Real identifiable brands
    assert is_identifiable_competitor("Lakmé") is True
    assert is_identifiable_competitor("Maybelline") is True
    assert is_identifiable_competitor("Swiss Beauty") is True
    assert is_identifiable_competitor("Acme Software Inc") is True
    assert is_identifiable_competitor("Zomato") is True

    # Generic or absence phrases that should NOT trigger web search
    assert is_identifiable_competitor("none") is False
    assert is_identifiable_competitor("None") is False
    assert is_identifiable_competitor("No direct competitors") is False
    assert is_identifiable_competitor("we don't have any competitors") is False
    assert is_identifiable_competitor("two local tiffin services") is False
    assert is_identifiable_competitor("other accounting firms") is False
    assert is_identifiable_competitor("local bakeries") is False
    assert is_identifiable_competitor("n/a") is False
    assert is_identifiable_competitor("") is False
    print("✓ TEST 1 PASSED: Correctly distinguishes identifiable brands from generic phrases.")


def test_2_extract_user_stated_context_for_competitor():
    print("\n--- TEST 2: Extract User-Stated Context for Competitor ---")
    user_answer = (
        "My main competitors are Lakmé, Maybelline, and Swiss Beauty. "
        "Lakmé has strong brand recognition, Maybelline has a broad product range, "
        "and Swiss Beauty competes on affordability."
    )

    ctx_lakme = extract_user_stated_context_for_competitor("Lakmé", user_answer)
    assert ctx_lakme is not None
    assert "strong brand recognition" in ctx_lakme

    ctx_maybelline = extract_user_stated_context_for_competitor("Maybelline", user_answer)
    assert ctx_maybelline is not None
    assert "broad product range" in ctx_maybelline

    ctx_swiss = extract_user_stated_context_for_competitor("Swiss Beauty", user_answer)
    assert ctx_swiss is not None
    assert "affordability" in ctx_swiss

    ctx_missing = extract_user_stated_context_for_competitor("Unknown Brand", user_answer)
    assert ctx_missing is None
    print("✓ TEST 2 PASSED: Extracted user-stated competitor observations accurately.")


def test_3_execute_tavily_search_mock_and_deduplication():
    print("\n--- TEST 3: Tavily Search Execution and Deduplication ---")
    mock_tavily_response = {
        "results": [
            {
                "title": "Lakmé Official Store",
                "url": "https://www.lakmeindia.com",
                "content": "Explore Lakmé makeup, skincare and cosmetics collections.",
                "score": 0.95,
            },
            {
                "title": "Lakmé Duplicate Link",
                "url": "https://www.lakmeindia.com",
                "content": "Duplicate entry with identical URL.",
                "score": 0.90,
            },
            {
                "title": "Lakmé Brand Review",
                "url": "https://www.beautyinsider.com/lakme",
                "content": "Analysis of Lakmé market positioning and pricing in India.",
                "score": 0.82,
            },
        ]
    }

    mock_client = MagicMock()
    mock_client.search.return_value = mock_tavily_response

    with patch("backend.app.services.tavily_research_service.TavilyClient", return_value=mock_client):
        results = execute_tavily_search("Lakmé cosmetics", api_key="test_key")
        assert len(results) == 2, f"Expected 2 deduplicated results, got {len(results)}"
        assert results[0]["url"] == "https://www.lakmeindia.com"
        assert results[1]["url"] == "https://www.beautyinsider.com/lakme"
        assert "retrieval_timestamp" in results[0]
    print("✓ TEST 3 PASSED: Tavily search deduplicated URLs and preserved metadata.")


def test_4_tavily_search_error_handling():
    print("\n--- TEST 4: Tavily Search Safe Error Handling ---")
    # 1. Missing API key returns [] safely
    results_no_key = execute_tavily_search("Any Query", api_key="")
    assert results_no_key == []

    # 2. Exception during search returns [] safely
    mock_client = MagicMock()
    mock_client.search.side_effect = Exception("Tavily API timeout 504")

    with patch("backend.app.services.tavily_research_service.TavilyClient", return_value=mock_client):
        results_err = execute_tavily_search("Any Query", api_key="valid_key")
        assert results_err == []
    print("✓ TEST 4 PASSED: Missing key and network/API errors handled gracefully without raising.")


def test_5_analyze_competitor_with_llm():
    print("\n--- TEST 5: LLM Competitor Analysis Parsing ---")
    mock_llm_json = """{
        "competitor_name": "Swiss Beauty",
        "official_website": "https://swissbeauty.in",
        "products_and_services": ["Lipsticks", "Liquid Concealers", "Eyeshadow Palettes"],
        "target_audience": "Gen-Z and young professionals looking for trendy budget cosmetics",
        "positioning": "Affordable trendy makeup with high pigment payoff",
        "pricing_information": "Budget-friendly (₹200 to ₹800 average price point)",
        "differentiators": ["High affordability", "Fast trend adoption", "Extensive shade range"],
        "marketing_channels_and_campaigns": ["Instagram Influencers", "Nykaa", "Amazon India"],
        "strengths": ["Strong Gen Z social media traction", "Competitive low pricing"],
        "weaknesses": ["Perceived lower luxury appeal compared to international brands"],
        "source_urls": ["https://swissbeauty.in", "https://beautyinsider.com/swissbeauty"],
        "confidence_score": 0.88,
        "evidence_quality": "high",
        "summary": "Swiss Beauty dominates the budget cosmetics segment with strong influencer marketing."
    }"""

    search_snippets = [
        {"title": "Swiss Beauty", "url": "https://swissbeauty.in", "content": "Trendy makeup at budget prices."}
    ]

    with patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_llm_json):
        analysis = analyze_competitor_with_llm(
            competitor_name="Swiss Beauty",
            search_results=search_snippets,
            business_context_summary="Client: Organic Cosmetics",
            user_stated_context="Competes heavily on affordability",
        )

        assert analysis is not None
        assert analysis.competitor_name == "Swiss Beauty"
        assert analysis.official_website == "https://swissbeauty.in"
        assert "Affordable trendy makeup" in analysis.positioning
        assert "Budget-friendly" in analysis.pricing_information
        assert analysis.user_stated_context == "Competes heavily on affordability"
        assert len(analysis.strengths) == 2
        assert len(analysis.weaknesses) == 1
        assert "https://swissbeauty.in" in analysis.source_urls
    print("✓ TEST 5 PASSED: Structured analysis item constructed with user-context separation.")


def test_6_conduct_competitor_research_orchestrator():
    print("\n--- TEST 6: Conduct Competitor Research Orchestrator & Idempotency ---")
    state = MarketingAgentState()
    state.business_context = BusinessContext(
        company_name="GlowNatural Skincare",
        product_or_service="Organic vegan skincare serum",
        marketing_goal="Increase D2C website sales",
        target_audience="Urban women 22-38",
        competitors=["Lakmé", "Swiss Beauty", "two local tiffin services"]
    )

    mock_search_results = [
        {"title": "Comp Brand", "url": "https://example.com", "content": "Leading cosmetic brand"}
    ]

    mock_analysis_json = """{
        "competitor_name": "Lakmé",
        "official_website": "https://lakmeindia.com",
        "products_and_services": ["Foundation", "Skincare"],
        "positioning": "Established legacy cosmetics",
        "pricing_information": "Mid to premium",
        "differentiators": ["High trust"],
        "strengths": ["High brand awareness"],
        "weaknesses": ["Less focused on 100% vegan formulations"],
        "source_urls": ["https://lakmeindia.com"],
        "confidence_score": 0.85,
        "evidence_quality": "high",
        "summary": "Dominant market incumbent."
    }"""

    with patch("backend.app.services.tavily_research_service.execute_tavily_search", return_value=mock_search_results) as mock_search, \
         patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_analysis_json):

        conduct_competitor_research(state, latest_user_answer="Lakmé has strong brand recognition, Swiss Beauty is affordable, and two local tiffin services are nearby.")

        # Verify only identifiable competitors researched (generic 'two local tiffin services' was skipped)
        assert state.competitor_research is not None
        researched = state.competitor_research.researched_names
        assert "lakmé" in researched
        assert "swiss beauty" in researched
        assert "two local tiffin services" not in researched
        assert mock_search.call_count == 2

        # Verify idempotency: calling a second time performs ZERO additional searches
        mock_search.reset_mock()
        conduct_competitor_research(state)
        assert mock_search.call_count == 0
        print("✓ TEST 6 PASSED: Orchestrator researched identifiable competitors and demonstrated idempotency.")


def test_7_no_competitors_does_not_call_tavily():
    print("\n--- TEST 7: No Competitors Stated ---")
    state = MarketingAgentState()
    state.business_context = BusinessContext(
        company_name="Solo Bakery",
        product_or_service="Handmade sourdough",
        competitors=[]
    )

    with patch("backend.app.services.tavily_research_service.execute_tavily_search") as mock_search:
        conduct_competitor_research(state)
        assert mock_search.call_count == 0
        assert state.competitor_research is None
    print("✓ TEST 7 PASSED: Cleanly skipped search when no competitors provided.")


def test_8_competitor_research_node_in_graph():
    print("\n--- TEST 8: LangGraph competitor_research_node ---")
    state = MarketingAgentState()
    state.business_context.competitors = ["Maybelline"]

    mock_analysis_json = """{
        "competitor_name": "Maybelline",
        "official_website": "https://maybelline.com",
        "positioning": "Global accessible beauty",
        "strengths": ["Mass market distribution"],
        "weaknesses": ["Synthetic ingredients"],
        "source_urls": ["https://maybelline.com"],
        "summary": "Global cosmetic giant."
    }"""

    with patch("backend.app.services.tavily_research_service.execute_tavily_search", return_value=[{"title": "M", "url": "https://maybelline.com", "content": "Beauty"}]), \
         patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_analysis_json):

        res_state = competitor_research_node(state)
        assert res_state.competitor_research is not None
        assert len(res_state.competitor_research.competitors) == 1
        assert res_state.competitor_research.competitors[0].competitor_name == "Maybelline"
    print("✓ TEST 8 PASSED: LangGraph competitor_research_node executed successfully.")


def test_9_strategy_generation_prompt_incorporates_research():
    print("\n--- TEST 9: Strategy Prompt Builder Incorporates Verified Competitor Research ---")
    state = MarketingAgentState()
    state.business_context = BusinessContext(
        company_name="Aura Natural",
        product_or_service="Clean organic lipsticks",
        marketing_goal="Generate 500 online orders",
        target_audience="Conscious consumers",
        competitors=["Lakmé"]
    )
    state.competitor_research = CompetitorResearchReport(
        competitors=[
            CompetitorAnalysisItem(
                competitor_name="Lakmé",
                user_stated_context="Known for huge retail reach",
                official_website="https://lakmeindia.com",
                positioning="Mainstream conventional beauty",
                products_and_services=["Lipsticks", "Kajal", "Foundations"],
                strengths=["Extensive offline salon and retail footprint"],
                weaknesses=["Lack of 100% certified organic clean formulations"],
                source_urls=["https://lakmeindia.com/products"],
                summary="Market leader with synthetic formulas."
            )
        ]
    )

    prompt = build_strategy_generation_prompt(state)
    assert "VERIFIED COMPETITOR WEB RESEARCH (VIA TAVILY):" in prompt
    assert "Competitor: Lakmé" in prompt
    assert "User-Reported Note: \"Known for huge retail reach\"" in prompt
    assert "Website: https://lakmeindia.com" in prompt
    assert "Evidence-Supported Weaknesses / Gaps: Lack of 100% certified organic clean formulations" in prompt
    assert "COMPETITOR RESEARCH STRATEGY DIRECTIVE:" in prompt
    assert "Ground differentiation in verified competitor findings" in prompt
    print("✓ TEST 9 PASSED: Strategy prompt accurately includes structured competitor findings and differentiation directives.")


def test_10_semantic_intent_detection_all_categories_and_variations():
    print("\n--- TEST 10: Semantic Intent Detection for All 4 Categories & Variations ---")
    
    # 1. Category 1: Specific competitor names
    mock_specific = '{"intent": "provided_specific", "competitor_names": ["Lakmé", "Maybelline", "Swiss Beauty"], "generic_category": null, "reasoning": "User named concrete brands"}'
    with patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_specific):
        res1 = classify_competitor_response("Lakmé, Maybelline, and Swiss Beauty")
        assert res1["intent"] == "provided_specific"
        assert res1["competitor_names"] == ["Lakmé", "Maybelline", "Swiss Beauty"]
        assert res1["generic_category"] is None

    # 2. Category 2: Request discovery
    mock_discovery = '{"intent": "request_discovery", "competitor_names": [], "generic_category": null, "reasoning": "User asked agent to identify rivals"}'
    with patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_discovery):
        res2 = classify_competitor_response("Can you discover who my competitors are?")
        assert res2["intent"] == "request_discovery"
        assert res2["competitor_names"] == []

    # Semantic variation for request discovery
    mock_discovery_var = '{"intent": "request_discovery", "competitor_names": [], "generic_category": null, "reasoning": "User requesting competitor discovery"}'
    with patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_discovery_var):
        res2_var = classify_competitor_response("Could you please research and suggest the top rivals in our niche?")
        assert res2_var["intent"] == "request_discovery"

    # 3. Category 3: Uncertain or Unknown
    mock_unknown = '{"intent": "uncertain_or_unknown", "competitor_names": [], "generic_category": null, "reasoning": "User expressed lack of knowledge"}'
    with patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_unknown):
        res3 = classify_competitor_response("I don't know who my competitors are, we just started.")
        assert res3["intent"] == "uncertain_or_unknown"
        assert res3["competitor_names"] == []

    # Semantic variation for uncertainty
    mock_unknown_var = '{"intent": "uncertain_or_unknown", "competitor_names": [], "generic_category": null, "reasoning": "User has no idea"}'
    with patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_unknown_var):
        res3_var = classify_competitor_response("Honestly not sure yet, haven't analyzed the market.")
        assert res3_var["intent"] == "uncertain_or_unknown"

    # 4. Category 4: Generic Category
    mock_generic = '{"intent": "generic_category", "competitor_names": [], "generic_category": "coffee brands", "reasoning": "User gave broad category"}'
    with patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_generic):
        res4 = classify_competitor_response("coffee brands")
        assert res4["intent"] == "generic_category"
        assert res4["generic_category"] == "coffee brands"
        assert res4["competitor_names"] == []

    # Semantic variation for generic category
    mock_generic_var = '{"intent": "generic_category", "competitor_names": [], "generic_category": "specialty coffee roasters", "reasoning": "Industry segment"}'
    with patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_generic_var):
        res4_var = classify_competitor_response("specialty coffee roasters")
        assert res4_var["intent"] == "generic_category"
        assert res4_var["generic_category"] == "specialty coffee roasters"

    # 5. Prioritization rule (Requirement 5): Specific names override generic descriptors
    mock_mixed = '{"intent": "provided_specific", "competitor_names": ["Blue Tokai"], "generic_category": "coffee brands", "reasoning": "Provided specific brand along with generic term"}'
    with patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_mixed):
        res5 = classify_competitor_response("Blue Tokai and other coffee brands")
        assert res5["intent"] == "provided_specific"
        assert res5["competitor_names"] == ["Blue Tokai"]

    # 6. Fallback heuristic resilience when LLM raises exception
    with patch("backend.app.services.tavily_research_service.get_llm_response", side_effect=Exception("API Timeout")):
        fb1 = classify_competitor_response("I don't know who my competitors are")
        assert fb1["intent"] == "uncertain_or_unknown"

        fb2 = classify_competitor_response("Please discover competitors for me")
        assert fb2["intent"] == "request_discovery"

        fb3 = classify_competitor_response("coffee brands")
        assert fb3["intent"] == "generic_category"

    print("✓ TEST 10 PASSED: Semantic intent classification handles all 4 categories, semantic variations, prioritization, and fallback heuristics.")


def test_11_discover_competitors_with_tavily_max_3_and_grounding():
    print("\n--- TEST 11: Competitor Discovery via Tavily (Max 3, Client Exclusion, Grounding) ---")
    state = MarketingAgentState()
    state.business_context = BusinessContext(
        company_name="BeanCraft Coffee",
        product_or_service="Specialty cold brew coffee bottles",
        target_audience="Urban professionals and coffee connoisseurs",
        marketing_goal="Scale direct-to-consumer online orders",
    )
    load_requirements_into_state(state)

    # Establish target market location
    tml = state.get_requirement_by_id("target_market_location")
    if tml:
        tml.status = RequirementStatus.KNOWN
        tml.value = "India"

    mock_search_results = [
        {"title": "Top Cold Brew Brands", "url": "https://example.com/coffee", "content": "Blue Tokai, Sleepy Owl, and Third Wave Coffee lead the market."}
    ]

    # LLM returns 4 candidates including client's own company name "BeanCraft Coffee"
    mock_discovery_json = """{
        "discovered_competitors": ["BeanCraft Coffee", "Blue Tokai", "Sleepy Owl", "Third Wave Coffee", "Rage Coffee"],
        "reasoning": "Top competing specialty coffee brands in India."
    }"""

    with patch("backend.app.services.tavily_research_service.execute_tavily_search", return_value=mock_search_results) as mock_search, \
         patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_discovery_json):

        discovered = discover_competitors_with_tavily(state, generic_category="coffee brands")

        # 1. Enforces maximum 3 unique competitors (Requirement 3)
        assert len(discovered) == 3, f"Expected exactly 3 competitors, got {len(discovered)}"

        # 2. Excludes client's own company name
        assert "BeanCraft Coffee" not in discovered
        assert "Blue Tokai" in discovered
        assert "Sleepy Owl" in discovered
        assert "Third Wave Coffee" in discovered

        # 3. Grounded in established target market without unestablished 'nearby' restriction
        called_query = mock_search.call_args[1]["query"]
        assert "India" in called_query
        assert "nearby" not in called_query
        assert "local" not in called_query

    # Test returning fewer if credible evidence is limited (e.g. only 2 credible competitors found)
    mock_fewer_json = """{
        "discovered_competitors": ["Blue Tokai", "Sleepy Owl"],
        "reasoning": "Only two direct specialty cold brew rivals found."
    }"""
    with patch("backend.app.services.tavily_research_service.execute_tavily_search", return_value=mock_search_results), \
         patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_fewer_json):
        discovered_fewer = discover_competitors_with_tavily(state)
        assert len(discovered_fewer) == 2
        assert discovered_fewer == ["Blue Tokai", "Sleepy Owl"]

    print("✓ TEST 11 PASSED: Competitor discovery enforces max 3 limit, client exclusion, credible evidence pruning, and geographic relevance.")


def test_12_orchestrator_category_1_specific_competitors_prioritized():
    print("\n--- TEST 12: Category 1 (Specific Names Provided) Prioritized Without Discovery ---")
    state = MarketingAgentState()
    state.business_context = BusinessContext(
        company_name="ChaiCulture",
        product_or_service="Artisanal spiced tea blends",
        competitors=[]
    )

    mock_intent_json = '{"intent": "provided_specific", "competitor_names": ["Vahdam Teas", "Chaayos"], "generic_category": null}'
    mock_search_results = [{"title": "Tea Brand", "url": "https://example.com", "content": "Leading tea brand"}]
    mock_analysis_json = """{
        "competitor_name": "Vahdam Teas",
        "official_website": "https://vahdam.com",
        "positioning": "Global export quality fresh Indian tea",
        "strengths": ["Strong US/Europe export presence"],
        "weaknesses": ["Higher price point"],
        "source_urls": ["https://vahdam.com"],
        "summary": "Global premium tea brand."
    }"""

    with patch("backend.app.services.tavily_research_service.classify_competitor_response", return_value={"intent": "provided_specific", "competitor_names": ["Vahdam Teas", "Chaayos"]}), \
         patch("backend.app.services.tavily_research_service.discover_competitors_with_tavily") as mock_discover, \
         patch("backend.app.services.tavily_research_service.execute_tavily_search", return_value=mock_search_results), \
         patch("backend.app.services.tavily_research_service.analyze_competitor_with_llm", return_value=CompetitorAnalysisItem(competitor_name="Vahdam Teas", official_website="https://vahdam.com")):

        conduct_competitor_research(state, latest_user_answer="We compete with Vahdam Teas and Chaayos.")

        # User-provided names are prioritized and researched
        assert "Vahdam Teas" in state.business_context.competitors
        assert "Chaayos" in state.business_context.competitors
        # Auto-discovery was NOT called because user explicitly provided names (Requirement 5)
        assert mock_discover.call_count == 0
        assert len(state.competitor_research.researched_names) >= 2

    print("✓ TEST 12 PASSED: User-provided competitor names prioritized and researched directly without discovery.")


def test_13_orchestrator_category_2_request_discovery():
    print("\n--- TEST 13: Category 2 (User Asks Agent to Discover Competitors) ---")
    state = MarketingAgentState()
    state.business_context = BusinessContext(
        company_name="SaaSMetrics",
        product_or_service="B2B Subscription Analytics Tool",
        marketing_goal="Acquire 100 SaaS customers",
        competitors=[]
    )

    mock_discovered = ["ProfitWell", "Baremetrics", "ChartMogul"]

    with patch("backend.app.services.tavily_research_service.classify_competitor_response", return_value={"intent": "request_discovery", "competitor_names": [], "generic_category": None}), \
         patch("backend.app.services.tavily_research_service.discover_competitors_with_tavily", return_value=mock_discovered) as mock_discover, \
         patch("backend.app.services.tavily_research_service.execute_tavily_search", return_value=[{"title": "Tool", "url": "https://example.com", "content": "Analytics"}]), \
         patch("backend.app.services.tavily_research_service.analyze_competitor_with_llm") as mock_analyze:

        mock_analyze.side_effect = lambda competitor_name, **kwargs: CompetitorAnalysisItem(
            competitor_name=competitor_name,
            official_website=f"https://{competitor_name.lower()}.com",
            positioning="SaaS revenue reporting",
            source_urls=[f"https://{competitor_name.lower()}.com"],
            summary=f"Analysis of {competitor_name}."
        )

        conduct_competitor_research(state, latest_user_answer="Can you discover who my competitors are?")

        # Discovery called and competitors added to state
        assert mock_discover.call_count == 1
        assert state.business_context.competitors == ["ProfitWell", "Baremetrics", "ChartMogul"]
        assert len(state.competitor_research.competitors) == 3
        assert mock_analyze.call_count == 3
        assert "profitwell" in state.competitor_research.researched_names

    print("✓ TEST 13 PASSED: Agent successfully discovered competitors upon user request and conducted full web research.")


def test_14_orchestrator_category_3_uncertain_or_unknown():
    print("\n--- TEST 14: Category 3 (User Uncertainty / Unknown Competitors) ---")
    state = MarketingAgentState()
    state.business_context = BusinessContext(
        company_name="NutriBites",
        product_or_service="High-protein gluten-free snack bars",
        marketing_goal="Drive D2C sales",
        competitors=[]
    )

    mock_discovered = ["Yoga Bar", "The Whole Truth", "Max Protein"]

    with patch("backend.app.services.tavily_research_service.classify_competitor_response", return_value={"intent": "uncertain_or_unknown", "competitor_names": [], "generic_category": None}), \
         patch("backend.app.services.tavily_research_service.discover_competitors_with_tavily", return_value=mock_discovered) as mock_discover, \
         patch("backend.app.services.tavily_research_service.execute_tavily_search", return_value=[{"title": "Bar", "url": "https://example.com", "content": "Snacks"}]), \
         patch("backend.app.services.tavily_research_service.analyze_competitor_with_llm") as mock_analyze:

        mock_analyze.side_effect = lambda competitor_name, **kwargs: CompetitorAnalysisItem(
            competitor_name=competitor_name,
            official_website=f"https://{competitor_name.lower().replace(' ', '')}.com",
            source_urls=[f"https://{competitor_name.lower().replace(' ', '')}.com"],
            summary=f"Market competitor {competitor_name}."
        )

        conduct_competitor_research(state, latest_user_answer="I don't know who my competitors are, we just launched.")

        # Discovered automatically without asking user again
        assert mock_discover.call_count == 1
        assert "Yoga Bar" in state.business_context.competitors
        assert "The Whole Truth" in state.business_context.competitors
        assert len(state.competitor_research.competitors) == 3

    print("✓ TEST 14 PASSED: Agent automatically discovered competitors when user expressed uncertainty.")


def test_15_orchestrator_category_4_generic_answer():
    print("\n--- TEST 15: Category 4 (Generic Answer Interpreted via Business Context) ---")
    state = MarketingAgentState()
    state.business_context = BusinessContext(
        company_name="Aura ColdBrew",
        product_or_service="Artisanal nitro cold brew cans",
        target_audience="Urban millennials",
        competitors=[]
    )

    mock_discovered = ["Sleepy Owl", "Blue Tokai", "Third Wave Coffee"]

    with patch("backend.app.services.tavily_research_service.classify_competitor_response", return_value={"intent": "generic_category", "competitor_names": [], "generic_category": "coffee brands"}), \
         patch("backend.app.services.tavily_research_service.discover_competitors_with_tavily", return_value=mock_discovered) as mock_discover, \
         patch("backend.app.services.tavily_research_service.execute_tavily_search", return_value=[{"title": "Cold Brew", "url": "https://example.com", "content": "Coffee"}]), \
         patch("backend.app.services.tavily_research_service.analyze_competitor_with_llm") as mock_analyze:

        mock_analyze.side_effect = lambda competitor_name, **kwargs: CompetitorAnalysisItem(
            competitor_name=competitor_name,
            official_website=f"https://{competitor_name.lower().replace(' ', '')}.com",
            source_urls=[f"https://{competitor_name.lower().replace(' ', '')}.com"],
            summary=f"Analysis of {competitor_name}."
        )

        conduct_competitor_research(state, latest_user_answer="coffee brands")

        # Discovered using generic category and business context
        assert mock_discover.call_count == 1
        call_kwargs = mock_discover.call_args[1]
        assert call_kwargs.get("generic_category") == "coffee brands"
        assert state.business_context.competitors == ["Sleepy Owl", "Blue Tokai", "Third Wave Coffee"]
        assert len(state.competitor_research.competitors) == 3

    print("✓ TEST 15 PASSED: Generic answer 'coffee brands' interpreted and concrete competitors identified and researched.")


def test_16_deduplication_and_research_failure_handling():
    print("\n--- TEST 16: Deduplication & Robust Error Handling ---")
    state = MarketingAgentState()
    state.business_context = BusinessContext(
        company_name="Sourdough Co",
        product_or_service="Artisan bread",
        competitors=["The Bread Box"]
    )
    state.competitor_research = CompetitorResearchReport(researched_names=["the bread box"])

    # 1. Deduplication: newly discovered list includes duplicate 'the bread box'
    mock_discovered = ["the bread box", "Crust & Co"]

    with patch("backend.app.services.tavily_research_service.classify_competitor_response", return_value={"intent": "request_discovery", "competitor_names": [], "generic_category": None}), \
         patch("backend.app.services.tavily_research_service.discover_competitors_with_tavily", return_value=mock_discovered), \
         patch("backend.app.services.tavily_research_service.execute_tavily_search", return_value=[]):

        conduct_competitor_research(state, latest_user_answer="Find my competitors")
        # 'The Bread Box' was already in researched_names, so only Crust & Co was newly added
        assert "Crust & Co" in state.business_context.competitors
        # Only 1 candidate researched
        assert len([c for c in state.business_context.competitors if c.lower() == "the bread box"]) == 1

    # 2. Tavily search network failure handling
    with patch("backend.app.services.tavily_research_service.execute_tavily_search", side_effect=Exception("Connection reset 502")):
        # Should not raise exception
        safe_state = conduct_competitor_research(state)
        assert safe_state is not None

    # 3. LLM analysis failure handling: fallback CompetitorAnalysisItem with source URLs
    search_snippets = [{"title": "Crust", "url": "https://crustco.com", "content": "Bakery"}]
    with patch("backend.app.services.tavily_research_service.get_llm_response", side_effect=Exception("OpenAI rate limit 429")):
        fb_item = analyze_competitor_with_llm(
            competitor_name="Crust & Co",
            search_results=search_snippets,
            business_context_summary="Bakery in London",
        )
        assert fb_item is not None
        assert fb_item.competitor_name == "Crust & Co"
        assert "https://crustco.com" in fb_item.source_urls
        assert fb_item.evidence_quality == "incomplete"

    print("✓ TEST 16 PASSED: Deduplication and graceful error recovery verified under network and LLM failures.")


def test_17_strategy_integration_with_discovered_competitors():
    print("\n--- TEST 17: Final Marketing Strategy Incorporates Discovered Competitors ---")
    state = MarketingAgentState()
    state.business_context = BusinessContext(
        company_name="PureAura Skincare",
        product_or_service="Vegan peptide face serum",
        marketing_goal="Reach 1000 orders/month",
        competitors=["The Ordinary", "Minimalist"]
    )
    state.competitor_research = CompetitorResearchReport(
        competitors=[
            CompetitorAnalysisItem(
                competitor_name="Minimalist",
                official_website="https://beminimalist.co",
                positioning="Active ingredient transparency at affordable pricing",
                products_and_services=["Niacinamide Serum", "Salicylic Acid Cleaners"],
                strengths=["High consumer trust in ingredient percentages"],
                weaknesses=["Minimal focus on plant-based / vegan botanical actives"],
                source_urls=["https://beminimalist.co", "https://beautyreview.com/minimalist"],
                summary="Major ingredient-led skincare competitor."
            )
        ]
    )

    prompt = build_strategy_generation_prompt(state)
    assert "VERIFIED COMPETITOR WEB RESEARCH (VIA TAVILY):" in prompt
    assert "Competitor: Minimalist" in prompt
    assert "Website: https://beminimalist.co" in prompt
    assert "Active ingredient transparency at affordable pricing" in prompt
    assert "Minimal focus on plant-based / vegan botanical actives" in prompt
    assert "Supporting Sources: https://beminimalist.co, https://beautyreview.com/minimalist" in prompt
    assert "Ground differentiation in verified competitor findings" in prompt

    print("✓ TEST 17 PASSED: Discovered competitor research and verified source URLs successfully incorporated into final strategy prompt.")


def run_all_tests():
    print("==================================================")
    print("RUNNING COMPETITOR WEB RESEARCH TEST SUITE")
    print("==================================================")
    test_1_is_identifiable_competitor()
    test_2_extract_user_stated_context_for_competitor()
    test_3_execute_tavily_search_mock_and_deduplication()
    test_4_tavily_search_error_handling()
    test_5_analyze_competitor_with_llm()
    test_6_conduct_competitor_research_orchestrator()
    test_7_no_competitors_does_not_call_tavily()
    test_8_competitor_research_node_in_graph()
    test_9_strategy_generation_prompt_incorporates_research()
    test_10_semantic_intent_detection_all_categories_and_variations()
    test_11_discover_competitors_with_tavily_max_3_and_grounding()
    test_12_orchestrator_category_1_specific_competitors_prioritized()
    test_13_orchestrator_category_2_request_discovery()
    test_14_orchestrator_category_3_uncertain_or_unknown()
    test_15_orchestrator_category_4_generic_answer()
    test_16_deduplication_and_research_failure_handling()
    test_17_strategy_integration_with_discovered_competitors()
    print("==================================================")
    print("ALL COMPETITOR WEB RESEARCH TESTS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    run_all_tests()
