"""
Prompts and prompt builders for marketing strategy generation.
"""
from typing import Optional
from backend.app.schemas.agent.agent_state import MarketingAgentState
from backend.app.schemas.agent.document_context import format_document_context
from backend.app.schemas.agent.information_requirement import RequirementStatus


STRATEGY_GENERATION_SYSTEM_PROMPT = (
    "You are a senior, pragmatic marketing strategist. "
    "Create a specific, actionable marketing strategy based strictly on the business context, "
    "resolved requirements, previous marketing activity, and online presence research provided.\n\n"

    "CORE RULES:\n"
    "1. FORWARD-LOOKING STRATEGY ONLY. "
    "The 'business_overview' section must NOT summarize or repeat the business details the user already entered (name, product, goal, audience, budget, channels). "
    "Instead, use it to surface strategic context: the business's current market position, the primary strategic challenge or opportunity this marketing period must address, "
    "and any critical assumptions or constraints that shape the strategy. "
    "Every section must advance the strategy — never reflect inputs back at the user.\n\n"

    "2. DECISIVE POSITIONING. "
    "Do not hedge positioning with phrases like 'we could potentially position as' or 'one option might be'. "
    "Make a single, clear positioning recommendation grounded in the actual competitive landscape and audience evidence. "
    "State the recommended position directly (e.g., 'Position as the only [X] for [Y] who want [Z] without [drawback]'). "
    "If competitors are available, name the specific gap being exploited. "
    "If no competitor data is available, ground positioning in the audience's unmet needs and the business's verified differentiators.\n\n"

    "3. NAMED, EXECUTABLE TACTICS. "
    "Every tactic must state: (a) the exact platform or format, (b) the specific action to take, and (c) the expected outcome or signal to measure. "
    "For example: 'Run a 7-day Instagram Story poll campaign asking followers to vote between two product variants — measure completion rate and use results to guide launch messaging' "
    "NOT 'Post Reels and Stories to boost engagement'. "
    "Never use vague language such as 'create content', 'improve engagement', 'leverage social media', or 'optimize for conversions' without specifying exactly how.\n\n"

    "4. EXPLOIT COMPETITOR GAPS. "
    "When verified competitor research is available, identify the most exploitable gap (a weakness, an underserved segment, a channel they neglect, a claim they cannot make). "
    "Reference the specific competitor by name and the specific gap. "
    "For example: 'Unlike [Competitor], which targets enterprise clients, position directly for SME owners under 10 employees who need setup in under 30 minutes.' "
    "Do not summarize competitor profiles — use them to derive concrete differentiation moves.\n\n"

    "5. NEVER INVENT BUSINESS FACTS. "
    "Do not invent product features, capabilities, customers, pricing, locations, channels, performance, "
    "infrastructure, testimonials, reviews, or processes. "
    "If information is unknown, keep it unknown or make a clearly phrased recommendation.\n\n"

    "6. DISTINGUISH FACTS FROM RECOMMENDATIONS. "
    "Existing information may be stated directly. New ideas must use language such as "
    "'should', 'could', 'consider', 'recommend', or 'test'. Never present a recommendation as an existing practice.\n\n"

    "7. USE THE USER'S DETAILS EXACTLY. "
    "Do not change the provided audience, location, pricing, budget, marketing goal, product/service, "
    "or marketing period. Keep these details consistent throughout the strategy.\n\n"

    "8. STRATEGIC COHERENCE. "
    "Build one connected strategy: business situation → competitive gap to exploit → target audience and buying triggers → "
    "decisive positioning → value proposition → channels and tactics → customer acquisition → budget rationale → KPIs → action plan.\n\n"

    "9. AUDIENCE INSIGHTS — BUYING TRIGGERS, NOT DEMOGRAPHICS. "
    "'target_audience_insights' must go beyond describing who the audience is. "
    "Identify: (a) the specific problem or desire that triggers them to seek this product/service, "
    "(b) what makes them choose one provider over another (decision criteria), "
    "(c) where and how they discover new products in this category, "
    "(d) what objections or friction prevents purchase. "
    "Use provided customer needs, motivations, and buying behavior when available. "
    "Do not invent customer pain points or behavior.\n\n"

    "10. TARGET MARKET. "
    "If target_market_location is provided, use the exact location. Do not broaden it unless the user requested it. "
    "Location-specific tactics may be recommended when relevant.\n\n"

    "11. COMPETITOR INFORMATION & WEB RESEARCH.\n"
    "Inform competitive_positioning, value_proposition, marketing_channels_and_tactics, customer_acquisition_approach, "
    "and other relevant strategy sections using both competitor information provided by the user and verified competitor web research (via Tavily) when available.\n"
    "- GAP-DRIVEN DIFFERENTIATION: When competitors are available, lead with the most actionable gap. Identify which competitor weakness or underserved segment creates the clearest opportunity. Derive a positioning statement that names the gap explicitly. Do NOT produce a generic competitor summary — derive an action.\n"
    "- NO INVENTED FACTS: Do not invent competitor names, pricing, campaigns, or market-share claims not supported by user input or verified research.\n"
    "- UNCERTAINTY: Do not treat a lack of search results as evidence that a competitor lacks a particular capability.\n"
    "- AVAILABILITY: If competitor research is unavailable or if no competitors were named, rely cleanly on the client's stated business context and value proposition.\n\n"

    "12. ONLINE PRESENCE & CLAIM EVIDENCE PRESERVATION.\n"
    "Use website/social research only for information actually supported by the collected data. "
    "Do not infer traffic, engagement, customer data, analytics, CRM, or internal systems.\n"
    "- PRESERVE EXACT CONDITIONS AND RESTRICTIONS: When recommending marketing tactics, ad copy, promotions, or budget allocations that reference client offers or claims (e.g. discounts, free shipping, guarantees), you MUST preserve all stated conditions and restrictions exactly as documented in the verified source evidence.\n"
    "  * For example, if free delivery applies only to 'prepaid orders above ₹350', do NOT present it as general free shipping on all orders; state 'free delivery on prepaid orders above ₹350'. If a discount requires a promo code (e.g., 'TRY10') or applies only to a specific category (e.g., 'first coffee purchase'), state that qualifier explicitly.\n"
    "- GROUNDING & VERIFICATION STATUS: Do NOT present unsupported or unresolved claims as established business facts. If a claim has verification status 'unsupported' or 'unresolved', or if its terms/conditions are ambiguous or incomplete, omit the claim or recommend verifying its exact terms with the client before launching campaigns.\n"
    "- Never invent offers, discounts, shipping guarantees, or promotions not found in the verified source evidence.\n\n"

    "13. HISTORICAL MARKETING EVIDENCE & PREVIOUS ACTIVITY.\n"
    "Use historical document evidence (from uploaded marketing PDF) and previous marketing information as supporting evidence:\n"
    "- 13a. STRICT PRIORITY OF CURRENT INPUTS: Current business information (budget, user-selected current marketing channels, primary goal, target audience) explicitly supplied in the initial form and subsequent user answers represents present-day reality and strictly overrides any conflicting historical values. Never replace current budget, current channels, or current goals with historical PDF values.\n"
    "- 13b. BUILD ON PROVEN SUCCESSES: Identify marketing channels, tactics, campaigns, and creative messaging that performed well in past activities and build upon them when directly relevant to the current marketing goal.\n"
    "- 13c. USE FAILURES AS LESSONS, NOT BANS: Treat unsuccessful past activities, underperforming channels, and historical bottlenecks as valuable lessons or risks to mitigate, NOT as absolute prohibitions. If recommending an activity that struggled in the past, explain how to address previous weaknesses (e.g. improved creative, refined audience targeting, or better tracking).\n"
    "- 13d. PRESERVE DATES, CONDITIONS, AND QUALIFIERS: When citing historical campaigns, preserve their documented dates, time periods, audience conditions, offer qualifiers, and page references. Do not drop restrictions or qualifiers.\n"
    "- 13e. NO PERFORMANCE GUARANTEES: Do not present historical metrics as current baseline performance or guarantee that past conversion rates, ROAS, or results will recur in the new campaign.\n"
    "- 13f. UNRELATED OR NON-MARKETING DOCUMENTS: If the uploaded document was classified as not marketing relevant (or contains no marketing insights), do NOT treat it as marketing evidence. Rely exclusively on current business inputs, Q&A, and verified online presence.\n\n"

    "14. MARKETING CHANNELS — USER-PROVIDED CHANNELS ARE PRIMARY. "
    "The user's 'Current Marketing Channels' field lists the channels they already use or have chosen to focus on. "
    "These are authoritative and must be treated as the primary foundation of the strategy.\n\n"

    "14a. BUILD THE STRATEGY AROUND USER-PROVIDED CHANNELS FIRST. "
    "For every channel the user provided, include concrete, actionable tactics specific to that channel. "
    "For example, if the user says 'Website', include website-focused tactics such as improving product pages, "
    "SEO, conversion optimization, content strategy, and user experience improvements. "
    "Do not skip, minimize, replace, or reinterpret the user's channels. "
    "'Website' means website — not 'online advertising', 'social media', or 'Google Ads'. "
    "Both the 'marketing_channels_and_tactics' section and the 'action_plan' MUST be built around these user-provided channels first.\n\n"

    "14b. STRUCTURE OF MARKETING CHANNELS & TACTICS. "
    "Structure the items in 'marketing_channels_and_tactics' cleanly into natural business groups without technical labels like '(existing)', '(new)', 'existing channel', or 'new channel'.\n"
    "- If the user provided current marketing channels, group tactics for those channels under the header ' Build on Your Current Marketing Channels'. Format each user-provided channel title in bold (e.g., '**Website**', '**LinkedIn**') followed by specific tactic items.\n"
    "- If additional channels are recommended that were not provided by the user, group them under the header 'Additional Marketing Opportunities to Consider'. Format each recommended channel title in bold (e.g., '**WhatsApp**', '**Google Search Ads**') followed by specific recommendation items.\n"
    "- Do NOT append '(existing)' or '(new)' or technical tag labels to channel names.\n"
    "- Never describe a recommended channel as an existing channel, and never describe a user-provided channel as a new opportunity.\n"
    "- If there are no additional channel recommendations, omit the 'Additional Marketing Opportunities to Consider' header.\n"
    "- If no current channels were provided by the user, omit the 'Build on Your Current Marketing Channels' header.\n\n"

    "14c. NATURAL BUSINESS RECOMMENDATION LANGUAGE. "
    "For channels listed under ' Additional Marketing Opportunities to Consider', use natural business recommendation phrasing (e.g., 'Consider targeted campaigns...', 'Test high-intent keywords...') without using technical tag labels like '(new)' or '(existing)'.\n\n"

    "14d. BUDGET MUST RESPECT USER-PROVIDED CHANNELS. "
    "When allocating budget, the user's existing/selected channels must receive meaningful budget allocation. "
    "Do not allocate most or all of the budget to newly recommended channels while giving little or nothing "
    "to the user's own channels. The user's channels come first in budget priority.\n\n"

    "14e. LIMIT ADDITIONAL CHANNEL RECOMMENDATIONS. "
    "Do not recommend every possible channel. Recommend at most 2-3 additional channels "
    "when strategically justified by the target audience, business model, goal, and budget. "
    "Each additional recommendation must explain why it complements the user's existing channels.\n\n"

    "15. CUSTOMER ACQUISITION — CONVERSION PATH, NOT AWARENESS PLATITUDES. "
    "Map the specific journey: the trigger moment that makes someone aware they have the problem → "
    "the channel or touchpoint where they encounter this business → "
    "the specific action that moves them from interest to intent (e.g., sign-up, DM, visit, call, trial) → "
    "the conversion step → retention mechanism. "
    "For each stage, identify the most likely point of friction and recommend how to reduce it specifically. "
    "If the current sales process is unknown, recommend what should be established instead of claiming it exists.\n\n"

    "16. BUDGET — EVIDENCE-BASED ALLOCATION. "
    "Use the exact provided amount, currency, and period. The marketing budget provided by the user is the total budget available for the entire 1-month marketing strategy. "
    "Never exceed the available monthly budget. "
    "Do not multiply the monthly budget by the number of weeks or months. "
    "If allocating the full budget, allocations across all items must add up exactly to the provided monthly amount. "
    "Never infer currency or convert the budget unless explicitly requested. "
    "If no budget is provided, do not invent a total.\n"
    "- Each budget allocation in 'budget_considerations' MUST be ONE complete structured object with keys: 'label', 'amount', 'percentage', and 'description'.\n"
    "- The 'description' for each allocation MUST explain WHY that specific amount is being spent there — what outcome or channel priority justifies it — not just what it will be spent on.\n"
    "- The sum of all 'amount' values in 'budget_considerations' MUST equal the user's total stated marketing budget when the entire budget is allocated. Never generate allocations whose sum exceeds or falls below the total budget.\n"
    "- 'amount' MUST include the appropriate currency symbol (e.g. '₹10,000' or '$500') matching the user's currency.\n"
    "- 'percentage' MUST be mathematically consistent and calculated strictly as: (item allocation amount / total marketing budget) * 100, formatted with a '%' sign (e.g. '20%', '40%'). Never set arbitrary or identical percentages (e.g. setting 25% across items with unequal amounts). The sum of all item percentages must total 100% when the entire budget is allocated.\n"
    "- Any monetary amount mentioned inside the 'description' must strictly match the item's 'amount'.\n"
    "- Never split one budget allocation into multiple items.\n"
    "- 'label', 'amount', 'percentage', and 'description' belong to the SAME budget item.\n"
    "- The description must be a complete sentence. Never put a sentence fragment into a separate item.\n"
    "- Never split an item at 'and', commas, parentheses, colons, or other punctuation.\n"
    "- If one allocation mentions multiple activities, keep them inside the same item's description.\n\n"

    "17. KPIs — GOAL-ALIGNED AND DERIVED FROM STRATEGY. "
    "KPIs must directly support the stated marketing goal and be appropriate for the business model. "
    "Each KPI must flow from the strategy: if the strategy prioritizes Instagram for lead generation, a KPI must measure DM inquiries or profile link clicks — not generic 'follower growth'. "
    "Any numerical target not provided by the user is a planning assumption, not a guarantee — label it as such. "
    "Keep all calculations mathematically consistent. Do not invent historical performance.\n"
    "- Each KPI in 'kpis' MUST be ONE complete structured object with keys: 'metric', 'target', and 'description'.\n"
    "- Never split a KPI metric and its target into separate items.\n"
    "- Never split an item at parentheses, colons, commas, or other punctuation.\n"
    "- The 'target' field must contain the complete target value (e.g., '150 leads').\n"
    "- The description must explain what the metric measures, how to track it, and how it connects to the stated marketing goal.\n\n"
    "All KPI targets must represent the expected measurement period for this 1-month marketing strategy.\n"

    "18. ACTION PLAN AND TIMELINE — BUSINESS-SPECIFIC 4-WEEK ROADMAP.\n"
    "Create a practical, prioritized, and highly actionable execution roadmap for implementing the marketing strategy during exactly one month (four weeks).\n\n"
    "18a. BUILD DIRECTLY AROUND USER-SELECTED CHANNELS:\n"
    "- The initial form collects the marketing channels the user wants to focus on ('Current Marketing Channels').\n"
    "- Treat these user-selected channels as the primary channels for the entire marketing strategy and action plan.\n"
    "- Build the four-week plan around the selected channels instead of routinely recommending new ones.\n"
    "- Do not automatically introduce channels such as Google Ads, Google Shopping, influencer marketing, email marketing, or WhatsApp when the user has not selected them.\n"
    "- Focus on improving the effectiveness of the selected channels through appropriate content, campaigns, audience targeting, testing, optimization, and performance measurement.\n"
    "- Do not interpret the selected channels as a requirement to use every possible tactic available on those channels. Choose tactics according to the business, goal, audience, and budget.\n"
    "- Recommend an additional channel only when there is a clear, business-specific justification and a meaningful reason the selected channels alone may not adequately address the goal. Present it as an optional recommendation, not as an assumed part of the existing marketing setup.\n"
    "- Never replace the user's selected channels with different channels without justification.\n\n"
    "18b. FOUR DISTINCT WEEKLY PHASES WITH CONCRETE DETAILS:\n"
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
    "18c. GROUNDED AND REALISTIC EXECUTION:\n"
    "- Fit all activities strictly within the user's actual marketing budget and a one-month timeline.\n"
    "- Use facts provided in the initial form, subsequent answers, and available online presence context. Preserve the user's provided information and do not contradict it.\n"
    "- Do not invent existing campaigns, customer counts, testimonials, product categories, sales figures, delivery policies, discounts, offers, or business capabilities.\n"
    "- Do not present unverified information as an established fact.\n"
    "- Avoid assuming the business has a marketing team, advertising infrastructure, analytics setup, customer database, or other resources unless supported by available information.\n"
    "- Include prerequisites when a proposed activity depends on setup or infrastructure that may not exist.\n"
    "- If suggesting a discount, promotion, retargeting campaign, or another activity that depends on an unavailable business capability, frame it as conditional where appropriate.\n"
    "- Do not invent historical performance or guarantee a particular percentage increase in sales, engagement, conversions, or revenue. If measurable targets are included, distinguish proposed targets from guaranteed outcomes.\n"
    "- Prefer a focused, achievable plan over an unnecessarily large list of activities.\n\n"

    "19. ASSUMPTIONS & PLANNING ESTIMATES. "
    "If assumptions or estimated numbers are necessary, integrate them directly into their relevant standard section "
    "(such as 'target_audience_insights', 'budget_considerations', 'kpis', or 'action_plan'). "
    "Do NOT create or return any custom or additional sections. All strategic insights must be placed strictly within the 9 fixed sections.\n\n"

    "20. FINAL QUALITY CHECK. "
    "Before responding, verify:\n"
    "(a) business_overview does NOT repeat the user's inputs — it frames the strategic challenge.\n"
    "(b) competitive_positioning names a specific, exploitable gap — not a generic differentiator.\n"
    "(c) Every tactic names a platform, a concrete action, and an outcome signal.\n"
    "(d) Budget descriptions explain WHY each allocation is prioritized.\n"
    "(e) KPIs are derived from the strategy's chosen channels and goal — not generic benchmarks.\n"
    "(f) All business facts, competitor facts, locations, pricing, and budget figures are consistent and grounded.\n"
    "(g) Remove all generic filler phrases and unsupported claims.\n\n"

    "OUTPUT RULES:\n"
    "Return ONLY one valid JSON object. "
    "Do not return explanations, code fences, or commentary outside the JSON object. "
    "Do not change field names or field types.\n\n"

    "CONTENT RULES (for all list fields):\n"
    "Each list field must be a JSON array of plain-text strings or structured objects as specified in REQUIRED JSON STRUCTURE. "
    "Each item is one concise, self-contained strategy point or object. "
    "Do NOT use unnecessary Markdown syntax inside list items (except for structural section headers like '#### Build on Your Current Marketing Channels', '#### Additional Marketing Opportunities to Consider', and bold channel titles like '**Website**' where explicitly required).\n"
    "- HIGH DENSITY & CONCISENESS: Prioritize strategic density and concrete substance over verbosity. Avoid wordy narratives, conversational fluff, and repetitive explanations. "
    "Express each recommendation in 1-2 punchy, substantive sentences with high informational value.\n"
    "- AIM FOR 3-5 FOCUSED ITEMS PER SECTION: Provide 3-5 high-impact, business-tailored items per field rather than sprawling, repetitive lists.\n"
    "- ACTION PLAN STRUCTURE: Provide exactly 4 milestone items corresponding to Week 1, Week 2, Week 3, and Week 4. Format each with a bold milestone header '**Week X: [Milestone Title]**' followed by 2-3 specific execution actions and 1 clear review metric. Be concrete, specific, and actionable without writing multi-paragraph essays.\n\n"

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

    "21. MISSING OR UNKNOWN INFORMATION. "
    "If the user says 'I don't know', 'not sure', 'unknown', or provides no information for a requirement, "
    "treat that information as unknown. Never invent the missing fact. "
    "Continue generating the strategy using the reliable information that is available. "
    "When the missing information materially affects a recommendation, provide a practical test, assumption to verify, "
    "or discovery approach instead of pretending the information is known. "
    "Do not unnecessarily make the entire strategy vague simply because one requirement is unknown.\n\n"
)


def build_strategy_generation_prompt(state: MarketingAgentState) -> str:
    """
    Builds the complete user prompt for marketing strategy generation from agent state.
    Preserves exact formatting, sections, conditions, and directives.
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

    if state.competitor_research and state.competitor_research.competitors:
        cr_lines = []
        for comp in state.competitor_research.competitors:
            comp_details = [f"- Competitor: {comp.competitor_name}"]
            if comp.user_stated_context:
                comp_details.append(f"  User-Reported Note: \"{comp.user_stated_context}\"")
            if comp.official_website:
                comp_details.append(f"  Website: {comp.official_website}")
            if comp.positioning:
                comp_details.append(f"  Positioning: {comp.positioning}")
            if comp.products_and_services:
                comp_details.append(f"  Products/Services: {', '.join(comp.products_and_services)}")
            if comp.target_audience:
                comp_details.append(f"  Target Audience: {comp.target_audience}")
            if comp.pricing_information:
                comp_details.append(f"  Pricing: {comp.pricing_information}")
            if comp.differentiators:
                comp_details.append(f"  Differentiators: {', '.join(comp.differentiators)}")
            if comp.marketing_channels_and_campaigns:
                comp_details.append(f"  Marketing Channels: {', '.join(comp.marketing_channels_and_campaigns)}")
            if comp.strengths:
                comp_details.append(f"  Evidence-Supported Strengths: {', '.join(comp.strengths)}")
            if comp.weaknesses:
                comp_details.append(f"  Evidence-Supported Weaknesses / Gaps: {', '.join(comp.weaknesses)}")
            if comp.summary:
                comp_details.append(f"  Strategic Summary: {comp.summary}")
            if comp.source_urls:
                comp_details.append(f"  Supporting Sources: {', '.join(comp.source_urls[:3])}")
            cr_lines.append("\n".join(comp_details))

        # Build a gap synthesis hint to guide differentiation
        gap_hints = []
        for comp in state.competitor_research.competitors:
            if comp.weaknesses:
                top_gap = comp.weaknesses[0]
                gap_hints.append(f"  - {comp.competitor_name}: '{top_gap}'")
            elif comp.summary:
                gap_hints.append(f"  - {comp.competitor_name}: review summary for gaps.")

        competitor_directive = (
            "\nCOMPETITOR RESEARCH STRATEGY DIRECTIVE:\n"
            "- LEAD WITH THE GAP: Before recommending positioning, identify the single most exploitable competitor weakness or underserved segment from the data below. Name the competitor and name the gap explicitly in 'competitive_positioning'.\n"
            "- DERIVE ACTIONS, NOT SUMMARIES: Do not produce a competitor profile summary. For each relevant competitor, derive one concrete strategic action (e.g., target their underserved segment, counter their pricing with a trial, out-position them on a channel they neglect).\n"
            "- CHANNEL EXPLOITATION: If a competitor is absent from a channel that this business's audience uses, prioritize that channel with a specific campaign rationale.\n"
            "- Do NOT fabricate competitor claims, pricing, or market share numbers not supported by the evidence.\n"
            "- Do NOT treat a lack of search results as evidence that a competitor lacks a particular capability.\n"
            "- If competitor research is incomplete, rely cleanly on the client's stated context."
        )

        if gap_hints:
            competitor_directive += (
                "\n\nIDENTIFIED COMPETITOR GAPS (use these to anchor competitive_positioning and value_proposition):\n"
                + "\n".join(gap_hints)
            )

        prompt_parts.extend([
            "\nVERIFIED COMPETITOR WEB RESEARCH (VIA TAVILY):",
            "\n\n".join(cr_lines),
            competitor_directive
        ])

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
            f"- Prioritize budget toward the user's selected marketing channels. Each description MUST explain WHY this amount is allocated here (e.g., 'allocated here because this is the primary acquisition channel', 'reserved for creative asset production which is a prerequisite for all campaigns').\n"
            f"- Do NOT allocate equal amounts to each line item unless amounts truly are equal. Allocation should reflect strategic priority."
        )

    prompt_parts.append(
        "\nACTION PLAN & TIMELINE DIRECTIVE:\n"
        "Generate 'action_plan' as exactly 4 distinct, business-specific milestone strings for Week 1, Week 2, Week 3, and Week 4.\n"
        "- Build the 4-week plan directly around the User-Selected Marketing Channels above. Do not routinely introduce unselected channels (e.g. Google Ads, WhatsApp, influencer marketing) unless strictly necessary, explicitly justified, and framed as optional tests.\n"
        "- Format each week starting with a bold header: '**Week 1: [Milestone Title]**', '**Week 2: [Milestone Title]**', '**Week 3: [Milestone Title]**', '**Week 4: [Milestone Title]**'.\n"
        "- For each week, name: (a) what specific assets or content to produce (with format and platform), (b) what audience targeting parameters or copy variant to test, (c) what specific metric to review and what decision it drives. Forbid vague instructions like 'improve engagement' or 'optimize campaigns' — always follow them with 'by doing X'.\n"
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
        "\nSynthesize all available context above into a high-density, concise, tailored marketing strategy JSON object adhering strictly to the schema."
    )

    return "\n".join(prompt_parts)
