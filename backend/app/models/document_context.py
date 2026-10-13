"""
Pydantic models and helper functions for HistoricalDocumentContext.

Represents structured historical marketing insights extracted from an uploaded PDF
document, keeping them strictly separated from current business context.
"""

from typing import Optional
from pydantic import BaseModel, Field

from backend.app.schemas.pdf_extraction import PDFInsights


class HistoricalDocumentContext(BaseModel):
    """
    Structured historical marketing insights retrieved from the uploaded document,
    maintained as distinct historical context separate from current business facts.
    """
    filename: Optional[str] = Field(
        default=None,
        description="Original filename of the uploaded PDF."
    )
    is_marketing_relevant: bool = Field(
        default=False,
        description="Whether the document was classified as marketing-relevant."
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Identified category or type of the document."
    )
    relevance_reason: Optional[str] = Field(
        default=None,
        description="Reasoning explaining why the document was or was not classified as marketing-relevant."
    )
    executive_summary: Optional[str] = Field(
        default=None,
        description="Narrative executive summary of the historical marketing insights."
    )
    pdf_insights: Optional[PDFInsights] = Field(
        default=None,
        description="Granular historical marketing insights (channels, campaigns, spend, results, offers, learnings)."
    )


def format_document_context(doc_ctx: Optional[HistoricalDocumentContext]) -> Optional[str]:
    """
    Formats the HistoricalDocumentContext into a clean, structured string block
    for injection into LLM prompts across agent workflow stages.
    Returns None if no document context exists.
    If the document is not marketing-relevant, returns a brief informational note.
    If marketing-relevant, formats all historical dimensions with source pages and qualifiers.
    """
    if not doc_ctx:
        return None

    if not doc_ctx.is_marketing_relevant or not doc_ctx.pdf_insights:
        if doc_ctx.relevance_reason:
            doc_label = doc_ctx.filename or "Uploaded PDF"
            doc_type_str = f" [{doc_ctx.document_type}]" if doc_ctx.document_type else ""
            return (
                f"DOCUMENT NOTE ({doc_label}{doc_type_str}): Classified as non-marketing. "
                f"Reason: {doc_ctx.relevance_reason}. (No historical marketing evidence available; do not treat as marketing history.)"
            )
        return None

    insights = doc_ctx.pdf_insights
    lines = [
        f"HISTORICAL MARKETING DOCUMENT CONTEXT (FROM UPLOADED FILE: {doc_ctx.filename or 'Document'}):",
        "NOTE: This information represents historical activity, previous experiments, and past results. "
        "It must NOT be confused with or overwrite the business's CURRENT inputs, current budget, or current marketing channels."
    ]

    if doc_ctx.document_type:
        lines.append(f"Document Type: {doc_ctx.document_type}")
    if doc_ctx.executive_summary:
        lines.append(f"Historical Executive Summary: {doc_ctx.executive_summary}")
    elif insights.executive_summary:
        lines.append(f"Historical Executive Summary: {insights.executive_summary}")

    # Historical Channels
    if insights.historical_channels:
        lines.append("\nHistorical Marketing Channels & Activities:")
        for ch in insights.historical_channels:
            page_ref = f" (Source: Page {', '.join(map(str, ch.source_pages))})" if ch.source_pages else ""
            details = []
            if ch.activities:
                details.append(f"Activities: {', '.join(ch.activities)}")
            if ch.spend:
                details.append(f"Past Spend: {ch.spend}")
            if ch.results:
                details.append(f"Past Results: {ch.results}")
            detail_str = f" | {'; '.join(details)}" if details else ""
            lines.append(f"  - {ch.channel}{detail_str}{page_ref}")
            if ch.supporting_text:
                lines.append(f"    Supporting quote: \"{ch.supporting_text}\"")

    # Past Campaigns
    if insights.campaigns:
        lines.append("\nPast Campaigns & Objectives:")
        for camp in insights.campaigns:
            page_ref = f" (Source: Page {', '.join(map(str, camp.source_pages))})" if camp.source_pages else ""
            dates = f" [{camp.period_or_dates}]" if camp.period_or_dates else ""
            details = []
            if camp.objective_or_goal:
                details.append(f"Goal: {camp.objective_or_goal}")
            if camp.budget_or_spend:
                details.append(f"Spend: {camp.budget_or_spend}")
            if camp.key_metrics_and_results:
                details.append(f"Results: {camp.key_metrics_and_results}")
            detail_str = f" | {'; '.join(details)}" if details else ""
            lines.append(f"  - {camp.campaign_name}{dates}{detail_str}{page_ref}")
            if camp.supporting_text:
                lines.append(f"    Supporting quote: \"{camp.supporting_text}\"")

    # Historical Budget and Spend Overview
    if insights.historical_budget_and_spend:
        lines.append(f"\nHistorical Budget & Spend Overview: {insights.historical_budget_and_spend}")

    # Key Results
    if insights.key_results:
        lines.append("\nHistorical Performance Benchmarks:")
        for kr in insights.key_results:
            page_ref = f" (Source: Page {', '.join(map(str, kr.source_pages))})" if kr.source_pages else ""
            ctx_str = f" ({kr.context_or_channel})" if kr.context_or_channel else ""
            lines.append(f"  - {kr.metric_name}: {kr.value}{ctx_str}{page_ref}")
            if kr.supporting_text:
                lines.append(f"    Supporting quote: \"{kr.supporting_text}\"")

    # Successful Activities
    if insights.successful_activities:
        lines.append("\nHistorical Activities That Succeeded (Proven in Past):")
        for sa in insights.successful_activities:
            page_ref = f" (Source: Page {', '.join(map(str, sa.source_pages))})" if sa.source_pages else ""
            lines.append(f"  - {sa.activity}: {sa.details}{page_ref}")
            if sa.supporting_text:
                lines.append(f"    Supporting quote: \"{sa.supporting_text}\"")

    # Unsuccessful Activities
    if insights.unsuccessful_activities:
        lines.append("\nHistorical Activities That Underperformed (Past Lessons/Warnings):")
        for ua in insights.unsuccessful_activities:
            page_ref = f" (Source: Page {', '.join(map(str, ua.source_pages))})" if ua.source_pages else ""
            lines.append(f"  - {ua.activity}: {ua.details}{page_ref}")
            if ua.supporting_text:
                lines.append(f"    Supporting quote: \"{ua.supporting_text}\"")

    # Audience Learnings
    if insights.audience_learnings:
        lines.append("\nAudience Demographics & Behavioral Learnings:")
        for al in insights.audience_learnings:
            page_ref = f" (Source: Page {', '.join(map(str, al.source_pages))})" if al.source_pages else ""
            lines.append(f"  - {al.segment_or_insight}: {al.takeaway}{page_ref}")
            if al.supporting_text:
                lines.append(f"    Supporting quote: \"{al.supporting_text}\"")

    # Offers and Messaging
    if insights.offers_and_messaging:
        lines.append("\nHistorical Offers, Creative Messaging & Tested Conditions:")
        for om in insights.offers_and_messaging:
            page_ref = f" (Source: Page {', '.join(map(str, om.source_pages))})" if om.source_pages else ""
            cond_str = f" [QUALIFIERS/CONDITIONS: {om.conditions_or_qualifiers}]" if om.conditions_or_qualifiers else " [NO QUALIFIERS STATED]"
            perf_str = f" | Result: {om.performance_note}" if om.performance_note else ""
            fmt_str = f" (Format/Channel: {om.channel_or_format})" if om.channel_or_format else ""
            lines.append(f"  - \"{om.offer_or_message}\"{fmt_str}{cond_str}{perf_str}{page_ref}")
            if om.supporting_text:
                lines.append(f"    Supporting quote: \"{om.supporting_text}\"")

    # Missing / Unreported Information
    if insights.missing_or_unreported_information:
        lines.append("\nHistorical Gaps & Unreported Data:")
        for gap in insights.missing_or_unreported_information:
            lines.append(f"  - {gap}")

    return "\n".join(lines)
