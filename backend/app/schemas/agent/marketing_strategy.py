from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BudgetItem(BaseModel):
    """
    Structured representation of a single budget allocation item.
    """
    label: str = Field(
        ...,
        description="Category, channel, or activity for the budget allocation."
    )
    amount: str = Field(
        ...,
        description="Monetary budget amount allocated (e.g. '₹20,000' or '$500')."
    )
    percentage: Optional[str] = Field(
        default=None,
        description="Percentage share of total budget allocated (e.g. '40%')."
    )
    description: str = Field(
        ...,
        description="Full explanation of how this budget will be deployed."
    )

    @field_validator("percentage", mode="before")
    @classmethod
    def format_percentage(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        v_str = str(v).strip()
        if not v_str:
            return None
        if not v_str.endswith("%"):
            try:
                float(v_str)
                return f"{v_str}%"
            except ValueError:
                pass
        return v_str


class KPIItem(BaseModel):
    """
    Structured representation of a single Key Performance Indicator (KPI).
    """
    metric: str = Field(
        ...,
        description="The KPI metric name (e.g. 'Qualified leads generated per month')."
    )
    target: str = Field(
        ...,
        description="Complete target goal value for the metric (e.g. '150 leads')."
    )
    description: str = Field(
        ...,
        description="Explanation of what this metric measures and how it will be tracked."
    )


class MarketingStrategy(BaseModel):
    """
    Represents the final structured marketing strategy synthesized by the agent.
    Contains exactly 9 fixed, standard sections in a fixed order.
    Configured to ignore extra fields (such as legacy additional_sections) for backward compatibility.
    """
    model_config = ConfigDict(extra="ignore")

    business_overview: Optional[List[str]] = Field(
        default=None,
        description="High-level overview of the business, its core market, and strategic context."
    )
    target_audience_insights: Optional[List[str]] = Field(
        default=None,
        description="Deep insights into customer personas, pain points, motivations, and behaviors."
    )
    competitive_positioning: Optional[List[str]] = Field(
        default=None,
        description="Market positioning relative to competitors and key differentiators."
    )
    value_proposition: Optional[List[str]] = Field(
        default=None,
        description="Core messaging angles and unique value proposition statements."
    )
    marketing_channels_and_tactics: Optional[List[str]] = Field(
        default=None,
        description="Recommended marketing channels (e.g. SEO, LinkedIn, Email) and tactical execution details."
    )
    customer_acquisition_approach: Optional[List[str]] = Field(
        default=None,
        description="Funnel strategy and tactics for turning prospects into paying customers."
    )
    budget_considerations: Optional[List[BudgetItem]] = Field(
        default=None,
        description="Structured budget allocation items and resource prioritization."
    )
    kpis: Optional[List[KPIItem]] = Field(
        default=None,
        description="Structured key performance indicators and target metrics."
    )
    action_plan: Optional[List[str]] = Field(
        default=None,
        description="Phased implementation roadmap and action steps."
    )

    @field_validator(
        "business_overview",
        "target_audience_insights",
        "competitive_positioning",
        "value_proposition",
        "marketing_channels_and_tactics",
        "customer_acquisition_approach",
        "action_plan",
        mode="before",
    )
    @classmethod
    def coerce_str_to_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v

    @field_validator("budget_considerations", mode="before")
    @classmethod
    def coerce_budget_considerations(cls, v: Any) -> Any:
        if v is None:
            return v
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            coerced = []
            for item in v:
                if isinstance(item, str):
                    coerced.append({
                        "label": "Budget Allocation",
                        "amount": "",
                        "percentage": None,
                        "description": item
                    })
                elif isinstance(item, (dict, BudgetItem)):
                    coerced.append(item)
            return coerced
        return v

    @field_validator("kpis", mode="before")
    @classmethod
    def coerce_kpis(cls, v: Any) -> Any:
        if v is None:
            return v
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            coerced = []
            for item in v:
                if isinstance(item, str):
                    coerced.append({
                        "metric": "KPI Metric",
                        "target": "",
                        "description": item
                    })
                elif isinstance(item, (dict, KPIItem)):
                    coerced.append(item)
            return coerced
        return v
