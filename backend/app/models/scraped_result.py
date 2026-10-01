from typing import Any, List, Optional
from pydantic import BaseModel, Field


class ApifyScrapeResult(BaseModel):
    """
    Normalized internal result from the Apify scraping service.
    """
    original_url: str = Field(
        ...,
        description="The original raw URL provided."
    )
    normalized_url: Optional[str] = Field(
        default=None,
        description="The normalized URL if valid."
    )
    platform: str = Field(
        ...,
        description="The identified platform ('website', 'instagram', 'facebook', 'linkedin', 'unknown')."
    )
    success: bool = Field(
        ...,
        description="True if scraping completed successfully."
    )
    data: Optional[List[Any]] = Field(
        default=None,
        description="The scraped dataset items list if successful."
    )
    error: Optional[str] = Field(
        default=None,
        description="Error details if scraping failed or was skipped."
    )
