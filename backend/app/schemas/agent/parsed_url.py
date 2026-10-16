from typing import Optional
from pydantic import BaseModel, Field


class ParsedURL(BaseModel):
    """
    Structured internal representation for a parsed and validated URL.
    """
    original_url: str = Field(
        ...,
        description="The original raw URL string provided by the user."
    )
    normalized_url: Optional[str] = Field(
        default=None,
        description="The normalized URL string if valid, or None if invalid."
    )
    platform: str = Field(
        default="unknown",
        description="Identified platform ('website', 'instagram', 'facebook', 'linkedin', 'unknown')."
    )
    is_valid: bool = Field(
        default=False,
        description="True if the URL is a valid HTTP/HTTPS URL."
    )
    error: Optional[str] = Field(
        default=None,
        description="Descriptive error message if the URL is invalid."
    )
