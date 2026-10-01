from pydantic import BaseModel, Field


class PDFExtractionResponse(BaseModel):
    """
    Response model for stateless PDF text extraction.
    """
    summary: str = Field(
        ...,
        description="The extracted or summarized marketing-relevant text from the PDF document."
    )
    character_count: int = Field(
        ...,
        description="The total character count of the original extracted text before summarization."
    )
    was_summarized: bool = Field(
        ...,
        description="Indicates whether the text was summarized by LLM."
    )


