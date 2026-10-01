import io
import logging
from fastapi import APIRouter, File, HTTPException, UploadFile, status
import pdfplumber

from backend.app.schemas.pdf_extraction import PDFExtractionResponse
from backend.app.services.llm_service import get_llm_response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["PDF Extraction"])

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB limit in bytes
MIN_CHARACTER_COUNT = 50
SUMMARIZATION_THRESHOLD = 1000
MAX_SUMMARY_LEN = 2000
LIGHTWEIGHT_MODEL = "openai/gpt-oss-20b"


@router.post("/extract-pdf-text", response_model=PDFExtractionResponse, status_code=status.HTTP_200_OK)
async def extract_pdf_text(file: UploadFile = File(...)):
    """
    Stateless endpoint to extract plain text from an uploaded PDF file using pdfplumber.
    Returns a short summary for texts longer than 1000 characters (compressed to <= 2000 chars)
    generated using lightweight model openai/gpt-oss-20b, or the extracted text as-is if <= 1000 characters.
    """
    filename = file.filename or ""
    content_type = file.content_type or ""

    # Validation step 3a: Content-Type and filename extension check
    if content_type.lower() != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a PDF document (content-type must be 'application/pdf' or file extension must be '.pdf')."
        )

    # Validation step 3b: Safely read file size and check 5 MB limit
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read the uploaded file."
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds the maximum limit of 5 MB."
        )

    # Validation step 3c: Open and parse PDF with pdfplumber
    try:
        pdf_stream = io.BytesIO(contents)
        pages_text = []
        with pdfplumber.open(pdf_stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error parsing PDF with pdfplumber: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to parse PDF file. The file may be corrupt, encrypted, or invalid."
        )

    # Step 4: Concatenate extracted page text with page breaks
    extracted_text = "\n\n".join(pages_text)
    original_char_count = len(extracted_text)

    # Step 5: Check for scanned / image-only PDF (< 50 characters stripped)
    if len(extracted_text.strip()) < MIN_CHARACTER_COUNT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The PDF contains no extractable text or fewer than 50 characters. It may be a scanned or image-only PDF."
        )

    # Summarization for text > 1000 characters
    summary_text = extracted_text
    was_summarized = False

    if original_char_count <= SUMMARIZATION_THRESHOLD:
        summary_text = extracted_text
        was_summarized = False
    else:
        logger.info(
            f"PDF text length ({original_char_count} chars) exceeds threshold ({SUMMARIZATION_THRESHOLD} chars). "
            f"Generating marketing summary using lightweight model '{LIGHTWEIGHT_MODEL}'."
        )
        system_prompt = (
            "You are an expert marketing strategy assistant. Your task is to extract and summarize "
            "only marketing-relevant information explicitly present in the provided document text."
        )
        prompt = (
            "Please analyze the following document text and extract ONLY marketing-relevant information "
            "explicitly present in the document, especially:\n"
            "- Previous marketing channels/activities\n"
            "- Campaign results\n"
            "- Successful and failed activities\n"
            "- Pricing/offers\n"
            "- Competitors\n"
            "- USP/positioning\n"
            "- Marketing goals\n"
            "- Important lessons or historical insights\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Summarize ONLY what is explicitly present in the document. Do NOT invent, assume, or add information.\n"
            "2. Produce a concise summary of approximately 1500-1800 characters, strictly keeping the final text within 2000 characters.\n\n"
            f"Document Text:\n{extracted_text}"
        )

        try:
            summary_res = get_llm_response(prompt=prompt, system_prompt=system_prompt, model=LIGHTWEIGHT_MODEL)
            if summary_res and len(summary_res.strip()) > 0:
                summary_text = summary_res.strip()
                was_summarized = True
                logger.info(f"PDF summarization by '{LIGHTWEIGHT_MODEL}' completed successfully ({len(summary_text)} chars).")
            else:
                logger.warning(
                    f"LLM summarization with '{LIGHTWEIGHT_MODEL}' returned empty text. "
                    f"Falling back to truncation (first {MAX_SUMMARY_LEN} characters)."
                )
                summary_text = extracted_text[:MAX_SUMMARY_LEN]
                was_summarized = False
        except Exception as e:
            logger.warning(
                f"LLM summarization with lightweight model '{LIGHTWEIGHT_MODEL}' failed ({e}). "
                f"Falling back to truncation (first {MAX_SUMMARY_LEN} characters)."
            )
            summary_text = extracted_text[:MAX_SUMMARY_LEN]
            was_summarized = False

        # If LLM returned > 2000 characters, compress/re-summarize to preserve information instead of plain slicing
        if was_summarized and len(summary_text) > MAX_SUMMARY_LEN:
            logger.warning(
                f"LLM summary from '{LIGHTWEIGHT_MODEL}' exceeded max limit ({len(summary_text)} chars > {MAX_SUMMARY_LEN} max). "
                f"Attempting compression using '{LIGHTWEIGHT_MODEL}' to preserve key information..."
            )
            compression_prompt = (
                "The following marketing summary exceeds the 2000-character limit. "
                "Please compress and re-summarize it to strictly under 2000 characters (target 1500-1800 characters) "
                "while preserving all critical marketing facts, channels, campaign results, pricing, competitors, goals, and positioning.\n"
                "Do NOT invent, assume, or add any new information.\n\n"
                f"Existing Summary:\n{summary_text}"
            )
            try:
                compressed_res = get_llm_response(prompt=compression_prompt, system_prompt=system_prompt, model=LIGHTWEIGHT_MODEL)
                if compressed_res and len(compressed_res.strip()) > 0:
                    summary_text = compressed_res.strip()
                    logger.info(f"Compression by '{LIGHTWEIGHT_MODEL}' completed successfully ({len(summary_text)} chars).")
                else:
                    logger.warning(f"Compression LLM call with '{LIGHTWEIGHT_MODEL}' returned empty text. Falling back to safety truncation.")
            except Exception as comp_err:
                logger.warning(f"Compression LLM call with '{LIGHTWEIGHT_MODEL}' failed ({comp_err}). Falling back to safety truncation.")

    # Guarantee final summary never exceeds MAX_SUMMARY_LEN (2000 chars)
    if len(summary_text) > MAX_SUMMARY_LEN:
        logger.warning(f"Summary text still exceeds {MAX_SUMMARY_LEN} characters after processing ({len(summary_text)} chars). Safety truncating.")
        summary_text = summary_text[:MAX_SUMMARY_LEN]

    # Step 6: Return JSON response with summary and original character_count
    return PDFExtractionResponse(
        summary=summary_text,
        character_count=original_char_count,
        was_summarized=was_summarized
    )
