import io
import json
import logging
import re
from typing import List, Optional, Tuple, Union
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
import pdfplumber

from backend.app.models.uploaded_document import UploadedDocument
from backend.app.db.session import SessionLocal
from backend.app.schemas.api.pdf_extraction import PDFExtractionResponse
from backend.app.schemas.agent.pdf_insights import PDFInsights
from backend.app.services.llm_service import get_llm_response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["PDF Extraction"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit in bytes
MIN_CHARACTER_COUNT = 50

PAGE_NUM_REGEX = re.compile(
    r"^(?:page\s+\d+(?:\s+of\s+\d+)?|\d+|\-\s*\d+\s*\-|\—\s*\d+\s*\—)$",
    re.IGNORECASE,
)

from backend.app.prompts.pdf_extraction import (
    PDF_INSIGHT_SYSTEM_PROMPT,
    build_pdf_extraction_prompt,
)



def format_table_as_text(table_data: list) -> str:
    """
    Converts a 2D table grid extracted by pdfplumber into readable pipe-separated text.
    """
    if not table_data:
        return ""
    formatted_rows = []
    for row in table_data:
        if not row:
            continue
        cleaned_cells = [
            re.sub(r"\s+", " ", str(cell).strip()) if cell is not None else ""
            for cell in row
        ]
        if any(cleaned_cells):
            formatted_rows.append(" | ".join(cleaned_cells))
    if not formatted_rows:
        return ""
    if len(formatted_rows) >= 2:
        col_count = len(table_data[0]) if table_data[0] else 1
        divider = " | ".join(["---"] * col_count)
        return "\n".join([formatted_rows[0], divider] + formatted_rows[1:])
    return "\n".join(formatted_rows)


def extract_and_clean_pdf(pdf_stream: io.BytesIO) -> Tuple[str, int, int]:
    """
    Extracts text page by page from a PDF using pdfplumber, extracting tables where available
    and cleaning standalone page-number noise and repeated headers/footers.
    Returns (cleaned_text_with_page_markers, total_char_count, total_pages).
    """
    with pdfplumber.open(pdf_stream) as pdf:
        total_pages = len(pdf.pages)
        if total_pages == 0:
            return "", 0, 0

        raw_page_lines: List[List[str]] = []
        page_tables_text: List[List[str]] = []

        for page in pdf.pages:
            # 1. Extract plain text lines and filter standalone page number noise
            page_text = page.extract_text() or ""
            lines = [line.strip() for line in page_text.splitlines() if line.strip()]
            filtered_lines = [line for line in lines if not PAGE_NUM_REGEX.match(line)]
            raw_page_lines.append(filtered_lines)

            # 2. Extract tables via pdfplumber
            formatted_tables = []
            try:
                tables = page.extract_tables() or []
                for tbl in tables:
                    tbl_str = format_table_as_text(tbl)
                    if tbl_str:
                        formatted_tables.append(tbl_str)
            except Exception as tbl_err:
                logger.warning(f"Error extracting tables on page: {tbl_err}")
            page_tables_text.append(formatted_tables)

    # 3. Clean recurring headers and footers across pages for documents with >= 3 pages
    if total_pages >= 3:
        # Check first line of each page
        first_lines = [lines[0] for lines in raw_page_lines if lines]
        if first_lines:
            most_common_header = max(set(first_lines), key=first_lines.count)
            if first_lines.count(most_common_header) > total_pages / 2:
                for lines in raw_page_lines:
                    if lines and lines[0] == most_common_header:
                        lines.pop(0)

        # Check last line of each page
        last_lines = [lines[-1] for lines in raw_page_lines if lines]
        if last_lines:
            most_common_footer = max(set(last_lines), key=last_lines.count)
            if last_lines.count(most_common_footer) > total_pages / 2:
                for lines in raw_page_lines:
                    if lines and lines[-1] == most_common_footer:
                        lines.pop()

    # 4. Assemble pages with clear page markers and table blocks
    page_sections = []
    total_chars = 0

    for i in range(total_pages):
        page_num = i + 1
        lines = raw_page_lines[i]
        tbls = page_tables_text[i]

        section_parts = [f"--- Page {page_num} ---"]
        page_body = "\n".join(lines).strip()
        if page_body:
            section_parts.append(page_body)
            total_chars += len(page_body)

        if tbls:
            tbl_block = f"\n[Tables on Page {page_num}]:\n" + "\n\n".join(tbls)
            section_parts.append(tbl_block)
            total_chars += len(tbl_block)

        page_sections.append("\n".join(section_parts))

    full_text = "\n\n".join(page_sections)
    return full_text, total_chars, total_pages


@router.post("/extract-pdf-text", response_model=PDFExtractionResponse, status_code=status.HTTP_200_OK)
async def extract_pdf_text(
    request: Request,
    file: UploadFile = File(...),
    strategy_request_id: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
):
    """
    Extracts text and tables from an uploaded PDF file and performs a single LLM call
    to determine marketing relevance and extract structured historical marketing insights.
    Associates LLM usage and uploaded document with available strategy_request_id or conversation_id.
    """
    filename = file.filename or ""
    content_type = file.content_type or ""

    # Resolve optional strategy_request_id and conversation_id from form or query params
    strat_id_raw = strategy_request_id or request.query_params.get("strategy_request_id")
    conv_id_raw = conversation_id or request.query_params.get("conversation_id")

    strat_uuid: Optional[uuid.UUID] = None
    if strat_id_raw:
        try:
            strat_uuid = uuid.UUID(str(strat_id_raw))
        except (ValueError, TypeError):
            logger.warning(f"Invalid strategy_request_id format in PDF extraction: {strat_id_raw}")

    conv_uuid: Optional[uuid.UUID] = None
    if conv_id_raw:
        try:
            conv_uuid = uuid.UUID(str(conv_id_raw))
        except (ValueError, TypeError):
            logger.warning(f"Invalid conversation_id format in PDF extraction: {conv_id_raw}")

    # If conversation_id is known but strategy_request_id is not, resolve strategy_request_id from the DB
    if conv_uuid and not strat_uuid:
        try:
            from backend.app.models.conversation import Conversation
            with SessionLocal() as db:
                conv_rec = db.query(Conversation).filter(Conversation.id == conv_uuid).first()
                if conv_rec and conv_rec.strategy_request_id:
                    strat_uuid = conv_rec.strategy_request_id
        except Exception as lookup_err:
            logger.debug(f"Could not resolve strategy_request_id from conversation in PDF extraction: {lookup_err}")

    # Validation: Content-Type and filename extension check
    if content_type.lower() != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a PDF document (content-type must be 'application/pdf' or file extension must be '.pdf')."
        )

    # Validation: Read file size safely and check 10 MB limit
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
            detail="File size exceeds the maximum limit of 10 MB."
        )

    # Parse and extract text and tables page-by-page
    try:
        pdf_stream = io.BytesIO(contents)
        extracted_text, total_chars, total_pages = extract_and_clean_pdf(pdf_stream)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error parsing PDF with pdfplumber: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to parse PDF file. The file may be corrupt, encrypted, or invalid."
        )

    # Check for scanned / image-only PDF (< 50 characters stripped)
    if len(extracted_text.strip()) < MIN_CHARACTER_COUNT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The PDF contains no extractable text or fewer than 50 characters. It may be a scanned or image-only PDF."
        )

    logger.info(
        f"PDF '{filename}' extracted successfully ({total_pages} pages, {total_chars} chars). "
        f"Calling LLM for marketing relevance classification and structured insight extraction."
    )

    user_prompt = build_pdf_extraction_prompt(
        filename=filename,
        total_pages=total_pages,
        extracted_text=extracted_text,
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=PDF_INSIGHT_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            operation="pdf_insight_extraction",
            strategy_request_id=strat_uuid,
            conversation_id=conv_uuid,
        )
        data = json.loads(raw_response)
    except Exception as llm_err:
        logger.error(f"LLM insight extraction failed: {llm_err}")
        doc_id = uuid.uuid4()
        summary_text = extracted_text[:1500] if len(extracted_text) > 1500 else extracted_text

        # Persist fallback document record
        try:
            with SessionLocal() as db:
                doc_record = UploadedDocument(
                    id=doc_id,
                    strategy_request_id=strat_uuid,
                    filename=filename,
                    file_size_bytes=len(contents),
                    page_count=total_pages,
                    is_marketing_relevant=True,
                    document_type="Document",
                    relevance_reason="Automated insight extraction encountered an issue; raw extracted text preserved.",
                    executive_summary=summary_text,
                    pdf_insights=None,
                    raw_extracted_text=extracted_text,
                )
                db.add(doc_record)
                db.commit()
                logger.info(f"Persisted fallback UploadedDocument '{doc_id}' for file '{filename}'.")
        except Exception as db_err:
            logger.error(f"Failed to persist UploadedDocument fallback to database: {db_err}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save uploaded document to the database."
            )

        return PDFExtractionResponse(
            is_marketing_relevant=True,
            relevance_reason="Automated insight extraction encountered an issue; raw extracted text preserved.",
            document_type="Document",
            summary=summary_text,
            pdf_insights=None,
            document_id=doc_id,
            character_count=total_chars,
            page_count=total_pages,
            was_summarized=False,
        )

    is_relevant = bool(data.get("is_marketing_relevant", True))
    relevance_reason = data.get("relevance_reason")
    doc_type = data.get("document_type")
    summary_text = data.get("summary")
    insights_raw = data.get("pdf_insights")

    pdf_insights_obj = None
    if is_relevant and isinstance(insights_raw, dict):
        try:
            pdf_insights_obj = PDFInsights.model_validate(insights_raw)
            if not summary_text and pdf_insights_obj.executive_summary:
                summary_text = pdf_insights_obj.executive_summary
        except Exception as val_err:
            logger.warning(f"Failed to validate PDFInsights with Pydantic: {val_err}. Retrying with sanitized dict.")
            try:
                sanitized_dict = {
                    k: v for k, v in insights_raw.items()
                    if k in PDFInsights.model_fields
                }
                pdf_insights_obj = PDFInsights(**sanitized_dict)
            except Exception as retry_err:
                logger.error(f"Sanitized PDFInsights validation failed: {retry_err}")
                pdf_insights_obj = None

    if not summary_text:
        if is_relevant:
            summary_text = "Historical marketing document analyzed successfully."
        else:
            summary_text = "The uploaded document does not appear to contain historical marketing or campaign information."

    # Persist document to uploaded_documents table
    doc_id = uuid.uuid4()
    insights_dict = pdf_insights_obj.model_dump() if pdf_insights_obj else (insights_raw if isinstance(insights_raw, dict) else None)

    try:
        with SessionLocal() as db:
            doc_record = UploadedDocument(
                id=doc_id,
                strategy_request_id=strat_uuid,
                filename=filename,
                file_size_bytes=len(contents),
                page_count=total_pages,
                is_marketing_relevant=is_relevant,
                document_type=doc_type,
                relevance_reason=relevance_reason,
                executive_summary=summary_text,
                pdf_insights=insights_dict,
                raw_extracted_text=extracted_text,
            )
            db.add(doc_record)
            db.commit()
            logger.info(f"Persisted UploadedDocument '{doc_id}' for file '{filename}'.")
    except Exception as db_err:
        logger.error(f"Failed to persist UploadedDocument to database: {db_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded document to the database."
        )

    return PDFExtractionResponse(
        is_marketing_relevant=is_relevant,
        relevance_reason=relevance_reason,
        document_type=doc_type,
        summary=summary_text,
        pdf_insights=pdf_insights_obj,
        document_id=doc_id,
        character_count=total_chars,
        page_count=total_pages,
        was_summarized=True,
    )
