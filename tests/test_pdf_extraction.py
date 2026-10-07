import io
import os
import sys
from unittest.mock import patch

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def make_pdf_bytes(text: str) -> bytes:
    """Helper to generate minimal valid PDF bytes with given text."""
    pdf_str = (
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        f"4 0 obj << /Length {len(text) + 27} >> stream\n"
        f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET\n"
        "endstream endobj\n"
        "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        "xref\n0 6\ntrailer << /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )
    return pdf_str.encode("latin1")


def test_pdf_extraction_endpoint():
    print("=== Testing Task 1.2 & Task 1.4: PDF Text Extraction & Summarization ===")

    # 1. Reject invalid file type (content-type not application/pdf AND filename not .pdf)
    print("\n--- Test 1: Reject Non-PDF File Type ---")
    response = client.post(
        "/extract-pdf-text",
        files={"file": ("sample.txt", b"This is plain text content", "text/plain")}
    )
    assert response.status_code == 400, f"Expected 400, got {response.status_code}: {response.text}"
    print(f"✓ Rejected non-PDF file: {response.json()['detail']}")

    # 2. Reject file exceeding 10 MB
    print("\n--- Test 2: Reject File Exceeding 10 MB ---")
    large_content = b"%PDF-1.4\n" + b"X" * (10 * 1024 * 1024 + 10)
    response = client.post(
        "/extract-pdf-text",
        files={"file": ("large.pdf", large_content, "application/pdf")}
    )
    assert response.status_code == 400, f"Expected 400, got {response.status_code}: {response.text}"
    assert "exceeds" in response.json()["detail"].lower()
    print(f"✓ Rejected oversized file: {response.json()['detail']}")

    # 3. Reject corrupt/invalid PDF file
    print("\n--- Test 3: Reject Invalid/Corrupt PDF ---")
    corrupt_content = b"%PDF-1.4 invalid pdf structure content that pdfplumber cannot open"
    response = client.post(
        "/extract-pdf-text",
        files={"file": ("corrupt.pdf", corrupt_content, "application/pdf")}
    )
    assert response.status_code == 400, f"Expected 400, got {response.status_code}: {response.text}"
    print(f"✓ Rejected corrupt PDF: {response.json()['detail']}")

    # 4. Reject scanned / short text PDF (< 50 characters)
    print("\n--- Test 4: Reject Scanned / Short Text PDF (< 50 chars) ---")
    short_pdf = make_pdf_bytes("Short text")
    response = client.post(
        "/extract-pdf-text",
        files={"file": ("short.pdf", short_pdf, "application/pdf")}
    )
    assert response.status_code == 400, f"Expected 400, got {response.status_code}: {response.text}"
    assert "scanned" in response.json()["detail"].lower() or "50" in response.json()["detail"]
    print(f"✓ Rejected short/scanned PDF: {response.json()['detail']}")

    # 5. Extract text from valid PDF (50 to 3000 characters) - returned as-is, was_summarized = False
    print("\n--- Test 5: Extract Text <= 3000 chars (As-Is) ---")
    sample_text = (
        "CleanSeas Ocean Cleanup Marketing Strategy Document 2026. "
        "Our goal is to recruit 500 volunteer cleanup captains across 4 university campuses."
    )
    valid_pdf = make_pdf_bytes(sample_text)
    response = client.post(
        "/extract-pdf-text",
        files={"file": ("strategy.pdf", valid_pdf, "application/pdf")}
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["was_summarized"] is False
    assert sample_text in data["summary"]
    assert data["character_count"] == len(sample_text)
    print("✓ Successfully returned text as-is (was_summarized=False):")
    print(f"  Character count: {data['character_count']}")

    # 6. Extract text > 3000 characters with successful LLM Summarization
    print("\n--- Test 6: Text > 3000 chars (Successful LLM Summarization) ---")
    long_text = "CleanSeas marketing campaign details. " * 100  # ~3900 chars
    long_pdf = make_pdf_bytes(long_text)
    mock_summary = "Summarized Key Marketing Highlights: CleanSeas recruited student captains via social media."

    with patch("backend.app.api.pdf_extraction.get_llm_response", return_value=mock_summary) as mock_llm:
        response = client.post(
            "/extract-pdf-text",
            files={"file": ("long_strategy.pdf", long_pdf, "application/pdf")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["was_summarized"] is True
        assert data["summary"] == mock_summary
        assert data["character_count"] > 3000
        mock_llm.assert_called_once()
        print("✓ Successfully summarized long PDF (was_summarized=True):")
        print(f"  Character count: {data['character_count']}")

    # 7. Extract text > 3000 characters with LLM Summarization Failure (Fallback Truncation)
    print("\n--- Test 7: Text > 3000 chars (LLM Failure Fallback to Truncation) ---")
    with patch("backend.app.api.pdf_extraction.get_llm_response", side_effect=Exception("API Error")):
        response = client.post(
            "/extract-pdf-text",
            files={"file": ("long_strategy.pdf", long_pdf, "application/pdf")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["was_summarized"] is False
        assert len(data["summary"]) == 2000
        assert data["character_count"] > 3000
        print("✓ Successfully handled LLM failure with truncation fallback (was_summarized=False):")
        print(f"  Truncated character count: {data['character_count']}")

    print("\n=== All PDF Extraction & Summarization Endpoint Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_pdf_extraction_endpoint()
