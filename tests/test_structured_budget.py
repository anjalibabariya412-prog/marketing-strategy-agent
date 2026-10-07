import os
import sys
import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.models.business_context import BusinessContext, MarketingBudget, CurrencyEnum
from backend.app.schemas.conversation import StartRequest

client = TestClient(app)


def test_marketing_budget_model_valid():
    """Test valid INR and USD budget model creation and string representation."""
    b_inr = MarketingBudget(amount=30000, currency=CurrencyEnum.INR)
    assert b_inr.amount == 30000
    assert b_inr.currency == CurrencyEnum.INR
    assert str(b_inr) == "₹30,000"

    b_usd = MarketingBudget(amount=5000, currency="USD")
    assert b_usd.amount == 5000
    assert b_usd.currency == CurrencyEnum.USD
    assert str(b_usd) == "$5,000"


def test_marketing_budget_model_invalid_currency():
    """Test that unsupported currencies (EUR, GBP, etc.) are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        MarketingBudget(amount=1000, currency="EUR")
    assert "Currency must be either 'INR' or 'USD'" in str(exc_info.value)


def test_marketing_budget_model_invalid_amount():
    """Test that amounts <= 0 are rejected."""
    with pytest.raises(ValidationError):
        MarketingBudget(amount=0, currency="INR")

    with pytest.raises(ValidationError):
        MarketingBudget(amount=-50, currency="USD")


def test_start_endpoint_budget_validation():
    """Test FastAPI /start endpoint structured budget validation behavior."""
    base_payload = {
        "company_name": "Test Bakery",
        "product_or_service": "Cakes",
        "marketing_goal": "Increase online sales",
        "target_audience": "Local customers"
    }

    # 1. Valid INR budget
    res_inr = client.post("/start", json={
        **base_payload,
        "budget_resources": {"amount": 30000, "currency": "INR"}
    })
    assert res_inr.status_code == 200, f"Expected 200, got {res_inr.status_code}: {res_inr.text}"

    # 2. Valid USD budget
    res_usd = client.post("/start", json={
        **base_payload,
        "budget_resources": {"amount": 2500, "currency": "USD"}
    })
    assert res_usd.status_code == 200, f"Expected 200, got {res_usd.status_code}: {res_usd.text}"

    # 3. Invalid currency EUR (should return 422 Unprocessable Entity)
    res_eur = client.post("/start", json={
        **base_payload,
        "budget_resources": {"amount": 2500, "currency": "EUR"}
    })
    assert res_eur.status_code == 422, f"Expected 422 for EUR, got {res_eur.status_code}"

    # 4. Invalid negative amount (should return 422 Unprocessable Entity)
    res_neg = client.post("/start", json={
        **base_payload,
        "budget_resources": {"amount": -100, "currency": "INR"}
    })
    assert res_neg.status_code == 422, f"Expected 422 for negative amount, got {res_neg.status_code}"

    # 5. Missing currency (should return 422 Unprocessable Entity)
    res_no_curr = client.post("/start", json={
        **base_payload,
        "budget_resources": {"amount": 5000}
    })
    assert res_no_curr.status_code == 422, f"Expected 422 for missing currency, got {res_no_curr.status_code}"
