"""Telco scenario tool handlers for the voice agent.

Mock implementations that return fixture data for demo purposes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCENARIO_PATH = Path(__file__).parent.parent.parent / "scenarios" / "telco_th.json"

_fixtures: dict[str, Any] = {}


def set_scenario_path(path: Path) -> None:
    """Point the fixtures at a specific scenario file (called at agent startup)."""
    global SCENARIO_PATH, _fixtures
    SCENARIO_PATH = Path(path)
    _fixtures = {}


def load_fixtures() -> dict[str, Any]:
    global _fixtures
    if not _fixtures:
        with SCENARIO_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        _fixtures = data["fixtures"]
    return _fixtures


def check_balance(customer_id: str) -> dict[str, Any]:
    """Return the customer's outstanding bill balance and due date."""
    fixtures = load_fixtures()
    customers = fixtures.get("customers", {})
    if customer_id not in customers:
        return {"error": "customer_not_found", "customer_id": customer_id}
    bill = fixtures.get("bills", {}).get(customer_id)
    if not bill:
        return {"error": "no_bill_on_file", "customer_id": customer_id}
    return {
        "customer_id": customer_id,
        "amount_thb": bill["amount_thb"],
        "due_date": bill["due_date"],
        "overdue": bill.get("overdue", False),
    }


def check_plan(customer_id: str) -> dict[str, Any]:
    """Return the customer's current mobile plan details."""
    fixtures = load_fixtures()
    customers = fixtures.get("customers", {})
    customer = customers.get(customer_id)
    if not customer:
        return {"error": "customer_not_found", "customer_id": customer_id}
    plan_id = customer.get("plan_id")
    plan = fixtures.get("plans", {}).get(plan_id)
    if not plan:
        return {"error": "plan_not_found", "customer_id": customer_id}
    return {
        "customer_id": customer_id,
        "plan_name": plan["name"],
        "monthly_fee_thb": plan["monthly_fee_thb"],
        "voice_minutes": plan["voice_minutes"],
        "data_gb": plan["data_gb"],
    }


def check_usage(customer_id: str) -> dict[str, Any]:
    """Return the customer's month-to-date usage and remaining quota."""
    fixtures = load_fixtures()
    customers = fixtures.get("customers", {})
    customer = customers.get(customer_id)
    if not customer:
        return {"error": "customer_not_found", "customer_id": customer_id}
    usage = fixtures.get("usage_mtd", {}).get(customer_id)
    if not usage:
        return {"error": "no_usage_on_file", "customer_id": customer_id}
    plan = fixtures.get("plans", {}).get(customer.get("plan_id"), {})
    return {
        "customer_id": customer_id,
        "voice_minutes_used": usage["voice_minutes_used"],
        "voice_minutes_remaining": max(
            0, plan.get("voice_minutes", 0) - usage["voice_minutes_used"]
        ),
        "data_gb_used": usage["data_gb_used"],
        "data_gb_remaining": max(0, plan.get("data_gb", 0) - usage["data_gb_used"]),
    }


def open_ticket(customer_id: str, category: str, summary: str) -> dict[str, Any]:
    """Open a support ticket and return the ticket ID."""
    fixtures = load_fixtures()
    customers = fixtures.get("customers", {})
    if customer_id not in customers:
        return {"error": "customer_not_found", "customer_id": customer_id}
    ticket_id = "AB-112"
    return {
        "customer_id": customer_id,
        "ticket_id": ticket_id,
        "category": category,
        "summary": summary,
        "status": "open",
    }
