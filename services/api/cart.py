from __future__ import annotations
"""
services/api/cart.py

UCP-compliant checkout using Pydantic models that mirror the real
Universal Commerce Protocol schema (https://ucp.dev/latest/).

Schema source: Google/Shopify UCP spec 2026-01-11
  - dev.ucp.shopping.checkout
  - dev.ucp.mock_payment  (test payment handler — no real charges)

To swap in the official SDK later:
    pip install ucp-python-sdk          (once published to PyPI)
    from ucp import CheckoutSession, LineItem, Order
    and delete the Pydantic model block below.
"""
import uuid, logging
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

UCP_VERSION = "2026-01-11"


# ═══════════════════════════════════════════════════════════════════════════════
# UCP Schema — Pydantic models that mirror the real UCP JSON structure
# ═══════════════════════════════════════════════════════════════════════════════

class UCPCapability(BaseModel):
    name: str    = "dev.ucp.shopping.checkout"
    version: str = UCP_VERSION


class UCPMeta(BaseModel):
    version:      str              = UCP_VERSION
    capabilities: list[UCPCapability] = Field(default_factory=lambda: [UCPCapability()])


class UCPItem(BaseModel):
    """Catalogue item reference inside a line item."""
    id:    str
    title: str
    price: int   # in cents — e.g. 1999 = €19.99


class UCPTotal(BaseModel):
    type:   Literal["subtotal", "tax", "shipping", "discount", "total"]
    amount: int  # cents


class LineItem(BaseModel):
    id:       str = Field(default_factory=lambda: f"li_{uuid.uuid4().hex[:9]}")
    item:     UCPItem
    quantity: int = 1
    totals:   list[UCPTotal] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.totals:
            subtotal = self.item.price * self.quantity
            self.totals = [
                UCPTotal(type="subtotal", amount=subtotal),
                UCPTotal(type="total",    amount=subtotal),
            ]


class UCPBuyer(BaseModel):
    full_name: str  = "Demo User"
    email:     str  = "demo@example.com"


class MockPaymentHandler(BaseModel):
    """dev.ucp.mock_payment — test handler, no real charges."""
    id:               str       = "mock_payment_handler"
    name:             str       = "dev.ucp.mock_payment"
    version:          str       = UCP_VERSION
    supported_tokens: list[str] = Field(
        default_factory=lambda: ["success_token", "fail_token"]
    )


class GooglePayHandler(BaseModel):
    """google.pay handler config (test environment)."""
    id:      str = "google_pay"
    name:    str = "google.pay"
    version: str = UCP_VERSION
    config: dict = Field(default_factory=lambda: {
        "api_version":       2,
        "api_version_minor": 0,
        "merchant_info": {
            "merchant_name":   "Demo Store",
            "merchant_id":     "TEST",
            "merchant_origin": "localhost",
        },
        "allowed_payment_methods": [{
            "type": "CARD",
            "parameters": {
                "allowedAuthMethods":  ["PAN_ONLY", "CRYPTOGRAM_3DS"],
                "allowedCardNetworks": ["VISA", "MASTERCARD"],
            },
        }],
    })


class UCPPayment(BaseModel):
    handlers:    list[Any] = Field(default_factory=lambda: [
        MockPaymentHandler(), GooglePayHandler()
    ])
    instruments: list[Any] = Field(default_factory=list)


class CheckoutSession(BaseModel):
    """Full UCP CheckoutSession — spec: dev.ucp.shopping.checkout"""
    ucp:        UCPMeta          = Field(default_factory=UCPMeta)
    id:         str              = Field(default_factory=lambda: f"chk_{uuid.uuid4().hex[:9]}")
    status:     str              = "ready_for_complete"
    currency:   str              = "EUR"
    line_items: list[LineItem]   = Field(default_factory=list)
    buyer:      UCPBuyer         = Field(default_factory=UCPBuyer)
    totals:     list[UCPTotal]   = Field(default_factory=list)
    payment:    UCPPayment       = Field(default_factory=UCPPayment)
    discounts:  dict             = Field(default_factory=dict)
    links:      list             = Field(default_factory=list)

    def recalculate_totals(self) -> None:
        subtotal = sum(
            li.item.price * li.quantity for li in self.line_items
        )
        self.totals = [
            UCPTotal(type="subtotal", amount=subtotal),
            UCPTotal(type="total",    amount=subtotal),
        ]


class Order(BaseModel):
    """Completed UCP order — returned after checkout completion."""
    id:          str
    checkout_id: str
    line_items:  list[LineItem]
    status:      str  = "confirmed"
    total:       float = 0.0        # in major currency units (€)
    payment_method: str = "dev.ucp.mock_payment"
    test_token_used: str = "success_token"


# ═══════════════════════════════════════════════════════════════════════════════
# In-memory session store
# ═══════════════════════════════════════════════════════════════════════════════
_sessions: dict[str, CheckoutSession] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — used by chatbot.py (via @tool) and main.py (REST endpoints)
# ═══════════════════════════════════════════════════════════════════════════════

def create_checkout_session(
    product_id:  str,
    title:       str,
    price_cents: int,
    quantity:    int = 1,
    buyer_name:  str = "Demo User",
    buyer_email: str = "demo@example.com",
) -> CheckoutSession:
    """Create a new UCP checkout session with one line item."""
    session = CheckoutSession(
        buyer=UCPBuyer(full_name=buyer_name, email=buyer_email),
        currency="EUR",
        line_items=[
            LineItem(
                item=UCPItem(id=product_id, title=title, price=price_cents),
                quantity=quantity,
            )
        ],
    )
    session.recalculate_totals()
    _sessions[session.id] = session
    logger.info("UCP session created: %s | %s × %d | €%.2f",
                session.id, title, quantity, price_cents * quantity / 100)
    return session


def add_line_item(
    session_id:  str,
    product_id:  str,
    title:       str,
    price_cents: int,
    quantity:    int = 1,
) -> CheckoutSession:
    """Append a line item to an existing checkout session."""
    session = _sessions[session_id]
    # Increment quantity if product already in session
    for li in session.line_items:
        if li.item.id == product_id:
            li.quantity += quantity
            li.model_post_init(None)   # recalculate line totals
            session.recalculate_totals()
            return session
    session.line_items.append(
        LineItem(
            item=UCPItem(id=product_id, title=title, price=price_cents),
            quantity=quantity,
        )
    )
    session.recalculate_totals()
    return session


def get_session(session_id: str) -> CheckoutSession | None:
    return _sessions.get(session_id)


def complete_checkout(
    session_id:   str,
    payment_token: str = "success_token",
) -> Order:
    """
    Complete a UCP checkout session.
    payment_token:
      'success_token' → confirmed  (dev.ucp.mock_payment)
      'fail_token'    → raises ValueError
    In production: forward the Google Pay token to your payment processor.
    """
    if payment_token == "fail_token":
        raise ValueError("Payment declined (fail_token used in test mode)")

    session = _sessions.pop(session_id, None)
    if session is None:
        raise KeyError(f"Checkout session '{session_id}' not found")

    total_cents = sum(li.item.price * li.quantity for li in session.line_items)
    order = Order(
        id=f"ord_{uuid.uuid4().hex[:9]}",
        checkout_id=session_id,
        line_items=session.line_items,
        status="confirmed",
        total=round(total_cents / 100, 2),
        payment_method="dev.ucp.mock_payment",
        test_token_used=payment_token,
    )
    logger.info("UCP order confirmed: %s | €%.2f", order.id, order.total)
    return order
