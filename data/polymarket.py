"""
Polymarket API client.

Wraps both:
 - Gamma REST API (public market data, no auth)
 - CLOB API         (order placement, requires Polygon key)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from loguru import logger

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        ApiCreds,
        MarketOrderArgs,
        OrderArgs,
        OrderType,
    )
    from py_clob_client.constants import POLYGON
    _CLOB_AVAILABLE = True
except ImportError:
    _CLOB_AVAILABLE = False
    ClobClient = None  # type: ignore[assignment,misc]

from config import settings


# ── Domain models ──────────────────────────────────────────────────────────────

@dataclass
class Market:
    condition_id: str
    question: str
    description: str
    category: str
    end_date: str
    active: bool
    # YES token price (0-1)  — None if no liquidity
    yes_price: Optional[float]
    no_price: Optional[float]
    volume_24h: float
    liquidity: float

    @property
    def spread(self) -> Optional[float]:
        if self.yes_price is not None and self.no_price is not None:
            return abs(1.0 - self.yes_price - self.no_price)
        return None


@dataclass
class Position:
    condition_id: str
    outcome: str          # "YES" or "NO"
    size: float
    entry_price: float
    current_price: float

    @property
    def pnl(self) -> float:
        return self.size * (self.current_price - self.entry_price)


# ── Polymarket client ───────────────────────────────────────────────────────────

class PolymarketClient:
    """
    Thin wrapper around the py-clob-client and Gamma REST API.
    Handles authentication, retries, and exposes a clean interface.
    """

    GAMMA = settings.gamma_host
    _RETRY_ATTEMPTS = 3
    _RETRY_DELAY = 1.5  # seconds

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "polymarket-bot/1.0"})
        self._clob: Optional[ClobClient] = None

    # ── CLOB (authenticated) ───────────────────────────────────────────────────

    def connect(self) -> None:
        """Initialise the CLOB client and derive/set API credentials."""
        if not _CLOB_AVAILABLE:
            raise RuntimeError(
                "py_clob_client is not installed. Run: pip install py-clob-client"
            )
        private_key = settings.polygon_private_key.get_secret_value()
        self._clob = ClobClient(
            host=settings.clob_host,
            chain_id=POLYGON,
            key=private_key,
        )

        # Use stored creds only if all three fields are present; otherwise derive
        _api_key = settings.polymarket_api_key
        _api_secret = settings.polymarket_api_secret.get_secret_value()
        _api_passphrase = settings.polymarket_api_passphrase.get_secret_value()
        if _api_key and _api_secret and _api_passphrase:
            creds = ApiCreds(
                api_key=_api_key,
                api_secret=_api_secret,
                api_passphrase=_api_passphrase,
            )
            self._clob.set_api_creds(creds)
            logger.info("Using stored API credentials (key: {}...)", _api_key[:8])
        else:
            logger.info("API secret/passphrase not set — deriving credentials from private key…")
            derived = self._clob.derive_api_key()
            creds = ApiCreds(
                api_key=derived.api_key,
                api_secret=derived.api_secret,
                api_passphrase=derived.api_passphrase,
            )
            self._clob.set_api_creds(creds)
            logger.info("Derived API key: {}...", derived.api_key[:8])

        logger.info("CLOB client connected. Address: {}", self._clob.get_address())

    @property
    def clob(self) -> ClobClient:
        if self._clob is None:
            raise RuntimeError("Call connect() before using the CLOB client.")
        return self._clob

    # ── Market data (Gamma, public) ────────────────────────────────────────────

    def get_markets(
        self,
        limit: int = 50,
        active: bool = True,
        category: Optional[str] = None,
    ) -> list[Market]:
        """Fetch open markets sorted by 24-h volume (descending)."""
        params: dict = {
            "active": str(active).lower(),
            "limit": limit,
            "order": "volume24hr",
            "ascending": "false",
        }
        if category:
            params["category"] = category

        raw = self._get(f"{self.GAMMA}/markets", params=params)
        markets: list[Market] = []
        for m in raw:
            try:
                markets.append(self._parse_market(m))
            except Exception as exc:
                logger.warning("Could not parse market {}: {}", m.get("conditionId"), exc)
        return markets

    def get_market(self, condition_id: str) -> Optional[Market]:
        """Fetch a single market by its condition ID."""
        raw = self._get(f"{self.GAMMA}/markets/{condition_id}")
        if not raw:
            return None
        return self._parse_market(raw)

    def get_order_book(self, token_id: str) -> dict:
        """Return the raw order book for a token (YES or NO)."""
        return self.clob.get_order_book(token_id)

    def get_usdc_balance(self) -> float:
        """Return the available USDC balance. Tries multiple API methods for compatibility."""
        # Try different method names across py-clob-client versions
        for method_name in ("get_balance", "get_collateral_balance", "get_usdc_balance"):
            method = getattr(self._clob, method_name, None)
            if method is None:
                continue
            try:
                result = method()
                if isinstance(result, dict):
                    for key in ("balance", "collateral_balance", "usdc", "amount"):
                        if key in result:
                            return float(result[key])
                if isinstance(result, (int, float, str)):
                    return float(result)
            except Exception as exc:
                logger.debug("Method {} failed: {}", method_name, exc)

        # Fallback: query the Gamma API for portfolio balance (public endpoint)
        try:
            address = self.clob.get_address()
            resp = self._session.get(
                f"https://data-api.polymarket.com/value?user={address}",
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                # API may return a list — take the first element
                if isinstance(data, list):
                    data = data[0] if data else {}
                if isinstance(data, dict):
                    return float(data.get("portfolioValue", data.get("value", 0)))
        except Exception as exc:
            logger.warning("Balance fallback also failed: {}", exc)

        logger.warning("Could not retrieve balance — assuming $0. Bot will operate in dry-run safety mode.")
        return 0.0

    # ── Order placement ────────────────────────────────────────────────────────

    def place_market_order(
        self,
        token_id: str,
        side: str,          # "BUY" or "SELL"
        amount_usdc: float,
    ) -> dict:
        """
        Place a market order.
        `token_id` is the conditional token ID for YES or NO.
        `amount_usdc` is the USDC amount to spend.
        """
        if settings.dry_run:
            logger.info("[DRY-RUN] Would place {} market order: token={} amount={:.2f} USDC",
                        side, token_id, amount_usdc)
            return {"dry_run": True, "side": side, "amount": amount_usdc}

        args = MarketOrderArgs(
            token_id=token_id,
            amount=amount_usdc,
            side=side.upper(),
        )
        signed = self.clob.create_market_order(args)
        result = self.clob.post_order(signed, OrderType.FOK)
        logger.info("Order placed: {}", result)
        return result

    def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> dict:
        """
        Place a GTC limit order.
        `price` is the probability (0-1).
        `size`  is the number of shares (= USDC / price for YES buys).
        """
        if settings.dry_run:
            logger.info("[DRY-RUN] Would place {} limit order: token={} price={:.3f} size={:.2f}",
                        side, token_id, price, size)
            return {"dry_run": True, "side": side, "price": price, "size": size}

        args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
        )
        signed = self.clob.create_order(args)
        result = self.clob.post_order(signed, OrderType.GTC)
        logger.info("Limit order placed: {}", result)
        return result

    # ── Positions & open orders ────────────────────────────────────────────────

    def get_open_orders(self) -> list[dict]:
        return self.clob.get_orders()

    def cancel_order(self, order_id: str) -> dict:
        if settings.dry_run:
            logger.info("[DRY-RUN] Would cancel order {}", order_id)
            return {"dry_run": True}
        return self.clob.cancel(order_id)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get(self, url: str, params: Optional[dict] = None) -> dict | list:
        for attempt in range(1, self._RETRY_ATTEMPTS + 1):
            try:
                resp = self._session.get(url, params=params, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                logger.warning("GET {} failed (attempt {}/{}): {}", url, attempt, self._RETRY_ATTEMPTS, exc)
                if attempt < self._RETRY_ATTEMPTS:
                    time.sleep(self._RETRY_DELAY)
        return {}

    @staticmethod
    def _parse_market(raw: dict) -> Market:
        yes_price = no_price = None

        # Format 1: tokens list (CLOB endpoint)
        tokens = raw.get("tokens", [])
        if tokens:
            for t in tokens:
                outcome = str(t.get("outcome", "")).upper()
                price = t.get("price")
                if price is not None:
                    price = float(price)
                if outcome == "YES":
                    yes_price = price
                elif outcome == "NO":
                    no_price = price

        # Format 2: outcomes + outcomePrices strings (Gamma REST endpoint)
        if yes_price is None:
            import json as _json
            outcomes_raw = raw.get("outcomes", "[]")
            prices_raw = raw.get("outcomePrices", "[]")
            try:
                outcomes = _json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
                prices = _json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                for outcome, price in zip(outcomes, prices):
                    outcome_upper = str(outcome).upper()
                    if outcome_upper in ("YES", "Y"):
                        yes_price = float(price)
                    elif outcome_upper in ("NO", "N"):
                        no_price = float(price)
            except Exception:
                pass

        return Market(
            condition_id=raw.get("conditionId", raw.get("id", "")),
            question=raw.get("question", ""),
            description=raw.get("description", ""),
            category=raw.get("category", ""),
            end_date=raw.get("endDate", ""),
            active=raw.get("active", True),
            yes_price=yes_price,
            no_price=no_price,
            volume_24h=float(raw.get("volume24hr", 0) or 0),
            liquidity=float(raw.get("liquidity", 0) or 0),
        )
