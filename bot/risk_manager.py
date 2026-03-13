"""
Risk manager.

Implements:
  - Fractional Kelly position sizing
  - Per-market exposure limits
  - Portfolio-level exposure cap
  - Trade deduplication (don't re-enter the same market twice)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from config import settings
from bot.analyzer import MarketOpportunity, TradeDirection


@dataclass
class TradeDecision:
    opportunity: MarketOpportunity
    usdc_amount: float           # Amount to risk in USDC
    approved: bool
    rejection_reason: Optional[str] = None


class RiskManager:
    """
    Given a MarketOpportunity and the current portfolio state, decides
    whether to trade and how much.
    """

    def __init__(self, bankroll: float) -> None:
        self._bankroll = bankroll
        self._active_condition_ids: set[str] = set()
        self._total_deployed: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self, opportunity: MarketOpportunity) -> TradeDecision:
        cid = opportunity.market.condition_id

        # 1. Deduplication — never hold two positions in the same market
        if cid in self._active_condition_ids:
            return TradeDecision(
                opportunity=opportunity,
                usdc_amount=0.0,
                approved=False,
                rejection_reason="Already have a position in this market",
            )

        # 2. Kelly position sizing
        usdc_amount = self._kelly_size(opportunity)
        if usdc_amount <= 0:
            return TradeDecision(
                opportunity=opportunity,
                usdc_amount=0.0,
                approved=False,
                rejection_reason="Kelly criterion returned zero or negative size",
            )

        # 3. Hard cap per trade
        usdc_amount = min(usdc_amount, settings.max_trade_size_usdc)

        # 4. Portfolio-level cap: don't deploy more than 50% of bankroll at once
        portfolio_cap = self._bankroll * 0.50
        if self._total_deployed + usdc_amount > portfolio_cap:
            usdc_amount = max(0.0, portfolio_cap - self._total_deployed)
            if usdc_amount < 1.0:
                return TradeDecision(
                    opportunity=opportunity,
                    usdc_amount=0.0,
                    approved=False,
                    rejection_reason="Portfolio cap reached",
                )

        # 5. Minimum trade size sanity check
        if usdc_amount < 1.0:
            return TradeDecision(
                opportunity=opportunity,
                usdc_amount=0.0,
                approved=False,
                rejection_reason="Trade size below $1 minimum",
            )

        logger.info(
            "RiskManager approved: {} {:.2f} USDC for '{}'",
            opportunity.direction.value,
            usdc_amount,
            opportunity.market.question[:60],
        )
        return TradeDecision(
            opportunity=opportunity,
            usdc_amount=round(usdc_amount, 2),
            approved=True,
        )

    def record_trade(self, condition_id: str, usdc_amount: float) -> None:
        """Call after a trade is successfully placed."""
        self._active_condition_ids.add(condition_id)
        self._total_deployed += usdc_amount

    def remove_position(self, condition_id: str, usdc_amount: float) -> None:
        """Call when a position is closed."""
        self._active_condition_ids.discard(condition_id)
        self._total_deployed = max(0.0, self._total_deployed - usdc_amount)

    # ── Kelly math ─────────────────────────────────────────────────────────────

    def _kelly_size(self, opp: MarketOpportunity) -> float:
        """
        Fractional Kelly criterion.

        For a binary bet:
          b   = net odds (payout per $1 wagered minus $1)
          p   = our estimated probability of winning
          q   = 1 - p
          f*  = (b*p - q) / b   (full Kelly fraction of bankroll)

        We use FRACTIONAL Kelly (multiply by max_kelly_fraction) to reduce
        variance significantly.
        """
        p = opp.estimated_probability
        market_price = opp.market_price

        if opp.direction == TradeDirection.BUY_YES:
            # b = (1 / market_price) - 1
            if market_price <= 0 or market_price >= 1:
                return 0.0
            b = (1.0 / market_price) - 1.0
            q = 1.0 - p
        elif opp.direction == TradeDirection.BUY_NO:
            # Buying NO: our "win probability" is (1-p), price is (1-market_price)
            p = 1.0 - p
            no_price = 1.0 - market_price
            if no_price <= 0 or no_price >= 1:
                return 0.0
            b = (1.0 / no_price) - 1.0
            q = 1.0 - p
        else:
            return 0.0

        kelly_fraction = (b * p - q) / b
        if kelly_fraction <= 0:
            return 0.0

        fractional_kelly = kelly_fraction * settings.max_kelly_fraction
        return self._bankroll * fractional_kelly
