"""
Trade executor.

Translates a TradeDecision into actual CLOB orders via the Polymarket client.
"""

from __future__ import annotations

from loguru import logger

from bot.analyzer import TradeDirection
from bot.risk_manager import TradeDecision
from data.polymarket import PolymarketClient


class TradeExecutor:
    """
    Turns approved TradeDecisions into market orders on Polymarket.
    """

    def __init__(self, client: PolymarketClient) -> None:
        self._client = client

    def execute(self, decision: TradeDecision) -> bool:
        """
        Place the trade.  Returns True if the order was submitted successfully
        (or in dry-run mode), False otherwise.
        """
        if not decision.approved:
            logger.debug("Skipping unapproved decision: {}", decision.rejection_reason)
            return False

        opp = decision.opportunity
        market = opp.market
        direction = opp.direction

        # Find the correct token ID for YES or NO
        token_id = self._get_token_id(market.condition_id, direction)
        if not token_id:
            logger.error("Could not find token ID for market {}", market.condition_id)
            return False

        side = "BUY"  # We always buy the mispriced side (YES or NO)

        logger.info(
            "Executing {} {} — {:.2f} USDC — market price {:.1%} — estimated {:.1%}",
            direction.value,
            market.question[:60],
            decision.usdc_amount,
            opp.market_price,
            opp.estimated_probability,
        )

        try:
            result = self._client.place_market_order(
                token_id=token_id,
                side=side,
                amount_usdc=decision.usdc_amount,
            )
            logger.info("Order result: {}", result)
            return True
        except Exception as exc:
            logger.error("Order failed for '{}': {}", market.question[:60], exc)
            return False

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_token_id(self, condition_id: str, direction: TradeDirection) -> str | None:
        """
        Look up the conditional token ID for the YES or NO outcome.
        The CLOB uses separate token IDs for each outcome.
        """
        try:
            market_data = self._client.clob.get_market(condition_id)
            tokens = market_data.get("tokens", [])
            target_outcome = "YES" if direction == TradeDirection.BUY_YES else "NO"
            for token in tokens:
                if str(token.get("outcome", "")).upper() == target_outcome:
                    return token.get("token_id")
        except Exception as exc:
            logger.error("Could not fetch token ID for {}: {}", condition_id, exc)
        return None
