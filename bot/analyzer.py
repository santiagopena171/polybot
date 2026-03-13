"""
Market inefficiency detector (the "edge finder").

Compares our estimated probability against the Polymarket price to
detect mispricings worth trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger

from config import settings
from data.polymarket import Market
from data.sources import DataAggregator, ExternalEvidence
from bot.estimator import ProbabilityEstimate, ProbabilityEstimator


class TradeDirection(str, Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    NO_TRADE = "NO_TRADE"


@dataclass
class MarketOpportunity:
    market: Market
    estimated_probability: float   # our best estimate of YES resolving
    market_price: float            # current Polymarket YES price
    edge: float                    # estimated_probability - market_price (signed)
    direction: TradeDirection
    confidence: str
    reasoning: str
    evidence: ExternalEvidence
    llm_estimate: ProbabilityEstimate

    @property
    def abs_edge(self) -> float:
        return abs(self.edge)

    @property
    def expected_value(self) -> float:
        """
        EV of a $1 bet in the direction of our edge.
        BUY_YES: payout = 1/price, cost = 1 → EV = est_prob/price - 1
        BUY_NO:  payout = 1/(1-price), cost = 1 → EV = (1-est_prob)/(1-price) - 1
        """
        if self.direction == TradeDirection.BUY_YES and self.market_price > 0:
            return self.estimated_probability / self.market_price - 1.0
        if self.direction == TradeDirection.BUY_NO and (1 - self.market_price) > 0:
            return (1 - self.estimated_probability) / (1 - self.market_price) - 1.0
        return 0.0

    def __str__(self) -> str:
        return (
            f"[{self.direction.value}] '{self.market.question[:60]}'\n"
            f"  market={self.market_price:.1%}  estimate={self.estimated_probability:.1%}  "
            f"edge={self.edge:+.1%}  EV={self.expected_value:+.1%}  conf={self.confidence}"
        )


class MarketAnalyzer:
    """
    For each market:
      1. Gather external evidence
      2. Estimate true probability with the LLM
      3. Compare with market price
      4. Return an opportunity if edge ≥ MIN_EDGE
    """

    # Minimum liquidity (USDC) required to even consider a market
    MIN_LIQUIDITY = 500.0
    # Minimum 24-h volume required
    MIN_VOLUME_24H = 200.0
    # Maximum spread tolerated (avoids illiquid, wide-spread markets)
    MAX_SPREAD = 0.05

    def __init__(self) -> None:
        self._aggregator = DataAggregator()
        self._estimator = ProbabilityEstimator()

    def analyze(self, market: Market) -> Optional[MarketOpportunity]:
        """
        Fully analyse one market.  Returns None if:
          - The market fails quality filters, or
          - Evidence / LLM call fails, or
          - The edge is below MIN_EDGE.
        """
        if not self._passes_filters(market):
            return None

        # Collect external evidence
        evidence = self._aggregator.collect(market.question, market.description)

        # Get LLM probability estimate
        llm_est = self._estimator.estimate(evidence)
        if llm_est is None:
            return None

        # Blended final probability
        final_prob = self._estimator.aggregate(llm_est, evidence)

        market_price = market.yes_price  # type: ignore[assignment]
        edge = final_prob - market_price

        direction = self._direction(edge)
        if direction == TradeDirection.NO_TRADE:
            logger.debug(
                "No trade for '{}': edge={:+.1%} < threshold={:.1%}",
                market.question[:60], edge, settings.min_edge
            )
            return None

        opp = MarketOpportunity(
            market=market,
            estimated_probability=final_prob,
            market_price=market_price,
            edge=edge,
            direction=direction,
            confidence=llm_est.confidence,
            reasoning=llm_est.reasoning,
            evidence=evidence,
            llm_estimate=llm_est,
        )
        logger.info("Opportunity found: {}", opp)
        return opp

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _passes_filters(self, market: Market) -> bool:
        """Pre-filters to skip markets that are obviously untradeable."""
        if market.yes_price is None or market.no_price is None:
            logger.debug("Skip '{}': no price data", market.question[:60])
            return False

        if market.liquidity < self.MIN_LIQUIDITY:
            logger.debug("Skip '{}': liquidity too low ({:.0f})", market.question[:60], market.liquidity)
            return False

        if market.volume_24h < self.MIN_VOLUME_24H:
            logger.debug("Skip '{}': 24h volume too low ({:.0f})", market.question[:60], market.volume_24h)
            return False

        spread = market.spread
        if spread is not None and spread > self.MAX_SPREAD:
            logger.debug("Skip '{}': spread too wide ({:.1%})", market.question[:60], spread)
            return False

        return True

    def _direction(self, edge: float) -> TradeDirection:
        min_edge = settings.min_edge
        if edge >= min_edge:
            return TradeDirection.BUY_YES
        if edge <= -min_edge:
            return TradeDirection.BUY_NO
        return TradeDirection.NO_TRADE
