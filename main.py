"""
Main bot orchestrator.

Starts the scan-analyse-trade loop that runs indefinitely.
"""

from __future__ import annotations

import sys
import time

import schedule
from loguru import logger

from config import settings
from data.polymarket import PolymarketClient
from bot.analyzer import MarketAnalyzer
from bot.risk_manager import RiskManager
from bot.trader import TradeExecutor
from utils.logger import setup_logging


# ── Bot ─────────────────────────────────────────────────────────────────────────

class PolymarketBot:
    """
    Top-level orchestrator.

    Each cycle:
      1. Fetch the top N markets by 24-h volume
      2. For each market, run the full analysis pipeline
      3. Approve/size trades through the risk manager
      4. Execute approved trades
    """

    def __init__(self) -> None:
        self._poly = PolymarketClient()
        self._analyzer = MarketAnalyzer()
        self._risk: RiskManager | None = None   # initialised after connecting
        self._executor: TradeExecutor | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        setup_logging()
        logger.info("=" * 60)
        logger.info("Polymarket Bot starting  [dry_run={}]", settings.dry_run)
        logger.info("=" * 60)

        # Connect to Polymarket CLOB
        self._poly.connect()

        # Initialise risk manager with current USDC balance
        bankroll = self._poly.get_usdc_balance()
        logger.info("USDC balance: ${:.2f}", bankroll)

        if bankroll < 5.0 and not settings.dry_run:
            logger.error("Insufficient balance (${:.2f}). Top up your wallet.", bankroll)
            return

        # Use a minimum $100 bankroll for Kelly calculations in dry-run
        effective_bankroll = max(bankroll, 100.0) if settings.dry_run else bankroll
        self._risk = RiskManager(bankroll=effective_bankroll)
        self._executor = TradeExecutor(client=self._poly)

        # ── Modo de ejecución ──────────────────────────────────────────────
        if settings.single_cycle:
            # GitHub Actions: un solo ciclo y salir limpiamente
            logger.info("SINGLE_CYCLE=true — running one cycle then exiting.")
            self._run_cycle()
            logger.info("Single cycle complete. Exiting.")
            return

        # Servidor/VPS: bucle continuo
        self._run_cycle()
        schedule.every(settings.scan_interval_seconds).seconds.do(self._run_cycle)

        logger.info(
            "Scheduled: scanning every {} seconds. Press Ctrl+C to stop.",
            settings.scan_interval_seconds,
        )
        try:
            while True:
                schedule.run_pending()
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")

    # ── Scan cycle ─────────────────────────────────────────────────────────────

    def _run_cycle(self) -> None:
        logger.info("─" * 50)
        logger.info("Starting scan cycle …")

        try:
            markets = self._poly.get_markets(limit=settings.markets_to_scan)
        except Exception as exc:
            logger.error("Failed to fetch markets: {}", exc)
            return

        logger.info("Fetched {} markets to analyse.", len(markets))
        opportunities_found = 0

        for market in markets:
            try:
                opp = self._analyzer.analyze(market)
                if opp is None:
                    continue

                opportunities_found += 1

                decision = self._risk.evaluate(opp)  # type: ignore[union-attr]
                if not decision.approved:
                    logger.info(
                        "Trade rejected for '{}': {}",
                        market.question[:60],
                        decision.rejection_reason,
                    )
                    continue

                success = self._executor.execute(decision)  # type: ignore[union-attr]
                if success:
                    self._risk.record_trade(  # type: ignore[union-attr]
                        market.condition_id, decision.usdc_amount
                    )
            except Exception as exc:
                logger.error("Error processing market '{}': {}", market.question[:60], exc)

            # Rate-limit: avoid hammering the LLM API
            time.sleep(1)

        logger.info(
            "Cycle complete. Opportunities found: {}. Next scan in {} s.",
            opportunities_found,
            settings.scan_interval_seconds,
        )


# ── Entrypoint ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = PolymarketBot()
    bot.start()
    sys.exit(0)
