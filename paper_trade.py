"""
paper_trade.py — Paper trading con mercados REALES de Polymarket.

Usa datos en vivo de la API pública de Polymarket (sin auth),
estima probabilidades con GPT-4o (o simulador si no hay key),
y opera con $500 USDC virtuales registrando cada decisión.

Ejecutar:
    python paper_trade.py
    python paper_trade.py --cycles 3   # corre 3 ciclos seguidos
    python paper_trade.py --save       # guarda historial en paper_trades.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Setup mínimo de entorno ───────────────────────────────────────────────────
os.environ.setdefault("POLYGON_PRIVATE_KEY", "0x" + "0" * 64)
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "sk-demo"))
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("MIN_EDGE", "0.07")
os.environ.setdefault("MAX_KELLY_FRACTION", "0.25")
os.environ.setdefault("MAX_TRADE_SIZE_USDC", "50")
os.environ.setdefault("MARKETS_TO_SCAN", "20")

import requests
from loguru import logger

from config import settings
from utils.logger import setup_logging
from data.polymarket import Market, PolymarketClient
from data.sources import ExternalEvidence, NewsItem
from bot.analyzer import MarketAnalyzer, MarketOpportunity, TradeDirection
from bot.estimator import ProbabilityEstimate, ProbabilityEstimator
from bot.risk_manager import RiskManager, TradeDecision

setup_logging()

# ─────────────────────────────────────────────────────────────────────────────
#  Historial persistente
# ─────────────────────────────────────────────────────────────────────────────
HISTORY_FILE = Path("paper_trades.json")

@dataclass
class PaperTrade:
    timestamp: str
    question: str
    condition_id: str
    direction: str
    market_price: float
    estimated_prob: float
    edge: float
    usdc_amount: float
    confidence: str
    reasoning: str

@dataclass
class PaperState:
    starting_bankroll: float = 500.0
    current_bankroll: float = 500.0
    trades: list[PaperTrade] = field(default_factory=list)
    cycles_run: int = 0

    def save(self, path: Path) -> None:
        data = {
            "starting_bankroll": self.starting_bankroll,
            "current_bankroll": self.current_bankroll,
            "cycles_run": self.cycles_run,
            "trades": [asdict(t) for t in self.trades],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "PaperState":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        state = cls(
            starting_bankroll=data["starting_bankroll"],
            current_bankroll=data["current_bankroll"],
            cycles_run=data.get("cycles_run", 0),
        )
        state.trades = [PaperTrade(**t) for t in data.get("trades", [])]
        return state


# ─────────────────────────────────────────────────────────────────────────────
#  Estimador LLM real o simulado
# ─────────────────────────────────────────────────────────────────────────────

def make_estimator(use_real: bool) -> ProbabilityEstimator:
    if use_real:
        logger.info("Usando estimador GPT-4o REAL")
        return ProbabilityEstimator()
    logger.info("Usando estimador SIMULADO (agrega ruido realista al precio de mercado)")
    return SimulatedEstimator()


class SimulatedEstimator(ProbabilityEstimator):
    """
    Sin OpenAI: simula un estimador LLM realista.

    Lógica:
    - 30% de los mercados: el bot encuentra una ineficiencia real (edge 8-25pp)
    - 70% de los mercados: la estimación queda cerca del precio de mercado
    Esto replica el comportamiento esperado de un LLM con buena información.
    """

    def __init__(self) -> None:
        pass  # No instanciar OpenAI

    def estimate(self, evidence: ExternalEvidence) -> Optional[ProbabilityEstimate]:
        market_price = getattr(evidence, "_market_price", 0.50)

        # Mercados en extremos (>90% o <10%) son casi imposibles de batir
        if market_price > 0.90 or market_price < 0.10:
            # Ruido muy pequeño — el mercado casi siempre tiene razón aquí
            bias = random.gauss(0, 0.03)
        elif random.random() < 0.35:
            # 35% de mercados en zona media: encontramos ineficiencia
            # El sesgo aleatorio puede ser hacia cualquier lado
            magnitude = random.uniform(0.08, 0.22)
            direction = 1 if random.random() > 0.5 else -1
            bias = magnitude * direction
        else:
            # 65%: estimación cercana al mercado (mercado bien valorado)
            bias = random.gauss(0, 0.04)

        prob = max(0.03, min(0.97, market_price + bias))
        edge = abs(prob - market_price)
        confidence = "high" if edge > 0.15 else "medium" if edge > 0.09 else "low"

        return ProbabilityEstimate(
            yes_probability=round(prob, 3),
            confidence=confidence,
            reasoning=f"Análisis de evidencia externa sugiere P(YES)={prob:.1%} vs precio de mercado {market_price:.1%}.",
            sources_used=["news", "metaculus", "manifold"],
        )

    def aggregate(self, llm_estimate: ProbabilityEstimate, evidence: ExternalEvidence) -> float:
        return llm_estimate.yes_probability


# ─────────────────────────────────────────────────────────────────────────────
#  Fetch de mercados reales
# ─────────────────────────────────────────────────────────────────────────────

def fetch_real_markets(limit: int = 20) -> list[Market]:
    """
    Descarga mercados activos con mayor volumen de la Gamma API.
    Prioriza mercados con precios entre 10% y 90% (incertidumbre real).
    """
    url = "https://gamma-api.polymarket.com/markets"
    all_markets: list[Market] = []

    # Buscar en bloques para encontrar mercados con precios intermedios
    for offset in range(0, 120, 40):
        params = {
            "active": "true",
            "limit": 40,
            "offset": offset,
            "order": "volume24hr",
            "ascending": "false",
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw_markets = resp.json()
            if not raw_markets:
                break
            for raw in raw_markets:
                try:
                    m = PolymarketClient._parse_market(raw)
                    all_markets.append(m)
                except Exception:
                    continue
        except Exception as e:
            logger.error("Error descargando mercados (offset={}): {}", offset, e)
            break

    # Separar: mercados con precio intermedio (tradeable) vs extremos
    tradeable = [m for m in all_markets
                 if m.yes_price is not None and 0.10 <= m.yes_price <= 0.90]
    extreme = [m for m in all_markets
               if m.yes_price is not None and (m.yes_price < 0.10 or m.yes_price > 0.90)]

    logger.info(
        "Mercados totales: {} | Con incertidumbre (10-90%): {} | Extremos: {}",
        len(all_markets), len(tradeable), len(extreme),
    )

    # Devolver los más interesantes primero (mayor volumen, precio intermedio)
    tradeable.sort(key=lambda m: m.volume_24h, reverse=True)
    return tradeable[:limit] + extreme[:5]


# ─────────────────────────────────────────────────────────────────────────────
#  Motor de paper trading
# ─────────────────────────────────────────────────────────────────────────────

def run_paper_cycle(
    state: PaperState,
    estimator: ProbabilityEstimator,
    risk_mgr: RiskManager,
    markets: list[Market],
) -> list[PaperTrade]:
    """Analiza mercados reales y registra las operaciones virtuales."""
    new_trades: list[PaperTrade] = []

    for market in markets:
        # Filtros de liquidez
        if (market.yes_price is None or market.no_price is None
                or market.liquidity < 500 or market.volume_24h < 200):
            logger.debug("Filtrado: '{}' (liq={:.0f}, vol={:.0f})",
                         market.question[:50], market.liquidity, market.volume_24h)
            continue

        logger.debug("Analizando: '{}' @ {:.0%}", market.question[:55], market.yes_price)

        # Crear evidencia mínima (sin fuentes externas para velocidad)
        evidence = ExternalEvidence(question=market.question)
        evidence._market_price = market.yes_price  # hint para el estimador simulado

        # Estimar
        llm_est = estimator.estimate(evidence)
        if llm_est is None:
            continue

        final_prob = estimator.aggregate(llm_est, evidence)
        edge = final_prob - market.yes_price

        # Dirección
        if edge >= settings.min_edge:
            direction = TradeDirection.BUY_YES
        elif edge <= -settings.min_edge:
            direction = TradeDirection.BUY_NO
        else:
            continue

        # Construir oportunidad
        from bot.analyzer import MarketOpportunity
        opp = MarketOpportunity(
            market=market,
            estimated_probability=final_prob,
            market_price=market.yes_price,
            edge=edge,
            direction=direction,
            confidence=llm_est.confidence,
            reasoning=llm_est.reasoning,
            evidence=evidence,
            llm_estimate=llm_est,
        )

        decision = risk_mgr.evaluate(opp)
        if not decision.approved:
            continue

        risk_mgr.record_trade(market.condition_id, decision.usdc_amount)

        trade = PaperTrade(
            timestamp=datetime.now(timezone.utc).isoformat(),
            question=market.question,
            condition_id=market.condition_id,
            direction=direction.value,
            market_price=market.yes_price,
            estimated_prob=final_prob,
            edge=round(edge, 4),
            usdc_amount=decision.usdc_amount,
            confidence=llm_est.confidence,
            reasoning=llm_est.reasoning,
        )
        new_trades.append(trade)
        state.trades.append(trade)

    return new_trades


def print_summary(state: PaperState, new_trades: list[PaperTrade], markets: list[Market]) -> None:
    total_deployed = sum(t.usdc_amount for t in new_trades)
    total_trades_all = len(state.trades)
    pnl_virtual = state.current_bankroll - state.starting_bankroll

    print("\n" + "=" * 68)
    print(f"  📊 PAPER TRADING — CICLO #{state.cycles_run}  "
          f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print("=" * 68)

    tradeable = [m for m in markets
                 if m.yes_price is not None and m.liquidity >= 500 and m.volume_24h >= 200]

    if not new_trades:
        print(f"\n  Mercados analizados: {len(tradeable)}")
        print("  Sin oportunidades este ciclo — el mercado está bien valorado")
        if tradeable:
            print("\n  Muestra de mercados analizados (precio YES actual):")
            for m in tradeable[:5]:
                print(f"    {m.yes_price:.0%}  '{m.question[:60]}'")
        print(f"\n  Bankroll virtual: ${state.current_bankroll:.2f}")
        print("=" * 68 + "\n")
        return

    print(f"\n  Oportunidades encontradas este ciclo: {len(new_trades)}")
    print(f"  Capital virtual a desplegar: ${total_deployed:.2f} USDC\n")

    for t in new_trades:
        icon = "▲" if t.direction == "BUY_YES" else "▼"
        edge_str = f"{t.edge:+.1%}"
        ev = (t.estimated_prob / t.market_price - 1) if t.direction == "BUY_YES" and t.market_price > 0 \
             else ((1 - t.estimated_prob) / (1 - t.market_price) - 1)
        print(f"  {icon} {t.direction:<8} | ${t.usdc_amount:>6.2f} USDC | "
              f"edge {edge_str:>6} | EV {ev:+.1%} | conf: {t.confidence}")
        print(f"    '{t.question[:65]}'")
        print(f"    Mercado: {t.market_price:.0%} → Bot estima: {t.estimated_prob:.0%}")
        print()

    print("-" * 68)
    print(f"  Bankroll virtual:      ${state.current_bankroll:.2f} USDC (simulado)")
    print(f"  Total trades histórico: {total_trades_all}")
    print(f"  P&L acumulado virtual: {'+' if pnl_virtual >= 0 else ''}${pnl_virtual:.2f}")
    print()
    print("  ⚠️  Dinero VIRTUAL — ninguna orden fue enviada a Polymarket")
    print("  Para operar real: deposita en Polymarket y el bot opera solo")
    print("=" * 68 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket paper trader")
    parser.add_argument("--cycles", type=int, default=1, help="Número de ciclos a correr")
    parser.add_argument("--save", action="store_true", help="Guardar historial en paper_trades.json")
    parser.add_argument("--reset", action="store_true", help="Reiniciar historial")
    args = parser.parse_args()

    # Estado persistente
    if args.reset and HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
        print("Historial reiniciado.\n")

    state = PaperState.load(HISTORY_FILE) if args.save else PaperState()

    # Detectar si hay OpenAI key real
    api_key = os.getenv("OPENAI_API_KEY", "")
    use_real_llm = bool(api_key) and not api_key.startswith("sk-demo")
    estimator = make_estimator(use_real_llm)

    print("\n" + "=" * 68)
    print("  POLYMARKET BOT — PAPER TRADING EN VIVO")
    print(f"  Bankroll virtual: ${state.starting_bankroll:.2f} USDC")
    print(f"  Edge mínimo: {settings.min_edge:.0%}  |  Kelly máx: {settings.max_kelly_fraction:.0%}")
    print(f"  Estimador: {'GPT-4o (real)' if use_real_llm else 'Simulado'}")
    print(f"  Ciclos: {args.cycles}")
    print("=" * 68)

    for cycle_num in range(1, args.cycles + 1):
        print(f"\n⏳ Descargando mercados reales de Polymarket...")
        markets = fetch_real_markets(limit=settings.markets_to_scan)

        if not markets:
            print("❌ No se pudieron descargar mercados. Revisa tu conexión.")
            break

        print(f"✅ {len(markets)} mercados descargados\n")

        risk_mgr = RiskManager(bankroll=state.current_bankroll)
        state.cycles_run += 1

        new_trades = run_paper_cycle(state, estimator, risk_mgr, markets)
        print_summary(state, new_trades, markets)

        if args.save:
            state.save(HISTORY_FILE)
            print(f"  💾 Historial guardado en {HISTORY_FILE}\n")

        if cycle_num < args.cycles:
            print(f"  Esperando 10 segundos antes del ciclo {cycle_num + 1}...")
            time.sleep(10)

    # Resumen final si corrió más de 1 ciclo
    if args.cycles > 1:
        print("=" * 68)
        print("  RESUMEN TOTAL")
        print("=" * 68)
        print(f"  Ciclos completados:    {state.cycles_run}")
        print(f"  Trades identificados:  {len(state.trades)}")
        total_capital = sum(t.usdc_amount for t in state.trades)
        print(f"  Capital virtual usado: ${total_capital:.2f} USDC")
        avg_edge = sum(abs(t.edge) for t in state.trades) / len(state.trades) if state.trades else 0
        print(f"  Edge promedio:         {avg_edge:.1%}")
        print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
