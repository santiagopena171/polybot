"""
demo.py — Simulación completa del bot sin APIs reales.

Ejecutar con:
    python demo.py

Lo que hace:
  1. Crea 12 mercados ficticios con precios simulados de Polymarket
  2. Simula evidencia externa (noticias, Metaculus, Manifold)
  3. Usa el estimador LLM (si hay OPENAI_API_KEY) o un estimador simulado
  4. Pasa todo por el risk manager y el ejecutor (dry-run)
  5. Resuelve cada mercado y calcula la P&L final
  6. Muestra un resumen detallado en consola
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ── Configuración mínima para la demo (sin .env) ─────────────────────────────
# Ponemos valores dummy ANTES de importar config para evitar errores de validación
os.environ.setdefault("POLYGON_PRIVATE_KEY", "0x" + "0" * 64)
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "sk-demo-not-real"))
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("MIN_EDGE", "0.07")
os.environ.setdefault("MAX_KELLY_FRACTION", "0.25")
os.environ.setdefault("MAX_TRADE_SIZE_USDC", "50")

from config import settings
from utils.logger import setup_logging
from data.polymarket import Market
from data.sources import ExternalEvidence, NewsItem
from bot.analyzer import MarketAnalyzer, MarketOpportunity, TradeDirection
from bot.estimator import ProbabilityEstimate, ProbabilityEstimator
from bot.risk_manager import RiskManager, TradeDecision

setup_logging()
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
#  Datos de mercados ficticios
#  resolution = None → se sortea al resolver según la probabilidad real implícita
# ─────────────────────────────────────────────────────────────────────────────
DEMO_SEED = 42
random.seed(DEMO_SEED)

@dataclass
class DemoMarket:
    """Un mercado con su precio de Polymarket y la 'verdad' subyacente."""
    market: Market
    # Probabilidad "real" del evento (la que queremos estimar)
    true_probability: float
    # Evidencia externa simulada
    metaculus_prob: Optional[float]
    manifold_prob: Optional[float]
    news_titles: list[str]
    # Se calculará después de la resolución
    resolved_yes: Optional[bool] = None


DEMO_MARKETS: list[DemoMarket] = [
    # ── Mercado subvalorado (YES barato → oportunidad BUY_YES) ────────────────
    DemoMarket(
        market=Market(
            condition_id="demo-001",
            question="Will the Fed cut interest rates by 25bp before June 2026?",
            description="Federal Reserve rate cut decision",
            category="economics",
            end_date="2026-06-01",
            active=True,
            yes_price=0.38,   # mercado cree 38%
            no_price=0.62,
            volume_24h=45_000,
            liquidity=120_000,
        ),
        true_probability=0.62,   # en realidad hay 62% de chance
        metaculus_prob=0.60,
        manifold_prob=0.58,
        news_titles=[
            "Fed signals possible rate relief as inflation cools to 2.1%",
            "Powell: 'We are watching employment data closely before next move'",
            "CME FedWatch tool shows 61% probability of cut in May FOMC meeting",
        ],
    ),
    # ── Mercado sobrecomprado (YES caro → oportunidad BUY_NO) ─────────────────
    DemoMarket(
        market=Market(
            condition_id="demo-002",
            question="Will Bitcoin reach $200,000 before end of 2026?",
            description="BTC price milestone",
            category="crypto",
            end_date="2026-12-31",
            active=True,
            yes_price=0.72,   # mercado cree 72%
            no_price=0.28,
            volume_24h=88_000,
            liquidity=250_000,
        ),
        true_probability=0.41,   # realmente solo 41% de probabilidad
        metaculus_prob=0.38,
        manifold_prob=0.44,
        news_titles=[
            "Bitcoin consolidates near $105k after recent rally; analysts divided",
            "On-chain data shows decreasing whale accumulation in last 30 days",
            "Macro headwinds could cap BTC upside in H2 2026, says Goldman",
        ],
    ),
    # ── Mercado correctamente valorado (sin edge) ─────────────────────────────
    DemoMarket(
        market=Market(
            condition_id="demo-003",
            question="Will Elon Musk remain CEO of Tesla through 2026?",
            description="Corporate leadership question",
            category="business",
            end_date="2026-12-31",
            active=True,
            yes_price=0.71,
            no_price=0.29,
            volume_24h=12_000,
            liquidity=35_000,
        ),
        true_probability=0.74,   # edge = +3%, menor que MIN_EDGE de 7%
        metaculus_prob=0.72,
        manifold_prob=0.73,
        news_titles=[
            "Musk focuses on Tesla robotaxi launch amid regulatory approvals",
        ],
    ),
    # ── Ineficiencia grande BUY_YES ────────────────────────────────────────────
    DemoMarket(
        market=Market(
            condition_id="demo-004",
            question="Will Spain win UEFA Euro 2028?",
            description="International football tournament",
            category="sports",
            end_date="2028-07-15",
            active=True,
            yes_price=0.11,
            no_price=0.89,
            volume_24h=22_000,
            liquidity=60_000,
        ),
        true_probability=0.22,   # doble del precio de mercado
        metaculus_prob=0.20,
        manifold_prob=0.19,
        news_titles=[
            "Spain retains most of its Euro 2024 winning squad for 2028 cycle",
            "La Roja ranks #1 in FIFA world rankings for 8 consecutive months",
            "Spain's youth pipeline produces three top-10 Ballon d'Or candidates",
        ],
    ),
    # ── BUY_NO fuerte ─────────────────────────────────────────────────────────
    DemoMarket(
        market=Market(
            condition_id="demo-005",
            question="Will the US enter a recession in 2026?",
            description="US recession probability",
            category="economics",
            end_date="2026-12-31",
            active=True,
            yes_price=0.55,
            no_price=0.45,
            volume_24h=31_000,
            liquidity=95_000,
        ),
        true_probability=0.30,   # mercado sobreestima la recesión
        metaculus_prob=0.28,
        manifold_prob=0.33,
        news_titles=[
            "US GDP Q4 2025 revised upward to 2.8% annual growth",
            "Unemployment stays at 3.8%, labor market resilient",
            "Consumer spending up 0.6% MoM in February 2026",
        ],
    ),
    # ── Mercado ilíquido (debe ser filtrado) ──────────────────────────────────
    DemoMarket(
        market=Market(
            condition_id="demo-006",
            question="Will a small-cap AI stock 10x in 2026?",
            description="Speculative stock question",
            category="finance",
            end_date="2026-12-31",
            active=True,
            yes_price=0.05,
            no_price=0.95,
            volume_24h=80,       # ← muy bajo, será filtrado
            liquidity=300,       # ← demasiado baja
        ),
        true_probability=0.10,
        metaculus_prob=None,
        manifold_prob=None,
        news_titles=[],
    ),
    # ── BUY_YES moderado ──────────────────────────────────────────────────────
    DemoMarket(
        market=Market(
            condition_id="demo-007",
            question="Will OpenAI release GPT-5 before July 2026?",
            description="AI model release timeline",
            category="technology",
            end_date="2026-07-01",
            active=True,
            yes_price=0.35,
            no_price=0.65,
            volume_24h=18_000,
            liquidity=52_000,
        ),
        true_probability=0.55,
        metaculus_prob=0.52,
        manifold_prob=0.57,
        news_titles=[
            "OpenAI reportedly finalizing pre-training of next-gen model",
            "Sam Altman hints at 'major announcement' in Q2 2026 earnings call",
            "Infrastructure buildout at OpenAI data centers points to Q2 launch",
        ],
    ),
    # ── BUY_NO moderado ───────────────────────────────────────────────────────
    DemoMarket(
        market=Market(
            condition_id="demo-008",
            question="Will Donald Trump be impeached in 2026?",
            description="US political event",
            category="politics",
            end_date="2026-12-31",
            active=True,
            yes_price=0.41,
            no_price=0.59,
            volume_24h=55_000,
            liquidity=180_000,
        ),
        true_probability=0.18,
        metaculus_prob=0.20,
        manifold_prob=0.22,
        news_titles=[
            "Republicans control both chambers; impeachment path remains blocked",
            "No formal inquiry opened as of March 2026",
        ],
    ),
    # ── BUY_YES pequeño (ineficiencia leve) ───────────────────────────────────
    DemoMarket(
        market=Market(
            condition_id="demo-009",
            question="Will Nvidia stock trade above $200 on Jan 1 2027?",
            description="Stock price milestone",
            category="finance",
            end_date="2027-01-01",
            active=True,
            yes_price=0.48,
            no_price=0.52,
            volume_24h=62_000,
            liquidity=210_000,
        ),
        true_probability=0.57,
        metaculus_prob=0.55,
        manifold_prob=0.59,
        news_titles=[
            "Nvidia reports record $45B quarter; data center demand unabated",
            "Blackwell GPU backlog extends into 2027 according to supply chain checks",
        ],
    ),
    # ── Mercado perfectamente arbitrado (sin edge significativo) ──────────────
    DemoMarket(
        market=Market(
            condition_id="demo-010",
            question="Will Apple release a new iPhone model in September 2026?",
            description="Annual Apple product cycle",
            category="technology",
            end_date="2026-10-01",
            active=True,
            yes_price=0.95,
            no_price=0.05,
            volume_24h=5_000,
            liquidity=14_000,
        ),
        true_probability=0.96,   # edge = +1%, ignorado
        metaculus_prob=0.97,
        manifold_prob=0.95,
        news_titles=["Apple's annual iPhone launch cycle has never been interrupted"],
    ),
    # ── Gran ineficiencia BUY_YES ──────────────────────────────────────────────
    DemoMarket(
        market=Market(
            condition_id="demo-011",
            question="Will China's GDP growth exceed 4.5% in 2026?",
            description="Chinese macroeconomic target",
            category="economics",
            end_date="2026-12-31",
            active=True,
            yes_price=0.28,
            no_price=0.72,
            volume_24h=29_000,
            liquidity=85_000,
        ),
        true_probability=0.62,
        metaculus_prob=0.58,
        manifold_prob=0.60,
        news_titles=[
            "China Q1 2026 GDP grows at 5.1%, beating government 4.5% target",
            "PBoC stimulus package shows early signs of boosting domestic consumption",
            "IMF revises China 2026 forecast to 5.0% from earlier 4.3% estimate",
        ],
    ),
    # ── BUY_NO con alta confianza ──────────────────────────────────────────────
    DemoMarket(
        market=Market(
            condition_id="demo-012",
            question="Will a nuclear weapon be used in a conflict before 2027?",
            description="Geopolitical tail risk",
            category="geopolitics",
            end_date="2027-01-01",
            active=True,
            yes_price=0.18,
            no_price=0.82,
            volume_24h=41_000,
            liquidity=130_000,
        ),
        true_probability=0.04,   # MUY improbable
        metaculus_prob=0.03,
        manifold_prob=0.05,
        news_titles=[
            "IAEA: no credible intelligence of imminent nuclear escalation",
            "US-Russia strategic stability talks resume in Geneva",
        ],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Estimador simulado (sin OpenAI)
# ─────────────────────────────────────────────────────────────────────────────

class MockEstimator(ProbabilityEstimator):
    """
    Reemplaza las llamadas reales a OpenAI con la probabilidad 'verdadera'
    del DemoMarket, más un ruido gaussiano pequeño para simular imperfección.
    """

    def __init__(self, demo_markets: list[DemoMarket]) -> None:
        # No llamamos super().__init__() para evitar instanciar el cliente OpenAI
        self._lookup = {dm.market.condition_id: dm for dm in demo_markets}

    def estimate(self, evidence: ExternalEvidence) -> Optional[ProbabilityEstimate]:
        # Identificar el mercado por su pregunta (buscamos match en lookup)
        dm = next(
            (v for v in self._lookup.values() if v.market.question == evidence.question),
            None,
        )
        if dm is None:
            return None

        # Agregar ruido realista (~±5pp)
        noise = random.gauss(0, 0.04)
        prob = max(0.02, min(0.98, dm.true_probability + noise))

        confidence = (
            "high" if abs(prob - dm.market.yes_price) > 0.15
            else "medium" if abs(prob - dm.market.yes_price) > 0.08
            else "low"
        )

        sources = ["news"]
        if dm.metaculus_prob is not None:
            sources.append("metaculus")
        if dm.manifold_prob is not None:
            sources.append("manifold")

        return ProbabilityEstimate(
            yes_probability=round(prob, 3),
            confidence=confidence,
            reasoning=(
                f"Based on {len(dm.news_titles)} news items and "
                f"peer market signals, the estimated probability is {prob:.1%}."
            ),
            sources_used=sources,
        )

    def aggregate(self, llm_estimate: ProbabilityEstimate, evidence: ExternalEvidence) -> float:
        """Replicar la lógica real de blending."""
        dm = next(
            (v for v in self._lookup.values() if v.market.question == evidence.question),
            None,
        )
        llm_w = {"low": 0.50, "medium": 0.65, "high": 0.80}.get(llm_estimate.confidence, 0.65)
        peers = []
        if dm and dm.metaculus_prob is not None:
            peers.append(dm.metaculus_prob)
        if dm and dm.manifold_prob is not None:
            peers.append(dm.manifold_prob)
        if not peers:
            return llm_estimate.yes_probability
        peer_mean = sum(peers) / len(peers)
        return round(llm_w * llm_estimate.yes_probability + (1 - llm_w) * peer_mean, 4)


# ─────────────────────────────────────────────────────────────────────────────
#  Agregador de evidencia simulado
# ─────────────────────────────────────────────────────────────────────────────

def make_evidence(dm: DemoMarket) -> ExternalEvidence:
    news = [
        NewsItem(
            title=title,
            description="",
            source="SimulatedSource",
            published_at="2026-03-13",
            url="https://example.com",
        )
        for title in dm.news_titles
    ]
    return ExternalEvidence(
        question=dm.market.question,
        news_items=news,
        metaculus_probability=dm.metaculus_prob,
        manifold_probability=dm.manifold_prob,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Resultado de trade y P&L
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    decision: TradeDecision
    resolved_yes: bool
    pnl: float
    outcome_label: str   # "WIN" | "LOSS" | "PUSH"


def resolve_and_compute_pnl(decision: TradeDecision, resolved_yes: bool) -> TradeResult:
    opp = decision.opportunity
    stake = decision.usdc_amount
    direction = opp.direction

    if direction == TradeDirection.BUY_YES:
        if resolved_yes:
            # Profit = stake * (1/price - 1)
            pnl = stake * ((1.0 / opp.market_price) - 1.0)
            label = "WIN"
        else:
            pnl = -stake
            label = "LOSS"
    elif direction == TradeDirection.BUY_NO:
        no_price = 1.0 - opp.market_price
        if not resolved_yes:
            pnl = stake * ((1.0 / no_price) - 1.0)
            label = "WIN"
        else:
            pnl = -stake
            label = "LOSS"
    else:
        pnl = 0.0
        label = "PUSH"

    return TradeResult(decision=decision, resolved_yes=resolved_yes, pnl=pnl, outcome_label=label)


# ─────────────────────────────────────────────────────────────────────────────
#  Motor de la demo
# ─────────────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    STARTING_BANKROLL = 500.0

    # Decidir qué estimador usar
    api_key = os.getenv("OPENAI_API_KEY", "")
    use_real_llm = api_key and not api_key.startswith("sk-demo")
    if use_real_llm:
        estimator = ProbabilityEstimator()
        logger.info("Usando estimador GPT-4o REAL (OPENAI_API_KEY detectada)")
    else:
        estimator = MockEstimator(DEMO_MARKETS)
        logger.info("Usando estimador SIMULADO (no se encontró OPENAI_API_KEY real)")

    risk_mgr = RiskManager(bankroll=STARTING_BANKROLL)

    print("\n")
    print("=" * 70)
    print("  POLYMARKET BOT — DEMO DE SIMULACIÓN")
    print(f"  Bankroll inicial: ${STARTING_BANKROLL:.2f} USDC")
    print(f"  Edge mínimo: {settings.min_edge:.0%}  |  Kelly máx: {settings.max_kelly_fraction:.0%}")
    print(f"  Mercados a analizar: {len(DEMO_MARKETS)}")
    print("=" * 70)

    trades_entered: list[tuple[TradeDecision, DemoMarket]] = []
    skipped: list[str] = []

    # ── Pipeline de análisis ────────────────────────────────────────────────
    print("\n📊 ANÁLISIS DE MERCADOS\n")

    for dm in DEMO_MARKETS:
        market = dm.market
        print(f"  • Analizando: '{market.question[:65]}'")
        print(f"    Precio YES en mercado: {market.yes_price:.1%}  |  "
              f"Volumen 24h: ${market.volume_24h:,.0f}  |  "
              f"Liquidez: ${market.liquidity:,.0f}")

        # Filtros básicos
        if market.liquidity < 500 or market.volume_24h < 200:
            print(f"    ⛔ FILTRADO — liquidez/volumen insuficiente\n")
            skipped.append(market.question[:65])
            continue

        # Evidencia
        evidence = make_evidence(dm)

        # Estimación
        llm_est = estimator.estimate(evidence)
        if llm_est is None:
            print(f"    ⚠️  LLM falló — omitido\n")
            skipped.append(market.question[:65])
            continue

        final_prob = estimator.aggregate(llm_est, evidence)
        edge = final_prob - market.yes_price

        print(f"    Prob estimada: {final_prob:.1%}  |  "
              f"Edge: {edge:+.1%}  |  Confianza LLM: {llm_est.confidence}")

        # Dirección
        if edge >= settings.min_edge:
            direction = TradeDirection.BUY_YES
        elif edge <= -settings.min_edge:
            direction = TradeDirection.BUY_NO
        else:
            direction = TradeDirection.NO_TRADE

        if direction == TradeDirection.NO_TRADE:
            print(f"    ➡️  Sin edge suficiente — se omite\n")
            skipped.append(market.question[:65])
            continue

        # Construir oportunidad
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

        # Risk manager
        decision = risk_mgr.evaluate(opp)
        if not decision.approved:
            print(f"    🚫 Rechazado por Risk Manager: {decision.rejection_reason}\n")
            skipped.append(market.question[:65])
            continue

        print(f"    ✅ TRADE APROBADO: {direction.value}  —  ${decision.usdc_amount:.2f} USDC")
        if dm.news_titles:
            print(f"    📰 Evidencia: {dm.news_titles[0][:75]}")
        print(f"    Razonamiento LLM: {llm_est.reasoning[:100]}")
        print()

        risk_mgr.record_trade(market.condition_id, decision.usdc_amount)
        trades_entered.append((decision, dm))

        time.sleep(0.05)  # pequeña pausa para efecto visual

    # ── Resolución de mercados ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ⏳ RESOLUCIÓN DE MERCADOS (simulada)")
    print("=" * 70 + "\n")

    results: list[TradeResult] = []
    for decision, dm in trades_entered:
        # Resolver: sorteo ponderado por la probabilidad real
        resolved_yes = random.random() < dm.true_probability
        dm.resolved_yes = resolved_yes
        result = resolve_and_compute_pnl(decision, resolved_yes)
        results.append(result)

        resolution_str = "✅ YES" if resolved_yes else "❌ NO"
        pnl_str = f"+${result.pnl:.2f}" if result.pnl >= 0 else f"-${abs(result.pnl):.2f}"
        print(f"  [{result.outcome_label}] {pnl_str:>8}  {resolution_str}  "
              f"'{decision.opportunity.market.question[:55]}'")

    # ── Resumen de P&L ──────────────────────────────────────────────────────
    total_pnl = sum(r.pnl for r in results)
    wins = [r for r in results if r.outcome_label == "WIN"]
    losses = [r for r in results if r.outcome_label == "LOSS"]
    total_staked = sum(r.decision.usdc_amount for r in results)
    final_bankroll = STARTING_BANKROLL + total_pnl
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

    print("\n" + "=" * 70)
    print("  📈 RESUMEN FINAL")
    print("=" * 70)
    print(f"  Bankroll inicial          ${STARTING_BANKROLL:.2f}")
    print(f"  Bankroll final            ${final_bankroll:.2f}")
    print(f"  P&L total                 {'+' if total_pnl >= 0 else ''}${total_pnl:.2f}")
    print(f"  ROI sobre capital apostado {roi:+.1f}%")
    print()
    print(f"  Mercados analizados       {len(DEMO_MARKETS)}")
    print(f"  Operaciones ejecutadas    {len(results)}")
    print(f"  Operaciones omitidas      {len(skipped)}")
    print(f"  Tasa de acierto           {len(wins)}/{len(results)}  "
          f"({len(wins)/len(results)*100:.0f}%)" if results else "  Tasa de acierto  —")
    print(f"  Capital total apostado    ${total_staked:.2f}")
    print(f"  Win promedio              ${sum(r.pnl for r in wins)/len(wins):.2f}" if wins else "")
    print(f"  Loss promedio             -${abs(sum(r.pnl for r in losses)/len(losses)):.2f}" if losses else "")

    print("\n  DETALLE DE OPERACIONES:\n")
    for r in results:
        opp = r.decision.opportunity
        direction_icon = "▲" if opp.direction == TradeDirection.BUY_YES else "▼"
        pnl_color = "+" if r.pnl >= 0 else ""
        print(
            f"  {direction_icon} {r.outcome_label:<4} | "
            f"Apostado ${r.decision.usdc_amount:>6.2f} | "
            f"PnL {pnl_color}${r.pnl:>7.2f} | "
            f"Precio {opp.market_price:.0%}→est {opp.estimated_probability:.0%} | "
            f"{opp.market.question[:45]}"
        )

    print()
    if total_pnl > 0:
        print(f"  🎉 El bot terminó con ganancia de ${total_pnl:.2f} USDC")
    elif total_pnl < 0:
        print(f"  📉 El bot terminó con pérdida de ${abs(total_pnl):.2f} USDC")
    else:
        print("  🤝 El bot terminó en tablas")

    print()
    print("  ⚠️  Nota: los resultados son SIMULADOS con datos ficticios.")
    print("  Para operar con dinero real, configura el .env y cambia DRY_RUN=false.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_demo()
