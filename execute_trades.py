"""
Manual trade executor — runs specific trades based on my analysis.
Ejecuta apostando NO en mercados sobrevaluados geopolíticos.
"""
from __future__ import annotations

import sys
import json
import requests
from loguru import logger

from utils.logger import setup_logging
from data.polymarket import PolymarketClient
from config import settings


# ── Mercados objetivo ──────────────────────────────────────────────────────────
# Seleccionados por análisis: precios del mercado muy por arriba de probabilidad real
TARGETS = [
    {
        "slug": "will-the-iranian-regime-fall-by-march-31",
        "question": "Will the Iranian regime fall by March 31?",
        "bet": "NO",
        "market_price_yes": 0.038,
        "my_estimate_yes": 0.015,
        "max_entry_price_yes": 0.12,  # no entrar si YES sube mucho
        "amount_usdc": 12.0,
        "reason": "17 días insuficientes para colapso de régimen; mercado sobreestima riesgo geopolítico",
    },
    # Nota: 'US forces enter Iran by March 31' eliminado — precio YES subió a 43% (serie condicional)
]


def get_market_by_slug(slug: str) -> dict | None:
    """Busca mercado en Gamma API por slug exacto."""
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"slug": slug},
            timeout=15,
        )
        if not resp.ok:
            logger.error("Gamma API error {}: {}", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        markets = data if isinstance(data, list) else [data]
        for m in markets:
            if (m.get("slug") or "").lower() == slug.lower():
                return m
        if markets:
            return markets[0]  # única coincidencia directa
    except Exception as exc:
        logger.error("Market search failed: {}", exc)
    return None


def get_no_token_id(clob, condition_id: str) -> str | None:
    """Obtiene el token ID del outcome NO via CLOB API."""
    try:
        mkt = clob.get_market(condition_id)
        tokens = mkt.get("tokens", [])
        for t in tokens:
            if str(t.get("outcome", "")).upper() == "NO":
                return t.get("token_id")
    except Exception as exc:
        logger.error("get_market failed for {}: {}", condition_id, exc)
    return None


def main():
    setup_logging()
    logger.info("=" * 60)
    logger.info("Manual Trade Executor — Iran NO bets")
    logger.info("DRY_RUN = {}", settings.dry_run)
    logger.info("=" * 60)

    client = PolymarketClient()
    client.connect()

    balance = client.get_usdc_balance()
    logger.info("Balance: ${:.2f} USDC", balance)

    total_to_spend = sum(t["amount_usdc"] for t in TARGETS)
    if balance > 0 and balance < total_to_spend and not settings.dry_run:
        logger.warning(
            "Balance ${:.2f} < required ${:.2f} — reduciendo tamaños proporcionalmente",
            balance, total_to_spend,
        )
        scale = balance / total_to_spend * 0.9
        for t in TARGETS:
            t["amount_usdc"] = round(t["amount_usdc"] * scale, 2)

    results = []
    for target in TARGETS:
        logger.info("")
        logger.info("─── {} ───", target["question"][:60])
        logger.info("  Precio mercado YES: {:.1%}", target["market_price_yes"])
        logger.info("  Mi estimación  YES: {:.1%}", target["my_estimate_yes"])
        edge = target["market_price_yes"] - target["my_estimate_yes"]
        logger.info("  Edge (NO):          {:.1%}", edge)
        logger.info("  Monto:              ${:.2f} USDC", target["amount_usdc"])
        logger.info("  Razón: {}", target["reason"])

        # 1. Buscar mercado por slug exacto
        mkt_data = get_market_by_slug(target["slug"])
        if not mkt_data:
            logger.error("  ✗ Mercado no encontrado: {}", target["slug"])
            results.append({"market": target["question"], "status": "NOT_FOUND"})
            continue

        condition_id = mkt_data.get("conditionId")
        if not condition_id:
            logger.error("  ✗ Sin conditionId para {}", target["slug"])
            results.append({"market": target["question"], "status": "NO_CONDITION_ID"})
            continue

        # Verificar precio actual
        current_prices_raw = mkt_data.get("outcomePrices", "[]")
        try:
            current_prices = json.loads(current_prices_raw) if isinstance(current_prices_raw, str) else current_prices_raw
            current_yes = float(current_prices[0]) if current_prices else target["market_price_yes"]
        except Exception:
            current_yes = target["market_price_yes"]
        logger.info("  Precio YES actual: {:.1%}", current_yes)

        # Sanity check: mercado ya resuelto o precio movido demasiado
        if current_yes < 0.005:
            logger.warning("  ✗ Mercado casi resuelto (YES < 0.5%), saltando")
            results.append({"market": target["question"], "status": "NEAR_RESOLVED"})
            continue
        max_entry = target.get("max_entry_price_yes", 0.30)
        if current_yes > max_entry:
            logger.warning(
                "  ✗ Precio YES {:.1%} supera máximo aceptable {:.1%}, saltando",
                current_yes, max_entry,
            )
            results.append({"market": target["question"], "status": "PRICE_MOVED"})
            continue

        # 2. Obtener token ID del NO
        no_token_id = get_no_token_id(client.clob, condition_id)
        if not no_token_id:
            # Fallback: intentar con clobTokenIds del Gamma data
            try:
                clob_ids = json.loads(mkt_data.get("clobTokenIds", "[]"))
                if len(clob_ids) >= 2:
                    no_token_id = clob_ids[1]  # index 1 = NO
                    logger.info("  Token NO (fallback Gamma): {}...{}", no_token_id[:8], no_token_id[-6:])
            except Exception:
                pass

        if not no_token_id:
            logger.error("  ✗ No se pudo obtener token ID para NO")
            results.append({"market": target["question"], "status": "NO_TOKEN_ID"})
            continue

        logger.info("  Token NO: {}...{}", no_token_id[:10], no_token_id[-8:])

        # 3. Ejecutar orden
        try:
            result = client.place_market_order(
                token_id=no_token_id,
                side="BUY",
                amount_usdc=target["amount_usdc"],
            )
            logger.info("  ✓ Orden enviada: {}", result)
            results.append({"market": target["question"], "status": "SUCCESS", "result": result})
        except Exception as exc:
            logger.error("  ✗ Orden fallida: {}", exc)
            results.append({"market": target["question"], "status": "ERROR", "error": str(exc)})

    # ── Resumen ────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESUMEN")
    logger.info("=" * 60)
    failed = [r for r in results if r["status"] not in ("SUCCESS",)]
    for r in results:
        status_icon = "✓" if r["status"] == "SUCCESS" else "✗"
        logger.info("  {} {} — {}", status_icon, r["market"][:55], r["status"])
        if r["status"] == "ERROR":
            logger.error("    Error details: {}", r.get("error", "unknown"))
    logger.info("=" * 60)
    if failed:
        logger.error("{} operacion(es) fallaron — revisar logs arriba", len(failed))
        sys.exit(1)
    logger.info("Todas las operaciones completadas exitosamente")


if __name__ == "__main__":
    main()
