# Polymarket Bot

Bot autónomo que opera Polymarket detectando ineficiencias de mercado comparando los precios
actuales con una probabilidad estimada a partir de múltiples fuentes externas.

---

## Arquitectura

```
main.py                  ← Orquestador principal (loop de escaneo)
config.py                ← Configuración centralizada vía .env
bot/
  analyzer.py            ← Detecta oportunidades (compara precio vs probabilidad estimada)
  estimator.py           ← Estima probabilidad real usando GPT-4o + fuentes externas
  risk_manager.py        ← Dimensionamiento de posiciones (Kelly fraccional) + límites
  trader.py              ← Ejecuta órdenes en la CLOB de Polymarket
data/
  polymarket.py          ← Cliente de la API de Polymarket (Gamma REST + CLOB)
  sources.py             ← Fuentes externas: NewsAPI, Metaculus, Manifold, Wikipedia
utils/
  logger.py              ← Logging con rotación de archivo
```

### Flujo de cada ciclo

```
Polymarket Markets API
        │
        ▼  (top N mercados por volumen 24h)
  MarketAnalyzer
        │
        ├─► DataAggregator ──► NewsAPI / Metaculus / Manifold / Wikipedia
        │
        ├─► ProbabilityEstimator (GPT-4o)
        │        └─ estima P(YES) basado en la evidencia
        │
        ├─► blend P(LLM) + P(peers)  →  probabilidad final
        │
        └─► si |P_final - P_mercado| ≥ MIN_EDGE:
                  │
                  ▼
            RiskManager  (Kelly fraccional, caps de cartera)
                  │
                  ▼
            TradeExecutor  →  orden en Polymarket CLOB
```

---

## Instalación

```bash
# 1. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
copy .env.example .env
# Editar .env con tus claves
```

---

## Configuración (.env)

| Variable | Descripción |
|---|---|
| `POLYGON_PRIVATE_KEY` | Clave privada de tu wallet Polygon (con USDC) |
| `OPENAI_API_KEY` | Clave de OpenAI para el estimador LLM |
| `NEWS_API_KEY` | Clave de newsapi.org (opcional pero recomendada) |
| `MIN_EDGE` | Edge mínimo para operar (0.07 = 7 puntos porcentuales) |
| `MAX_KELLY_FRACTION` | Fracción Kelly máxima (0.25 = Kelly 25%) |
| `MAX_TRADE_SIZE_USDC` | Tamaño máximo por operación en USDC |
| `MARKETS_TO_SCAN` | Número de mercados a analizar por ciclo |
| `SCAN_INTERVAL_SECONDS` | Segundos entre ciclos de escaneo |
| `DRY_RUN` | `true` = simular sin ejecutar órdenes reales |

---

## Ejecución

```bash
# Modo simulación (DRY_RUN=true por defecto — recomendado para pruebas)
python main.py

# Modo producción (después de validar con dry-run)
# Cambiar DRY_RUN=false en .env y ejecutar:
python main.py
```

---

## Fuentes externas de probabilidad

1. **NewsAPI** — titulares recientes relacionados con el mercado
2. **Metaculus** — probabilidad comunitaria de expertos en forecasting  
3. **Manifold Markets** — otro mercado de predicción para triangular
4. **Wikipedia** — contexto de fondo para el LLM
5. **GPT-4o** — fusiona todo lo anterior en una estimación calibrada

La probabilidad final es un promedio ponderado entre el LLM y los mercados pares (mayor peso al LLM cuando su confianza es alta).

---

## Gestión del riesgo

- **Kelly fraccional**: el tamaño de cada posición es `(kelly_fraction * bankroll * MAX_KELLY_FRACTION)`
- **Cap por operación**: nunca se arriesga más de `MAX_TRADE_SIZE_USDC`
- **Cap de cartera**: máximo 50% del bankroll desplegado simultáneamente
- **Sin duplicados**: no se abre una segunda posición en el mismo mercado
- **Spread máximo**: se evitan mercados con spread > 5% (ilíquidos)

---

## Advertencia

Este bot opera capital real en un mercado de predicciones. Los mercados de predicción
son intrínsecamente arriesgados. Usa `DRY_RUN=true` hasta estar completamente seguro
del comportamiento del sistema. El autor no se responsabiliza de pérdidas.
