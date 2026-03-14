import requests
r = requests.get("https://gamma-api.polymarket.com/markets?active=true&limit=8&order=volume24hr&ascending=false", timeout=15)
markets = r.json()
for m in markets[:8]:
    tokens = m.get("tokens", [])
    prices = [(t.get("outcome"), t.get("price")) for t in tokens]
    liq = m.get("liquidity") or m.get("liquidityNum") or "N/A"
    vol = m.get("volume24hr") or m.get("volume") or "N/A"
    print(f'Q: {str(m.get("question",""))[:65]}')
    print(f'   liquidity={liq}  vol24={vol}  prices={prices}')
    print(f'   raw keys: {list(m.keys())[:12]}')
    print()
