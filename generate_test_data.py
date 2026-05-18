"""
Genera datos sintéticos de velas para probar el sistema.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

np.random.seed(42)
N = 3000
price = 33.0
rows = []
t = datetime(2025, 1, 1, tzinfo=timezone.utc)

for i in range(N):
    o = price * (1 + np.random.normal(0, 0.001))
    h = o * (1 + abs(np.random.normal(0, 0.002)))
    l = o * (1 - abs(np.random.normal(0, 0.002)))
    c = np.random.uniform(l, h)
    vol = abs(np.random.normal(1000, 300))
    rows.append({
        "open_time": t,
        "open":  round(o, 5),
        "high":  round(h, 5),
        "low":   round(l, 5),
        "close": round(c, 5),
        "volume": round(vol, 2),
        "EMA_200": "",
        "RSI_14":  round(50 + np.random.normal(0, 10), 5),
        "ATR_14":  round(abs(np.random.normal(0.2, 0.05)), 5),
        "num_trades": np.random.randint(100, 600),
    })
    price = c
    t += timedelta(minutes=1)

df = pd.DataFrame(rows)
df.to_csv("test_candles.csv", index=False)
print(f"✅ Generadas {N} velas en test_candles.csv")
