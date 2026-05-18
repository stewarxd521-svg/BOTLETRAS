#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║        Binance HYPEUSDT — Simulador de Trading en Vivo          ║
║        Candle Pattern Predictor → Long/Short → P&L Stats        ║
╚══════════════════════════════════════════════════════════════════╝

Uso:
    # Descargar de Binance y simular (tu máquina con internet):
    python binance_simulator.py

    # Usar CSV local (demo / sin internet):
    python binance_simulator.py --csv hype_data.csv

    # Ajustar parámetros:
    python binance_simulator.py --tp 0.015 --sl 0.01 --confidence 0.45
"""

import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Import del sistema predictor ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from candle_predictor import CandlePatternSystem, CandleEncoder, NgramPredictor, SYMBOLS, MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

BINANCE_API   = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL        = "HYPEUSDT"
INTERVAL      = "1m"
N_KLINES      = 1500
WARMUP        = 100          # velas de calentamiento antes de simular
TP_PCT        = 0.009         # take profit 1%
SL_PCT        = 0.01         # stop loss  1%
MAX_HOLD      = 60           # velas máximas en posición
MIN_CONFIDENCE = 0.50        # confianza mínima del modelo para abrir posición
CAPITAL       = 1_000.0      # USDT inicial
POS_SIZE_PCT  = 0.012         # tamaño de posición = 10% del capital actual
FEE_PCT       = 0.00       # 0.1% comisión Binance por lado (0.2% round-trip)
DIR_THRESHOLD = 0.02         # sesgo mínimo de % body para direccionar long/short
TRAIN_RATIO   = 0.70         # fracción de datos para entrenar si no hay modelo


# ══════════════════════════════════════════════════════════════════════════════
#  DESCARGA DE BINANCE
# ══════════════════════════════════════════════════════════════════════════════

def download_binance(symbol: str = SYMBOL, interval: str = INTERVAL,
                     limit: int = N_KLINES) -> pd.DataFrame:
    """Descarga klines de Binance REST API y retorna DataFrame con columnas OHLCV."""
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests")

    log.info(f"Descargando {limit} klines de {symbol} ({interval}) desde Binance...")
    resp = requests.get(
        BINANCE_API,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "num_trades",
        "taker_base", "taker_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df["EMA_200"] = ""
    df["RSI_14"] = 0.0
    df["ATR_14"] = 0.0

    df = df[["open_time","open","high","low","close","volume",
             "EMA_200","RSI_14","ATR_14","num_trades"]].copy()
    log.info(f"Descargadas {len(df)} velas │ {df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]}")
    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    log.info(f"CSV cargado: {len(df)} velas │ {df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECCIONALIDAD DE SÍMBOLOS
# ══════════════════════════════════════════════════════════════════════════════

def compute_symbol_directions(encoder: CandleEncoder,
                               threshold: float = DIR_THRESHOLD) -> dict[str, str]:
    """
    Para cada símbolo (cluster), determina si el patrón es LONG, SHORT o NEUTRAL.

    Método: invierte la transformación del centroide y extrae las columnas
    'pct_body' (índices 1, 5, 9, … 37). La suma de esos valores indica
    si el patrón de 10 velas tiene sesgo alcista o bajista.
    """
    centroids_scaled = encoder.kmeans.cluster_centers_          # (20, 40)
    centroids        = encoder.scaler.inverse_transform(centroids_scaled)

    # Columnas pct_body: índices 1, 5, 9, 13, 17, 21, 25, 29, 33, 37
    body_cols = list(range(1, 40, 4))
    dirs = {}
    for k, sym in enumerate(SYMBOLS[:encoder.n_clusters]):
        body_sum = centroids[k, body_cols].sum()
        if body_sum > threshold:
            dirs[sym] = "LONG"
        elif body_sum < -threshold:
            dirs[sym] = "SHORT"
        else:
            dirs[sym] = "NEUTRAL"
    return dirs


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULACIÓN DE TRADE
# ══════════════════════════════════════════════════════════════════════════════

def simulate_trade(df: pd.DataFrame, entry_idx: int, direction: str,
                   tp_pct: float = TP_PCT, sl_pct: float = SL_PCT,
                   max_hold: int = MAX_HOLD,
                   fee_pct: float = FEE_PCT) -> dict:
    """
    Simula un trade a partir de entry_idx.

    - Entrada: open de la vela entry_idx.
    - TP/SL: se verifica intra-vela (high/low).
    - Si SL y TP pueden ocurrir en la misma vela → SL primero (conservador).
    - Tiempo límite: max_hold velas.
    """
    if entry_idx >= len(df):
        return None

    entry_price = df.iloc[entry_idx]["open"]
    if entry_price <= 0:
        return None

    if direction == "LONG":
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:  # SHORT
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

    last_idx = min(entry_idx + max_hold, len(df)) - 1

    for j in range(entry_idx, last_idx + 1):
        row = df.iloc[j]
        h, l, c = row["high"], row["low"], row["close"]
        duration = j - entry_idx + 1

        if direction == "LONG":
            if l <= sl_price:
                exit_price = sl_price
                result = "LOSS"
                pnl_pct = -sl_pct - 2 * fee_pct
                break
            if h >= tp_price:
                exit_price = tp_price
                result = "WIN"
                pnl_pct = tp_pct - 2 * fee_pct
                break
        else:  # SHORT
            if h >= sl_price:
                exit_price = sl_price
                result = "LOSS"
                pnl_pct = -sl_pct - 2 * fee_pct
                break
            if l <= tp_price:
                exit_price = tp_price
                result = "WIN"
                pnl_pct = tp_pct - 2 * fee_pct
                break
    else:
        # Timeout: cerrar al close de la última vela
        exit_price = df.iloc[last_idx]["close"]
        duration   = last_idx - entry_idx + 1
        if direction == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price - 2 * fee_pct
        else:
            pnl_pct = (entry_price - exit_price) / entry_price - 2 * fee_pct
        result = "WIN" if pnl_pct > 0 else "LOSS"
        if abs(pnl_pct) < 0.0005:
            result = "BREAK_EVEN"

    return {
        "direction":    direction,
        "entry_idx":    entry_idx,
        "entry_price":  entry_price,
        "exit_price":   exit_price,
        "result":       result,
        "pnl_pct":      round(pnl_pct, 6),
        "duration":     duration,
        "entry_time":   df.iloc[entry_idx]["open_time"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULACIÓN VELA A VELA
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(
    df: pd.DataFrame,
    system: CandlePatternSystem,
    dirs: dict[str, str],
    warmup: int = WARMUP,
    min_confidence: float = MIN_CONFIDENCE,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    max_hold: int = MAX_HOLD,
    fee_pct: float = FEE_PCT,
    capital: float = CAPITAL,
    pos_size_pct: float = POS_SIZE_PCT,
) -> list[dict]:
    """
    Simulación walk-forward vela a vela.

    Lógica:
      1. Codifica todas las velas → secuencia de símbolos.
      2. Desde el símbolo warmup_sym en adelante:
         - Predice el siguiente símbolo.
         - Si confianza ≥ threshold y dirección ≠ NEUTRAL → señal.
         - Abre trade en la vela siguiente a la ventana actual.
         - No abre nueva operación si ya hay una activa (1 a la vez).
    """
    WINDOW = system.encoder.window_size
    MAX_N  = system.predictor.max_n

    log.info("Codificando todas las velas a secuencia de símbolos...")
    sequence = system.encoder.transform(df)
    n_sym = len(sequence)
    log.info(f"Secuencia: {n_sym} símbolos (de {len(df)} velas)")

    # Índice en símbolo desde el que empezamos a operar
    warmup_sym = max(MAX_N + 1, warmup - WINDOW + 1)
    log.info(f"Calentamiento: {warmup_sym} símbolos ({warmup} velas) antes de operar")

    trades: list[dict]  = []
    equity: list[float] = [capital]
    cap    = capital
    busy_until = -1   # índice de vela hasta donde ya hay posición abierta

    print("\n" + "═" * 80)
    print(f"  {'#':>4}  {'Dir':6}  {'Entry':>8}  {'Exit':>8}  {'P&L%':>7}  "
          f"{'P&L$':>8}  {'Dur':>4}  {'Capital':>10}  Ctx→Pred")
    print("─" * 80)

    for i in range(warmup_sym, n_sym - 1):
        # Índice de la primera vela del trade si se dispara señal
        # La ventana actual ocupa velas [i : i+WINDOW]; next = i+WINDOW
        trade_entry_idx = i + WINDOW
        if trade_entry_idx >= len(df) - 1:
            break

        # No abrir si hay trade activo
        if trade_entry_idx <= busy_until:
            continue

        # Predicción del siguiente símbolo
        ctx  = sequence[max(0, i - MAX_N) : i + 1]
        pred = system.predictor.predict(list(ctx), top_k=5)
        if not pred:
            continue

        best       = pred[0]
        confidence = best["probability"]
        next_sym   = best["symbol"]
        n_order    = best["n_used"]

        if confidence < min_confidence:
            continue

        direction = dirs.get(next_sym, "NEUTRAL")
        if direction == "NEUTRAL":
            continue

        # Ejecutar trade
        trade = simulate_trade(
            df, trade_entry_idx, direction,
            tp_pct=tp_pct, sl_pct=sl_pct,
            max_hold=max_hold, fee_pct=fee_pct,
        )
        if trade is None:
            continue

        busy_until = trade_entry_idx + trade["duration"] - 1

        # P&L en USDT
        pos_usdt = cap * pos_size_pct
        pnl_usdt = pos_usdt * trade["pnl_pct"]
        cap      = cap + pnl_usdt
        trade["pnl_usdt"]     = round(pnl_usdt, 4)
        trade["capital_after"] = round(cap, 4)
        trade["confidence"]   = confidence
        trade["n_used"]       = n_order
        trade["context_tail"] = "".join(ctx[-4:])
        trade["pred_sym"]     = next_sym

        equity.append(cap)
        trades.append(trade)

        # Mostrar fila de trade
        idx   = len(trades)
        d_sym = "▲ LONG " if direction == "LONG" else "▼ SHORT"
        r_sym = "✅" if trade["result"] == "WIN" else ("❌" if trade["result"] == "LOSS" else "⚠️")
        pnl_s = f"{trade['pnl_pct']*100:+.2f}%"
        usd_s = f"{pnl_usdt:+.2f}$"
        ctx_s = f"{''.join(ctx[-3:])}→{next_sym}"

        print(
            f"  {idx:>4}  {d_sym}  {trade['entry_price']:>8.4f}  "
            f"{trade['exit_price']:>8.4f}  {pnl_s:>7}  {usd_s:>8}  "
            f"{trade['duration']:>4}  {cap:>10.2f}  {r_sym} {ctx_s} ({confidence:.0%})"
        )

    print("─" * 80)
    return trades, equity


# ══════════════════════════════════════════════════════════════════════════════
#  ESTADÍSTICAS
# ══════════════════════════════════════════════════════════════════════════════

def print_statistics(trades: list[dict], equity: list[float],
                     initial_capital: float = CAPITAL) -> None:
    """Imprime estadísticas completas de la simulación."""
    if not trades:
        print("\n⚠️  Sin operaciones en esta simulación.")
        return

    df_t = pd.DataFrame(trades)
    final_cap = equity[-1]
    n = len(trades)

    wins   = df_t[df_t["result"] == "WIN"]
    losses = df_t[df_t["result"] == "LOSS"]
    n_win  = len(wins)
    n_los  = len(losses)
    n_be   = n - n_win - n_los

    gross_win  = wins["pnl_usdt"].sum()
    gross_loss = abs(losses["pnl_usdt"].sum())
    net_pnl    = df_t["pnl_usdt"].sum()
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    n_long  = len(df_t[df_t["direction"] == "LONG"])
    n_short = len(df_t[df_t["direction"] == "SHORT"])
    long_wins  = len(df_t[(df_t["direction"] == "LONG") & (df_t["result"] == "WIN")])
    short_wins = len(df_t[(df_t["direction"] == "SHORT") & (df_t["result"] == "WIN")])

    # Drawdown
    eq_arr  = np.array(equity)
    running_max  = np.maximum.accumulate(eq_arr)
    drawdowns    = (eq_arr - running_max) / running_max
    max_dd       = drawdowns.min()
    max_dd_usdt  = (eq_arr - running_max).min()

    # Avg trade stats
    avg_win  = wins["pnl_usdt"].mean()   if n_win > 0  else 0
    avg_loss = losses["pnl_usdt"].mean() if n_los > 0  else 0
    avg_dur  = df_t["duration"].mean()
    avg_conf = df_t["confidence"].mean()

    # Ratio R
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Equity curve ASCII
    eq_min = min(equity)
    eq_max = max(equity)
    width  = 60
    height = 8
    chart_rows = []
    for row_i in range(height):
        threshold = eq_max - (eq_max - eq_min) * row_i / (height - 1)
        row = ""
        for j, v in enumerate(equity):
            step = int(j * width / len(equity))
            if len(row) <= step:
                row += "█" if v >= threshold else " "
        chart_rows.append((threshold, row))

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + "  📊  ESTADÍSTICAS DE SIMULACIÓN — HYPEUSDT 1m".center(78) + "║")
    print("╠" + "═" * 78 + "╣")

    def row(label, value, unit=""):
        s = f"  {label:<35} {value}{unit}"
        print("║" + s.ljust(78) + "║")

    print("║" + "  RESUMEN GENERAL".ljust(78) + "║")
    print("║" + "─" * 78 + "║")
    row("Capital inicial",    f"${initial_capital:>10,.2f}")
    row("Capital final",      f"${final_cap:>10,.2f}")
    row("P&L neto",           f"${net_pnl:>+10,.2f}   ({(final_cap/initial_capital-1)*100:+.2f}%)")
    row("Máximo Drawdown",    f"${max_dd_usdt:>+10,.2f}  ({max_dd*100:.2f}%)")
    row("Profit Factor",      f"{profit_factor:>10.2f}")

    print("║" + "─" * 78 + "║")
    print("║" + "  OPERACIONES".ljust(78) + "║")
    print("║" + "─" * 78 + "║")
    row("Total operaciones",  f"{n:>10}")
    row("  ✅ Ganadoras",     f"{n_win:>10}  ({n_win/n*100:.1f}%)")
    row("  ❌ Perdedoras",    f"{n_los:>10}  ({n_los/n*100:.1f}%)")
    row("  ⚠️  Break-even",   f"{n_be:>10}  ({n_be/n*100:.1f}%)" if n_be else "           0")
    row("Win Rate",           f"{n_win/n*100:>9.1f}%")
    row("Ganancia media/op",  f"${avg_win:>+10,.4f}")
    row("Pérdida media/op",   f"${avg_loss:>+10,.4f}")
    row("Ratio Reward/Risk",  f"{rr:>10.2f}")
    row("Duración media",     f"{avg_dur:>9.1f}  velas")
    row("Confianza promedio", f"{avg_conf:>9.1%}")

    print("║" + "─" * 78 + "║")
    print("║" + "  LONG vs SHORT".ljust(78) + "║")
    print("║" + "─" * 78 + "║")
    row("Operaciones LONG",   f"{n_long:>4}  │  Ganadoras: {long_wins}  ({long_wins/n_long*100:.1f}%)" if n_long else "           0")
    row("Operaciones SHORT",  f"{n_short:>4}  │  Ganadoras: {short_wins}  ({short_wins/n_short*100:.1f}%)" if n_short else "           0")
    row("P&L LONG",           f"${df_t[df_t['direction']=='LONG']['pnl_usdt'].sum():>+10,.4f}")
    row("P&L SHORT",          f"${df_t[df_t['direction']=='SHORT']['pnl_usdt'].sum():>+10,.4f}")

    print("║" + "─" * 78 + "║")
    print("║" + "  SÍMBOLOS MÁS OPERADOS".ljust(78) + "║")
    print("║" + "─" * 78 + "║")
    sym_stats = df_t.groupby("pred_sym").agg(
        trades=("pnl_usdt", "count"),
        pnl=("pnl_usdt", "sum"),
        wins=("result", lambda x: (x == "WIN").sum()),
    ).sort_values("trades", ascending=False).head(8)
    for sym, r in sym_stats.iterrows():
        wr = r["wins"] / r["trades"] * 100
        row(f"  Símbolo [{sym}]",
            f"{int(r['trades']):>3} ops │ WR: {wr:4.1f}% │ P&L: ${r['pnl']:>+8,.2f}")

    print("║" + "─" * 78 + "║")
    print("║" + "  CURVA DE EQUITY".ljust(78) + "║")
    print("║" + "─" * 78 + "║")
    for threshold, bar in chart_rows:
        label = f"${threshold:>8,.2f} │"
        print(f"║  {label}{bar[:width]}  ║")
    print("║" + f"{'':>10} └" + "─" * width + "  ║")
    print("║" + f"{'':>11}{'▸ Operaciones →':^{width}}  ║")

    print("╚" + "═" * 78 + "╝")


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECCIÓN DE SÍMBOLOS — REPORTE
# ══════════════════════════════════════════════════════════════════════════════

def print_symbol_map(dirs: dict[str, str]) -> None:
    longs   = [s for s, d in dirs.items() if d == "LONG"]
    shorts  = [s for s, d in dirs.items() if d == "SHORT"]
    neutral = [s for s, d in dirs.items() if d == "NEUTRAL"]
    print("\n📋 Mapa de símbolos:")
    print(f"   ▲ LONG    ({len(longs):>2}): {' '.join(longs)}")
    print(f"   ▼ SHORT   ({len(shorts):>2}): {' '.join(shorts)}")
    print(f"   ─ NEUTRAL ({len(neutral):>2}): {' '.join(neutral)}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Binance HYPEUSDT Simulator")
    parser.add_argument("--csv",        default=None,  help="CSV local en lugar de Binance API")
    parser.add_argument("--model-dir",  default="candle_model", help="Directorio del modelo")
    parser.add_argument("--symbol",     default=SYMBOL)
    parser.add_argument("--interval",   default=INTERVAL)
    parser.add_argument("--limit",      default=N_KLINES, type=int)
    parser.add_argument("--warmup",     default=WARMUP,   type=int)
    parser.add_argument("--tp",         default=TP_PCT,   type=float, help="Take profit (ej: 0.01 = 1%%)")
    parser.add_argument("--sl",         default=SL_PCT,   type=float, help="Stop loss (ej: 0.01 = 1%%)")
    parser.add_argument("--max-hold",   default=MAX_HOLD, type=int)
    parser.add_argument("--confidence", default=MIN_CONFIDENCE, type=float)
    parser.add_argument("--capital",    default=CAPITAL,  type=float)
    parser.add_argument("--pos-size",   default=POS_SIZE_PCT, type=float)
    parser.add_argument("--fee",        default=FEE_PCT,  type=float)
    parser.add_argument("--no-fee",     action="store_true", help="Simular sin comisiones")
    parser.add_argument("--dir-thresh", default=DIR_THRESHOLD, type=float)
    parser.add_argument("--retrain",    action="store_true", help="Re-entrenar con datos descargados")
    args = parser.parse_args()

    fee = 0.0 if args.no_fee else args.fee

    print("\n" + "═" * 60)
    print(f"  🚀 Binance HYPEUSDT Candle Pattern Simulator")
    print(f"  TP: {args.tp*100:.1f}%  │  SL: {args.sl*100:.1f}%  │  "
          f"MaxHold: {args.max_hold}  │  Confianza: {args.confidence:.0%}")
    print(f"  Capital: ${args.capital:,.0f}  │  PosSize: {args.pos_size*100:.0f}%  │  "
          f"Fee: {fee*100:.2f}%/lado")
    print("═" * 60)

    # ── 1. Obtener datos ─────────────────────────────────────────────────────
    if args.csv:
        df = load_csv(args.csv)
    else:
        df = download_binance(args.symbol, args.interval, args.limit)

    if len(df) < args.warmup + 20:
        log.error(f"Datos insuficientes: {len(df)} velas (mínimo {args.warmup + 20})")
        sys.exit(1)

    # ── 2. Cargar o entrenar modelo ──────────────────────────────────────────
    model_path = Path(args.model_dir)
    system     = CandlePatternSystem(model_dir=model_path)

    if not model_path.exists() or args.retrain:
        split = int(len(df) * TRAIN_RATIO)
        train_df = df.iloc[:split].copy()
        log.info(f"Modelo no encontrado. Entrenando con primeras {len(train_df)} velas...")
        system.train_df(train_df)   # (método auxiliar definido abajo)
    else:
        log.info(f"Cargando modelo desde {model_path}/")
        system = CandlePatternSystem.load(model_path)

    # ── 3. Calcular direccionalidad de símbolos ──────────────────────────────
    dirs = compute_symbol_directions(system.encoder, threshold=args.dir_thresh)
    print_symbol_map(dirs)

    # ── 4. Simulación vela a vela ────────────────────────────────────────────
    print(f"\n  Simulando {len(df) - args.warmup} velas "
          f"(con {args.warmup} de calentamiento)...\n")
    trades, equity = run_simulation(
        df, system, dirs,
        warmup=args.warmup,
        min_confidence=args.confidence,
        tp_pct=args.tp, sl_pct=args.sl,
        max_hold=args.max_hold,
        fee_pct=fee,
        capital=args.capital,
        pos_size_pct=args.pos_size,
    )

    # ── 5. Estadísticas ──────────────────────────────────────────────────────
    print_statistics(trades, equity, initial_capital=args.capital)
    print(f"\n✅ Simulación completada — {len(trades)} operaciones sobre {len(df)} velas")


# ── Método auxiliar de entrenamiento ────────────────────────────────────────

def _train_df(self, df: pd.DataFrame, val_split: float = 0.15) -> dict:
    """Entrena el sistema directamente desde un DataFrame (sin CSV)."""
    log.info(f"Entrenando sobre {len(df)} velas...")
    sequence = self.encoder.fit_transform(df)
    split    = int(len(sequence) * (1 - val_split))
    self.predictor.fit(sequence[:split])
    self.save()
    if len(sequence) - split > self.predictor.max_n + 1:
        return self.predictor.evaluate(sequence[split:])
    return {}

CandlePatternSystem.train_df = _train_df   # monkey-patch


if __name__ == "__main__":
    main()
