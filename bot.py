#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          HYPEUSDT Candle Pattern Bot — Render Web Service                   ║
║                                                                              ║
║  • Carga (o entrena) el modelo CandlePatternSystem automáticamente.          ║
║  • Loop cada 60 s: descarga velas de Binance → predice → simula trades.     ║
║  • Expone dashboard web en / y endpoints JSON en /api/*.                    ║
║                                                                              ║
║  Variables de entorno (todas opcionales, tienen defaults):                   ║
║    SYMBOL, INTERVAL, CAPITAL, TP_PCT, SL_PCT, MIN_CONFIDENCE,               ║
║    POS_SIZE_PCT, FEE_PCT, MAX_HOLD, DIR_THRESHOLD, LOOP_INTERVAL,           ║
║    N_KLINES_INIT, N_KLINES_LIVE, MODEL_DIR, PORT                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, Response

# ── Importar el sistema predictor (mismo directorio) ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from candle_predictor import CandlePatternSystem, CandleEncoder, NgramPredictor,SYMBOLS

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN VÍA ENTORNO
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL          = os.getenv("SYMBOL",          "HYPEUSDT")
INTERVAL        = os.getenv("INTERVAL",        "1m")
MODEL_DIR       = Path(os.getenv("MODEL_DIR",  "candle_model"))
CAPITAL         = float(os.getenv("CAPITAL",         "1000.0"))
TP_PCT          = float(os.getenv("TP_PCT",          "0.009"))
SL_PCT          = float(os.getenv("SL_PCT",          "0.010"))
MIN_CONFIDENCE  = float(os.getenv("MIN_CONFIDENCE",  "0.50"))
POS_SIZE_PCT    = float(os.getenv("POS_SIZE_PCT",    "0.012"))
FEE_PCT         = float(os.getenv("FEE_PCT",         "0.0004"))
MAX_HOLD        = int(os.getenv("MAX_HOLD",          "60"))
DIR_THRESHOLD   = float(os.getenv("DIR_THRESHOLD",   "0.02"))
LOOP_INTERVAL   = int(os.getenv("LOOP_INTERVAL",     "60"))   # segundos entre loops
N_KLINES_INIT   = int(os.getenv("N_KLINES_INIT",     "1500")) # klines para entrenamiento inicial
N_KLINES_LIVE   = int(os.getenv("N_KLINES_LIVE",     "200"))  # klines para predicción en vivo
PORT            = int(os.getenv("PORT",               "10000"))
TRAIN_RATIO     = float(os.getenv("TRAIN_RATIO",      "0.85"))
BINANCE_API     = "https://fapi.binance.com/fapi/v1/klines"
MAX_TRADES_LOG  = 200  # máximo de trades guardados en memoria

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bot")

# ══════════════════════════════════════════════════════════════════════════════
#  ESTADO GLOBAL (compartido entre el loop y Flask)
# ══════════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()

state = {
    # Financiero
    "capital":          CAPITAL,
    "initial_capital":  CAPITAL,
    "equity":           [CAPITAL],
    "trades":           [],            # lista de dicts con historial completo
    # Posición abierta (None si no hay)
    "position":         None,
    # Última vela procesada
    "last_candle":      None,
    "last_price":       None,
    # Última señal generada
    "last_signal":      None,
    # Metadata del bot
    "status":           "iniciando",
    "error":            None,
    "started_at":       datetime.now(timezone.utc).isoformat(),
    "last_loop_at":     None,
    "loops_completed":  0,
    "model_loaded":     False,
    "symbol_dirs":      {},            # {"A": "LONG", "B": "SHORT", ...}
    "total_candles":    0,
}

# ══════════════════════════════════════════════════════════════════════════════
#  LÓGICA DE BINANCE
# ══════════════════════════════════════════════════════════════════════════════

def download_candles(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Descarga klines de Binance Futures REST API."""
    resp = requests.get(
        BINANCE_API,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=20,
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
    # Columnas requeridas por el encoder (valores placeholder si no hay datos reales)
    df["EMA_200"] = ""
    df["RSI_14"]  = 0.0
    df["ATR_14"]  = 0.0
    df = df[["open_time","open","high","low","close","volume",
             "EMA_200","RSI_14","ATR_14","num_trades"]].copy()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECCIONALIDAD DE CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_symbol_directions(encoder: CandleEncoder,
                               threshold: float = DIR_THRESHOLD) -> dict:
    """Devuelve {symbol: 'LONG'|'SHORT'|'NEUTRAL'} analizando centroides KMeans."""
    centroids_scaled = encoder.kmeans.cluster_centers_
    centroids        = encoder.scaler.inverse_transform(centroids_scaled)
    body_cols        = list(range(1, 40, 4))   # índices pct_body en el vector de 40 features
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
#  GESTIÓN DE MODELO
# ══════════════════════════════════════════════════════════════════════════════

def load_or_train_model() -> tuple[CandlePatternSystem, dict]:
    """
    1. Intenta cargar el modelo desde MODEL_DIR.
    2. Si no existe, descarga datos de Binance y entrena desde cero.
    Devuelve (system, dirs).
    """
    if MODEL_DIR.exists() and (MODEL_DIR / "encoder_meta.json").exists():
        log.info(f"Cargando modelo desde {MODEL_DIR}/...")
        system = CandlePatternSystem.load(MODEL_DIR)
        log.info("✅ Modelo cargado exitosamente.")
    else:
        log.info(f"Modelo no encontrado. Descargando {N_KLINES_INIT} velas para entrenar...")
        with _lock:
            state["status"] = "entrenando"
        df = download_candles(SYMBOL, INTERVAL, N_KLINES_INIT)
        log.info(f"  Descargadas {len(df)} velas. Iniciando entrenamiento...")
        system = CandlePatternSystem(model_dir=MODEL_DIR)
        sequence = system.encoder.fit_transform(df)
        split    = int(len(sequence) * TRAIN_RATIO)
        system.predictor.fit(sequence[:split])
        system.save()
        log.info("✅ Modelo entrenado y guardado.")

    dirs = compute_symbol_directions(system.encoder, threshold=DIR_THRESHOLD)
    long_syms  = [s for s, d in dirs.items() if d == "LONG"]
    short_syms = [s for s, d in dirs.items() if d == "SHORT"]
    log.info(f"Direcciones — LONG: {long_syms}  SHORT: {short_syms}")
    return system, dirs


# ══════════════════════════════════════════════════════════════════════════════
#  GESTIÓN DE POSICIÓN
# ══════════════════════════════════════════════════════════════════════════════

def check_position(candle: pd.Series) -> Optional[dict]:
    """
    Verifica si la posición abierta (si existe) alcanzó TP, SL o timeout.
    Devuelve el trade cerrado (dict) o None si la posición sigue abierta.
    Modifica state["position"] y state["capital"] in-place (bajo lock externo).
    """
    pos = state["position"]
    if pos is None:
        return None

    h = candle["high"]
    l = candle["low"]
    c = candle["close"]
    pos["candles_held"] += 1

    trade_result = None

    if pos["direction"] == "LONG":
        if l <= pos["sl_price"]:                       # Stop Loss primero (conservador)
            exit_price   = pos["sl_price"]
            pnl_pct      = -SL_PCT - 2 * FEE_PCT
            trade_result = _close_position(exit_price, pnl_pct, "LOSS", candle)
        elif h >= pos["tp_price"]:                     # Take Profit
            exit_price   = pos["tp_price"]
            pnl_pct      = TP_PCT - 2 * FEE_PCT
            trade_result = _close_position(exit_price, pnl_pct, "WIN", candle)

    else:  # SHORT
        if h >= pos["sl_price"]:
            exit_price   = pos["sl_price"]
            pnl_pct      = -SL_PCT - 2 * FEE_PCT
            trade_result = _close_position(exit_price, pnl_pct, "LOSS", candle)
        elif l <= pos["tp_price"]:
            exit_price   = pos["tp_price"]
            pnl_pct      = TP_PCT - 2 * FEE_PCT
            trade_result = _close_position(exit_price, pnl_pct, "WIN", candle)

    # Timeout
    if trade_result is None and pos["candles_held"] >= MAX_HOLD:
        if pos["direction"] == "LONG":
            pnl_pct = (c - pos["entry_price"]) / pos["entry_price"] - 2 * FEE_PCT
        else:
            pnl_pct = (pos["entry_price"] - c) / pos["entry_price"] - 2 * FEE_PCT
        result = "WIN" if pnl_pct > 0.0005 else ("LOSS" if pnl_pct < -0.0005 else "BREAK_EVEN")
        trade_result = _close_position(c, pnl_pct, result, candle)

    return trade_result


def _close_position(exit_price: float, pnl_pct: float,
                    result: str, candle: pd.Series) -> dict:
    """Cierra la posición actual y actualiza el estado."""
    pos      = state["position"]
    pos_usdt = state["capital"] * POS_SIZE_PCT
    pnl_usdt = pos_usdt * pnl_pct
    state["capital"] = round(state["capital"] + pnl_usdt, 4)
    state["equity"].append(state["capital"])

    trade = {
        "id":            len(state["trades"]) + 1,
        "direction":     pos["direction"],
        "entry_price":   pos["entry_price"],
        "exit_price":    round(exit_price, 6),
        "entry_time":    pos["entry_time"],
        "exit_time":     candle["open_time"].isoformat(),
        "candles_held":  pos["candles_held"],
        "result":        result,
        "pnl_pct":       round(pnl_pct * 100, 4),
        "pnl_usdt":      round(pnl_usdt, 4),
        "capital_after": state["capital"],
        "confidence":    pos["confidence"],
        "pred_symbol":   pos["pred_symbol"],
        "context":       pos["context"],
    }
    state["trades"].append(trade)
    # Limitar historial en memoria
    if len(state["trades"]) > MAX_TRADES_LOG:
        state["trades"] = state["trades"][-MAX_TRADES_LOG:]

    state["position"] = None
    emoji = "✅" if result == "WIN" else ("❌" if result == "LOSS" else "⚠️")
    log.info(
        f"{emoji} TRADE {result} │ {pos['direction']} │ "
        f"entry={pos['entry_price']:.4f} exit={exit_price:.4f} │ "
        f"P&L={pnl_pct*100:+.2f}% (${pnl_usdt:+.4f}) │ "
        f"capital=${state['capital']:,.2f}"
    )
    return trade


def open_position(direction: str, entry_price: float,
                  confidence: float, pred_symbol: str,
                  context: str, candle_time: str) -> None:
    """Abre una nueva posición simulada."""
    if direction == "LONG":
        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)
    else:
        tp_price = entry_price * (1 - TP_PCT)
        sl_price = entry_price * (1 + SL_PCT)

    state["position"] = {
        "direction":    direction,
        "entry_price":  entry_price,
        "tp_price":     round(tp_price, 6),
        "sl_price":     round(sl_price, 6),
        "entry_time":   candle_time,
        "candles_held": 0,
        "confidence":   confidence,
        "pred_symbol":  pred_symbol,
        "context":      context,
    }
    log.info(
        f"📂 ABRIENDO {direction} │ precio={entry_price:.4f} │ "
        f"TP={tp_price:.4f} SL={sl_price:.4f} │ "
        f"confianza={confidence:.1%} (símbolo={pred_symbol})"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LOOP PRINCIPAL DE TRADING
# ══════════════════════════════════════════════════════════════════════════════

def trading_loop(system: CandlePatternSystem, dirs: dict) -> None:
    """
    Ejecuta el loop de trading en un hilo de fondo.
    Cada iteración:
      1. Descarga las últimas N_KLINES_LIVE velas de Binance.
      2. Comprueba si la posición abierta alcanzó TP/SL/timeout.
      3. Si no hay posición, genera señal y abre trade si aplica.
    """
    MAX_N  = system.predictor.max_n
    WINDOW = system.encoder.window_size

    log.info("🟢 Loop de trading iniciado.")

    while True:
        loop_start = time.time()
        try:
            # ── 1. Descargar velas recientes ───────────────────────────────
            df = download_candles(SYMBOL, INTERVAL, N_KLINES_LIVE)
            if len(df) < WINDOW + MAX_N + 5:
                log.warning(f"Velas insuficientes ({len(df)}). Esperando...")
                time.sleep(LOOP_INTERVAL)
                continue

            last_candle = df.iloc[-1]
            current_price = float(last_candle["close"])

            with _lock:
                state["last_candle"] = last_candle["open_time"].isoformat()
                state["last_price"]  = current_price
                state["total_candles"] += len(df)
                state["status"]      = "activo"
                state["error"]       = None

                # ── 2. Verificar posición abierta ──────────────────────────
                if state["position"] is not None:
                    closed = check_position(last_candle)
                    if closed:
                        log.info(f"   Trade #{closed['id']} cerrado.")
                    # Si la posición sigue abierta, no buscamos señal nueva
                    if state["position"] is not None:
                        _update_loop_meta()
                        continue

                # ── 3. Generar señal ───────────────────────────────────────
                try:
                    sequence = system.encoder.transform(df)
                except Exception as enc_err:
                    log.warning(f"Error codificando velas: {enc_err}")
                    _update_loop_meta()
                    continue

                if len(sequence) < MAX_N + 1:
                    _update_loop_meta()
                    continue

                ctx  = sequence[-(MAX_N):]
                pred = system.predictor.predict(list(ctx), top_k=5)

                if not pred:
                    _update_loop_meta()
                    continue

                best       = pred[0]
                confidence = best["probability"]
                next_sym   = best["symbol"]
                direction  = dirs.get(next_sym, "NEUTRAL")

                signal = {
                    "time":        last_candle["open_time"].isoformat(),
                    "price":       current_price,
                    "pred_symbol": next_sym,
                    "direction":   direction,
                    "confidence":  round(confidence, 4),
                    "context":     "".join(ctx[-4:]),
                    "n_used":      best.get("n_used", 0),
                    "top5": [
                        {"sym": p["symbol"],
                         "prob": round(p["probability"], 4),
                         "dir": dirs.get(p["symbol"], "NEUTRAL")}
                        for p in pred
                    ],
                }
                state["last_signal"] = signal

                log.info(
                    f"📊 Señal │ ctx={''.join(ctx[-4:])}→{next_sym} │ "
                    f"dir={direction} │ conf={confidence:.1%}"
                )

                # ── 4. Abrir posición si cumple criterios ──────────────────
                if confidence >= MIN_CONFIDENCE and direction in ("LONG", "SHORT"):
                    entry_price = float(df.iloc[-1]["close"])  # entra al precio de cierre actual
                    open_position(
                        direction    = direction,
                        entry_price  = entry_price,
                        confidence   = confidence,
                        pred_symbol  = next_sym,
                        context      = "".join(ctx[-4:]),
                        candle_time  = last_candle["open_time"].isoformat(),
                    )
                else:
                    reason = (
                        f"confianza baja ({confidence:.1%} < {MIN_CONFIDENCE:.1%})"
                        if confidence < MIN_CONFIDENCE
                        else f"dirección NEUTRAL para {next_sym}"
                    )
                    log.info(f"   Sin señal operativa: {reason}")

                _update_loop_meta()

        except requests.exceptions.RequestException as e:
            log.error(f"Error de red: {e}")
            with _lock:
                state["error"]  = f"Error de red: {str(e)[:120]}"
                state["status"] = "error_red"

        except Exception as e:
            log.exception(f"Error inesperado en el loop: {e}")
            with _lock:
                state["error"]  = str(e)[:200]
                state["status"] = "error"

        # Esperar hasta el próximo ciclo
        elapsed = time.time() - loop_start
        sleep_s = max(5, LOOP_INTERVAL - elapsed)
        log.info(f"⏳ Próximo ciclo en {sleep_s:.0f}s")
        time.sleep(sleep_s)


def _update_loop_meta():
    """Actualiza metadata del loop (llamar dentro de _lock)."""
    state["last_loop_at"]    = datetime.now(timezone.utc).isoformat()
    state["loops_completed"] += 1


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK — DASHBOARD Y API
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)


def _stats() -> dict:
    """Calcula estadísticas agregadas de los trades."""
    trades = state["trades"]
    n = len(trades)
    if n == 0:
        return {"total": 0, "win_rate": 0, "profit_factor": 0,
                "net_pnl": 0, "max_dd_pct": 0}

    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    gross_win  = sum(t["pnl_usdt"] for t in wins)
    gross_loss = abs(sum(t["pnl_usdt"] for t in losses))

    equity = state["equity"]
    eq_arr = np.array(equity, dtype=float)
    running_max = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - running_max) / running_max
    max_dd = float(dd.min()) * 100

    return {
        "total":          n,
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       round(len(wins) / n * 100, 1),
        "net_pnl":        round(sum(t["pnl_usdt"] for t in trades), 4),
        "net_pnl_pct":    round((state["capital"] / state["initial_capital"] - 1) * 100, 2),
        "profit_factor":  round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "avg_duration":   round(sum(t["candles_held"] for t in trades) / n, 1),
        "avg_confidence": round(sum(t["confidence"] for t in trades) / n * 100, 1),
        "max_dd_pct":     round(max_dd, 2),
    }


@app.route("/health")
def health():
    return jsonify({"status": "ok", "bot_status": state["status"]}), 200


@app.route("/api/status")
def api_status():
    with _lock:
        pos = state["position"]
        position_info = None
        if pos:
            current_price = state.get("last_price") or pos["entry_price"]
            if pos["direction"] == "LONG":
                unrealized_pct = (current_price - pos["entry_price"]) / pos["entry_price"] * 100
            else:
                unrealized_pct = (pos["entry_price"] - current_price) / pos["entry_price"] * 100
            position_info = {**pos, "unrealized_pct": round(unrealized_pct, 3)}

        return jsonify({
            "symbol":           SYMBOL,
            "interval":         INTERVAL,
            "status":           state["status"],
            "error":            state["error"],
            "model_loaded":     state["model_loaded"],
            "started_at":       state["started_at"],
            "last_loop_at":     state["last_loop_at"],
            "loops_completed":  state["loops_completed"],
            "last_candle":      state["last_candle"],
            "last_price":       state["last_price"],
            "capital":          round(state["capital"], 2),
            "initial_capital":  state["initial_capital"],
            "position":         position_info,
            "last_signal":      state["last_signal"],
            "symbol_dirs":      state["symbol_dirs"],
            "config": {
                "tp_pct":         TP_PCT,
                "sl_pct":         SL_PCT,
                "min_confidence": MIN_CONFIDENCE,
                "pos_size_pct":   POS_SIZE_PCT,
                "fee_pct":        FEE_PCT,
                "max_hold":       MAX_HOLD,
                "loop_interval":  LOOP_INTERVAL,
            },
            "stats": _stats(),
        })


@app.route("/api/trades")
def api_trades():
    with _lock:
        return jsonify({
            "count":  len(state["trades"]),
            "trades": list(reversed(state["trades"])),  # más reciente primero
        })


@app.route("/api/equity")
def api_equity():
    with _lock:
        return jsonify({
            "initial_capital": state["initial_capital"],
            "current_capital": round(state["capital"], 2),
            "equity":          state["equity"][-500:],  # últimos 500 puntos
        })


@app.route("/")
def dashboard():
    """Dashboard HTML servido en la raíz."""
    html = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>🤖 HYPEUSDT Bot</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Courier New', monospace; background: #0d1117; color: #c9d1d9; }
  .header { background: #161b22; border-bottom: 1px solid #30363d;
            padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  .header h1 { font-size: 1.2rem; color: #f0f6fc; }
  .badge { padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; }
  .badge.green  { background: #1a4731; color: #3fb950; border: 1px solid #3fb950; }
  .badge.yellow { background: #4b3504; color: #d29922; border: 1px solid #d29922; }
  .badge.red    { background: #5c1b1e; color: #f85149; border: 1px solid #f85149; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
          gap: 12px; padding: 20px 24px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 16px; }
  .card .label { font-size: 0.7rem; color: #8b949e; text-transform: uppercase;
                 letter-spacing: 1px; margin-bottom: 6px; }
  .card .value { font-size: 1.4rem; color: #f0f6fc; font-weight: bold; }
  .card .sub   { font-size: 0.75rem; color: #8b949e; margin-top: 4px; }
  .green-val { color: #3fb950 !important; }
  .red-val   { color: #f85149 !important; }
  .section { padding: 0 24px 20px; }
  .section h2 { font-size: 0.85rem; color: #8b949e; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 12px; border-bottom: 1px solid #30363d;
                padding-bottom: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  th { color: #8b949e; text-align: left; padding: 6px 10px;
       border-bottom: 1px solid #30363d; font-weight: normal; }
  td { padding: 6px 10px; border-bottom: 1px solid #21262d; }
  tr:hover td { background: #1c2128; }
  .win  { color: #3fb950; }
  .loss { color: #f85149; }
  .be   { color: #d29922; }
  .long  { color: #3fb950; }
  .short { color: #f85149; }
  .pos-box { background: #1c2128; border: 1px solid #30363d; border-radius: 8px;
             padding: 16px; font-size: 0.82rem; line-height: 1.8; }
  .pos-box .dir-long  { color: #3fb950; font-weight: bold; font-size: 1rem; }
  .pos-box .dir-short { color: #f85149; font-weight: bold; font-size: 1rem; }
  .no-pos { color: #8b949e; font-style: italic; }
  .refresh { font-size: 0.7rem; color: #8b949e; padding: 0 24px 16px; }
  .signal-box { background: #1c2128; border: 1px solid #30363d;
                border-radius: 8px; padding: 14px; font-size: 0.8rem; }
</style>
</head>
<body>

<div id="app">Cargando...</div>

<script>
async function load() {
  const [s, t] = await Promise.all([
    fetch('/api/status').then(r => r.json()),
    fetch('/api/trades').then(r => r.json()),
  ]);

  const pnl = s.capital - s.initial_capital;
  const pnlPct = ((s.capital / s.initial_capital) - 1) * 100;
  const stats = s.stats;

  const statusColor = s.status === 'activo' ? 'green'
                    : s.status.includes('error') ? 'red' : 'yellow';

  const trades = t.trades.slice(0, 50);

  const posHtml = s.position ? `
    <div class="pos-box">
      <span class="${s.position.direction === 'LONG' ? 'dir-long' : 'dir-short'}">
        ${s.position.direction === 'LONG' ? '▲ LONG' : '▼ SHORT'}
      </span><br>
      Entrada: <b>$${s.position.entry_price.toFixed(4)}</b> &nbsp;|&nbsp;
      TP: <span class="win">$${s.position.tp_price.toFixed(4)}</span> &nbsp;|&nbsp;
      SL: <span class="loss">$${s.position.sl_price.toFixed(4)}</span><br>
      Velas en posición: <b>${s.position.candles_held}</b> / ${s.config.max_hold}<br>
      Confianza: <b>${(s.position.confidence * 100).toFixed(1)}%</b>
      &nbsp;| Símbolo predicho: <b>${s.position.pred_symbol}</b>
      (contexto: <b>${s.position.context}</b>)<br>
      ${s.position.unrealized_pct !== undefined ?
        `P&L no realizado: <b class="${s.position.unrealized_pct >= 0 ? 'win' : 'loss'}">
          ${s.position.unrealized_pct >= 0 ? '+' : ''}${s.position.unrealized_pct.toFixed(3)}%
        </b>` : ''
      }
    </div>` : '<p class="no-pos">Sin posición abierta</p>';

  const sigHtml = s.last_signal ? `
    <div class="signal-box">
      Tiempo: ${s.last_signal.time}<br>
      Precio: <b>$${s.last_signal.price?.toFixed(4) ?? 'N/A'}</b> &nbsp;|&nbsp;
      Contexto: <b>${s.last_signal.context}</b> → <b>${s.last_signal.pred_symbol}</b><br>
      Dirección: <span class="${s.last_signal.direction === 'LONG' ? 'win' : s.last_signal.direction === 'SHORT' ? 'loss' : ''}">
        <b>${s.last_signal.direction}</b></span> &nbsp;|&nbsp;
      Confianza: <b>${(s.last_signal.confidence * 100).toFixed(1)}%</b>
      &nbsp;(n-gram orden ${s.last_signal.n_used})<br>
      Top 5: ${(s.last_signal.top5 || []).map(p =>
        `<span style="color:${p.dir==='LONG'?'#3fb950':p.dir==='SHORT'?'#f85149':'#8b949e'}">
          ${p.sym}(${(p.prob*100).toFixed(0)}%)</span>`).join(' ')}
    </div>` : '<p class="no-pos">Sin señal generada aún</p>';

  const tradeRows = trades.map(t => `
    <tr>
      <td>${t.id}</td>
      <td class="${t.direction === 'LONG' ? 'long' : 'short'}">${t.direction === 'LONG' ? '▲' : '▼'} ${t.direction}</td>
      <td>$${t.entry_price.toFixed(4)}</td>
      <td>$${t.exit_price.toFixed(4)}</td>
      <td class="${t.result === 'WIN' ? 'win' : t.result === 'LOSS' ? 'loss' : 'be'}">
        ${t.result === 'WIN' ? '✅' : t.result === 'LOSS' ? '❌' : '⚠️'} ${t.result}
      </td>
      <td class="${t.pnl_pct >= 0 ? 'win' : 'loss'}">${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(3)}%</td>
      <td class="${t.pnl_usdt >= 0 ? 'win' : 'loss'}">${t.pnl_usdt >= 0 ? '+' : ''}$${t.pnl_usdt.toFixed(3)}</td>
      <td>${t.candles_held}</td>
      <td>${(t.confidence * 100).toFixed(1)}%</td>
      <td style="color:#8b949e;font-size:0.7rem">${t.exit_time?.slice(11,16) ?? ''}</td>
    </tr>`).join('');

  document.getElementById('app').innerHTML = `
  <div class="header">
    <h1>🤖 ${s.symbol} Bot</h1>
    <span class="badge ${statusColor}">${s.status.toUpperCase()}</span>
    ${s.error ? `<span style="color:#f85149;font-size:0.75rem">${s.error}</span>` : ''}
    <span style="flex:1"></span>
    <span style="font-size:0.72rem;color:#8b949e">Precio: <b style="color:#f0f6fc">$${s.last_price?.toFixed(4) ?? 'N/A'}</b>
      &nbsp;| Loop #${s.loops_completed}</span>
  </div>

  <div class="grid">
    <div class="card">
      <div class="label">Capital</div>
      <div class="value">$${s.capital.toLocaleString('es', {minimumFractionDigits:2, maximumFractionDigits:2})}</div>
      <div class="sub ${pnl >= 0 ? 'green-val' : 'red-val'}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)} (${pnlPct.toFixed(2)}%)</div>
    </div>
    <div class="card">
      <div class="label">Operaciones</div>
      <div class="value">${stats.total}</div>
      <div class="sub">✅ ${stats.wins} ganadoras &nbsp; ❌ ${stats.losses} perdedoras</div>
    </div>
    <div class="card">
      <div class="label">Win Rate</div>
      <div class="value ${stats.win_rate >= 50 ? 'green-val' : 'red-val'}">${stats.win_rate}%</div>
      <div class="sub">Conf. media: ${stats.avg_confidence}%</div>
    </div>
    <div class="card">
      <div class="label">Profit Factor</div>
      <div class="value ${(stats.profit_factor ?? 0) >= 1 ? 'green-val' : 'red-val'}">${stats.profit_factor ?? '—'}</div>
      <div class="sub">Max DD: ${stats.max_dd_pct.toFixed(2)}%</div>
    </div>
    <div class="card">
      <div class="label">Configuración</div>
      <div class="value" style="font-size:1rem">TP ${(s.config.tp_pct*100).toFixed(1)}% / SL ${(s.config.sl_pct*100).toFixed(1)}%</div>
      <div class="sub">Conf. mín: ${(s.config.min_confidence*100).toFixed(0)}% | PosSize: ${(s.config.pos_size_pct*100).toFixed(1)}%</div>
    </div>
    <div class="card">
      <div class="label">Última vela</div>
      <div class="value" style="font-size:0.9rem">${s.last_candle?.slice(11,19) ?? 'N/A'}</div>
      <div class="sub">Actualiza cada ${s.config.loop_interval}s | Loop #${s.loops_completed}</div>
    </div>
  </div>

  <div class="section">
    <h2>Posición Actual</h2>
    ${posHtml}
  </div>

  <div class="section">
    <h2>Última Señal del Modelo</h2>
    ${sigHtml}
  </div>

  <div class="section">
    <h2>Historial de Trades (últimos 50)</h2>
    ${trades.length === 0
      ? '<p class="no-pos">Sin trades aún.</p>'
      : `<table>
          <thead><tr>
            <th>#</th><th>Dir</th><th>Entrada</th><th>Salida</th>
            <th>Resultado</th><th>P&L%</th><th>P&L$</th>
            <th>Velas</th><th>Conf</th><th>Hora cierre</th>
          </tr></thead>
          <tbody>${tradeRows}</tbody>
        </table>`
    }
  </div>

  <div class="refresh">⟳ Auto-recarga cada 30 s | ${new Date().toLocaleTimeString()}</div>
  `;
}

load().catch(e => {
  document.getElementById('app').innerHTML =
    '<p style="padding:24px;color:#f85149">Error cargando datos: ' + e + '</p>';
});
</script>
</body>
</html>"""
    return Response(html, mimetype="text/html")


# ══════════════════════════════════════════════════════════════════════════════
#  ARRANQUE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info(f"  🚀 HYPEUSDT Candle Pattern Bot")
    log.info(f"  Símbolo: {SYMBOL} | Intervalo: {INTERVAL}")
    log.info(f"  Capital: ${CAPITAL:,.0f} | TP: {TP_PCT*100:.1f}% | SL: {SL_PCT*100:.1f}%")
    log.info(f"  Conf. mín: {MIN_CONFIDENCE:.0%} | PosSize: {POS_SIZE_PCT*100:.1f}%")
    log.info(f"  Loop: cada {LOOP_INTERVAL}s | MaxHold: {MAX_HOLD} velas")
    log.info("=" * 60)

    # ── Cargar / entrenar modelo ───────────────────────────────────────────────
    try:
        system, dirs = load_or_train_model()
        with _lock:
            state["model_loaded"] = True
            state["symbol_dirs"]  = dirs
            state["status"]       = "modelo_listo"
    except Exception as e:
        log.exception(f"❌ Error cargando/entrenando modelo: {e}")
        with _lock:
            state["status"] = "error_modelo"
            state["error"]  = str(e)
        # No matamos el proceso; Flask sigue corriendo para health checks
        system = None
        dirs   = {}

    # ── Iniciar loop en hilo de fondo ──────────────────────────────────────────
    if system is not None:
        t = threading.Thread(
            target=trading_loop,
            args=(system, dirs),
            daemon=True,
            name="trading-loop",
        )
        t.start()
    else:
        log.error("Loop de trading NO iniciado (modelo no disponible).")

    # ── Iniciar Flask ──────────────────────────────────────────────────────────
    log.info(f"🌐 Dashboard disponible en http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
