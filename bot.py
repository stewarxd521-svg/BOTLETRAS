#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          HYPEUSDT Candle Pattern Bot — Render Web Service                   ║
║                                                                              ║
║  Lógica idéntica a run_simulation / simulate_trade de binance_simulator:    ║
║   1. Descarga N_KLINES_LIVE velas de Binance cada LOOP_INTERVAL segundos.  ║
║   2. Si hay posición abierta → recorre TODAS las velas desde la entrada     ║
║      vela a vela buscando TP/SL (SL primero, conservador). Timeout si       ║
║      transcurren MAX_HOLD velas sin alcanzar ninguno.                       ║
║   3. Si no hay posición → codifica las últimas velas con el encoder,        ║
║      predice el próximo símbolo (N-gram), y abre trade si la confianza      ║
║      supera MIN_CONFIDENCE y la dirección no es NEUTRAL.                    ║
║   4. Expone dashboard web en / y endpoints JSON en /api/*.                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, Response

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, Path.cwd()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from candle_predictor import CandlePatternSystem, CandleEncoder, NgramPredictor
except ModuleNotFoundError:
    print(f"\n❌ No se encuentra candle_predictor.py — sys.path={sys.path}\n")
    raise

# Definido localmente para no depender del import de SYMBOLS desde candle_predictor
SYMBOLS: list = list("ABCDEFGHIJKLMNOPQRST")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN VÍA ENTORNO
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL          = os.getenv("SYMBOL",          "HYPEUSDT")
INTERVAL        = os.getenv("INTERVAL",        "1m")
MODEL_DIR       = Path(os.getenv("MODEL_DIR",  "candle_model"))
CAPITAL         = float(os.getenv("CAPITAL",         "1000.0"))
TP_PCT          = float(os.getenv("TP_PCT",          "0.010"))
SL_PCT          = float(os.getenv("SL_PCT",          "0.010"))
MIN_CONFIDENCE  = float(os.getenv("MIN_CONFIDENCE",  "0.50"))
POS_SIZE_PCT    = float(os.getenv("POS_SIZE_PCT",    "0.012"))
FEE_PCT         = float(os.getenv("FEE_PCT",         "0.0004"))
MAX_HOLD        = int(os.getenv("MAX_HOLD",          "1000"))
DIR_THRESHOLD   = float(os.getenv("DIR_THRESHOLD",   "0.02"))
LOOP_INTERVAL   = int(os.getenv("LOOP_INTERVAL",     "60"))
N_KLINES_INIT   = int(os.getenv("N_KLINES_INIT",     "1500"))
N_KLINES_LIVE   = int(os.getenv("N_KLINES_LIVE",     "1400"))  # suficiente para el encoder + historial de la posición
TRAIN_RATIO     = float(os.getenv("TRAIN_RATIO",      "0.85"))
PORT            = int(os.getenv("PORT",               "10000"))
BINANCE_API     = "https://fapi.binance.com/fapi/v1/klines"
MAX_TRADES_LOG  = 200

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
#  ESTADO GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()

state = {
    "capital":          CAPITAL,
    "initial_capital":  CAPITAL,
    "equity":           [CAPITAL],
    "trades":           [],
    # Posición abierta — None si no hay
    # {direction, entry_price, tp_price, sl_price,
    #  entry_time (str ISO), entry_ts (pd.Timestamp UTC),
    #  candles_held, confidence, pred_symbol, context}
    "position":         None,
    "last_price":       None,
    "last_candle_time": None,
    "last_signal":      None,
    "status":           "iniciando",
    "error":            None,
    "started_at":       datetime.now(timezone.utc).isoformat(),
    "last_loop_at":     None,
    "loops_completed":  0,
    "model_loaded":     False,
    "symbol_dirs":      {},
}

# ══════════════════════════════════════════════════════════════════════════════
#  DESCARGA DE VELAS
# ══════════════════════════════════════════════════════════════════════════════

def download_candles(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    resp = requests.get(
        BINANCE_API,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=20,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json(), columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","num_trades",
        "taker_base","taker_quote","ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open","high","low","close","volume"):
        df[col] = df[col].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df["EMA_200"] = ""
    df["RSI_14"]  = 0.0
    df["ATR_14"]  = 0.0
    df = df[["open_time","open","high","low","close","volume",
             "EMA_200","RSI_14","ATR_14","num_trades"]].reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECCIONALIDAD DE CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_symbol_directions(encoder: CandleEncoder,
                               threshold: float = DIR_THRESHOLD) -> dict:
    """Devuelve {symbol: 'LONG'|'SHORT'|'NEUTRAL'} analizando centroides KMeans."""
    centroids = encoder.scaler.inverse_transform(encoder.kmeans.cluster_centers_)
    body_cols  = list(range(1, 40, 4))   # índices pct_body dentro del vector de 40 dims
    dirs = {}
    for k, sym in enumerate(SYMBOLS[:encoder.n_clusters]):
        body_sum = centroids[k, body_cols].sum()
        if   body_sum >  threshold: dirs[sym] = "LONG"
        elif body_sum < -threshold: dirs[sym] = "SHORT"
        else:                       dirs[sym] = "NEUTRAL"
    return dirs


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA / ENTRENAMIENTO DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

def load_or_train_model() -> tuple:
    if MODEL_DIR.exists() and (MODEL_DIR / "encoder_meta.json").exists():
        log.info(f"Cargando modelo desde {MODEL_DIR}/...")
        system = CandlePatternSystem.load(MODEL_DIR)
        log.info("✅ Modelo cargado.")
    else:
        log.info(f"Modelo no encontrado. Descargando {N_KLINES_INIT} velas para entrenar...")
        with _lock:
            state["status"] = "entrenando"
        df  = download_candles(SYMBOL, INTERVAL, N_KLINES_INIT)
        log.info(f"  {len(df)} velas descargadas. Entrenando...")
        system   = CandlePatternSystem(model_dir=MODEL_DIR)
        sequence = system.encoder.fit_transform(df)
        split    = int(len(sequence) * TRAIN_RATIO)
        system.predictor.fit(sequence[:split])
        system.save()
        log.info("✅ Modelo entrenado y guardado.")

    dirs = compute_symbol_directions(system.encoder)
    log.info(f"  LONG : {[s for s,d in dirs.items() if d=='LONG']}")
    log.info(f"  SHORT: {[s for s,d in dirs.items() if d=='SHORT']}")
    return system, dirs


# ══════════════════════════════════════════════════════════════════════════════
#  GESTIÓN DE POSICIÓN  (idéntica a simulate_trade de binance_simulator)
# ══════════════════════════════════════════════════════════════════════════════

def open_position(direction: str, entry_price: float,
                  entry_ts: pd.Timestamp,
                  confidence: float, pred_symbol: str, context: str) -> None:
    """
    Abre una posición simulada.
    entry_ts: timestamp de la vela de señal (la entrada es la SIGUIENTE vela,
              por eso filtramos df[open_time > entry_ts] al verificar).
    """
    tp_price = entry_price * (1 + TP_PCT) if direction == "LONG" else entry_price * (1 - TP_PCT)
    sl_price = entry_price * (1 - SL_PCT) if direction == "LONG" else entry_price * (1 + SL_PCT)

    state["position"] = {
        "direction":   direction,
        "entry_price": round(entry_price, 6),
        "tp_price":    round(tp_price, 6),
        "sl_price":    round(sl_price, 6),
        "entry_time":  entry_ts.isoformat(),   # string para JSON
        "entry_ts":    entry_ts,               # pd.Timestamp para filtrar df
        "candles_held": 0,
        "confidence":  confidence,
        "pred_symbol": pred_symbol,
        "context":     context,
    }
    log.info(
        f"📂 ABRE {direction} │ entrada={entry_price:.5f} │ "
        f"TP={tp_price:.5f} SL={sl_price:.5f} │ "
        f"conf={confidence:.1%} ctx={context}→{pred_symbol}"
    )


def scan_position(df: pd.DataFrame) -> Optional[dict]:
    """
    ★ LÓGICA PRINCIPAL ★
    Recorre TODAS las velas del df posteriores a entry_ts, vela a vela,
    buscando TP/SL en cada una (igual que simulate_trade).
    SL se verifica ANTES que TP (conservador).
    Si se superan MAX_HOLD velas → cierre por timeout al close de esa vela.
    Retorna el trade cerrado o None si la posición sigue abierta.
    """
    pos = state["position"]
    if pos is None:
        return None

    # Velas POSTERIORES a la señal (la entrada es la vela siguiente a la señal)
    future = df[df["open_time"] > pos["entry_ts"]].reset_index(drop=True)

    if future.empty:
        log.info(f"   ⏳ Posición {pos['direction']} abierta — esperando primera vela post-entrada")
        return None

    for i in range(len(future)):
        row  = future.iloc[i]
        h    = float(row["high"])
        l    = float(row["low"])
        c    = float(row["close"])
        held = i + 1   # cuántas velas han transcurrido desde la entrada

        if pos["direction"] == "LONG":
            if l <= pos["sl_price"]:                        # ← SL primero
                return _register_close(pos["sl_price"],
                                       -SL_PCT - 2*FEE_PCT, "LOSS", row, held)
            if h >= pos["tp_price"]:
                return _register_close(pos["tp_price"],
                                        TP_PCT - 2*FEE_PCT, "WIN",  row, held)
        else:  # SHORT
            if h >= pos["sl_price"]:                        # ← SL primero
                return _register_close(pos["sl_price"],
                                       -SL_PCT - 2*FEE_PCT, "LOSS", row, held)
            if l <= pos["tp_price"]:
                return _register_close(pos["tp_price"],
                                        TP_PCT - 2*FEE_PCT, "WIN",  row, held)

        # Timeout: MAX_HOLD velas sin tocar TP ni SL
        if held >= MAX_HOLD:
            if pos["direction"] == "LONG":
                pnl_pct = (c - pos["entry_price"]) / pos["entry_price"] - 2*FEE_PCT
            else:
                pnl_pct = (pos["entry_price"] - c) / pos["entry_price"] - 2*FEE_PCT
            result = ("WIN" if pnl_pct > 0.0005
                      else "LOSS" if pnl_pct < -0.0005
                      else "BREAK_EVEN")
            return _register_close(c, pnl_pct, result, row, held)

    # Sigue abierta — actualizar contador y loguear estado
    pos["candles_held"] = len(future)
    current_price = float(df.iloc[-1]["close"])
    if pos["direction"] == "LONG":
        unrealized = (current_price - pos["entry_price"]) / pos["entry_price"] * 100
    else:
        unrealized = (pos["entry_price"] - current_price) / pos["entry_price"] * 100
    log.info(
        f"   ⏳ {pos['direction']} activa │ "
        f"velas={pos['candles_held']}/{MAX_HOLD} │ "
        f"entrada={pos['entry_price']:.5f} actual={current_price:.5f} │ "
        f"P&L no realizado={unrealized:+.3f}%"
    )
    return None


def _register_close(exit_price: float, pnl_pct: float,
                    result: str, row: pd.Series, candles_held: int) -> dict:
    """Registra el cierre de una posición y actualiza el capital."""
    pos      = state["position"]
    pos_usdt = state["capital"] * POS_SIZE_PCT
    pnl_usdt = round(pos_usdt * pnl_pct, 4)
    state["capital"] = round(state["capital"] + pnl_usdt, 4)
    state["equity"].append(state["capital"])

    trade = {
        "id":            len(state["trades"]) + 1,
        "direction":     pos["direction"],
        "entry_price":   pos["entry_price"],
        "exit_price":    round(float(exit_price), 6),
        "entry_time":    pos["entry_time"],
        "exit_time":     row["open_time"].isoformat(),
        "candles_held":  candles_held,
        "result":        result,
        "pnl_pct":       round(pnl_pct * 100, 4),
        "pnl_usdt":      pnl_usdt,
        "capital_after": state["capital"],
        "confidence":    pos["confidence"],
        "pred_symbol":   pos["pred_symbol"],
        "context":       pos["context"],
    }
    state["trades"].append(trade)
    if len(state["trades"]) > MAX_TRADES_LOG:
        state["trades"] = state["trades"][-MAX_TRADES_LOG:]

    state["position"] = None

    emoji = "✅" if result == "WIN" else ("❌" if result == "LOSS" else "⚠️")
    log.info(
        f"{emoji} CIERRE {result} │ {pos['direction']} │ "
        f"entrada={pos['entry_price']:.5f} salida={exit_price:.5f} │ "
        f"velas={candles_held} │ "
        f"P&L={pnl_pct*100:+.3f}% (${pnl_usdt:+.4f}) │ "
        f"capital=${state['capital']:,.2f}"
    )
    return trade


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE SEÑAL
# ══════════════════════════════════════════════════════════════════════════════

def generate_signal(df: pd.DataFrame, system: CandlePatternSystem,
                    dirs: dict) -> Optional[dict]:
    """
    Codifica el df completo → secuencia de símbolos → predice el siguiente.
    Retorna un dict con la señal o None si no hay señal operable.
    """
    MAX_N  = system.predictor.max_n
    WINDOW = system.encoder.window_size

    if len(df) < WINDOW + MAX_N + 5:
        log.warning(f"   Velas insuficientes para señal ({len(df)} < {WINDOW+MAX_N+5})")
        return None

    try:
        sequence = system.encoder.transform(df)
    except Exception as e:
        log.warning(f"   Error en encoder: {e}")
        return None

    if len(sequence) < MAX_N + 1:
        return None

    # Contexto: últimos MAX_N símbolos de la secuencia
    ctx  = sequence[-MAX_N:]
    pred = system.predictor.predict(list(ctx), top_k=5)
    if not pred:
        return None

    best       = pred[0]
    confidence = best["probability"]
    next_sym   = best["symbol"]
    direction  = dirs.get(next_sym, "NEUTRAL")

    signal = {
        "time":        df.iloc[-1]["open_time"].isoformat(),
        "price":       float(df.iloc[-1]["close"]),
        "pred_symbol": next_sym,
        "direction":   direction,
        "confidence":  round(confidence, 4),
        "context":     "".join(ctx[-4:]),
        "n_used":      best.get("n_used", 0),
        "top5": [
            {"sym": p["symbol"],
             "prob": round(p["probability"], 4),
             "dir":  dirs.get(p["symbol"], "NEUTRAL")}
            for p in pred
        ],
    }

    log.info(
        f"📊 Señal │ ctx={''.join(ctx[-4:])}→{next_sym} │ "
        f"dir={direction} │ conf={confidence:.1%} │ n={best.get('n_used',0)}"
    )
    return signal


# ══════════════════════════════════════════════════════════════════════════════
#  LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def trading_loop(system: CandlePatternSystem, dirs: dict) -> None:
    """
    Loop de fondo que se ejecuta cada LOOP_INTERVAL segundos.

    Flujo en cada iteración
    ───────────────────────
    1. Descarga N_KLINES_LIVE velas de Binance (>= 300 para cubrir
       el warmup del encoder + hasta MAX_HOLD velas de la posición).

    2. Si hay posición abierta:
       → scan_position(df) recorre vela a vela desde entry_ts buscando
         TP/SL/timeout, exactamente como simulate_trade.
       → Si la posición sigue abierta, logueamos estado y continuamos.
       → Si la posición se cerró dentro de este mismo batch de velas,
         caemos al paso 3 para buscar una nueva señal de inmediato.

    3. Si no hay posición:
       → generate_signal(df) codifica y predice.
       → Si la señal supera MIN_CONFIDENCE y la dirección es operable,
         abrimos posición. La entrada es el CLOSE de la última vela
         (aproximación del open de la siguiente vela).
    """
    log.info("🟢 Loop de trading iniciado.")

    while True:
        loop_start = time.time()
        try:
            # ── 1. Descargar velas ─────────────────────────────────────────
            df = download_candles(SYMBOL, INTERVAL, N_KLINES_LIVE)
            last_row      = df.iloc[-1]
            current_price = float(last_row["close"])

            with _lock:
                state["last_price"]       = current_price
                state["last_candle_time"] = last_row["open_time"].isoformat()
                state["status"]           = "activo"
                state["error"]            = None

                # ── 2. Verificar posición abierta ──────────────────────────
                if state["position"] is not None:
                    closed = scan_position(df)
                    if closed:
                        log.info(f"   Trade #{closed['id']} cerrado en este batch.")
                        # Posición cerrada: buscamos nueva señal en el mismo ciclo ↓
                    else:
                        # Sigue abierta → no generamos señal nueva
                        _tick()
                        # Seguir al sleep sin continue para no bloquear el lock
                        # (el continue está fuera del with)

                # ── 3. Generar señal (solo si no hay posición) ─────────────
                if state["position"] is None:
                    signal = generate_signal(df, system, dirs)
                    state["last_signal"] = signal

                    if signal and signal["confidence"] >= MIN_CONFIDENCE \
                               and signal["direction"] in ("LONG", "SHORT"):
                        open_position(
                            direction    = signal["direction"],
                            entry_price  = signal["price"],           # close de la última vela
                            entry_ts     = last_row["open_time"],     # timestamp para filtrar futuras velas
                            confidence   = signal["confidence"],
                            pred_symbol  = signal["pred_symbol"],
                            context      = signal["context"],
                        )
                    elif signal:
                        reason = (
                            f"confianza baja ({signal['confidence']:.1%} < {MIN_CONFIDENCE:.1%})"
                            if signal["confidence"] < MIN_CONFIDENCE
                            else f"dirección NEUTRAL para {signal['pred_symbol']}"
                        )
                        log.info(f"   Sin entrada: {reason}")

                _tick()

        except requests.exceptions.RequestException as e:
            log.error(f"Error de red: {e}")
            with _lock:
                state["error"]  = f"Error de red: {str(e)[:150]}"
                state["status"] = "error_red"

        except Exception as e:
            log.exception(f"Error en loop: {e}")
            with _lock:
                state["error"]  = str(e)[:200]
                state["status"] = "error"

        elapsed = time.time() - loop_start
        sleep_s = max(5, LOOP_INTERVAL - elapsed)
        log.info(f"⏳ Próximo ciclo en {sleep_s:.0f}s  (loop tomó {elapsed:.1f}s)")
        time.sleep(sleep_s)


def _tick():
    state["last_loop_at"]    = datetime.now(timezone.utc).isoformat()
    state["loops_completed"] += 1


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK — DASHBOARD Y API
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)


def _stats() -> dict:
    trades = state["trades"]
    n = len(trades)
    if n == 0:
        return {"total":0,"wins":0,"losses":0,"win_rate":0,
                "net_pnl":0,"net_pnl_pct":0,"profit_factor":None,
                "avg_duration":0,"avg_confidence":0,"max_dd_pct":0}
    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    gw = sum(t["pnl_usdt"] for t in wins)
    gl = abs(sum(t["pnl_usdt"] for t in losses))
    eq = np.array(state["equity"], dtype=float)
    rm = np.maximum.accumulate(eq)
    dd = float(((eq - rm) / rm).min()) * 100
    return {
        "total":          n,
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       round(len(wins)/n*100, 1),
        "net_pnl":        round(sum(t["pnl_usdt"] for t in trades), 4),
        "net_pnl_pct":    round((state["capital"]/state["initial_capital"]-1)*100, 2),
        "profit_factor":  round(gw/gl, 2) if gl > 0 else None,
        "avg_duration":   round(sum(t["candles_held"] for t in trades)/n, 1),
        "avg_confidence": round(sum(t["confidence"] for t in trades)/n*100, 1),
        "max_dd_pct":     round(dd, 2),
    }


@app.route("/health")
def health():
    return jsonify({"status": "ok", "bot_status": state["status"]}), 200


@app.route("/api/status")
def api_status():
    with _lock:
        pos = state["position"]
        pos_out = None
        if pos:
            cp = state["last_price"] or pos["entry_price"]
            if pos["direction"] == "LONG":
                unreal = (cp - pos["entry_price"]) / pos["entry_price"] * 100
            else:
                unreal = (pos["entry_price"] - cp) / pos["entry_price"] * 100
            pos_out = {k: v for k, v in pos.items() if k != "entry_ts"}
            pos_out["unrealized_pct"] = round(unreal, 3)

        return jsonify({
            "symbol":          SYMBOL,
            "interval":        INTERVAL,
            "status":          state["status"],
            "error":           state["error"],
            "model_loaded":    state["model_loaded"],
            "started_at":      state["started_at"],
            "last_loop_at":    state["last_loop_at"],
            "loops_completed": state["loops_completed"],
            "last_candle_time":state["last_candle_time"],
            "last_price":      state["last_price"],
            "capital":         round(state["capital"], 2),
            "initial_capital": state["initial_capital"],
            "position":        pos_out,
            "last_signal":     state["last_signal"],
            "symbol_dirs":     state["symbol_dirs"],
            "config": {
                "tp_pct":         TP_PCT,
                "sl_pct":         SL_PCT,
                "min_confidence": MIN_CONFIDENCE,
                "pos_size_pct":   POS_SIZE_PCT,
                "fee_pct":        FEE_PCT,
                "max_hold":       MAX_HOLD,
                "loop_interval":  LOOP_INTERVAL,
                "n_klines_live":  N_KLINES_LIVE,
            },
            "stats": _stats(),
        })


@app.route("/api/trades")
def api_trades():
    with _lock:
        return jsonify({
            "count":  len(state["trades"]),
            "trades": list(reversed(state["trades"])),
        })


@app.route("/api/equity")
def api_equity():
    with _lock:
        return jsonify({
            "initial_capital": state["initial_capital"],
            "current_capital": round(state["capital"], 2),
            "equity":          state["equity"][-500:],
        })


@app.route("/")
def dashboard():
    html = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>🤖 HYPEUSDT Bot</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Courier New',monospace;background:#0d1117;color:#c9d1d9}
.hdr{background:#161b22;border-bottom:1px solid #30363d;
     padding:14px 22px;display:flex;align-items:center;gap:10px}
.hdr h1{font-size:1.1rem;color:#f0f6fc}
.badge{padding:3px 9px;border-radius:10px;font-size:.72rem;font-weight:bold}
.g{background:#1a4731;color:#3fb950;border:1px solid #3fb950}
.y{background:#4b3504;color:#d29922;border:1px solid #d29922}
.r{background:#5c1b1e;color:#f85149;border:1px solid #f85149}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
      gap:10px;padding:16px 22px}
.card{background:#161b22;border:1px solid #30363d;border-radius:7px;padding:14px}
.lbl{font-size:.68rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:5px}
.val{font-size:1.35rem;color:#f0f6fc;font-weight:bold}
.sub{font-size:.72rem;color:#8b949e;margin-top:3px}
.gv{color:#3fb950!important}.rv{color:#f85149!important}
.sec{padding:0 22px 16px}
.sec h2{font-size:.78rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;
        margin-bottom:10px;border-bottom:1px solid #30363d;padding-bottom:5px}
table{width:100%;border-collapse:collapse;font-size:.76rem}
th{color:#8b949e;text-align:left;padding:5px 8px;border-bottom:1px solid #30363d;font-weight:normal}
td{padding:5px 8px;border-bottom:1px solid #21262d}
tr:hover td{background:#1c2128}
.win{color:#3fb950}.loss{color:#f85149}.be{color:#d29922}
.long{color:#3fb950}.short{color:#f85149}
.pb{background:#1c2128;border:1px solid #30363d;border-radius:7px;
    padding:14px;font-size:.8rem;line-height:1.9}
.dl{color:#3fb950;font-weight:bold;font-size:.95rem}
.ds{color:#f85149;font-weight:bold;font-size:.95rem}
.np{color:#8b949e;font-style:italic}
.sb{background:#1c2128;border:1px solid #30363d;border-radius:7px;
    padding:12px;font-size:.78rem;line-height:1.8}
.ref{font-size:.68rem;color:#8b949e;padding:0 22px 14px}
</style>
</head>
<body>
<div id="app">Cargando...</div>
<script>
async function load(){
  const [s,t]=await Promise.all([
    fetch('/api/status').then(r=>r.json()),
    fetch('/api/trades').then(r=>r.json()),
  ]);
  const p=s.stats; const pnl=s.capital-s.initial_capital;
  const pnlPct=((s.capital/s.initial_capital)-1)*100;
  const sc=s.status==='activo'?'g':s.status.includes('error')?'r':'y';
  const trades=t.trades.slice(0,60);

  const posHtml=s.position?`
    <div class="pb">
      <span class="${s.position.direction==='LONG'?'dl':'ds'}">
        ${s.position.direction==='LONG'?'▲ LONG':'▼ SHORT'}
      </span><br>
      Entrada: <b>$${s.position.entry_price.toFixed(5)}</b>
      &nbsp;|&nbsp; TP: <span class="win">$${s.position.tp_price.toFixed(5)}</span>
      &nbsp;|&nbsp; SL: <span class="loss">$${s.position.sl_price.toFixed(5)}</span><br>
      Velas en posición: <b>${s.position.candles_held}</b> / ${s.config.max_hold}<br>
      Confianza: <b>${(s.position.confidence*100).toFixed(1)}%</b>
      &nbsp;| Símbolo: <b>${s.position.pred_symbol}</b>
      &nbsp;| Contexto: <b>${s.position.context}</b><br>
      P&L no realizado:
      <b class="${s.position.unrealized_pct>=0?'win':'loss'}">
        ${s.position.unrealized_pct>=0?'+':''}${s.position.unrealized_pct.toFixed(3)}%
      </b>
    </div>`:'<p class="np">Sin posición abierta</p>';

  const sigHtml=s.last_signal?`
    <div class="sb">
      Hora: ${s.last_signal.time?.slice(11,19)??''}
      &nbsp;| Precio: <b>$${s.last_signal.price?.toFixed(5)??'N/A'}</b>
      &nbsp;| n-gram orden: ${s.last_signal.n_used}<br>
      Ctx: <b>${s.last_signal.context}</b> → <b>${s.last_signal.pred_symbol}</b>
      &nbsp;|&nbsp;
      Dir: <span class="${s.last_signal.direction==='LONG'?'win':s.last_signal.direction==='SHORT'?'loss':''}">
        <b>${s.last_signal.direction}</b></span>
      &nbsp;| Conf: <b>${(s.last_signal.confidence*100).toFixed(1)}%</b><br>
      Top5: ${(s.last_signal.top5||[]).map(x=>
        `<span style="color:${x.dir==='LONG'?'#3fb950':x.dir==='SHORT'?'#f85149':'#8b949e'}">
          ${x.sym}(${(x.prob*100).toFixed(0)}%)</span>`).join(' ')}
    </div>`:'<p class="np">Sin señal generada.</p>';

  const rows=trades.map(t=>`<tr>
    <td>${t.id}</td>
    <td class="${t.direction==='LONG'?'long':'short'}">${t.direction==='LONG'?'▲':'▼'} ${t.direction}</td>
    <td>$${t.entry_price.toFixed(5)}</td>
    <td>$${t.exit_price.toFixed(5)}</td>
    <td class="${t.result==='WIN'?'win':t.result==='LOSS'?'loss':'be'}">
      ${t.result==='WIN'?'✅':t.result==='LOSS'?'❌':'⚠️'} ${t.result}</td>
    <td class="${t.pnl_pct>=0?'win':'loss'}">${t.pnl_pct>=0?'+':''}${t.pnl_pct.toFixed(3)}%</td>
    <td class="${t.pnl_usdt>=0?'win':'loss'}">${t.pnl_usdt>=0?'+':''}$${t.pnl_usdt.toFixed(3)}</td>
    <td>${t.candles_held}</td>
    <td>${(t.confidence*100).toFixed(1)}%</td>
    <td style="color:#8b949e;font-size:.68rem">${t.exit_time?.slice(11,16)??''}</td>
  </tr>`).join('');

  document.getElementById('app').innerHTML=`
  <div class="hdr">
    <h1>🤖 ${s.symbol} Bot</h1>
    <span class="badge ${sc}">${s.status.toUpperCase()}</span>
    ${s.error?`<span style="color:#f85149;font-size:.72rem">${s.error}</span>`:''}
    <span style="flex:1"></span>
    <span style="font-size:.7rem;color:#8b949e">
      Precio: <b style="color:#f0f6fc">$${s.last_price?.toFixed(5)??'N/A'}</b>
      &nbsp;| Loop #${s.loops_completed}
    </span>
  </div>

  <div class="grid">
    <div class="card">
      <div class="lbl">Capital</div>
      <div class="val">$${s.capital.toLocaleString('es',{minimumFractionDigits:2,maximumFractionDigits:2})}</div>
      <div class="sub ${pnl>=0?'gv':'rv'}">${pnl>=0?'+':''}$${pnl.toFixed(2)} (${pnlPct.toFixed(2)}%)</div>
    </div>
    <div class="card">
      <div class="lbl">Operaciones</div>
      <div class="val">${p.total}</div>
      <div class="sub">✅ ${p.wins} &nbsp; ❌ ${p.losses}</div>
    </div>
    <div class="card">
      <div class="lbl">Win Rate</div>
      <div class="val ${p.win_rate>=50?'gv':'rv'}">${p.win_rate}%</div>
      <div class="sub">Conf. media: ${p.avg_confidence}%</div>
    </div>
    <div class="card">
      <div class="lbl">Profit Factor</div>
      <div class="val ${(p.profit_factor??0)>=1?'gv':'rv'}">${p.profit_factor??'—'}</div>
      <div class="sub">Max DD: ${p.max_dd_pct.toFixed(2)}%</div>
    </div>
    <div class="card">
      <div class="lbl">TP / SL / MaxHold</div>
      <div class="val" style="font-size:1rem">${(s.config.tp_pct*100).toFixed(1)}% / ${(s.config.sl_pct*100).toFixed(1)}% / ${s.config.max_hold}</div>
      <div class="sub">Conf. mín: ${(s.config.min_confidence*100).toFixed(0)}% | PosSize: ${(s.config.pos_size_pct*100).toFixed(1)}%</div>
    </div>
    <div class="card">
      <div class="lbl">Última vela</div>
      <div class="val" style="font-size:.9rem">${s.last_candle_time?.slice(11,19)??'N/A'}</div>
      <div class="sub">Cada ${s.config.loop_interval}s | ${s.config.n_klines_live} velas/ciclo</div>
    </div>
  </div>

  <div class="sec"><h2>Posición Actual</h2>${posHtml}</div>
  <div class="sec"><h2>Última Señal del Modelo</h2>${sigHtml}</div>

  <div class="sec">
    <h2>Historial de Trades</h2>
    ${trades.length===0?'<p class="np">Sin trades aún.</p>':
    `<table>
      <thead><tr>
        <th>#</th><th>Dir</th><th>Entrada</th><th>Salida</th>
        <th>Resultado</th><th>P&L%</th><th>P&L$</th>
        <th>Velas</th><th>Conf</th><th>Cierre</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`}
  </div>
  <div class="ref">⟳ Auto-recarga 30s | ${new Date().toLocaleTimeString()}</div>`;
}
load().catch(e=>{
  document.getElementById('app').innerHTML=
    '<p style="padding:24px;color:#f85149">Error: '+e+'</p>';
});
</script>
</body>
</html>"""
    return Response(html, mimetype="text/html")


# ══════════════════════════════════════════════════════════════════════════════
#  ARRANQUE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 62)
    log.info(f"  🚀 {SYMBOL} Candle Pattern Bot")
    log.info(f"  TP={TP_PCT*100:.1f}% | SL={SL_PCT*100:.1f}% | MaxHold={MAX_HOLD} velas")
    log.info(f"  Capital=${CAPITAL:,.0f} | PosSize={POS_SIZE_PCT*100:.1f}% | Fee={FEE_PCT*100:.3f}%/lado")
    log.info(f"  Conf≥{MIN_CONFIDENCE:.0%} | Loop={LOOP_INTERVAL}s | Velas/ciclo={N_KLINES_LIVE}")
    log.info("=" * 62)

    try:
        system, dirs = load_or_train_model()
        with _lock:
            state["model_loaded"] = True
            state["symbol_dirs"]  = dirs
            state["status"]       = "modelo_listo"
    except Exception as e:
        log.exception(f"❌ Error cargando modelo: {e}")
        with _lock:
            state["status"] = "error_modelo"
            state["error"]  = str(e)
        system, dirs = None, {}

    if system is not None:
        t = threading.Thread(
            target=trading_loop, args=(system, dirs),
            daemon=True, name="trading-loop",
        )
        t.start()
    else:
        log.error("⛔ Loop NO iniciado — modelo no disponible.")

    log.info(f"🌐 Dashboard en http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
