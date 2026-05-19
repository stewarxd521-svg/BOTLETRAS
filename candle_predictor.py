#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║           Candle Pattern Predictor System                ║
║                                                          ║
║  Pipeline:                                               ║
║   1. CSV (OHLCV) → Velas en % de cambio                 ║
║   2. Ventanas de 10 velas → Símbolo (A-T, 20 max)       ║
║   3. Secuencia de símbolos → N-gram predictor            ║
║   4. Guarda y actualiza modelo incrementalmente          ║
╚══════════════════════════════════════════════════════════╝

Uso:
    python candle_predictor.py train   datos.csv
    python candle_predictor.py update  datos_nuevos.csv
    python candle_predictor.py predict datos_recientes.csv
    python candle_predictor.py eval    datos_test.csv
"""

import argparse
import json
import logging
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Configuración de logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constantes globales ────────────────────────────────────────────────────────
SYMBOLS     = list("ABCDEFGHIJKLMNOPQRST")   # 20 símbolos posibles
WINDOW_SIZE = 10                              # velas por patrón
N_CLUSTERS  = 20                              # máximo de clusters
MAX_N       = 3                               # orden máximo del n-grama
MODEL_DIR   = Path("candle_model")


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODER: OHLC → % cambio → símbolo
# ══════════════════════════════════════════════════════════════════════════════

class CandleEncoder:
    """
    Convierte velas OHLC crudas en una secuencia de símbolos (A-T).

    Pasos:
      1. Calcula 4 métricas de cambio porcentual por vela:
         - pct_gap    : brecha open_i vs close_{i-1}   (contexto inter-vela)
         - pct_body   : cuerpo (open→close)
         - pct_wick_h : mecha superior
         - pct_wick_l : mecha inferior (negativa = bajo el open)
      2. Agrupa ventanas de `window_size` velas consecutivas (vector de 40 dims).
      3. Aplica KMeans(k=20) para asignar un cluster → símbolo.
    """

    def __init__(self, n_clusters: int = N_CLUSTERS, window_size: int = WINDOW_SIZE):
        self.n_clusters  = n_clusters
        self.window_size = window_size
        self.scaler      = StandardScaler()
        self.kmeans: Optional[KMeans] = None
        self.is_fitted   = False

    # ── Transformaciones ────────────────────────────────────────────────────

    def _to_pct(self, df: pd.DataFrame) -> pd.DataFrame:
        """OHLC → 4 columnas de cambio porcentual por vela."""
        prev_close = df["close"].shift(1)
        pct = pd.DataFrame(index=df.index)

        pct["pct_gap"]    = (df["open"]  - prev_close) / prev_close * 100
        pct["pct_body"]   = (df["close"] - df["open"])  / df["open"] * 100
        pct["pct_wick_h"] = (df["high"]  - df["open"])  / df["open"] * 100
        pct["pct_wick_l"] = (df["low"]   - df["open"])  / df["open"] * 100

        return pct.dropna()

    def _build_windows(self, pct: pd.DataFrame) -> np.ndarray:
        """
        Ventana deslizante → matriz de forma (n_ventanas, window_size × 4).
        Cada fila es el vector aplanado de `window_size` velas consecutivas.
        """
        data = pct.values
        n = len(data) - self.window_size + 1
        if n <= 0:
            raise ValueError(
                f"Datos insuficientes: se necesitan >{self.window_size} velas, "
                f"hay {len(data)} tras calcular % cambio."
            )
        return np.array([data[i : i + self.window_size].flatten() for i in range(n)])

    def _cluster_to_symbol(self, cluster_ids: np.ndarray) -> list[str]:
        return [SYMBOLS[c] for c in cluster_ids]

    # ── API pública ─────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "CandleEncoder":
        log.info("Calculando velas en % de cambio...")
        pct = self._to_pct(df)

        log.info(f"Construyendo ventanas de {self.window_size} velas...")
        windows = self._build_windows(pct)
        log.info(f"  → {len(windows)} ventanas × {windows.shape[1]} features")

        log.info(f"Ajustando KMeans(k={self.n_clusters})...")
        windows_scaled = self.scaler.fit_transform(windows)
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=25,
            max_iter=500,
            random_state=42,
        )
        self.kmeans.fit(windows_scaled)
        self.is_fitted = True

        # Distribución de clusters
        dist = Counter(
            SYMBOLS[c] for c in self.kmeans.labels_
        )
        log.info(f"Distribución de símbolos: {dict(sorted(dist.items()))}")
        inertia = self.kmeans.inertia_
        log.info(f"Inercia KMeans: {inertia:.2f}")
        return self

    def transform(self, df: pd.DataFrame) -> list[str]:
        """Codifica un DataFrame de velas a una lista de símbolos."""
        if not self.is_fitted:
            raise RuntimeError("Encoder no entrenado. Llama a fit() primero.")
        pct     = self._to_pct(df)
        windows = self._build_windows(pct)
        scaled  = self.scaler.transform(windows)
        return self._cluster_to_symbol(self.kmeans.predict(scaled))

    def fit_transform(self, df: pd.DataFrame) -> list[str]:
        self.fit(df)
        return self.transform(df)

    # ── Persistencia ────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.kmeans, directory / "kmeans.joblib")
        joblib.dump(self.scaler, directory / "scaler.joblib")
        meta = {"n_clusters": self.n_clusters, "window_size": self.window_size}
        (directory / "encoder_meta.json").write_text(json.dumps(meta, indent=2))
        log.info(f"Encoder guardado en {directory}/")

    @classmethod
    def load(cls, directory: Path) -> "CandleEncoder":
        directory = Path(directory)
        meta = json.loads((directory / "encoder_meta.json").read_text())
        enc = cls(n_clusters=meta["n_clusters"], window_size=meta["window_size"])
        enc.kmeans   = joblib.load(directory / "kmeans.joblib")
        enc.scaler   = joblib.load(directory / "scaler.joblib")
        enc.is_fitted = True
        log.info(f"Encoder cargado desde {directory}/")
        return enc


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTOR: N-gram con backoff
# ══════════════════════════════════════════════════════════════════════════════

class NgramPredictor:
    """
    Predictor de secuencias basado en N-gramas con backoff jerárquico.

    Estrategia:
      - Almacena conteos para n = 1 … max_n.
      - Al predecir, busca primero el contexto más largo disponible;
        si no existe, retrocede al siguiente orden (backoff).
      - Si no hay ningún contexto conocido, usa distribución unigrama.
      - Soporta actualización incremental sin reentrenar desde cero.
    """

    def __init__(self, max_n: int = MAX_N, symbols: list[str] = SYMBOLS):
        self.max_n    = max_n
        self.symbols  = symbols
        # counts[n][ctx_tuple] = Counter({next_symbol: freq})
        self.counts: dict[int, dict] = {}
        self.total_trained = 0
        self.is_fitted     = False

    # ── Entrenamiento ───────────────────────────────────────────────────────

    def _accumulate(self, sequence: list[str]) -> None:
        """Acumula conteos de n-gramas para todos los órdenes."""
        for n in range(1, self.max_n + 1):
            if n not in self.counts:
                self.counts[n] = defaultdict(Counter)
            for i in range(len(sequence) - n):
                ctx    = tuple(sequence[i : i + n])
                target = sequence[i + n]
                self.counts[n][ctx][target] += 1

    def fit(self, sequence: list[str]) -> "NgramPredictor":
        log.info(f"Entrenando N-gram predictor (max_n={self.max_n}) con {len(sequence)} símbolos...")
        self.counts = {}
        self._accumulate(sequence)
        self.total_trained = len(sequence)
        self.is_fitted = True
        self._log_top_patterns()
        return self

    def update(self, new_sequence: list[str]) -> "NgramPredictor":
        """Actualización incremental con nuevos datos."""
        if not self.is_fitted:
            return self.fit(new_sequence)
        log.info(f"Actualizando modelo con {len(new_sequence)} nuevos símbolos...")
        self._accumulate(new_sequence)
        self.total_trained += len(new_sequence)
        log.info(f"Total de símbolos procesados: {self.total_trained:,}")
        return self

    # ── Predicción ──────────────────────────────────────────────────────────

    def predict(
        self, context: list[str], top_k: int = 5
    ) -> list[dict]:
        """
        Predice los próximos `top_k` símbolos más probables dado un contexto.

        Returns:
            Lista de dicts: [{"symbol": "A", "probability": 0.42, "n_used": 3}, ...]
        """
        if not self.is_fitted:
            raise RuntimeError("Predictor no entrenado. Llama a fit() primero.")

        # Backoff: intenta del orden más alto al más bajo
        for n in range(min(self.max_n, len(context)), 0, -1):
            ctx = tuple(context[-n:])
            if ctx in self.counts.get(n, {}):
                counter = self.counts[n][ctx]
                total   = sum(counter.values())
                top     = counter.most_common(top_k)
                return [
                    {"symbol": sym, "probability": cnt / total, "n_used": n, "count": cnt}
                    for sym, cnt in top
                ]

        # Fallback: distribución unigrama global
        uni: Counter = Counter()
        for ctx_dict in self.counts.get(1, {}).values():
            uni.update(ctx_dict)
        total = sum(uni.values())
        top   = uni.most_common(top_k)
        return [
            {"symbol": sym, "probability": cnt / total, "n_used": 0, "count": cnt}
            for sym, cnt in top
        ]

    # ── Evaluación ──────────────────────────────────────────────────────────

    def evaluate(self, sequence: list[str]) -> dict:
        """
        Calcula métricas de predicción sobre una secuencia de test.

        Métricas:
          - top1_accuracy: acierta el primer candidato
          - top3_accuracy: el símbolo real está entre los 3 primeros
          - avg_rank     : posición promedio del símbolo real
          - perplexity   : exp(-media(log P(real)))
        """
        if not self.is_fitted:
            raise RuntimeError("Predictor no entrenado.")

        correct_1 = correct_3 = 0
        ranks: list[int] = []
        log_probs: list[float] = []
        total = 0

        for i in range(self.max_n, len(sequence) - 1):
            ctx    = sequence[max(0, i - self.max_n) : i]
            actual = sequence[i]
            preds  = self.predict(ctx, top_k=len(self.symbols))
            syms   = [p["symbol"] for p in preds]
            probs  = [p["probability"] for p in preds]

            if syms and syms[0] == actual:
                correct_1 += 1
            if actual in syms[:3]:
                correct_3 += 1

            if actual in syms:
                rank = syms.index(actual) + 1
                prob = probs[syms.index(actual)]
            else:
                rank = len(self.symbols) + 1
                prob = 1e-10

            ranks.append(rank)
            log_probs.append(np.log(prob + 1e-10))
            total += 1

        return {
            "total_predictions": total,
            "top1_accuracy":     round(correct_1 / total, 4) if total else 0,
            "top3_accuracy":     round(correct_3 / total, 4) if total else 0,
            "avg_rank":          round(float(np.mean(ranks)), 2) if ranks else 0,
            "perplexity":        round(float(np.exp(-np.mean(log_probs))), 2) if log_probs else 0,
        }

    # ── Análisis del diccionario ────────────────────────────────────────────

    def top_patterns(self, n: int = 3, top: int = 10) -> list[dict]:
        """Retorna los patrones n-gram más frecuentes."""
        if n not in self.counts:
            return []
        results = []
        for ctx, counter in self.counts[n].items():
            best_sym, best_cnt = counter.most_common(1)[0]
            total = sum(counter.values())
            results.append({
                "context":     "".join(ctx),
                "next_symbol": best_sym,
                "count":       best_cnt,
                "probability": round(best_cnt / total, 3),
                "total_obs":   total,
            })
        return sorted(results, key=lambda x: x["total_obs"], reverse=True)[:top]

    def _log_top_patterns(self) -> None:
        for n in range(2, min(4, self.max_n + 1)):
            top = self.top_patterns(n=n, top=3)
            if top:
                log.info(f"Top patrones {n}-gram: {[(p['context'], p['next_symbol'], p['probability']) for p in top]}")

    # ── Persistencia ────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        # Convertir defaultdict/Counter → dict serializable
        serializable = {}
        for n, ctx_dict in self.counts.items():
            serializable[str(n)] = {
                "|".join(ctx): dict(cnt)
                for ctx, cnt in ctx_dict.items()
            }
        (directory / "ngram_counts.json").write_text(
            json.dumps(serializable, separators=(",", ":"))
        )
        meta = {
            "max_n":         self.max_n,
            "symbols":       self.symbols,
            "total_trained": self.total_trained,
        }
        (directory / "predictor_meta.json").write_text(json.dumps(meta, indent=2))
        log.info(f"Predictor guardado en {directory}/")

    @classmethod
    def load(cls, directory: Path) -> "NgramPredictor":
        directory = Path(directory)
        meta = json.loads((directory / "predictor_meta.json").read_text())
        pred = cls(max_n=meta["max_n"], symbols=meta["symbols"])
        pred.total_trained = meta.get("total_trained", 0)

        raw = json.loads((directory / "ngram_counts.json").read_text())
        pred.counts = {}
        for n_str, ctx_dict in raw.items():
            n = int(n_str)
            pred.counts[n] = defaultdict(Counter)
            for ctx_str, cnt_dict in ctx_dict.items():
                ctx = tuple(ctx_str.split("|")) if ctx_str else ()
                pred.counts[n][ctx] = Counter(cnt_dict)

        pred.is_fitted = True
        log.info(f"Predictor cargado desde {directory}/ (entrenado con {pred.total_trained:,} símbolos)")
        return pred


# ══════════════════════════════════════════════════════════════════════════════
#  SISTEMA COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

class CandlePatternSystem:
    """
    Sistema end-to-end: CSV de velas → predicción del próximo patrón.

    Modos:
      - train  : Entrena encoder + predictor desde cero.
      - update : Actualiza el predictor con nuevos datos (encoder fijo).
      - predict: Predice el próximo símbolo dado un CSV reciente.
      - eval   : Evalúa el modelo sobre datos de prueba.
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.encoder   = CandleEncoder()
        self.predictor = NgramPredictor()

    # ── Carga de CSV ────────────────────────────────────────────────────────

    @staticmethod
    def _load_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Columnas faltantes en el CSV: {missing}")
        log.info(f"CSV cargado: {len(df):,} velas desde {df['open_time'].iloc[0]} hasta {df['open_time'].iloc[-1]}")
        return df

    # ── Entrenamiento desde cero ─────────────────────────────────────────────

    def train(self, csv_path: str, val_split: float = 0.15) -> dict:
        """
        Entrena el sistema completo:
          1. Ajusta el encoder (KMeans) sobre todos los datos.
          2. Genera la secuencia de símbolos.
          3. Divide en train/val.
          4. Entrena el predictor sobre el split de train.
          5. Evalúa sobre val.
        """
        log.info("═" * 55)
        log.info("  ENTRENAMIENTO DESDE CERO")
        log.info("═" * 55)

        df = self._load_csv(csv_path)
        sequence = self.encoder.fit_transform(df)

        log.info(f"Secuencia total: {len(sequence)} símbolos")
        log.info(f"Distribución: {dict(sorted(Counter(sequence).items()))}")

        split = int(len(sequence) * (1 - val_split))
        train_seq, val_seq = sequence[:split], sequence[split:]
        log.info(f"Split: {len(train_seq)} train │ {len(val_seq)} validación")

        self.predictor.fit(train_seq)

        metrics = {}
        if len(val_seq) > self.predictor.max_n + 1:
            log.info("Evaluando en validación...")
            metrics = self.predictor.evaluate(val_seq)
            log.info(f"Top-1 accuracy : {metrics['top1_accuracy']:.2%}")
            log.info(f"Top-3 accuracy : {metrics['top3_accuracy']:.2%}")
            log.info(f"Perplexity     : {metrics['perplexity']}")
        else:
            log.warning("Validación demasiado pequeña, se omite evaluación.")

        self.save()
        return metrics

    # ── Actualización incremental ────────────────────────────────────────────

    def update(self, csv_path: str) -> dict:
        """
        Actualiza el predictor con nuevos datos.
        El encoder (KMeans) permanece fijo para mantener el espacio de símbolos estable.
        """
        log.info("═" * 55)
        log.info("  ACTUALIZACIÓN INCREMENTAL")
        log.info("═" * 55)

        df = self._load_csv(csv_path)
        new_sequence = self.encoder.transform(df)
        log.info(f"Nuevos símbolos: {len(new_sequence)}")

        self.predictor.update(new_sequence)
        metrics = self.predictor.evaluate(new_sequence)
        log.info(f"Top-1 accuracy (datos nuevos): {metrics['top1_accuracy']:.2%}")
        log.info(f"Top-3 accuracy (datos nuevos): {metrics['top3_accuracy']:.2%}")

        self.save()
        return metrics

    # ── Predicción ──────────────────────────────────────────────────────────

    def predict(self, csv_path: str, top_k: int = 5) -> dict:
        """
        Genera predicciones del próximo símbolo dado un archivo de velas recientes.
        """
        df = self._load_csv(csv_path)
        sequence = self.encoder.transform(df)

        ctx_size = min(self.predictor.max_n, len(sequence))
        context  = sequence[-ctx_size:]

        preds = self.predictor.predict(context, top_k=top_k)

        return {
            "sequence_length":   len(sequence),
            "context_used":      context,
            "predictions":       preds,
            "total_trained_on":  self.predictor.total_trained,
        }

    # ── Evaluación completa ──────────────────────────────────────────────────

    def evaluate(self, csv_path: str) -> dict:
        """Evalúa el modelo sobre un CSV de test."""
        log.info("═" * 55)
        log.info("  EVALUACIÓN")
        log.info("═" * 55)

        df = self._load_csv(csv_path)
        sequence = self.encoder.transform(df)
        metrics  = self.predictor.evaluate(sequence)

        log.info(f"Top-1 accuracy : {metrics['top1_accuracy']:.2%}")
        log.info(f"Top-3 accuracy : {metrics['top3_accuracy']:.2%}")
        log.info(f"Rank promedio  : {metrics['avg_rank']}")
        log.info(f"Perplexity     : {metrics['perplexity']}")
        return metrics

    # ── Análisis del diccionario ─────────────────────────────────────────────

    def show_dictionary(self, n: int = 3, top: int = 15) -> list[dict]:
        """Muestra los patrones más frecuentes del diccionario."""
        return self.predictor.top_patterns(n=n, top=top)

    # ── Persistencia ────────────────────────────────────────────────────────

    def save(self) -> None:
        self.encoder.save(self.model_dir)
        self.predictor.save(self.model_dir)
        log.info(f"✅ Modelo completo guardado en {self.model_dir}/")

    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR) -> "CandlePatternSystem":
        system = cls(model_dir=model_dir)
        system.encoder   = CandleEncoder.load(model_dir)
        system.predictor = NgramPredictor.load(model_dir)
        return system


# ══════════════════════════════════════════════════════════════════════════════
#  INTERFAZ DE LÍNEA DE COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def _separator():
    print("═" * 55)


def cmd_train(args):
    system = CandlePatternSystem(model_dir=args.model_dir)
    metrics = system.train(args.csv, val_split=args.val_split)
    _separator()
    print("✅  ENTRENAMIENTO COMPLETADO")
    if metrics:
        print(f"   Top-1 Accuracy : {metrics['top1_accuracy']:.2%}")
        print(f"   Top-3 Accuracy : {metrics['top3_accuracy']:.2%}")
        print(f"   Perplexity     : {metrics['perplexity']}")
    _separator()


def cmd_update(args):
    if not Path(args.model_dir).exists():
        print(f"❌  Modelo no encontrado en '{args.model_dir}'. Ejecuta 'train' primero.")
        return
    system = CandlePatternSystem.load(model_dir=args.model_dir)
    metrics = system.update(args.csv)
    _separator()
    print("✅  MODELO ACTUALIZADO")
    print(f"   Top-1 Accuracy : {metrics['top1_accuracy']:.2%}")
    print(f"   Top-3 Accuracy : {metrics['top3_accuracy']:.2%}")
    _separator()


def cmd_predict(args):
    if not Path(args.model_dir).exists():
        print(f"❌  Modelo no encontrado en '{args.model_dir}'. Ejecuta 'train' primero.")
        return
    system  = CandlePatternSystem.load(model_dir=args.model_dir)
    result  = system.predict(args.csv, top_k=args.top_k)

    _separator()
    print("🔮  PREDICCIÓN DEL PRÓXIMO PATRÓN")
    _separator()
    print(f"   Contexto usado : {''.join(result['context_used'])}")
    print(f"   Símbolos en secuencia : {result['sequence_length']}")
    print(f"   Entrenado con : {result['total_trained_on']:,} símbolos")
    print()
    print("   Candidatos:")
    for i, p in enumerate(result["predictions"], 1):
        bar = "█" * int(p["probability"] * 30)
        print(
            f"   {i}. [{p['symbol']}]  {p['probability']:.1%}  {bar:<30}"
            f"  (n={p['n_used']}, obs={p['count']})"
        )
    _separator()


def cmd_eval(args):
    if not Path(args.model_dir).exists():
        print(f"❌  Modelo no encontrado en '{args.model_dir}'. Ejecuta 'train' primero.")
        return
    system  = CandlePatternSystem.load(model_dir=args.model_dir)
    metrics = system.evaluate(args.csv)

    _separator()
    print("📊  EVALUACIÓN DEL MODELO")
    _separator()
    print(f"   Predicciones totales : {metrics['total_predictions']:,}")
    print(f"   Top-1 Accuracy       : {metrics['top1_accuracy']:.2%}")
    print(f"   Top-3 Accuracy       : {metrics['top3_accuracy']:.2%}")
    print(f"   Rank promedio        : {metrics['avg_rank']}")
    print(f"   Perplexity           : {metrics['perplexity']}")
    _separator()


def cmd_dict(args):
    if not Path(args.model_dir).exists():
        print(f"❌  Modelo no encontrado en '{args.model_dir}'.")
        return
    system   = CandlePatternSystem.load(model_dir=args.model_dir)
    patterns = system.show_dictionary(n=args.ngram, top=args.top)

    _separator()
    print(f"📖  TOP {args.top} PATRONES {args.ngram}-GRAM")
    _separator()
    print(f"  {'Contexto':<15} {'→':>3} {'Siguiente':>10}  {'P':>7}  {'Obs':>6}")
    print("  " + "─" * 48)
    for p in patterns:
        print(
            f"  {p['context']:<15} {'→':>3} {p['next_symbol']:>10}"
            f"  {p['probability']:>6.1%}  {p['total_obs']:>6,}"
        )
    _separator()


def main():
    parser = argparse.ArgumentParser(
        description="Candle Pattern Predictor — Sistema de predicción de patrones de velas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", default="candle_model", help="Directorio del modelo (default: candle_model)")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Entrena el modelo desde cero")
    p_train.add_argument("csv", help="CSV con datos de velas (OHLCV)")
    p_train.add_argument("--val-split", type=float, default=0.15,
                         help="Fracción de datos para validación (default: 0.15)")
    p_train.set_defaults(func=cmd_train)

    # ── update ───────────────────────────────────────────────────────────────
    p_upd = sub.add_parser("update", help="Actualiza el predictor con nuevos datos")
    p_upd.add_argument("csv", help="CSV con nuevas velas")
    p_upd.set_defaults(func=cmd_update)

    # ── predict ──────────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Predice el próximo símbolo")
    p_pred.add_argument("csv", help="CSV con velas recientes")
    p_pred.add_argument("--top-k", type=int, default=5, help="Número de candidatos (default: 5)")
    p_pred.set_defaults(func=cmd_predict)

    # ── eval ─────────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Evalúa el modelo sobre datos de prueba")
    p_eval.add_argument("csv", help="CSV de prueba")
    p_eval.set_defaults(func=cmd_eval)

    # ── dict ─────────────────────────────────────────────────────────────────
    p_dict = sub.add_parser("dict", help="Muestra el diccionario de patrones aprendidos")
    p_dict.add_argument("--ngram", type=int, default=3, help="Orden del n-grama (default: 3)")
    p_dict.add_argument("--top", type=int, default=15, help="Número de patrones a mostrar (default: 15)")
    p_dict.set_defaults(func=cmd_dict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
