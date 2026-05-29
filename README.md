# 🤖 HYPEUSDT Candle Pattern Bot — Deploy en Render

## Estructura del repositorio

```
tu-repo/
├── bot.py                 ← servidor web + loop de trading  (NUEVO)
├── candle_predictor.py    ← sistema de predicción          (ya tienes)
├── binance_simulator.py   ← simulador histórico            (ya tienes)
├── generate_test_data.py  ← generador de datos de prueba   (ya tienes)
├── requirements.txt       ← dependencias actualizadas      (NUEVO)
├── render.yaml            ← configuración de Render        (NUEVO)
├── Procfile               ← comando de arranque            (NUEVO)
└── candle_model/          ← modelo entrenado (opcional*)
    ├── encoder_meta.json
    ├── kmeans.joblib
    ├── scaler.joblib
    ├── predictor_meta.json
    └── ngram_counts.json
```

> *Si subes la carpeta `candle_model/` al repo, el bot la carga directamente.
> Si no la subes, el bot descarga 1 500 velas de Binance y entrena solo al arrancar.

---

## Paso 1 — Preparar el repositorio

```bash
git init
git add bot.py candle_predictor.py binance_simulator.py \
        generate_test_data.py requirements.txt render.yaml Procfile

# Opcional: incluir el modelo pre-entrenado (evita re-entrenar en cada deploy)
# git add candle_model/

git commit -m "feat: add render trading bot"
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

---

## Paso 2 — Crear el servicio en Render

1. Ve a [render.com](https://render.com) → **New → Web Service**
2. Conecta tu repositorio de GitHub/GitLab
3. Render detecta `render.yaml` automáticamente
4. Haz clic en **Create Web Service**

El build tarda ~2-3 minutos. En los logs verás:

```
Cargando modelo desde candle_model/...        ← si incluiste el modelo
  -- o --
Descargando 1500 velas para entrenar...       ← si no lo incluiste
✅ Modelo entrenado y guardado.
🟢 Loop de trading iniciado.
🌐 Dashboard disponible en http://0.0.0.0:10000
```

---

## Paso 3 — Verificar el dashboard

Abre la URL que te da Render (ej. `https://hypeusdt-candle-bot.onrender.com`).

Verás un dashboard con:
- **Capital actual** y P&L acumulado
- **Operaciones** totales, win rate, profit factor
- **Posición abierta** con TP/SL en tiempo real
- **Última señal** del modelo (símbolo, dirección, confianza)
- **Historial** de los últimos 50 trades

La página se auto-recarga cada 30 segundos.

---

## Endpoints disponibles

| Endpoint        | Descripción                                      |
|-----------------|--------------------------------------------------|
| `GET /`         | Dashboard HTML                                   |
| `GET /health`   | Health check (retorna `{"status":"ok"}`)         |
| `GET /api/status` | Estado completo del bot en JSON                |
| `GET /api/trades` | Historial de trades en JSON                    |
| `GET /api/equity` | Curva de equity en JSON                        |

---

## Variables de entorno (ajustables en Render → Environment)

| Variable         | Default    | Descripción                              |
|------------------|------------|------------------------------------------|
| `SYMBOL`         | HYPEUSDT   | Par de trading (Binance Futures)         |
| `INTERVAL`       | 1m         | Temporalidad de velas                    |
| `CAPITAL`        | 1000.0     | Capital inicial simulado en USDT         |
| `TP_PCT`         | 0.009      | Take profit (0.9%)                       |
| `SL_PCT`         | 0.010      | Stop loss (1.0%)                         |
| `MIN_CONFIDENCE` | 0.50       | Confianza mínima del modelo para operar  |
| `POS_SIZE_PCT`   | 0.012      | % del capital por trade                  |
| `FEE_PCT`        | 0.0004     | Comisión por lado                        |
| `MAX_HOLD`       | 60         | Velas máximas en posición (timeout)      |
| `LOOP_INTERVAL`  | 60         | Segundos entre ciclos del bot            |
| `N_KLINES_INIT`  | 1500       | Velas para entrenamiento inicial         |
| `N_KLINES_LIVE`  | 1400       | Velas descargadas en cada ciclo live     |
| `NORMAL_COUNT`   | 2          | Operaciones que respetan la dirección del modelo por ciclo |
| `INVERT_COUNT`   | 2          | Operaciones que invierten LONG↔SHORT por ciclo             |

---

## Plan de Render recomendado

| Plan     | Precio | Duerme      | Recomendado para           |
|----------|--------|-------------|---------------------------|
| Free     | $0/mes | Sí (15 min) | Pruebas, no producción     |
| Starter  | $7/mes | No          | ✅ Producción / bot 24/7   |

> Con el plan Free el bot se duerme y pierde el estado en memoria.
> Usa **Starter** o superior para operación continua.

---

## Notas importantes

- El bot realiza **operaciones 100% simuladas** (no mueve dinero real).
- El bot alterna el sesgo de variabilidad en vivo: por defecto abre **2 operaciones normales** con la dirección del modelo y luego **2 operaciones invertidas** (LONG↔SHORT), repitiendo el ciclo sin abrir más de una posición simultánea.
- El modelo se re-entrena solo si no existe `candle_model/` en el repositorio.
- El estado (trades, capital, posición) vive **en memoria**: se resetea en cada deploy.
  Para persistencia duradera, activa el **disco persistente** en `render.yaml` (ver comentarios).
- Si cambias `SYMBOL` a un par que no sea de Binance Futures, actualiza `BINANCE_API` en `bot.py`.
