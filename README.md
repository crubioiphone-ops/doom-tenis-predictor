# Doom — Tennis Match Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

Predictor de resultados de tenis (ATP + WTA) basado en 26 años de datos históricos y machine learning. Incluye análisis de value bet, Kelly Criterion y registro de apuestas con gráficas de rendimiento.

---

## Características principales

- **Modelo XGBoost calibrado** — entrenado sobre partidos 2000–2022, evaluado en 2023–2026 (sin data leakage)
- **ELO dinámico** — rating global y por superficie, actualizado cronológicamente partido a partido
- **Análisis de valor** — Expected Value y Kelly Criterion frente a las cuotas de la casa
- **Actualización automática** — descarga diaria de datos nuevos via Windows Task Scheduler
- **Interfaz web Streamlit** — predicción interactiva, radar comparativo y estadísticas detalladas
- **Registro de apuestas** — historial persistente con evolución del bankroll y análisis de ROI

### Rendimiento del modelo

| Tour | Precisión global | Precisión con confianza >= 70% |
|------|-----------------|-------------------------------|
| ATP  | 65.5%           | ~74.6%                        |
| WTA  | 65.5%           | ~76.0%                        |

*Evaluado sobre partidos 2023–2026, nunca vistos durante el entrenamiento.*

---

## Instalación

```bash
git clone https://github.com/TU_USUARIO/doom-tennis-predictor.git
cd doom-tennis-predictor
pip install -r requirements.txt
```

---

## Configuración inicial (primera vez)

```bash
# 1. Descargar datos históricos ATP y WTA (2000–2026)
python download_data.py

# 2. Calcular features (~2 min)
python src/features.py

# 3. Entrenar modelos (~5 min)
python src/model.py
```

---

## Cómo ejecutar

**Opción A — Doble clic** en `iniciar.bat` (Windows). Abre el navegador automáticamente.

**Opción B — Terminal:**
```bash
streamlit run app.py
```

La app estará disponible en `http://localhost:8501`.

---

## Predicción desde terminal

```bash
# Predecir un partido con análisis de valor
python src/predict.py "Carlos Alcaraz" "Jannik Sinner" --surface clay --odds1 1.75 --odds2 2.10

# Buscar el nombre exacto de un jugador
python src/predict.py --search "alcaraz" --tour atp
```

---

## Actualización de datos

```bash
# Manual
python src/update_data.py

# Programar actualización automática diaria a las 7:00 AM (Windows, ejecutar una vez)
python setup_scheduler.py --time 07:00
```

---

## Estructura del proyecto

```
doom-tennis-predictor/
├── data/                       # Datos (generados localmente, no en el repo)
│   ├── atp/                    # CSVs ATP por año
│   ├── wta/                    # CSVs WTA por año
│   ├── features_*.parquet      # Features procesadas
│   ├── state_*.pkl             # Estado ELO e historial
│   └── apuestas.json           # Registro de apuestas personal
├── models/                     # Modelos entrenados (generados localmente)
│   ├── xgb_atp.joblib
│   ├── xgb_cal_atp.joblib
│   └── ...
├── src/
│   ├── data_loader.py          # Carga y limpieza de CSVs
│   ├── features.py             # ELO, forma, H2H, stats de servicio
│   ├── model.py                # Entrenamiento XGBoost + calibración Platt
│   ├── predict.py              # CLI de predicción
│   └── update_data.py          # Descarga incremental de datos nuevos
├── .streamlit/
│   └── config.toml             # Tema oscuro personalizado
├── app.py                      # Interfaz Streamlit
├── download_data.py            # Descarga inicial de todos los datos
├── setup_scheduler.py          # Configura Windows Task Scheduler
├── iniciar.bat                 # Lanzador con doble clic (Windows)
├── guia.md                     # Guía de interpretación (cargada en la app)
├── requirements.txt
└── README.md
```

---

## Cómo funciona el modelo

### Features (diferencias jugador 1 − jugador 2)

| Categoría | Variables |
|-----------|-----------|
| ELO | Global, por superficie |
| Forma reciente | Win rate últimos 5, 10 y 20 partidos |
| Superficie | Win rate en la superficie del partido |
| H2H | Diferencia de victorias directas |
| Servicio | 1er serv. %, ganados con 1er/2º serv., BP salvados, aces %, DFs % |

### Pipeline

1. **XGBoost** (500 árboles, profundidad 4, lr 0.05) — entrenado con augmentación simétrica
2. **Calibración de Platt** — Logistic Regression sobre probabilidades brutas del XGBoost
3. **Split temporal** — train < 2023-01-01, test >= 2023-01-01

### Análisis de valor

```
EV  = P_modelo × (cuota − 1) − (1 − P_modelo)
Kelly = (P × b − (1−P)) / b     donde b = cuota − 1
Kelly/4 = versión conservadora recomendada
```

---

## Fuente de datos

Repositorios públicos de [Jeff Sackmann](https://github.com/JeffSackmann):
- [`tennis_atp`](https://github.com/JeffSackmann/tennis_atp) — partidos ATP desde 2000
- [`tennis_wta`](https://github.com/JeffSackmann/tennis_wta) — partidos WTA desde 2000

Cada fila es un partido con: jugadores, ranking, estadísticas de servicio/resto, superficie, torneo, ronda y resultado.

---

## Disclaimer

> Este proyecto es **exclusivamente de uso educativo y de investigación**. No constituye asesoramiento financiero ni de apuestas. El uso de este software para realizar apuestas reales es responsabilidad exclusiva del usuario. Los resultados históricos del modelo no garantizan rendimiento futuro.
