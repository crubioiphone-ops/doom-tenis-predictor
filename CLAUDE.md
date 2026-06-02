# Tennis Match Predictor — Contexto del proyecto

## Objetivo
Predictor de resultados de partidos de tenis (ATP + WTA) usando datos históricos y machine learning.

## Estado actual
- [x] Paso 1 — Estructura del proyecto + descarga de datos
- [x] Paso 2 — Limpieza de datos (`src/data_loader.py`)
- [x] Paso 3 — Feature engineering (`src/features.py`)
- [x] Paso 4 — Modelo XGBoost calibrado + LR (`src/model.py`)
- [x] Paso 5 — Predicción con análisis de value bet (`src/predict.py`)
- [x] Actualización diaria automática (`src/update_data.py` + `setup_scheduler.py`)
- [ ] Paso 6 — Interfaz web con Streamlit (opcional)

## Flujo de uso completo
```
# Primera vez (orden obligatorio):
pip install -r requirements.txt
python download_data.py          # ya hecho — datos en data/atp/ y data/wta/
python src/features.py           # ~2 min — genera data/features_*.parquet + data/state_*.pkl
python src/model.py              # ~5 min — genera models/xgb_*.joblib

# Predicción diaria:
python src/update_data.py        # actualiza datos del año en curso
python src/predict.py "Carlos Alcaraz" "Jannik Sinner" --surface clay --odds1 1.75 --odds2 2.10

# Buscar nombre exacto de jugador:
python src/predict.py --search "alcaraz" --tour atp

# Programar actualización automática diaria (Windows, ejecutar 1 sola vez):
python setup_scheduler.py --time 07:00
```

## Datos
- Fuente: repositorios de Jeff Sackmann (tennis_atp / tennis_wta en GitHub)
- Rango: 2000–2025, ATP y WTA, ~26 años de partidos
- Ubicación: `data/atp/` y `data/wta/` (CSVs por año, ya descargados)

## Features planificadas para Paso 3
- ELO dinámico por jugador (global + por superficie)
- Forma reciente: win rate últimos 5/10/20 partidos
- Head-to-head: historial directo entre los dos jugadores, también por superficie
- Stats de servicio/resto normalizadas (1st serve %, aces, break points)
- Fatiga: días desde último partido, rondas jugadas en el torneo actual

## Archivos clave
- `download_data.py` — descarga los CSVs
- `src/data_loader.py` — carga, limpia y normaliza los datos; expone `load_all()` → `(df_atp, df_wta)`
- `src/features.py` — (por crear) construye el DataFrame de features para el modelo
- `src/model.py` — (por crear) entrena y evalúa el modelo
- `src/predict.py` — (por crear) predice un partido dado dos jugadores

## Decisiones técnicas tomadas
- Datos limpios: se eliminan retiros (RET), walkovers (W/O) y partidos sin score
- Superficie normalizada a: clay, hard, grass, carpet, unknown
- El modelo recibirá features **simétricas** (player_a vs player_b, con target binario)
