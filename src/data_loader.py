"""
Carga y limpia los CSVs de partidos ATP y WTA.
Produce un DataFrame unificado con columnas estandarizadas.
"""

import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Columnas mínimas que necesitamos para el modelo
REQUIRED_COLS = [
    "tourney_id", "tourney_name", "surface", "tourney_date",
    "winner_id", "winner_name", "winner_rank", "winner_rank_points",
    "loser_id", "loser_name", "loser_rank", "loser_rank_points",
    "score", "round", "best_of",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]

# Columnas que deben ser numéricas
NUMERIC_COLS = [
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]


def load_tour(tour: str) -> pd.DataFrame:
    """Carga todos los CSVs de un tour (atp o wta) y los concatena."""
    pattern = os.path.join(DATA_DIR, tour, f"{tour}_matches_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos en data/{tour}/. "
            "Ejecuta primero: python download_data.py"
        )

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df["tour"] = tour.upper()
            dfs.append(df)
        except Exception as e:
            print(f"  Aviso: no se pudo leer {os.path.basename(f)}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[{tour.upper()}] {len(files)} archivos | {len(combined):,} partidos cargados")
    return combined


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y normaliza el DataFrame."""

    # Añadir columnas faltantes como NaN (no todos los años tienen todas las stats)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Convertir fecha
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")

    # Asegurar tipos numéricos
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalizar superficie
    surface_map = {
        "Clay": "clay", "Hard": "hard", "Grass": "grass",
        "Carpet": "carpet", "Indoor Hard": "hard",
    }
    df["surface"] = df["surface"].map(surface_map).fillna("unknown")

    # Eliminar partidos sin resultado claro (retiros sin score)
    df = df[df["score"].notna()]
    df = df[~df["score"].str.contains("W/O|RET|DEF|ABN|nbsp", na=True, case=False)]

    # Eliminar duplicados exactos
    df = df.drop_duplicates()

    # Ordenar por fecha
    df = df.sort_values("tourney_date").reset_index(drop=True)

    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga y limpia ATP y WTA. Devuelve (df_atp, df_wta)."""
    df_atp = clean(load_tour("atp"))
    df_wta = clean(load_tour("wta"))
    return df_atp, df_wta


def summary(df: pd.DataFrame, label: str):
    sep = "-" * 50
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  Partidos totales : {len(df):,}")
    print(f"  Rango de fechas  : {df['tourney_date'].min().date()} - {df['tourney_date'].max().date()}")
    print(f"  Superficies      : {df['surface'].value_counts().to_dict()}")
    print(f"  Jugadores unicos : {pd.concat([df['winner_name'], df['loser_name']]).nunique():,}")
    print(f"  % con stats srv  : {df['w_ace'].notna().mean()*100:.1f}%")
    print(f"  Columnas totales : {len(df.columns)}")


if __name__ == "__main__":
    print("Cargando datos...\n")
    df_atp, df_wta = load_all()
    summary(df_atp, "ATP")
    summary(df_wta, "WTA")
    sep = "-" * 50
    print(f"\n{sep}")
    print("  Muestra de columnas disponibles:")
    print(f"  {list(df_atp.columns[:15])}")
    print(sep)
    print("\nDatos listos. Siguiente paso: python src/features.py")
