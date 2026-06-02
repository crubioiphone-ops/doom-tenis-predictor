"""
Actualización diaria de datos y estado del modelo.
Descarga los CSVs del año en curso y reconstruye el estado de los jugadores.
El modelo NO se reentrena (ejecutar model.py mensualmente si se desea).

Uso:
  python src/update_data.py
"""

import os
import sys
import time
from datetime import date

import requests

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from data_loader import load_tour, clean
from features import compute_features, save_state

DATA_DIR = os.path.join(_SRC, "..", "data")

_BASE = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv",
}


def download_csv(tour: str, year: int) -> bool:
    """Descarga CSV de un año. Devuelve True si hubo cambios."""
    out_dir = os.path.join(DATA_DIR, tour)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tour}_matches_{year}.csv")
    url = _BASE[tour].format(year=year)

    print(f"  {tour.upper()} {year}... ", end="", flush=True)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        new = resp.content
        if os.path.exists(path):
            with open(path, "rb") as f:
                if f.read() == new:
                    print("sin cambios.")
                    return False
        with open(path, "wb") as f:
            f.write(new)
        print(f"actualizado ({len(new)//1024} KB)")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def rebuild_state(tour: str):
    """Reconstruye estado de jugadores procesando todos los CSVs disponibles."""
    print(f"\n[{tour.upper()}] Reconstruyendo estado... ", end="", flush=True)
    t0 = time.time()
    df = clean(load_tour(tour))
    _, state = compute_features(df)
    out = os.path.join(DATA_DIR, f"state_{tour}.pkl")
    save_state(state, out)
    print(f"listo en {time.time()-t0:.1f}s | {len(state['name_to_id'])} jugadores")


def run():
    today = date.today()
    years = [today.year - 1, today.year]  # año anterior por si hay partidos tardíos

    print(f"{'='*50}")
    print(f"  Actualización diaria — {today}")
    print(f"{'='*50}\n")
    print("Comprobando datos nuevos...")

    changed = False
    for tour in ["atp", "wta"]:
        for year in years:
            if download_csv(tour, year):
                changed = True

    if changed:
        print("\nDatos nuevos detectados. Actualizando estados...")
        for tour in ["atp", "wta"]:
            rebuild_state(tour)
        print(f"\nActualizacion completada: {today}")
    else:
        print("\nNo hay datos nuevos. Estado actual sigue vigente.")


if __name__ == "__main__":
    run()
