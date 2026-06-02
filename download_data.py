"""
Descarga los CSVs históricos de partidos ATP y WTA desde los repositorios
de Jeff Sackmann en GitHub. Los guarda en data/atp/ y data/wta/.
"""

import os
import requests
from tqdm import tqdm

BASE_ATP = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
BASE_WTA = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"

START_YEAR = 2000
END_YEAR = 2025

TARGETS = [
    ("atp", BASE_ATP, "atp_matches_{year}.csv"),
    ("wta", BASE_WTA, "wta_matches_{year}.csv"),
]


def download_file(url: str, dest_path: str) -> bool:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(r.content)
            return True
        return False
    except requests.RequestException as e:
        print(f"  Error al descargar {url}: {e}")
        return False


def main():
    for tour, base_url, filename_tpl in TARGETS:
        dest_dir = os.path.join("data", tour)
        os.makedirs(dest_dir, exist_ok=True)

        years = range(START_YEAR, END_YEAR + 1)
        ok, fail = 0, 0

        print(f"\n{'='*50}")
        print(f"Descargando datos {tour.upper()} ({START_YEAR}–{END_YEAR})")
        print(f"{'='*50}")

        for year in tqdm(years, desc=tour.upper()):
            filename = filename_tpl.format(year=year)
            url = f"{base_url}/{filename}"
            dest = os.path.join(dest_dir, filename)

            if os.path.exists(dest):
                ok += 1
                continue

            if download_file(url, dest):
                ok += 1
            else:
                fail += 1
                tqdm.write(f"  No disponible: {filename}")

        print(f"  Descargados/existentes: {ok} | No disponibles: {fail}")

    print("\nDescarga completada. Ejecuta 'python src/data_loader.py' para explorar los datos.")


if __name__ == "__main__":
    main()
