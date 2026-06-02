"""
Predictor de partidos de tenis con análisis de value bet.

Uso:
  python src/predict.py "Carlos Alcaraz" "Jannik Sinner" --surface clay --tour atp
  python src/predict.py "Alcaraz" "Sinner" --surface clay --odds1 1.75 --odds2 2.10
  python src/predict.py --search "alcaraz" --tour atp   # buscar nombre exacto
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from features import load_state, get_player_stats, find_players, DIFF_FEAT_KEYS

DATA_DIR = os.path.join(_SRC, "..", "data")
MODEL_DIR = os.path.join(_SRC, "..", "models")


def _load_model(tour: str):
    try:
        xgb_model = joblib.load(os.path.join(MODEL_DIR, f"xgb_{tour}.joblib"))
        xgb_cal = joblib.load(os.path.join(MODEL_DIR, f"xgb_cal_{tour}.joblib"))
        medians = joblib.load(os.path.join(MODEL_DIR, f"medians_{tour}.joblib"))
        feature_names = joblib.load(os.path.join(MODEL_DIR, f"feature_names_{tour}.joblib"))

        class _Model:
            def predict_proba(self, X):
                raw = xgb_model.predict_proba(X)[:, 1].reshape(-1, 1)
                p1 = xgb_cal.predict_proba(raw)[:, 1]
                return np.column_stack([1 - p1, p1])

        return _Model(), medians, feature_names
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Modelos no encontrados. Ejecuta primero:\n"
            f"  python src/features.py\n  python src/model.py"
        )


def _load_state(tour: str):
    path = os.path.join(DATA_DIR, f"state_{tour}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Estado no encontrado: {path}\nEjecuta primero: python src/features.py"
        )
    return load_state(path)


def _build_row(s1: dict, s2: dict, surface: str) -> dict:
    row = {}
    for feat in DIFF_FEAT_KEYS:
        v1 = s1.get(feat, np.nan)
        v2 = s2.get(feat, np.nan)
        row[f"diff_{feat}"] = (v1 - v2) if (not np.isnan(v1) and not np.isnan(v2)) else np.nan
    row["h2h_adv"] = s1.get("h2h_wins", 0) - s2.get("h2h_wins", 0)
    row["surface_clay"] = int(surface == "clay")
    row["surface_hard"] = int(surface == "hard")
    row["surface_grass"] = int(surface == "grass")
    return row


def predict_match(
    player1: str, player2: str, surface: str, tour: str = "atp",
    odds1: float = None, odds2: float = None,
) -> dict:
    state = _load_state(tour)
    model, medians, feature_names = _load_model(tour)

    s1 = get_player_stats(player1, surface, state, opponent_name=player2)
    s2 = get_player_stats(player2, surface, state, opponent_name=player1)

    if s1 is None:
        candidates = find_players(player1, state)
        return {"error": f"'{player1}' no encontrado.", "sugerencias": candidates[:8]}
    if s2 is None:
        candidates = find_players(player2, state)
        return {"error": f"'{player2}' no encontrado.", "sugerencias": candidates[:8]}

    row = _build_row(s1, s2, surface)
    X = pd.DataFrame([row])[feature_names]
    X = X.fillna(medians[feature_names])

    prob1 = float(model.predict_proba(X)[0, 1])
    prob2 = 1.0 - prob1

    result = {
        "player1": player1,
        "player2": player2,
        "surface": surface,
        "tour": tour.upper(),
        "prob1": round(prob1, 4),
        "prob2": round(prob2, 4),
        "fair_odds1": round(1 / prob1, 3) if prob1 > 0 else None,
        "fair_odds2": round(1 / prob2, 3) if prob2 > 0 else None,
        "stats1": {
            "elo": round(s1.get("elo", 0)),
            "surface_elo": round(s1.get("surface_elo", 0)),
            "form10": round(s1.get("form10", 0.5), 3),
            "surface_form": round(s1.get("surface_form", 0.5), 3),
            "h2h_wins": s1.get("h2h_wins", 0),
        },
        "stats2": {
            "elo": round(s2.get("elo", 0)),
            "surface_elo": round(s2.get("surface_elo", 0)),
            "form10": round(s2.get("form10", 0.5), 3),
            "surface_form": round(s2.get("surface_form", 0.5), 3),
            "h2h_wins": s2.get("h2h_wins", 0),
        },
    }

    # Value bet: EV = P_model * (odds - 1) - (1 - P_model)
    if odds1 is not None:
        ev1 = prob1 * (odds1 - 1) - prob2
        result["odds1"] = odds1
        result["ev1"] = round(ev1, 4)
        result["value1"] = ev1 > 0

    if odds2 is not None:
        ev2 = prob2 * (odds2 - 1) - prob1
        result["odds2"] = odds2
        result["ev2"] = round(ev2, 4)
        result["value2"] = ev2 > 0

    return result


def _print(r: dict):
    if "error" in r:
        print(f"\nError: {r['error']}")
        if r.get("sugerencias"):
            print("  Jugadores similares encontrados:")
            for s in r["sugerencias"]:
                print(f"    - {s}")
        return

    W = 58
    sep = "=" * W
    thin = "—" * W
    p1, p2 = r["player1"], r["player2"]
    print(f"\n{sep}")
    print(f"  {r['tour']} | Superficie: {r['surface'].upper()}")
    print(f"{thin}")
    bar1 = "#" * round(r["prob1"] * 20)
    bar2 = "#" * round(r["prob2"] * 20)
    print(f"  {p1[:26]:<26}  {r['prob1']*100:5.1f}%  {bar1}")
    print(f"  {p2[:26]:<26}  {r['prob2']*100:5.1f}%  {bar2}")
    print(f"  Cuota justa:  {p1[:12]} -> {r['fair_odds1']}  |  {p2[:12]} -> {r['fair_odds2']}")
    print(f"{thin}")

    s1, s2 = r["stats1"], r["stats2"]
    print(f"  {'':14}  ELO    sELO  form10  surf%  H2H")
    print(f"  {p1[:14]:<14}  {s1['elo']:5.0f}  {s1['surface_elo']:5.0f}  "
          f"{s1['form10']:.2f}   {s1['surface_form']:.2f}   {s1['h2h_wins']}")
    print(f"  {p2[:14]:<14}  {s2['elo']:5.0f}  {s2['surface_elo']:5.0f}  "
          f"{s2['form10']:.2f}   {s2['surface_form']:.2f}   {s2['h2h_wins']}")

    has_odds = "odds1" in r or "odds2" in r
    if has_odds:
        print(f"{thin}")
        print(f"  ANÁLISIS DE APUESTA")

    if "odds1" in r:
        tag = "  ** VALUE BET **" if r["value1"] else ""
        sign = "+" if r["ev1"] > 0 else ""
        print(f"  {p1[:20]:<20}  cuota {r['odds1']}  EV: {sign}{r['ev1']*100:.1f}%{tag}")

    if "odds2" in r:
        tag = "  ** VALUE BET **" if r["value2"] else ""
        sign = "+" if r["ev2"] > 0 else ""
        print(f"  {p2[:20]:<20}  cuota {r['odds2']}  EV: {sign}{r['ev2']*100:.1f}%{tag}")

    print(f"{sep}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictor de partidos de tenis")
    parser.add_argument("player1", nargs="?", help="Jugador 1")
    parser.add_argument("player2", nargs="?", help="Jugador 2")
    parser.add_argument("--surface", "-s", default="hard",
                        choices=["clay", "hard", "grass", "carpet"],
                        help="Superficie (default: hard)")
    parser.add_argument("--tour", "-t", default="atp", choices=["atp", "wta"],
                        help="Tour (default: atp)")
    parser.add_argument("--odds1", type=float, default=None, help="Cuota bookmaker para jugador 1")
    parser.add_argument("--odds2", type=float, default=None, help="Cuota bookmaker para jugador 2")
    parser.add_argument("--search", help="Buscar jugador por nombre parcial")
    args = parser.parse_args()

    if args.search:
        state = _load_state(args.tour)
        results = find_players(args.search, state)
        if results:
            print(f"\nJugadores encontrados para '{args.search}':")
            for name in results[:20]:
                print(f"  {name}")
        else:
            print(f"Ningún jugador encontrado para '{args.search}'")
        sys.exit(0)

    if not args.player1 or not args.player2:
        parser.print_help()
        sys.exit(1)

    result = predict_match(
        args.player1, args.player2, args.surface, args.tour,
        odds1=args.odds1, odds2=args.odds2,
    )
    _print(result)
