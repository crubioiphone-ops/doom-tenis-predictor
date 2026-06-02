"""
Feature engineering: ELO, forma reciente, H2H, stats de servicio.
Sin data leakage: las features de cada partido usan solo historia previa.
"""

import os
import sys
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from data_loader import load_all

ELO_INIT = 1500.0
ELO_K = 32.0
SERVE_WINDOW = 15
SURFACE_FORM_WINDOW = 30

DIFF_FEAT_KEYS = [
    "elo", "surface_elo", "rank_points",
    "form5", "form10", "form20", "surface_form",
    "serve1_pct", "serve1_won_pct", "serve2_won_pct",
    "bp_save_pct", "ace_pct", "df_pct",
]


def _elo_win_prob(ra, rb):
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _win_rate(history, n, surface=None):
    filtered = [(w, s) for w, s in history if s == surface] if surface else history
    recent = filtered[-n:]
    if not recent:
        return 0.5
    return sum(w for w, _ in recent) / len(recent)


def _serve_avg(serve_hist):
    keys = ["serve1_pct", "serve1_won_pct", "serve2_won_pct", "bp_save_pct", "ace_pct", "df_pct"]
    recent = serve_hist[-SERVE_WINDOW:]
    if not recent:
        return {k: np.nan for k in keys}
    result = {}
    for k in keys:
        vals = [r[k] for r in recent if not np.isnan(r.get(k, np.nan))]
        result[k] = float(np.mean(vals)) if vals else np.nan
    return result


def _extract_serve(row, prefix):
    svpt = row.get(f"{prefix}_svpt", np.nan)
    if pd.isna(svpt) or svpt == 0:
        return None
    first_in = row.get(f"{prefix}_1stIn", np.nan)
    first_won = row.get(f"{prefix}_1stWon", np.nan)
    second_won = row.get(f"{prefix}_2ndWon", np.nan)
    bp_saved = row.get(f"{prefix}_bpSaved", np.nan)
    bp_faced = row.get(f"{prefix}_bpFaced", np.nan)
    ace = row.get(f"{prefix}_ace", np.nan)
    df_val = row.get(f"{prefix}_df", np.nan)
    second_pts = svpt - (first_in if not pd.isna(first_in) else 0)
    return {
        "serve1_pct": first_in / svpt if not pd.isna(first_in) else np.nan,
        "serve1_won_pct": first_won / first_in if (not pd.isna(first_won) and not pd.isna(first_in) and first_in > 0) else np.nan,
        "serve2_won_pct": second_won / second_pts if (not pd.isna(second_won) and second_pts > 0) else np.nan,
        "bp_save_pct": bp_saved / bp_faced if (not pd.isna(bp_saved) and not pd.isna(bp_faced) and bp_faced > 0) else np.nan,
        "ace_pct": ace / svpt if not pd.isna(ace) else np.nan,
        "df_pct": df_val / svpt if not pd.isna(df_val) else np.nan,
    }


def make_initial_state():
    return {
        "elo": defaultdict(lambda: ELO_INIT),
        "surface_elo": defaultdict(lambda: ELO_INIT),
        "history": defaultdict(list),
        "serve_history": defaultdict(list),
        "h2h": defaultdict(lambda: [0, 0]),
        "last_rank_points": {},
        "name_to_id": {},
        "id_to_name": {},
    }


def compute_features(df: pd.DataFrame, state: dict = None) -> tuple:
    """
    Procesa partidos en orden cronológico.
    Devuelve (feat_df, estado_actualizado).
    Pasa state existente para procesamiento incremental.
    """
    if state is None:
        state = make_initial_state()

    elo = state["elo"]
    surface_elo = state["surface_elo"]
    history = state["history"]
    serve_history = state["serve_history"]
    h2h = state["h2h"]
    last_rank_pts = state["last_rank_points"]
    name_to_id = state["name_to_id"]
    id_to_name = state["id_to_name"]

    rows = []
    for _, m in df.iterrows():
        w_id = m["winner_id"]
        l_id = m["loser_id"]
        surface = m["surface"]

        name_to_id[m.get("winner_name", "")] = w_id
        name_to_id[m.get("loser_name", "")] = l_id
        id_to_name[w_id] = m.get("winner_name", "")
        id_to_name[l_id] = m.get("loser_name", "")

        hkey = (min(w_id, l_id), max(w_id, l_id))
        hw = h2h[hkey]
        if w_id == hkey[0]:
            w_h2h_wins, l_h2h_wins = hw[0], hw[1]
        else:
            w_h2h_wins, l_h2h_wins = hw[1], hw[0]

        w_serve = _serve_avg(serve_history[w_id])
        l_serve = _serve_avg(serve_history[l_id])

        row = {
            "p1_elo": elo[w_id],
            "p1_surface_elo": surface_elo[(w_id, surface)],
            "p1_rank_points": last_rank_pts.get(w_id, m.get("winner_rank_points", np.nan)),
            "p1_form5": _win_rate(history[w_id], 5),
            "p1_form10": _win_rate(history[w_id], 10),
            "p1_form20": _win_rate(history[w_id], 20),
            "p1_surface_form": _win_rate(history[w_id], SURFACE_FORM_WINDOW, surface),
            "p1_h2h_wins": w_h2h_wins,
            **{f"p1_{k}": v for k, v in w_serve.items()},
            "p2_elo": elo[l_id],
            "p2_surface_elo": surface_elo[(l_id, surface)],
            "p2_rank_points": last_rank_pts.get(l_id, m.get("loser_rank_points", np.nan)),
            "p2_form5": _win_rate(history[l_id], 5),
            "p2_form10": _win_rate(history[l_id], 10),
            "p2_form20": _win_rate(history[l_id], 20),
            "p2_surface_form": _win_rate(history[l_id], SURFACE_FORM_WINDOW, surface),
            "p2_h2h_wins": l_h2h_wins,
            **{f"p2_{k}": v for k, v in l_serve.items()},
            "surface": surface,
            "tourney_date": m["tourney_date"],
            "winner_id": w_id,
            "loser_id": l_id,
            "tour": m.get("tour", ""),
        }
        rows.append(row)

        # Actualizar ELO global
        w_exp = _elo_win_prob(elo[w_id], elo[l_id])
        elo[w_id] += ELO_K * (1 - w_exp)
        elo[l_id] -= ELO_K * (1 - w_exp)

        # Actualizar ELO por superficie
        w_sexp = _elo_win_prob(surface_elo[(w_id, surface)], surface_elo[(l_id, surface)])
        surface_elo[(w_id, surface)] += ELO_K * (1 - w_sexp)
        surface_elo[(l_id, surface)] -= ELO_K * (1 - w_sexp)

        history[w_id].append((True, surface))
        history[l_id].append((False, surface))

        w_srv = _extract_serve(m, "w")
        l_srv = _extract_serve(m, "l")
        if w_srv:
            serve_history[w_id].append(w_srv)
        if l_srv:
            serve_history[l_id].append(l_srv)

        if w_id == hkey[0]:
            h2h[hkey][0] += 1
        else:
            h2h[hkey][1] += 1

        # Guardar último ranking conocido
        wrp = m.get("winner_rank_points", np.nan)
        lrp = m.get("loser_rank_points", np.nan)
        if not pd.isna(wrp):
            last_rank_pts[w_id] = wrp
        if not pd.isna(lrp):
            last_rank_pts[l_id] = lrp

    return pd.DataFrame(rows), state


def to_diff_matrix(feat_df: pd.DataFrame, augment: bool = True) -> tuple:
    """
    Transforma features p1/p2 en diferencias (p1 - p2).
    augment=True dobla el dataset con p1/p2 invertidos (label=0) para entrenamiento no sesgado.
    """
    def _diff_frame(df, flip):
        p1, p2 = ("p2", "p1") if flip else ("p1", "p2")
        X = pd.DataFrame(index=range(len(df)))
        for feat in DIFF_FEAT_KEYS:
            X[f"diff_{feat}"] = df[f"{p1}_{feat}"].values - df[f"{p2}_{feat}"].values
        X["h2h_adv"] = df[f"{p1}_h2h_wins"].values - df[f"{p2}_h2h_wins"].values
        X["surface_clay"] = (df["surface"] == "clay").astype(int).values
        X["surface_hard"] = (df["surface"] == "hard").astype(int).values
        X["surface_grass"] = (df["surface"] == "grass").astype(int).values
        y = np.zeros(len(df)) if flip else np.ones(len(df))
        return X, y

    X_orig, y_orig = _diff_frame(feat_df, flip=False)
    if not augment:
        return X_orig, y_orig.astype(int)
    X_flip, y_flip = _diff_frame(feat_df, flip=True)
    X = pd.concat([X_orig, X_flip], ignore_index=True)
    y = np.concatenate([y_orig, y_flip]).astype(int)
    return X, y


def get_player_stats(player_name: str, surface: str, state: dict, opponent_name: str = None) -> dict:
    """Devuelve las estadísticas actuales de un jugador para predicción."""
    pid = state["name_to_id"].get(player_name)
    if pid is None:
        return None

    h2h_wins = 0
    if opponent_name:
        opp_id = state["name_to_id"].get(opponent_name)
        if opp_id:
            hkey = (min(pid, opp_id), max(pid, opp_id))
            hw = state["h2h"].get(hkey, [0, 0])
            h2h_wins = hw[0] if pid == hkey[0] else hw[1]

    hist = state["history"].get(pid, [])
    serve_hist = state["serve_history"].get(pid, [])

    elo_val = state["elo"].get(pid, ELO_INIT) if isinstance(state["elo"], dict) else state["elo"][pid]
    surf_elo_val = state["surface_elo"].get((pid, surface), ELO_INIT) if isinstance(state["surface_elo"], dict) else state["surface_elo"][(pid, surface)]

    return {
        "elo": elo_val,
        "surface_elo": surf_elo_val,
        "rank_points": state.get("last_rank_points", {}).get(pid, np.nan),
        "form5": _win_rate(hist, 5),
        "form10": _win_rate(hist, 10),
        "form20": _win_rate(hist, 20),
        "surface_form": _win_rate(hist, SURFACE_FORM_WINDOW, surface),
        "h2h_wins": h2h_wins,
        **_serve_avg(serve_hist),
    }


def find_players(name: str, state: dict) -> list:
    """Busca jugadores cuyo nombre contiene el texto dado."""
    q = name.lower()
    return sorted(n for n in state["name_to_id"] if q in n.lower())


def save_state(state: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    serializable = {
        "elo": dict(state["elo"]),
        "surface_elo": dict(state["surface_elo"]),
        "history": {pid: hist[-50:] for pid, hist in state["history"].items()},
        "serve_history": {pid: hist[-30:] for pid, hist in state["serve_history"].items()},
        "h2h": dict(state["h2h"]),
        "last_rank_points": state.get("last_rank_points", {}),
        "name_to_id": state["name_to_id"],
        "id_to_name": state["id_to_name"],
    }
    with open(path, "wb") as f:
        pickle.dump(serializable, f)


def load_state(path: str) -> dict:
    with open(path, "rb") as f:
        s = pickle.load(f)
    elo_d = defaultdict(lambda: ELO_INIT)
    elo_d.update(s.get("elo", {}))
    se_d = defaultdict(lambda: ELO_INIT)
    se_d.update(s.get("surface_elo", {}))
    hist_d = defaultdict(list)
    hist_d.update(s.get("history", {}))
    srv_d = defaultdict(list)
    srv_d.update(s.get("serve_history", {}))
    h2h_d = defaultdict(lambda: [0, 0])
    h2h_d.update(s.get("h2h", {}))
    return {
        "elo": elo_d,
        "surface_elo": se_d,
        "history": hist_d,
        "serve_history": srv_d,
        "h2h": h2h_d,
        "last_rank_points": s.get("last_rank_points", {}),
        "name_to_id": s.get("name_to_id", {}),
        "id_to_name": s.get("id_to_name", {}),
    }


if __name__ == "__main__":
    import time

    DATA_DIR = os.path.join(_SRC, "..", "data")
    print("Cargando datos...")
    df_atp, df_wta = load_all()

    for tour, df in [("atp", df_atp), ("wta", df_wta)]:
        print(f"\n[{tour.upper()}] Computando features ({len(df):,} partidos)...")
        t0 = time.time()
        feat_df, state = compute_features(df)
        elapsed = time.time() - t0
        print(f"  Listo en {elapsed:.1f}s | {len(state['name_to_id'])} jugadores")

        out_feat = os.path.join(DATA_DIR, f"features_{tour}.parquet")
        out_state = os.path.join(DATA_DIR, f"state_{tour}.pkl")
        feat_df.to_parquet(out_feat, index=False)
        save_state(state, out_state)
        print(f"  Guardado: {out_feat}")
        print(f"  Guardado: {out_state}")

    print("\nSiguiente paso: python src/model.py")
