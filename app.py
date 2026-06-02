"""
Tennis Predictor - Interfaz Streamlit
Arranca con: streamlit run app.py  (o doble clic en iniciar.bat)
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

ROOT     = os.path.dirname(os.path.abspath(__file__))
SRC      = os.path.join(ROOT, "src")
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR= os.path.join(ROOT, "models")
GUIA_PATH= os.path.join(ROOT, "guia.md")
BETS_PATH= os.path.join(DATA_DIR, "apuestas.json")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

from features import load_state, get_player_stats, DIFF_FEAT_KEYS

st.set_page_config(page_title="Tennis Predictor", layout="centered",
                   initial_sidebar_state="expanded")

# ── Paleta ────────────────────────────────────────────────────────────────────
BG      = "#060d1a"
CARD    = "#0d1b2e"
CARD2   = "#111d30"
BORDER  = "#1e3a5f"
ACCENT  = "#38bdf8"
P2_COL  = "#7dd3fc"
GREEN   = "#34d399"
AMBER   = "#fbbf24"
RED     = "#f87171"
TEXT    = "#f1f5f9"
MUTED   = "#64748b"

def rgba(hex_col, alpha):
    h = hex_col.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
  }}

  h1 {{
    font-weight: 300;
    font-size: 2rem;
    letter-spacing: .08em;
    color: {TEXT};
  }}

  /* Boton principal */
  div[data-testid="stButton"] > button[kind="primary"] {{
    background: linear-gradient(135deg, #1d4ed8 0%, {ACCENT} 100%);
    border: none;
    color: white;
    font-weight: 600;
    font-size: .85rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    padding: .8rem 1rem;
    border-radius: 8px;
    transition: opacity .2s;
    box-shadow: 0 4px 20px {rgba(ACCENT, 0.25)};
  }}
  div[data-testid="stButton"] > button[kind="primary"]:hover {{
    opacity: .85;
  }}

  /* Boton secundario (sidebar) */
  div[data-testid="stButton"] > button:not([kind="primary"]) {{
    background: {CARD2};
    border: 1px solid {BORDER};
    color: {TEXT};
    border-radius: 6px;
    font-size: .8rem;
    letter-spacing: .06em;
    transition: border-color .2s;
  }}
  div[data-testid="stButton"] > button:not([kind="primary"]):hover {{
    border-color: {ACCENT};
    color: {ACCENT};
  }}

  /* Tabs */
  div[data-testid="stTabs"] button {{
    font-size: .82rem;
    letter-spacing: .06em;
    font-weight: 500;
  }}

  /* Expander */
  details summary {{
    font-size: .82rem;
    color: {MUTED};
    letter-spacing: .04em;
  }}

  /* Ocultar decoracion Streamlit */
  #MainMenu, footer, .stDeployButton {{ display: none; visibility: hidden; }}

  hr.thin {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 28px 0;
  }}
</style>
""", unsafe_allow_html=True)

# ── Helpers HTML inline (evita problemas con clases CSS en Streamlit) ─────────
def card(val, label, color=ACCENT, sublabel=None, badge=False):
    badge_html = f"""
      <span style="display:inline-block;background:{rgba(GREEN,0.15)};border:1px solid
        {rgba(GREEN,0.5)};color:{GREEN};font-size:.65rem;font-weight:600;
        letter-spacing:.1em;padding:2px 8px;border-radius:3px;text-transform:uppercase;
        margin-top:6px;">VALUE BET</span>""" if badge else ""
    sub_html = f"<div style='font-size:.75rem;color:{MUTED};margin-top:4px;'>{sublabel}</div>" if sublabel else ""
    return f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;
                padding:20px 24px;text-align:center;height:100%;">
      <div style="font-size:2rem;font-weight:500;color:{color};line-height:1.1;">{val}</div>
      <div style="font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;
                  color:{MUTED};margin-top:6px;">{label}</div>
      {sub_html}
      {badge_html}
    </div>"""

def conf_pill(max_p):
    if max_p >= 0.70:
        c, msg = GREEN, f"Confianza alta ({max_p*100:.0f}%) — precision historica ~75%"
    elif max_p >= 0.60:
        c, msg = AMBER, f"Confianza media ({max_p*100:.0f}%) — precision historica ~70%"
    else:
        c, msg = MUTED, f"Partido igualado ({max_p*100:.0f}%) — precision historica ~66%"
    return f"""<div style="display:inline-flex;align-items:center;gap:7px;
                 background:{rgba(c,0.12)};border:1px solid {rgba(c,0.3)};
                 border-radius:20px;padding:5px 14px;margin-top:14px;">
      <span style="width:7px;height:7px;border-radius:50%;background:{c};display:inline-block;"></span>
      <span style="font-size:.78rem;color:{c};">{msg}</span>
    </div>"""

# ── Cache ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources(tour):
    state     = load_state(os.path.join(DATA_DIR, f"state_{tour}.pkl"))
    xgb_m     = joblib.load(os.path.join(MODEL_DIR, f"xgb_{tour}.joblib"))
    xgb_c     = joblib.load(os.path.join(MODEL_DIR, f"xgb_cal_{tour}.joblib"))
    medians   = joblib.load(os.path.join(MODEL_DIR, f"medians_{tour}.joblib"))
    feat_names= joblib.load(os.path.join(MODEL_DIR, f"feature_names_{tour}.joblib"))
    players   = sorted(k for k in state["name_to_id"] if k)
    return state, xgb_m, xgb_c, medians, feat_names, players

def predict(p1, p2, surface, tour, odds1=None, odds2=None):
    state, xgb_m, xgb_c, medians, feat_names, _ = load_resources(tour)
    s1 = get_player_stats(p1, surface, state, opponent_name=p2)
    s2 = get_player_stats(p2, surface, state, opponent_name=p1)
    if not s1: return {"error": f"'{p1}' no encontrado."}
    if not s2: return {"error": f"'{p2}' no encontrado."}

    row = {}
    for f in DIFF_FEAT_KEYS:
        v1 = float(s1.get(f) or np.nan)
        v2 = float(s2.get(f) or np.nan)
        row[f"diff_{f}"] = (v1-v2) if not (np.isnan(v1) or np.isnan(v2)) else np.nan
    row.update({"h2h_adv": s1.get("h2h_wins",0)-s2.get("h2h_wins",0),
                "surface_clay":  int(surface=="clay"),
                "surface_hard":  int(surface=="hard"),
                "surface_grass": int(surface=="grass")})

    X     = pd.DataFrame([row])[feat_names].fillna(medians[feat_names])
    raw   = xgb_m.predict_proba(X)[:,1].reshape(-1,1)
    prob1 = float(xgb_c.predict_proba(raw)[0,1])
    prob2 = 1.0 - prob1

    def srv(s): return {k: s.get(k) for k in
        ["serve1_pct","serve1_won_pct","serve2_won_pct","bp_save_pct","h2h_wins"]}

    res = {"player1":p1,"player2":p2,"surface":surface,"tour":tour.upper(),
           "prob1":round(prob1,4),"prob2":round(prob2,4),
           "fair1":round(1/prob1,3),"fair2":round(1/prob2,3),
           "s1":{**{"elo":round(s1.get("elo",0)),"surface_elo":round(s1.get("surface_elo",0)),
                    "form10":s1.get("form10",.5),"form20":s1.get("form20",.5),
                    "surface_form":s1.get("surface_form",.5)},**srv(s1)},
           "s2":{**{"elo":round(s2.get("elo",0)),"surface_elo":round(s2.get("surface_elo",0)),
                    "form10":s2.get("form10",.5),"form20":s2.get("form20",.5),
                    "surface_form":s2.get("surface_form",.5)},**srv(s2)}}

    for k, prob, opp, odds in [("1",prob1,prob2,odds1),("2",prob2,prob1,odds2)]:
        if odds and odds > 1:
            ev    = prob*(odds-1) - opp
            kelly = max(0.0, ev/(odds-1))
            res.update({f"odds{k}":odds, f"ev{k}":round(ev,4),
                        f"val{k}":ev>0, f"kelly{k}":kelly})
    return res

# ── Apuestas (persistencia) ───────────────────────────────────────────────────
_BETS_COLS = ["fecha","tour","superficie","jugador","rival","cuota","stake","resultado","profit"]

def load_bets():
    if not os.path.exists(BETS_PATH):
        return pd.DataFrame(columns=_BETS_COLS)
    with open(BETS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return pd.DataFrame(columns=_BETS_COLS)
    return pd.DataFrame(data)[_BETS_COLS]


def save_bet(record):
    bets = []
    if os.path.exists(BETS_PATH):
        with open(BETS_PATH, "r", encoding="utf-8") as f:
            bets = json.load(f)
    bets.append(record)
    with open(BETS_PATH, "w", encoding="utf-8") as f:
        json.dump(bets, f, ensure_ascii=False, indent=2)


def delete_bet(idx):
    with open(BETS_PATH, "r", encoding="utf-8") as f:
        bets = json.load(f)
    if 0 <= idx < len(bets):
        bets.pop(idx)
        with open(BETS_PATH, "w", encoding="utf-8") as f:
            json.dump(bets, f, ensure_ascii=False, indent=2)


# ── Charts ────────────────────────────────────────────────────────────────────
LAYOUT = dict(plot_bgcolor=BG, paper_bgcolor=BG,
              font=dict(color=TEXT, family="Inter, sans-serif"))

def chart_probs(r):
    fig = go.Figure()
    for prob, name, color, tcolor in [
        (r["prob1"]*100, r["player1"], ACCENT,  "white"),
        (r["prob2"]*100, r["player2"], CARD2,   "#94a3b8"),
    ]:
        fig.add_trace(go.Bar(
            x=[prob], y=[name], orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            text=[f"{prob:.1f}%"], textposition="inside",
            textfont=dict(size=16, color=tcolor, family="Inter"),
            showlegend=False,
        ))
    fig.add_vline(x=50, line_width=1, line_dash="dot", line_color=rgba(TEXT, 0.15))
    fig.update_xaxes(range=[0,100], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=13))
    fig.update_layout(**LAYOUT, height=160, bargap=0.38,
                      margin=dict(l=10,r=10,t=14,b=10))
    return fig

def chart_radar(r):
    s1, s2 = r["s1"], r["s2"]

    def ne(v): return max(0.0, min(1.0, (v-1300)/1000))
    def ns(s):
        vals = [float(s.get(k)) for k in
                ["serve1_pct","serve1_won_pct","serve2_won_pct","bp_save_pct"]
                if s.get(k) is not None and not np.isnan(float(s.get(k)))]
        return float(np.mean(vals)) if vals else 0.6

    cats = ["ELO Global","ELO Superficie","Forma 10","Forma superficie","Servicio"]
    v1 = [ne(s1["elo"]), ne(s1["surface_elo"]), s1["form10"], s1["surface_form"], ns(s1)]
    v2 = [ne(s2["elo"]), ne(s2["surface_elo"]), s2["form10"], s2["surface_form"], ns(s2)]

    fig = go.Figure()
    for v, name, color, alpha in [
        (v1, r["player1"], ACCENT,   0.18),
        (v2, r["player2"], "#94a3b8", 0.10),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=v+[v[0]], theta=cats+[cats[0]],
            fill="toself",
            fillcolor=rgba(color, alpha),   # FIX: rgba en lugar de hex+alpha
            line=dict(color=color, width=2),
            name=name[:24],
        ))

    fig.update_layout(
        **LAYOUT,
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(visible=True, range=[0,1], showticklabels=False,
                            gridcolor=rgba(TEXT,0.07), linecolor=rgba(TEXT,0.07)),
            angularaxis=dict(gridcolor=rgba(TEXT,0.07), linecolor=rgba(TEXT,0.07),
                             tickfont=dict(size=11, color=MUTED)),
        ),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.14,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        height=320, margin=dict(l=20,r=20,t=20,b=55),
    )
    return fig

def chart_ev(r):
    items = []
    if "ev1" in r: items.append((r["player1"], r["ev1"]*100, r["val1"]))
    if "ev2" in r: items.append((r["player2"], r["ev2"]*100, r["val2"]))
    if not items: return None

    colors = [rgba(GREEN,0.85) if i[2] else rgba(RED,0.7) for i in items]
    fig = go.Figure(go.Bar(
        x=[i[0] for i in items], y=[i[1] for i in items],
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{i[1]:+.1f}%" for i in items],
        textposition="outside", textfont=dict(size=13, color=TEXT),
        showlegend=False,
    ))
    fig.add_hline(y=0, line_color=rgba(TEXT,0.15), line_width=1)
    fig.update_yaxes(title="EV (%)", gridcolor=rgba(TEXT,0.07), zeroline=False, color=MUTED)
    fig.update_xaxes(showgrid=False, color=MUTED)
    fig.update_layout(**LAYOUT, height=220, margin=dict(l=10,r=10,t=14,b=10))
    return fig

def chart_confidence():
    xs      = [50,55,60,65,70]
    atp_acc = [65.5,67.2,69.2,71.6,74.6]
    wta_acc = [65.5,67.3,69.5,72.9,76.0]
    fig = go.Figure()
    for acc, name, color in [(atp_acc,"ATP",ACCENT),(wta_acc,"WTA","#94a3b8")]:
        fig.add_trace(go.Scatter(
            x=xs, y=acc, mode="lines+markers", name=name,
            line=dict(color=color, width=2),
            marker=dict(size=8, line=dict(width=2, color=BG)),
        ))
    fig.add_hrect(y0=70, y1=80, fillcolor=rgba(GREEN,0.05), line_width=0)
    fig.add_hline(y=50, line_dash="dot", line_color=rgba(TEXT,0.15),
                  annotation_text="nivel azar", annotation_font_size=10,
                  annotation_font_color=MUTED)
    fig.update_xaxes(title="Confianza del modelo (%)",
                     tickvals=xs, ticktext=[f"{x}%" for x in xs],
                     gridcolor=rgba(TEXT,0.06), color=MUTED)
    fig.update_yaxes(title="Precision historica (%)", range=[48,80],
                     gridcolor=rgba(TEXT,0.06), color=MUTED)
    fig.update_layout(**LAYOUT, height=290,
                      margin=dict(l=10,r=10,t=14,b=10),
                      legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.15, bgcolor="rgba(0,0,0,0)"))
    return fig

def chart_bankroll(df):
    df = df.sort_values("fecha").reset_index(drop=True).copy()
    df["cp"] = df["profit"].cumsum()
    positive = float(df["cp"].iloc[-1]) >= 0
    lc = GREEN if positive else RED
    fc = rgba(GREEN if positive else RED, 0.10)
    pt_colors = [GREEN if float(p) > 0 else RED for p in df["profit"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index + 1, y=df["cp"].round(2),
        fill="tozeroy", fillcolor=fc,
        line=dict(color=lc, width=2),
        mode="lines+markers",
        marker=dict(size=9, color=pt_colors, line=dict(width=2, color=BG)),
        hovertemplate="Apuesta #%{x}<br>Acumulado: %{y:+.2f}u<extra></extra>",
        name="",
    ))
    fig.add_hline(y=0, line_color=rgba(TEXT, 0.2), line_width=1, line_dash="dot")
    fig.update_xaxes(title="N apuesta", showgrid=False, color=MUTED, tickvals=list(df.index+1))
    fig.update_yaxes(title="Unidades", gridcolor=rgba(TEXT, 0.07), zeroline=False, color=MUTED)
    fig.update_layout(**LAYOUT, height=280,
                      margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig


def chart_surface_results(df):
    surfs = sorted(df["superficie"].unique())
    wins_map  = df[df["resultado"] == "win"].groupby("superficie").size().to_dict()
    loss_map  = df[df["resultado"] == "loss"].groupby("superficie").size().to_dict()
    wins   = [wins_map.get(s, 0) for s in surfs]
    losses = [loss_map.get(s, 0) for s in surfs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=surfs, y=wins, name="Acertadas",
        marker=dict(color=rgba(GREEN, 0.8), line=dict(width=0)),
        text=wins, textposition="auto", textfont=dict(color="white", size=13),
    ))
    fig.add_trace(go.Bar(
        x=surfs, y=losses, name="Falladas",
        marker=dict(color=rgba(RED, 0.75), line=dict(width=0)),
        text=losses, textposition="auto", textfont=dict(color="white", size=13),
    ))
    fig.update_layout(**LAYOUT, height=240, barmode="group",
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.12, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(showgrid=False, color=MUTED)
    fig.update_yaxes(showgrid=False, color=MUTED, dtick=1)
    return fig


def chart_odds_profit(df):
    fig = go.Figure()
    for mask, name, color in [
        (df["resultado"] == "win",  "Acertadas", GREEN),
        (df["resultado"] == "loss", "Falladas",  RED),
    ]:
        sub = df[mask]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["cuota"], y=sub["profit"].round(2),
            mode="markers", name=name,
            marker=dict(size=12, color=rgba(color, 0.85),
                        line=dict(width=1.5, color=BG)),
            hovertemplate=(
                "%{customdata}<br>"
                "Cuota: %{x:.2f} | Resultado: %{y:+.2f}u<extra></extra>"
            ),
            customdata=(sub["jugador"] + " vs " + sub["rival"]).tolist(),
        ))
    fig.add_hline(y=0, line_color=rgba(TEXT, 0.15), line_width=1, line_dash="dot")
    fig.update_xaxes(title="Cuota", gridcolor=rgba(TEXT, 0.07), color=MUTED)
    fig.update_yaxes(title="Beneficio (u)", gridcolor=rgba(TEXT, 0.07), color=MUTED)
    fig.update_layout(**LAYOUT, height=240, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.12, bgcolor="rgba(0,0,0,0)"))
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<p style='color:{MUTED};font-size:.75rem;letter-spacing:.1em;"
                f"text-transform:uppercase;margin-bottom:4px;'>Panel</p>",
                unsafe_allow_html=True)
    st.markdown("---")
    if st.button("Recargar datos", use_container_width=True,
                 help="Limpia la cache y recarga modelos y estados del disco"):
        st.cache_resource.clear()
        st.rerun()
    st.markdown("---")
    st.caption("Actualizacion automatica diaria a las 7:00 AM (Task Scheduler).")
    st.caption("Manual: `python src/update_data.py`")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Tennis Predictor")
st.markdown(
    f"<p style='color:{MUTED};font-size:.85rem;margin-top:-8px;margin-bottom:28px;'>"
    "Modelo basado en ELO, forma reciente, H2H y estadisticas de servicio &mdash; 2000 a 2026</p>",
    unsafe_allow_html=True,
)

tab_pred, tab_bets, tab_guide = st.tabs(["Prediccion", "Mis Apuestas", "Guia de interpretacion"])

# ── TAB PREDICCION ────────────────────────────────────────────────────────────
with tab_pred:
    col_tour, col_surf = st.columns(2)
    with col_tour:
        tour = st.selectbox("Tour", ["ATP","WTA"])
    with col_surf:
        surf_label = st.selectbox("Superficie", ["Clay","Hard","Grass"])

    surface  = surf_label.lower()
    tour_key = tour.lower()

    try:
        state, _,_,_,_, players = load_resources(tour_key)
    except FileNotFoundError:
        st.error("Modelos no encontrados. Ejecuta: python src/features.py && python src/model.py")
        st.stop()

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p1 = st.selectbox("Jugador 1", [""]+players, help="Escribe para filtrar")
    with col_p2:
        p2 = st.selectbox("Jugador 2", [""]+players, help="Escribe para filtrar")

    with st.expander("Cuotas de la casa  (opcional — activa el analisis de valor)"):
        oc1, oc2 = st.columns(2)
        with oc1:
            odds1 = st.number_input(f"Cuota {p1 or 'Jugador 1'}",
                                    min_value=1.01, max_value=200.0,
                                    value=None, placeholder="ej: 1.85", key="o1")
        with oc2:
            odds2 = st.number_input(f"Cuota {p2 or 'Jugador 2'}",
                                    min_value=1.01, max_value=200.0,
                                    value=None, placeholder="ej: 2.20", key="o2")

    st.markdown("<br>", unsafe_allow_html=True)
    go_btn = st.button("PREDECIR", use_container_width=True, type="primary")

    if go_btn:
        if not p1 or not p2:
            st.warning("Selecciona ambos jugadores.")
        elif p1 == p2:
            st.warning("Los dos jugadores deben ser distintos.")
        else:
            with st.spinner(""):
                result = predict(p1, p2, surface, tour_key, odds1=odds1, odds2=odds2)

            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown("<hr class='thin'>", unsafe_allow_html=True)

                # Cabecera resultado
                st.markdown(
                    f"<h3 style='font-weight:400;letter-spacing:.03em;margin-bottom:2px;'>"
                    f"{result['player1']} &nbsp;vs&nbsp; {result['player2']}</h3>"
                    f"<p style='color:{MUTED};font-size:.8rem;margin-top:0;margin-bottom:16px;'>"
                    f"{result['tour']} &nbsp;/&nbsp; {result['surface'].capitalize()}</p>",
                    unsafe_allow_html=True,
                )

                # Barras de probabilidad
                st.plotly_chart(chart_probs(result), width="stretch",
                                config={"displayModeBar": False})

                # Cuotas justas
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(card(result["fair1"],
                                     f"Cuota justa &mdash; {result['player1'].split()[-1]}"),
                                unsafe_allow_html=True)
                with c2:
                    st.markdown(card(result["fair2"],
                                     f"Cuota justa &mdash; {result['player2'].split()[-1]}"),
                                unsafe_allow_html=True)

                # Analisis de valor
                if "ev1" in result or "ev2" in result:
                    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
                    st.markdown(
                        f"<p style='font-weight:500;letter-spacing:.04em;'>"
                        f"Analisis de valor</p>",
                        unsafe_allow_html=True,
                    )
                    ev_fig = chart_ev(result)
                    if ev_fig:
                        st.plotly_chart(ev_fig, width="stretch",
                                        config={"displayModeBar": False})

                    ev1_col, ev2_col = st.columns(2)
                    for col, ekey, okey, vkey, kkey, player in [
                        (ev1_col,"ev1","odds1","val1","kelly1",result["player1"]),
                        (ev2_col,"ev2","odds2","val2","kelly2",result["player2"]),
                    ]:
                        if ekey not in result:
                            continue
                        ev_v  = result[ekey]
                        color = GREEN if ev_v > 0 else RED
                        kpct  = result.get(kkey, 0) * 100
                        with col:
                            st.markdown(
                                card(f"{ev_v*100:+.1f}%",
                                     f"EV &mdash; {player.split()[-1]} @ {result[okey]}",
                                     color=color,
                                     sublabel=f"Kelly: {kpct:.1f}%  |  Kelly/4: {kpct/4:.1f}%",
                                     badge=result[vkey]),
                                unsafe_allow_html=True,
                            )

                # Radar
                st.markdown("<hr class='thin'>", unsafe_allow_html=True)
                st.markdown(
                    f"<p style='font-weight:500;letter-spacing:.04em;'>Perfil comparativo</p>"
                    f"<p style='font-size:.75rem;color:{MUTED};margin-top:-8px;'>"
                    f"Escala 0-1. ELO normalizado en rango tipico 1300-2300.</p>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(chart_radar(result), width="stretch",
                                config={"displayModeBar": False})

                # Stats detalladas
                with st.expander("Estadisticas detalladas"):
                    def fmt(v, pct=False):
                        if v is None: return "---"
                        try:
                            fv = float(v)
                        except: return "---"
                        if np.isnan(fv): return "---"
                        return f"{fv*100:.1f}%" if pct else str(int(fv))

                    s1d, s2d = result["s1"], result["s2"]
                    st.dataframe(pd.DataFrame({
                        "Metrica": [
                            "ELO Global","ELO Superficie",
                            "Forma 10","Forma 20","Forma en superficie",
                            "H2H (victorias)","1er Servicio %",
                            "Ganados 1er serv.","Ganados 2do serv.","Break points salvados",
                        ],
                        result["player1"]: [
                            s1d["elo"], s1d["surface_elo"],
                            fmt(s1d["form10"],True), fmt(s1d["form20"],True),
                            fmt(s1d["surface_form"],True), s1d.get("h2h_wins",0),
                            fmt(s1d.get("serve1_pct"),True), fmt(s1d.get("serve1_won_pct"),True),
                            fmt(s1d.get("serve2_won_pct"),True), fmt(s1d.get("bp_save_pct"),True),
                        ],
                        result["player2"]: [
                            s2d["elo"], s2d["surface_elo"],
                            fmt(s2d["form10"],True), fmt(s2d["form20"],True),
                            fmt(s2d["surface_form"],True), s2d.get("h2h_wins",0),
                            fmt(s2d.get("serve1_pct"),True), fmt(s2d.get("serve1_won_pct"),True),
                            fmt(s2d.get("serve2_won_pct"),True), fmt(s2d.get("bp_save_pct"),True),
                        ],
                    }), width="stretch", hide_index=True)

                # Pill de confianza
                st.markdown(conf_pill(max(result["prob1"], result["prob2"])),
                            unsafe_allow_html=True)

    # Grafica de precision (siempre visible abajo)
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    with st.expander("Precision historica del modelo por nivel de confianza"):
        st.plotly_chart(chart_confidence(), width="stretch",
                        config={"displayModeBar": False})
        st.caption("Medido sobre partidos 2023-2026 no usados en el entrenamiento.")

# ── TAB MIS APUESTAS ─────────────────────────────────────────────────────────
with tab_bets:
    st.markdown(
        f"<p style='color:{MUTED};font-size:.8rem;margin-top:-4px;margin-bottom:24px;'>"
        "Registra cada apuesta y analiza tu rendimiento. Los datos se guardan "
        "en <code>data/apuestas.json</code>.</p>",
        unsafe_allow_html=True,
    )

    # ── Formulario ────────────────────────────────────────────────────────────
    with st.form("nueva_apuesta", clear_on_submit=True):
        st.markdown(
            f"<p style='font-weight:500;letter-spacing:.04em;margin-bottom:4px;'>"
            f"Nueva apuesta</p>",
            unsafe_allow_html=True,
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            fa_jugador = st.text_input("Apostaste por", placeholder="Ej: Alcaraz")
            fa_tour    = st.selectbox("Tour", ["ATP", "WTA"], key="fa_tour")
        with col_b:
            fa_rival   = st.text_input("Rival", placeholder="Ej: Sinner")
            fa_surf    = st.selectbox("Superficie", ["Clay", "Hard", "Grass"], key="fa_surf")
        with col_c:
            fa_cuota   = st.number_input("Cuota", min_value=1.01, max_value=50.0,
                                          value=1.85, step=0.05, key="fa_cuota")
            fa_stake   = st.number_input("Unidades apostadas", min_value=0.1,
                                          max_value=100.0, value=1.0, step=0.1, key="fa_stake")

        col_d, col_e = st.columns([1, 2])
        with col_d:
            fa_fecha = st.date_input("Fecha", value=datetime.date.today(), key="fa_fecha")
        with col_e:
            fa_result = st.radio("Resultado", ["Victoria", "Derrota"],
                                  horizontal=True, key="fa_result")

        submitted = st.form_submit_button(
            "GUARDAR APUESTA", use_container_width=True, type="primary"
        )
        if submitted:
            if fa_jugador.strip() and fa_rival.strip():
                resultado = "win" if fa_result == "Victoria" else "loss"
                profit    = round(
                    float(fa_stake) * (float(fa_cuota) - 1) if resultado == "win"
                    else -float(fa_stake), 4
                )
                save_bet({
                    "fecha":      fa_fecha.strftime("%Y-%m-%d"),
                    "tour":       fa_tour,
                    "superficie": fa_surf.lower(),
                    "jugador":    fa_jugador.strip(),
                    "rival":      fa_rival.strip(),
                    "cuota":      float(fa_cuota),
                    "stake":      float(fa_stake),
                    "resultado":  resultado,
                    "profit":     profit,
                })
                st.success("Apuesta registrada.")
                st.rerun()
            else:
                st.warning("Rellena los campos 'Apostaste por' y 'Rival'.")

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # ── Datos ─────────────────────────────────────────────────────────────────
    bets_df = load_bets()

    if bets_df.empty:
        st.markdown(
            f"<div style='text-align:center;padding:48px 24px;color:{MUTED};'>"
            f"<div style='font-size:3rem;margin-bottom:14px;opacity:.3;'>[ ]</div>"
            f"<div style='font-size:.95rem;'>Aun no hay apuestas registradas.<br>"
            f"Usa el formulario de arriba para añadir la primera.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        # ── Tarjetas resumen ──────────────────────────────────────────────────
        total      = len(bets_df)
        n_wins     = int((bets_df["resultado"] == "win").sum())
        pct_wins   = n_wins / total * 100
        total_stk  = float(bets_df["stake"].sum())
        net_profit = float(bets_df["profit"].sum())
        roi        = net_profit / total_stk * 100 if total_stk > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(card(str(total), "Apuestas"), unsafe_allow_html=True)
        with c2:
            cwr = GREEN if pct_wins >= 55 else (AMBER if pct_wins >= 45 else RED)
            st.markdown(card(f"{pct_wins:.1f}%", "% Aciertos", color=cwr), unsafe_allow_html=True)
        with c3:
            cr = GREEN if roi > 0 else RED
            st.markdown(card(f"{roi:+.1f}%", "ROI", color=cr), unsafe_allow_html=True)
        with c4:
            cn = GREEN if net_profit > 0 else RED
            st.markdown(card(f"{net_profit:+.2f}u", "Beneficio neto", color=cn),
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Evolucion bankroll ────────────────────────────────────────────────
        st.markdown(
            f"<p style='font-weight:500;letter-spacing:.04em;'>Evolucion del bankroll</p>"
            f"<p style='font-size:.75rem;color:{MUTED};margin-top:-8px;'>"
            f"Beneficio acumulado en unidades. Punto verde = apuesta ganada, rojo = perdida.</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(chart_bankroll(bets_df), width="stretch",
                        config={"displayModeBar": False})

        # ── Graficas secundarias ──────────────────────────────────────────────
        if total >= 2:
            col_s, col_o = st.columns(2)
            with col_s:
                st.markdown(
                    f"<p style='font-weight:500;letter-spacing:.04em;'>Resultados por superficie</p>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(chart_surface_results(bets_df), width="stretch",
                                config={"displayModeBar": False})
            with col_o:
                st.markdown(
                    f"<p style='font-weight:500;letter-spacing:.04em;'>Cuota vs resultado</p>"
                    f"<p style='font-size:.75rem;color:{MUTED};margin-top:-8px;'>"
                    f"Pasa el cursor para ver el partido.</p>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(chart_odds_profit(bets_df), width="stretch",
                                config={"displayModeBar": False})

        st.markdown("<hr class='thin'>", unsafe_allow_html=True)

        # ── Historial ─────────────────────────────────────────────────────────
        st.markdown(
            f"<p style='font-weight:500;letter-spacing:.04em;'>Historial completo</p>",
            unsafe_allow_html=True,
        )

        # Ordenar desc para mostrar, guardando indice original para borrar
        bets_sorted = (
            bets_df
            .reset_index()           # columna "index" = posicion en JSON
            .sort_values("fecha", ascending=False)
            .reset_index(drop=True)
        )
        show_df = bets_sorted[
            ["fecha","tour","superficie","jugador","rival","cuota","stake","resultado","profit"]
        ].copy()
        show_df["resultado"] = show_df["resultado"].map({"win": "Victoria", "loss": "Derrota"})
        show_df.rename(columns={
            "fecha": "Fecha", "tour": "Tour", "superficie": "Superficie",
            "jugador": "Apostado", "rival": "Rival", "cuota": "Cuota",
            "stake": "Stake", "resultado": "Resultado", "profit": "Beneficio",
        }, inplace=True)
        show_df.index = range(1, len(show_df) + 1)
        st.dataframe(show_df, width="stretch", hide_index=False)

        # ── Eliminar ──────────────────────────────────────────────────────────
        with st.expander("Eliminar una apuesta"):
            st.caption(
                "El numero de fila corresponde al indice de la tabla de arriba "
                "(1 = apuesta mas reciente)."
            )
            del_row = st.number_input(
                "Fila a eliminar", min_value=1, max_value=total, step=1, value=1
            )
            if st.button("Eliminar esta apuesta", key="del_bet"):
                orig_idx = int(bets_sorted.loc[int(del_row) - 1, "index"])
                delete_bet(orig_idx)
                st.rerun()


# ── TAB GUIA ──────────────────────────────────────────────────────────────────
with tab_guide:
    try:
        with open(GUIA_PATH, encoding="utf-8") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.error("guia.md no encontrado.")
