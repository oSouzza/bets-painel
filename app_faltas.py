import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime

try:
    from drive_utils import salvar_csv_drive, drive_disponivel
    _DRIVE_OK = True
except ImportError:
    _DRIVE_OK = False

st.set_page_config(page_title="Faltas Analyzer", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0a0c10; color: #c9d1d9; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1c2128; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] label { font-size:11px !important; text-transform:uppercase !important; letter-spacing:.08em !important; color:#6e7681 !important; font-weight:500 !important; }
.main { background: #0a0c10; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }
div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] > div > div { background:#0d1117 !important; border:1px solid #1c2128 !important; color:#c9d1d9 !important; border-radius:6px !important; font-family:'DM Mono',monospace !important; font-size:14px !important; }
.stNumberInput label, .stSelectbox label, .stTextInput label { font-size:11px !important; text-transform:uppercase !important; letter-spacing:.08em !important; color:#6e7681 !important; font-weight:500 !important; }
.stTabs [data-baseweb="tab-list"] { background:#0d1117; border-radius:8px; padding:3px; border:1px solid #1c2128; }
.stTabs [data-baseweb="tab"] { color:#6e7681 !important; font-size:13px !important; font-weight:500 !important; }
.stTabs [aria-selected="true"] { background:#161b22 !important; color:#c9d1d9 !important; border-radius:6px !important; }
.stButton > button[kind="primary"] { background:#238636 !important; border:1px solid #2ea043 !important; color:#fff !important; font-weight:600 !important; font-size:14px !important; border-radius:8px !important; height:44px !important; }
.stButton > button[kind="secondary"] { background:#161b22 !important; border:1px solid #1c2128 !important; color:#c9d1d9 !important; font-size:13px !important; border-radius:6px !important; }

/* Card de jogo na fila */
.jogo-card { background:#0d1117; border:1px solid #1c2128; border-radius:12px; padding:1.1rem 1.4rem; margin-bottom:1rem; }
.jogo-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:.8rem; }
.jogo-title { font-size:15px; font-weight:600; color:#e6edf3; }
.jogo-sub { font-size:11px; color:#6e7681; margin-top:2px; font-family:'DM Mono',monospace; }
.input-label { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.1em; color:#6e7681; margin-bottom:3px; }

/* Resultado inline */
.result-card { background:#0d1117; border:1px solid #1c2128; border-radius:12px; padding:1.2rem 1.4rem; }
.result-title { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.12em; color:#6e7681; margin-bottom:.6rem; }
.lambda-val { font-family:'DM Mono',monospace; font-size:28px; font-weight:500; color:#e6edf3; line-height:1; }
.lambda-sub { font-size:12px; color:#6e7681; margin-top:4px; }
.edge-row { display:flex; gap:8px; margin-top:10px; flex-wrap:wrap; align-items:center; }
.edge-label { font-size:12px; color:#6e7681; }
.badge-pos { font-family:'DM Mono',monospace; font-size:13px; font-weight:500; background:#0d2818; color:#3fb950; border:1px solid #2ea043; border-radius:5px; padding:2px 10px; }
.badge-neg { font-family:'DM Mono',monospace; font-size:13px; font-weight:500; background:#2d1111; color:#f85149; border:1px solid #da3633; border-radius:5px; padding:2px 10px; }
.badge-neu { font-family:'DM Mono',monospace; font-size:13px; font-weight:500; background:#131d2e; color:#79c0ff; border:1px solid #388bfd; border-radius:5px; padding:2px 10px; }
.prob-line { font-size:12px; color:#6e7681; margin-top:6px; font-family:'DM Mono',monospace; }

/* Boas apostas */
.stake-card { background:#0d1117; border:1px solid #2ea043; border-radius:10px; padding:1rem 1.2rem; margin-bottom:.6rem; }
.stake-header { font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.1em; color:#3fb950; margin-bottom:.5rem; }
.stake-match { font-size:15px; font-weight:600; color:#e6edf3; }
.stake-detail { font-size:12px; color:#6e7681; font-family:'DM Mono',monospace; margin-top:4px; }
.stake-val { font-family:'DM Mono',monospace; font-size:22px; font-weight:500; color:#3fb950; }

.stat-box { background:#0d1117; border:1px solid #1c2128; border-radius:8px; padding:.8rem 1rem; text-align:center; }
.stat-box .val { font-family:'DM Mono',monospace; font-size:20px; color:#e6edf3; }
.stat-box .lbl { font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:#6e7681; margin-top:2px; }
.sec { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.12em; color:#6e7681; border-bottom:1px solid #1c2128; padding-bottom:6px; margin:1.2rem 0 .8rem; }
</style>
""", unsafe_allow_html=True)

# ─── FUNÇÕES DO MODELO ────────────────────────────────────────────────────────

def weighted_mean(series):
    n = len(series)
    if n == 0: return 0
    weights = list(range(1, n + 1))
    return (series * weights).sum() / sum(weights)

def home_stats(df, team):
    g = df[df["HomeTeam"] == team].tail(10)
    return weighted_mean(g["HomeFouls"]), weighted_mean(g["AwayFouls"])

def away_stats(df, team):
    g = df[df["AwayTeam"] == team].tail(10)
    return weighted_mean(g["AwayFouls"]), weighted_mean(g["HomeFouls"])

def overall_stats(df, team):
    hg = df[df["HomeTeam"] == team].tail(10)
    ag = df[df["AwayTeam"] == team].tail(10)
    comm = weighted_mean(pd.concat([hg["HomeFouls"], ag["AwayFouls"]]))
    conc = weighted_mean(pd.concat([hg["AwayFouls"], ag["HomeFouls"]]))
    return comm, conc

def team_pace(df, team):
    hg = df[df["HomeTeam"] == team].tail(10)
    ag = df[df["AwayTeam"] == team].tail(10)
    return weighted_mean(pd.concat([hg["TotalFouls"], ag["TotalFouls"]]))

def predict_fouls(df, home_team, away_team, referee_avg):
    league_avg = df["TotalFouls"].mean()
    hc, hnc = home_stats(df, home_team)
    ac, anc = away_stats(df, away_team)
    ht, hnt = overall_stats(df, home_team)
    at, ant = overall_stats(df, away_team)
    w_ha, w_t = 0.7, 0.3
    home_comm = hc * w_ha + ht * w_t
    home_conc = hnc * w_ha + hnt * w_t
    away_comm = ac * w_ha + at * w_t
    away_conc = anc * w_ha + ant * w_t
    def_home = home_conc / (league_avg / 2)
    def_away = away_conc / (league_avg / 2)
    pace = ((team_pace(df, home_team) + team_pace(df, away_team)) / 2) / league_avg
    ref_factor = referee_avg / league_avg
    return home_comm * def_away * pace * ref_factor, away_comm * def_home * pace * ref_factor

def calculate_edge(lv, line, odd_over, odd_under):
    k = int(line)
    pu = poisson.cdf(k, lv)
    po = 1 - pu
    return {"prob_over": po, "prob_under": pu,
            "fair_over": 1/po if po > 0 else 0, "fair_under": 1/pu if pu > 0 else 0,
            "edge_over": po * odd_over - 1, "edge_under": pu * odd_under - 1}

def calcular_edge_total(lt, line, odd_over, odd_under):
    k = int(line)
    pu = poisson.cdf(k, lt); po = 1 - pu
    return po, pu, (po * odd_over) - 1, (pu * odd_under) - 1

def stake_por_diff(diff):
    if diff < 1: return 0
    elif diff < 2: return 1
    elif diff < 3: return 1
    elif diff < 4: return 1.25
    elif diff < 5: return 1.5
    elif diff < 6: return 1.75
    else: return 2

def snap_half(val):
    return round(int(val) + 0.5, 1) if val % 1 != 0.5 else val

def badge(val):
    pct = f"{val*100:+.1f}%"
    if val > 0.03: return f'<span class="badge-pos">{pct}</span>'
    elif val < -0.03: return f'<span class="badge-neg">{pct}</span>'
    else: return f'<span class="badge-neu">{pct}</span>'

def result_card(titulo, lam, linha, ed):
    return f"""<div class="result-card">
        <div class="result-title">{titulo}</div>
        <div class="lambda-val">{lam:.2f}</div>
        <div class="lambda-sub">λ esperado · linha {linha}</div>
        <div class="edge-row"><span class="edge-label">Over</span>{badge(ed['edge_over'])}<span class="edge-label">Under</span>{badge(ed['edge_under'])}</div>
        <div class="prob-line">P(over) {ed['prob_over']*100:.1f}% · P(under) {ed['prob_under']*100:.1f}% &nbsp;|&nbsp; fair {ed['fair_over']:.2f} / {ed['fair_under']:.2f}</div>
    </div>"""

def stat_box(val, label):
    return f'<div class="stat-box"><div class="val">{val}</div><div class="lbl">{label}</div></div>'

COLS_MODELO = [
    "Rodada","HomeTeam","AwayTeam","Date","Arbitro","Referee_Avg",
    "Lambda_Home","Lambda_Away","Lambda_Total",
    "Total_Line","Odd_Over_Total","Odd_Under_Total",
    "Line_Home","Odd_Over_Home","Odd_Under_Home",
    "Line_Away","Odd_Over_Away","Odd_Under_Away",
    "Prob_Over_Total","Prob_Under_Total","Edge_Over_Total","Edge_Under_Total",
    "Prob_Over_Home","Prob_Under_Home","Edge_Over_Home","Edge_Under_Home",
    "Prob_Over_Away","Prob_Under_Away","Edge_Over_Away","Edge_Under_Away",
    "diff","stake","Real_Fouls_Home","Real_Fouls_Away","Real_Fouls_Total"
]

# ─── SESSION STATE ────────────────────────────────────────────────────────────

if "ligas" not in st.session_state:
    st.session_state.ligas = {}
if "jogos_ocultos" not in st.session_state:
    # set de chaves "liga|home|away|rodada" ocultados manualmente
    st.session_state.jogos_ocultos = set()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚽ Faltas Analyzer")
    st.markdown("---")
    st.markdown("### Ligas")

    with st.expander("➕  Adicionar liga", expanded=len(st.session_state.ligas) == 0):
        nome_liga   = st.text_input("Nome da liga", placeholder="Ex: Alemanha, Brasil...", key="inp_nome")
        f_faltas_up = st.file_uploader("CSV histórico de faltas", type=["csv"], key="up_faltas",
                                        help="Colunas: HomeTeam, AwayTeam, HomeFouls, AwayFouls, TotalFouls (opcional: arbitro)")
        f_jogos_up  = st.file_uploader("CSV de jogos da rodada", type=["csv"], key="up_jogos",
                                        help="Colunas: rodada, HomeTeam, AwayTeam (opcional: date, arbitro)")
        f_model_up  = st.file_uploader("CSV model — opcional", type=["csv"], key="up_model")

        if st.button("Adicionar", key="btn_add"):
            if not nome_liga:
                st.warning("Digite um nome.")
            elif not f_faltas_up:
                st.warning("Faça upload do CSV de faltas.")
            elif not f_jogos_up:
                st.warning("Faça upload do CSV de jogos.")
            elif nome_liga in st.session_state.ligas:
                st.warning("Liga já existe.")
            else:
                df_f = pd.read_csv(f_faltas_up)
                if "TotalFouls" not in df_f.columns:
                    df_f["TotalFouls"] = df_f["HomeFouls"] + df_f["AwayFouls"]

                df_j = pd.read_csv(f_jogos_up)
                # normalizar colunas do df de jogos
                col_map = {"homeTeam": "HomeTeam", "awayTeam": "AwayTeam",
                           "Rodada": "rodada", "Round": "rodada"}
                df_j = df_j.rename(columns=col_map)

                df_h = pd.read_csv(f_model_up) if f_model_up else pd.DataFrame(columns=COLS_MODELO)

                tem_arbitro_jogos = "arbitro" in df_j.columns
                tem_arbitro_faltas = "arbitro" in df_f.columns

                st.session_state.ligas[nome_liga] = {
                    "df":                df_f,
                    "df_jogos":          df_j,
                    "historico":         df_h,
                    "historico_original": df_h.copy(),
                    "tem_arbitro_jogos": tem_arbitro_jogos,
                    "tem_arbitro_faltas": tem_arbitro_faltas,
                }
                st.success(f"'{nome_liga}' adicionada!")
                st.rerun()

    if st.session_state.ligas:
        st.markdown("**Ligas carregadas:**")
        for nome in list(st.session_state.ligas.keys()):
            c1, c2 = st.columns([4, 1])
            with c1:
                n = len(st.session_state.ligas[nome]["historico"])
                st.markdown(f"<span style='font-size:13px'>⚽ {nome}</span> <span style='font-size:11px;color:#6e7681'>({n} reg.)</span>", unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"rm_{nome}"):
                    del st.session_state.ligas[nome]
                    st.rerun()

# ─── SEM LIGAS ────────────────────────────────────────────────────────────────

if not st.session_state.ligas:
    st.markdown('<h1 style="color:#e6edf3;font-size:22px;font-weight:600;">Faltas Analyzer</h1>', unsafe_allow_html=True)
    st.info("⬅️  Adicione pelo menos uma liga na sidebar para começar.")
    st.stop()

# ─── SELETOR DE LIGA ──────────────────────────────────────────────────────────

st.markdown('<h1 style="color:#e6edf3;font-size:22px;font-weight:600;margin-bottom:6px;">Faltas Analyzer</h1>', unsafe_allow_html=True)

col_liga, col_info = st.columns([3, 7])
with col_liga:
    liga_ativa = st.selectbox("Liga ativa", list(st.session_state.ligas.keys()), key="liga_ativa")

liga_data        = st.session_state.ligas[liga_ativa]
df               = liga_data["df"]
df_jogos         = liga_data["df_jogos"]
historico        = liga_data["historico"]
tem_arb_jogos    = liga_data["tem_arbitro_jogos"]
tem_arb_faltas   = liga_data["tem_arbitro_faltas"]
arbitros_faltas  = sorted(df["arbitro"].dropna().unique().tolist()) if tem_arb_faltas else []
league_avg       = df["TotalFouls"].mean()

with col_info:
    st.markdown(
        f'<p style="color:#6e7681;font-size:13px;margin-top:32px;">'
        f'{len(df)} partidas no histórico &nbsp;·&nbsp; {len(df_jogos)} jogos na fila &nbsp;·&nbsp; {len(historico)} registros no model</p>',
        unsafe_allow_html=True
    )

# ─── ABAS ─────────────────────────────────────────────────────────────────────

aba_analise, aba_apostas, aba_historico = st.tabs(["⚡  Análise", "🎯  Boas Apostas", "📋  Histórico"])

# ══════════════════════════════════════════════════════════════════════════════
# ABA ANÁLISE — fila de jogos
# ══════════════════════════════════════════════════════════════════════════════
with aba_analise:

    # Identificar jogos já calculados no model
    if not historico.empty and "HomeTeam" in historico.columns:
        jogos_calculados = set(
            zip(historico["HomeTeam"].astype(str), historico["AwayTeam"].astype(str))
        )
    else:
        jogos_calculados = set()

    # Filtrar fila: remover já calculados e ocultos manualmente
    fila = []
    for _, row in df_jogos.iterrows():
        home = str(row["HomeTeam"])
        away = str(row["AwayTeam"])
        rodada_j = str(row.get("rodada", ""))
        chave = f"{liga_ativa}|{home}|{away}|{rodada_j}"
        par = (home, away)
        if par in jogos_calculados:
            continue
        if chave in st.session_state.jogos_ocultos:
            continue
        fila.append(row)

    n_calculados = len(df_jogos) - len(fila) - len([
        r for r in df_jogos.itertuples()
        if f"{liga_ativa}|{r.HomeTeam}|{r.AwayTeam}|{getattr(r,'rodada','')}" in st.session_state.jogos_ocultos
        and (str(r.HomeTeam), str(r.AwayTeam)) not in jogos_calculados
    ])

    col_fila, col_ocultos = st.columns([6, 4])
    with col_fila:
        st.markdown(f'<div class="sec">Fila de jogos — {len(fila)} pendentes · {len(jogos_calculados)} calculados</div>', unsafe_allow_html=True)
    with col_ocultos:
        if st.button("↺  Restaurar ocultos", use_container_width=True):
            # Limpa só os ocultos desta liga
            st.session_state.jogos_ocultos = {
                k for k in st.session_state.jogos_ocultos if not k.startswith(f"{liga_ativa}|")
            }
            st.rerun()

    if not fila:
        st.info("Todos os jogos desta liga já foram calculados ou ocultados.")
    
    ESTREANTE = "⭐ Estreante — média manual"

    for idx_fila, row in enumerate(fila):
        home    = str(row["HomeTeam"])
        away    = str(row["AwayTeam"])
        rodada_j = str(row.get("rodada", "—"))
        date_j  = str(row.get("date", row.get("Date", "")))
        arb_csv = str(row.get("arbitro", "")) if tem_arb_jogos else ""
        chave   = f"{liga_ativa}|{home}|{away}|{rodada_j}"
        prefix  = f"{liga_ativa}_{home}_{away}_{idx_fila}".replace(" ", "_")

        # Stats rápidas dos times
        fk_home = df[df["HomeTeam"] == home]
        fk_away = df[df["AwayTeam"] == away]
        mh = fk_home["HomeFouls"].mean()
        sh = fk_home["AwayFouls"].mean()
        ma = fk_away["AwayFouls"].mean()
        sa = fk_away["HomeFouls"].mean()

        with st.container():
            st.markdown(f"""
            <div class="jogo-card">
                <div class="jogo-header">
                    <div>
                        <div class="jogo-title">🏠 {home} &nbsp;vs&nbsp; ✈️ {away}</div>
                        <div class="jogo-sub">Rodada {rodada_j}{f" · {date_j[:10]}" if date_j else ""}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Linha 1: stats
            s1, s2, s3, s4 = st.columns(4)
            with s1: st.markdown(stat_box(f"{mh:.1f}" if not pd.isna(mh) else "—", f"{home[:13]} faltas casa"), unsafe_allow_html=True)
            with s2: st.markdown(stat_box(f"{sh:.1f}" if not pd.isna(sh) else "—", f"Sofridas {home[:11]} casa"), unsafe_allow_html=True)
            with s3: st.markdown(stat_box(f"{ma:.1f}" if not pd.isna(ma) else "—", f"{away[:13]} faltas fora"), unsafe_allow_html=True)
            with s4: st.markdown(stat_box(f"{sa:.1f}" if not pd.isna(sa) else "—", f"Sofridas {away[:11]} fora"), unsafe_allow_html=True)

            st.markdown("")

            # Linha 2: árbitro
            if tem_arb_jogos and arb_csv and arb_csv != "nan":
                # árbitro já vem no df de jogos
                jogos_arb = df[df["arbitro"] == arb_csv] if tem_arb_faltas else pd.DataFrame()
                if not jogos_arb.empty:
                    ref_avg = float(jogos_arb["TotalFouls"].mean())
                else:
                    ref_avg = league_avg
                st.markdown(
                    f'<p style="font-size:12px;color:#6e7681;font-family:\'DM Mono\',monospace;">'
                    f'Árbitro: {arb_csv} &nbsp;·&nbsp; média {ref_avg:.2f} faltas/jogo'
                    f'{" (liga)" if jogos_arb.empty else f" ({len(jogos_arb)} jogos)"}</p>',
                    unsafe_allow_html=True
                )
                arbitro_nome = arb_csv
            else:
                # Sem árbitro no df de jogos — dropdown ou manual
                arb_col1, arb_col2 = st.columns([3, 3])
                with arb_col1:
                    if arbitros_faltas:
                        opcoes_arb = arbitros_faltas + [ESTREANTE]
                        sel_arb = st.selectbox("Árbitro", opcoes_arb, key=f"{prefix}_arb")
                    else:
                        sel_arb = ESTREANTE
                        st.markdown('<p style="font-size:11px;color:#6e7681;">Sem árbitros no histórico — use média manual</p>', unsafe_allow_html=True)

                if sel_arb == ESTREANTE:
                    with arb_col2:
                        ref_avg = st.number_input(
                            "Média faltas árbitro",
                            min_value=0.0, value=round(float(league_avg), 2),
                            step=0.1, format="%.2f", key=f"{prefix}_ref_avg"
                        )
                    arbitro_nome = st.text_input("Nome (opcional)", key=f"{prefix}_arb_nome", placeholder="Ex: João Silva") or "Estreante"
                else:
                    jogos_arb = df[df["arbitro"] == sel_arb] if tem_arb_faltas else pd.DataFrame()
                    ref_avg = float(jogos_arb["TotalFouls"].mean()) if not jogos_arb.empty else league_avg
                    arbitro_nome = sel_arb

            st.markdown("")

            # Linha 3: inputs de linhas e odds — tudo visível de uma vez
            # Total | Casa | Fora | Árbitro (já resolvido acima)
            # Colunas: [linha_total, odd_ot, odd_ut] [linha_casa, odd_oh, odd_uh] [linha_fora, odd_oa, odd_ua]
            c_tot, c_casa, c_fora = st.columns(3)

            with c_tot:
                st.markdown('<div class="input-label">🏟️ Total</div>', unsafe_allow_html=True)
                _lt = st.number_input("Linha total", value=18.5, step=0.5, format="%.1f", key=f"{prefix}_lt", label_visibility="collapsed")
                line_jogo = snap_half(_lt)
                if line_jogo != _lt: st.caption(f"↳ {line_jogo}")
                odd_jogo_over  = st.number_input("Over total", value=1.90, step=0.01, format="%.2f", key=f"{prefix}_oto", label_visibility="collapsed", placeholder="Odd Over")
                odd_jogo_under = st.number_input("Under total", value=1.83, step=0.01, format="%.2f", key=f"{prefix}_otu", label_visibility="collapsed", placeholder="Odd Under")

            with c_casa:
                st.markdown('<div class="input-label">🏠 Casa</div>', unsafe_allow_html=True)
                _lc = st.number_input("Linha casa", value=9.5, step=0.5, format="%.1f", key=f"{prefix}_lc", label_visibility="collapsed")
                line_casa = snap_half(_lc)
                if line_casa != _lc: st.caption(f"↳ {line_casa}")
                odd_over_home  = st.number_input("Over casa", value=2.00, step=0.01, format="%.2f", key=f"{prefix}_oho", label_visibility="collapsed")
                odd_under_home = st.number_input("Under casa", value=1.75, step=0.01, format="%.2f", key=f"{prefix}_ohu", label_visibility="collapsed")

            with c_fora:
                st.markdown('<div class="input-label">✈️ Fora</div>', unsafe_allow_html=True)
                _lf = st.number_input("Linha fora", value=9.5, step=0.5, format="%.1f", key=f"{prefix}_lf", label_visibility="collapsed")
                line_fora = snap_half(_lf)
                if line_fora != _lf: st.caption(f"↳ {line_fora}")
                odd_over_away  = st.number_input("Over fora", value=2.10, step=0.01, format="%.2f", key=f"{prefix}_oao", label_visibility="collapsed")
                odd_under_away = st.number_input("Under fora", value=1.67, step=0.01, format="%.2f", key=f"{prefix}_oau", label_visibility="collapsed")

            st.markdown("")

            # Botões
            btn1, btn2 = st.columns([5, 1])
            with btn1:
                calcular = st.button("⚡  Calcular Edge", type="primary", use_container_width=True, key=f"{prefix}_calc")
            with btn2:
                ocultar = st.button("🗑️", use_container_width=True, key=f"{prefix}_ocultar", help="Remover da fila")

            if ocultar:
                st.session_state.jogos_ocultos.add(chave)
                st.rerun()

            if calcular:
                lh, la = predict_fouls(df, home, away, ref_avg)
                lt = lh + la
                he = calculate_edge(lh, line_casa, odd_over_home, odd_under_home)
                ae = calculate_edge(la, line_fora, odd_over_away, odd_under_away)
                po_t, pu_t, eo_t, eu_t = calcular_edge_total(lt, line_jogo, odd_jogo_over, odd_jogo_under)
                te = {"prob_over": po_t, "prob_under": pu_t,
                      "fair_over": 1/po_t if po_t > 0 else 0,
                      "fair_under": 1/pu_t if pu_t > 0 else 0,
                      "edge_over": eo_t, "edge_under": eu_t}

                diff  = abs(lt - line_jogo)
                stake = stake_por_diff(diff)

                r1, r2, r3 = st.columns(3)
                with r1: st.markdown(result_card(f"🏟️ Total · diff {diff:.2f} · {stake}u", lt, line_jogo, te), unsafe_allow_html=True)
                with r2: st.markdown(result_card(f"🏠 {home}", lh, line_casa, he), unsafe_allow_html=True)
                with r3: st.markdown(result_card(f"✈️ {away}", la, line_fora, ae), unsafe_allow_html=True)

                # Usar data do CSV de jogos se disponível, senão data de hoje
                data_jogo = date_j[:10] if date_j and date_j.strip() not in ("","nan") else datetime.now().strftime("%Y-%m-%d")

                registro = {
                    "Rodada": rodada_j, "HomeTeam": home, "AwayTeam": away,
                    "Date": data_jogo,
                    "Arbitro": arbitro_nome, "Referee_Avg": round(ref_avg, 2),
                    "Lambda_Home": round(lh, 4), "Lambda_Away": round(la, 4), "Lambda_Total": round(lt, 4),
                    "Total_Line": line_jogo, "Odd_Over_Total": odd_jogo_over, "Odd_Under_Total": odd_jogo_under,
                    "Line_Home": line_casa, "Odd_Over_Home": odd_over_home, "Odd_Under_Home": odd_under_home,
                    "Line_Away": line_fora, "Odd_Over_Away": odd_over_away, "Odd_Under_Away": odd_under_away,
                    "Prob_Over_Total": round(po_t, 4), "Prob_Under_Total": round(pu_t, 4),
                    "Edge_Over_Total": round(eo_t, 4), "Edge_Under_Total": round(eu_t, 4),
                    "Prob_Over_Home": round(he["prob_over"], 4), "Prob_Under_Home": round(he["prob_under"], 4),
                    "Edge_Over_Home": round(he["edge_over"], 4), "Edge_Under_Home": round(he["edge_under"], 4),
                    "Prob_Over_Away": round(ae["prob_over"], 4), "Prob_Under_Away": round(ae["prob_under"], 4),
                    "Edge_Over_Away": round(ae["edge_over"], 4), "Edge_Under_Away": round(ae["edge_under"], 4),
                    "diff": round(diff, 4), "stake": stake,
                    "Real_Fouls_Home": None, "Real_Fouls_Away": None, "Real_Fouls_Total": None
                }

                novo_hist = pd.concat(
                    [st.session_state.ligas[liga_ativa]["historico"], pd.DataFrame([registro])],
                    ignore_index=True
                )
                st.session_state.ligas[liga_ativa]["historico"] = novo_hist
                st.success(f"✓  {home} vs {away} salvo.")
                st.rerun()

            st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ABA BOAS APOSTAS
# ══════════════════════════════════════════════════════════════════════════════
with aba_apostas:

    hist = st.session_state.ligas[liga_ativa]["historico"]

    if hist.empty or "diff" not in hist.columns:
        st.info("Calcule pelo menos um jogo na aba Análise para ver as apostas recomendadas.")
    else:
        def render_apostas(df_render, titulo, stake_fixo=None):
            if df_render.empty: return
            st.markdown(f'<div class="sec">{titulo}</div>', unsafe_allow_html=True)
            for _, row in df_render.iterrows():
                lado_over = row["Lambda_Total"] > row["Total_Line"]
                lado      = "Over" if lado_over else "Under"
                odd_lado  = row["Odd_Over_Total"] if lado_over else row["Odd_Under_Total"]
                edge_lado = row["Edge_Over_Total"] if lado_over else row["Edge_Under_Total"]
                stake_show = stake_fixo if stake_fixo else row["stake"]
                st.markdown(f"""
                <div class="stake-card">
                    <div class="stake-header">⚽ Aposta recomendada</div>
                    <div class="stake-match">{row['HomeTeam']} vs {row['AwayTeam']}</div>
                    <div class="stake-detail">
                        {lado} {row['Total_Line']} &nbsp;·&nbsp; odd {odd_lado:.2f}
                        &nbsp;·&nbsp; {badge(edge_lado)}
                        &nbsp;·&nbsp; λ {row['Lambda_Total']:.2f} vs linha {row['Total_Line']}
                        &nbsp;·&nbsp; diff {row['diff']:.2f}
                    </div>
                    <div style="margin-top:8px;">
                        <span style="font-size:11px;color:#6e7681;text-transform:uppercase;letter-spacing:.08em;">Stake &nbsp;</span>
                        <span class="stake-val">{stake_show}u</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        rodadas_disponiveis = sorted(hist["Rodada"].dropna().unique().tolist(), key=lambda x: str(x), reverse=True)
        rodada_sel = st.selectbox("Rodada", rodadas_disponiveis, key="rodada_apostas") if rodadas_disponiveis else None

        if rodada_sel:
            hist_r = hist[hist["Rodada"].astype(str) == str(rodada_sel)]
            df_2   = hist_r[hist_r["diff"] >= 2].copy()
            df_1   = hist_r[(hist_r["diff"] >= 1) & (hist_r["diff"] < 2)].copy()

            if df_2.empty and df_1.empty:
                st.info(f"Nenhum jogo com diff ≥ 1 na rodada {rodada_sel}.")
            else:
                render_apostas(df_2, f"diff ≥ 2 — {liga_ativa} · Rodada {rodada_sel}")
                render_apostas(df_1, f"diff ≥ 1 — {liga_ativa} · Rodada {rodada_sel}", stake_fixo=1)

                df_resumo = pd.concat([df_2, df_1])
                cols_show = ["HomeTeam","AwayTeam","Lambda_Total","Total_Line","diff","stake","Odd_Over_Total","Odd_Under_Total"]
                cols_show = [c for c in cols_show if c in df_resumo.columns]
                st.markdown('<div class="sec">Resumo</div>', unsafe_allow_html=True)
                st.dataframe(df_resumo[cols_show].round(2), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABA HISTÓRICO
# ══════════════════════════════════════════════════════════════════════════════
with aba_historico:

    hist = st.session_state.ligas[liga_ativa]["historico"]

    if hist.empty:
        st.info("Nenhum registro ainda para esta liga.")
    else:
        st.markdown(f'<div class="sec">Histórico — {liga_ativa} ({len(hist)} registros)</div>', unsafe_allow_html=True)

        cols_view = ["Rodada","HomeTeam","AwayTeam","Arbitro","Referee_Avg",
                     "Lambda_Home","Lambda_Away","Lambda_Total","diff","stake",
                     "Total_Line","Odd_Over_Total","Odd_Under_Total",
                     "Line_Home","Odd_Over_Home","Odd_Under_Home",
                     "Line_Away","Odd_Over_Away","Odd_Under_Away",
                     "Real_Fouls_Home","Real_Fouls_Away","Real_Fouls_Total"]
        cols_view = [c for c in cols_view if c in hist.columns]
        st.dataframe(hist[cols_view].round(4), use_container_width=True, hide_index=True)

        with st.expander("✏️  Preencher resultados reais"):
            idx_opcoes = [f"{i} · {r['HomeTeam']} vs {r['AwayTeam']}" for i, r in hist.iterrows()]
            sel   = st.selectbox("Jogo", idx_opcoes, key="sel_jogo")
            idx_s = int(sel.split(" · ")[0])
            rc1, rc2, rc3 = st.columns(3)
            with rc1: real_h = st.number_input("Faltas Casa",  min_value=0, step=1, key="rh")
            with rc2: real_a = st.number_input("Faltas Fora",  min_value=0, step=1, key="ra")
            with rc3: real_t = st.number_input("Total Faltas", min_value=0, step=1, key="rt", value=int(real_h + real_a))
            if st.button("Salvar resultado real"):
                st.session_state.ligas[liga_ativa]["historico"].at[idx_s, "Real_Fouls_Home"]  = real_h
                st.session_state.ligas[liga_ativa]["historico"].at[idx_s, "Real_Fouls_Away"]  = real_a
                st.session_state.ligas[liga_ativa]["historico"].at[idx_s, "Real_Fouls_Total"] = real_t
                st.success("Resultado salvo!")
                st.rerun()

        st.markdown("")
        a1, a2, a3 = st.columns(3)

        with a1:
            nome_csv  = f"model_{liga_ativa.lower().replace(' ','_')}FALTAS.csv"
            csv_bytes = hist.to_csv(index=False).encode("utf-8")
            st.download_button(f"⬇️  Exportar {liga_ativa}", data=csv_bytes,
                               file_name=nome_csv, mime="text/csv", use_container_width=True)

        with a2:
            if _DRIVE_OK and drive_disponivel():
                if st.button("☁️  Salvar no Drive", use_container_width=True, key="drive_save"):
                    subtipo = "faltas_br" if "brasil" in liga_ativa.lower() or "br" in liga_ativa.lower() else "faltas_eu"
                    nome_drive = f"model_{liga_ativa.lower().replace(' ','_')}FALTAS.csv"
                    ok = salvar_csv_drive(hist, nome_drive, subtipo)
                    if ok:
                        st.success(f"✓ Salvo no Drive: {nome_drive}")
            else:
                if st.button("🗑️  Apagar último jogo", use_container_width=True):
                    st.session_state.ligas[liga_ativa]["historico"] = hist.iloc[:-1].reset_index(drop=True)
                    st.success("Último jogo removido.")
                    st.rerun()

        with a3:
            if _DRIVE_OK and drive_disponivel():
                if st.button("🗑️  Apagar último jogo", use_container_width=True):
                    st.session_state.ligas[liga_ativa]["historico"] = hist.iloc[:-1].reset_index(drop=True)
                    st.success("Último jogo removido.")
                    st.rerun()
            else:
                if st.button("🗑️  Limpar sessão", use_container_width=True):
                    original = st.session_state.ligas[liga_ativa].get("historico_original")
                    st.session_state.ligas[liga_ativa]["historico"] = original.copy() if original is not None else pd.DataFrame(columns=COLS_MODELO)
                    st.success("Sessão limpa.")
                    st.rerun()
