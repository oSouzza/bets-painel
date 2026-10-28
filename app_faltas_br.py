import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import LinearRegression
from datetime import date, datetime

# ─── DRIVE (opcional) ─────────────────────────────────────────────────────────
try:
    from drive_utils import salvar_csv_drive, drive_disponivel
    _DRIVE_OK = True
except ImportError:
    _DRIVE_OK = False

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Faltas BR Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0a0c10; color: #c9d1d9; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1c2128; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] label { font-size: 11px !important; text-transform: uppercase !important; letter-spacing: .08em !important; color: #6e7681 !important; font-weight: 500 !important; }
.main { background: #0a0c10; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }
div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] > div > div { background: #0d1117 !important; border: 1px solid #1c2128 !important; color: #c9d1d9 !important; border-radius: 6px !important; font-family: 'DM Mono', monospace !important; font-size: 14px !important; }
.stNumberInput label, .stSelectbox label, .stTextInput label { font-size: 11px !important; text-transform: uppercase !important; letter-spacing: .08em !important; color: #6e7681 !important; font-weight: 500 !important; }
.stTabs [data-baseweb="tab-list"] { background: #0d1117; border-radius: 8px; padding: 3px; border: 1px solid #1c2128; }
.stTabs [data-baseweb="tab"] { color: #6e7681 !important; font-size: 13px !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: #161b22 !important; color: #c9d1d9 !important; border-radius: 6px !important; }
.stButton > button[kind="primary"] { background: #238636 !important; border: 1px solid #2ea043 !important; color: #fff !important; font-weight: 600 !important; font-size: 14px !important; border-radius: 8px !important; height: 48px !important; }
.stButton > button[kind="secondary"] { background: #161b22 !important; border: 1px solid #1c2128 !important; color: #c9d1d9 !important; font-size: 13px !important; border-radius: 6px !important; }
.result-card { background: #0d1117; border: 1px solid #1c2128; border-radius: 12px; padding: 1.2rem 1.4rem; }
.result-title { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .12em; color: #6e7681; margin-bottom: .6rem; }
.lambda-val { font-family: 'DM Mono', monospace; font-size: 30px; font-weight: 500; color: #e6edf3; line-height: 1; }
.lambda-sub { font-size: 12px; color: #6e7681; margin-top: 4px; }
.edge-row { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; align-items: center; }
.edge-label { font-size: 12px; color: #6e7681; }
.badge-pos  { font-family:'DM Mono',monospace; font-size:13px; font-weight:500; background:#0d2818; color:#3fb950; border:1px solid #2ea043; border-radius:5px; padding:2px 10px; }
.badge-neg  { font-family:'DM Mono',monospace; font-size:13px; font-weight:500; background:#2d1111; color:#f85149; border:1px solid #da3633; border-radius:5px; padding:2px 10px; }
.badge-neu  { font-family:'DM Mono',monospace; font-size:13px; font-weight:500; background:#131d2e; color:#79c0ff; border:1px solid #388bfd; border-radius:5px; padding:2px 10px; }
.prob-line  { font-size:12px; color:#6e7681; margin-top:6px; font-family:'DM Mono',monospace; }
.stake-card { background: #0d1117; border: 1px solid #2ea043; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: .6rem; }
.stake-header { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .1em; color: #3fb950; margin-bottom: .5rem; }
.stake-match  { font-size: 15px; font-weight: 600; color: #e6edf3; }
.stake-detail { font-size: 12px; color: #6e7681; font-family: 'DM Mono', monospace; margin-top: 4px; }
.stake-val    { font-family: 'DM Mono', monospace; font-size: 22px; font-weight: 500; color: #3fb950; }
.stat-box { background: #0d1117; border: 1px solid #1c2128; border-radius: 8px; padding: .8rem 1rem; text-align: center; }
.stat-box .val { font-family:'DM Mono',monospace; font-size:20px; color:#e6edf3; }
.stat-box .lbl { font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:#6e7681; margin-top:2px; }
.sec { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.12em; color:#6e7681; border-bottom:1px solid #1c2128; padding-bottom:6px; margin:1.2rem 0 .8rem; }
.cal-box { background:#0d1117; border:1px solid #2d3748; border-radius:8px; padding:.7rem 1rem; font-size:12px; font-family:'DM Mono',monospace; color:#6e7681; margin-top:.5rem; }
.jogo-card { background:#0d1117; border:1px solid #1c2128; border-radius:10px; padding:1rem 1.2rem; margin-bottom:.8rem; }
.jogo-card-header { font-size:13px; font-weight:600; color:#e6edf3; margin-bottom:.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── CONFIGURAÇÕES ────────────────────────────────────────────────────────────

N_JOGOS   = 10
W_HA      = 0.7
W_TOT     = 0.3
MIN_JOGOS = 7
MIN_CAL   = 20

COL_MAP = {
    "HomeTeam":  ["HomeTeam", "homeTeam"],
    "AwayTeam":  ["AwayTeam", "awayTeam"],
    "HomeFouls": ["HomeFouls", "homeFouls"],
    "AwayFouls": ["AwayFouls", "awayFouls"],
    "Referee":   ["Referee", "arbitro", "referee", "Arbitro"],
    "Rodada":    ["Rodada", "rodada", "round"],
    "id":        ["id", "Id", "ID"],
    "Date":      ["Date", "date", "Data"],
}

COLUNAS_MODELO = [
    "Date", "Rodada", "HomeTeam", "AwayTeam", "Referee", "Arbitro", "Referee_Avg",
    "Lambda_Home", "Lambda_Away", "Lambda_Total",
    "Line_Home", "Line_Away", "Total_Line",
    "Odd_Over_Total", "Odd_Under_Total",
    "Odd_Over_Home", "Odd_Under_Home",
    "Odd_Over_Away", "Odd_Under_Away",
    "Prob_Over_Total", "Prob_Under_Total", "Edge_Over_Total", "Edge_Under_Total",
    "Prob_Over_Home",  "Prob_Under_Home",  "Edge_Over_Home",  "Edge_Under_Home",
    "Prob_Over_Away",  "Prob_Under_Away",  "Edge_Over_Away",  "Edge_Under_Away",
    "diff", "stake",
    "Real_Fouls_Home", "Real_Fouls_Away", "Real_Fouls_Total",
]

# ─── NORMALIZAÇÃO DE COLUNAS ──────────────────────────────────────────────────

def normalizar_colunas(df):
    rename = {}
    for padrao, alternativas in COL_MAP.items():
        for alt in alternativas:
            if alt in df.columns and padrao not in df.columns:
                rename[alt] = padrao
                break
    return df.rename(columns=rename)

def normalizar_jogos(df):
    """Normaliza CSV de jogos da rodada (pode ter colunas diferentes)."""
    col_map_jogos = {
        "HomeTeam": ["HomeTeam", "homeTeam", "home", "Home", "mandante", "casa"],
        "AwayTeam": ["AwayTeam", "awayTeam", "away", "Away", "visitante", "fora"],
        "Referee":  ["Referee", "arbitro", "referee", "Arbitro"],
        "Rodada":   ["Rodada", "rodada", "round", "Round"],
        "Date":     ["Date", "date", "Data", "data"],
    }
    rename = {}
    for padrao, alternativas in col_map_jogos.items():
        for alt in alternativas:
            if alt in df.columns and padrao not in df.columns:
                rename[alt] = padrao
                break
    return df.rename(columns=rename)

# ─── FUNÇÕES DO MODELO ────────────────────────────────────────────────────────

def weighted_mean(series):
    series = series.dropna()
    n = len(series)
    if n == 0:
        return np.nan
    weights = np.arange(1, n + 1)
    return np.average(series.values, weights=weights)

def home_stats(df, team):
    g = df[df["HomeTeam"] == team].tail(N_JOGOS)
    return weighted_mean(g["HomeFouls"]), weighted_mean(g["AwayFouls"])

def away_stats(df, team):
    g = df[df["AwayTeam"] == team].tail(N_JOGOS)
    return weighted_mean(g["AwayFouls"]), weighted_mean(g["HomeFouls"])

def overall_stats(df, team):
    hg = df[df["HomeTeam"] == team].tail(N_JOGOS)
    ag = df[df["AwayTeam"] == team].tail(N_JOGOS)
    comm = weighted_mean(pd.concat([hg["HomeFouls"], ag["AwayFouls"]]))
    conc = weighted_mean(pd.concat([hg["AwayFouls"], ag["HomeFouls"]]))
    return comm, conc

def team_pace(df, team):
    hg = df[df["HomeTeam"] == team].tail(N_JOGOS)
    ag = df[df["AwayTeam"] == team].tail(N_JOGOS)
    h_tot = hg["HomeFouls"] + hg["AwayFouls"]
    a_tot = ag["HomeFouls"] + ag["AwayFouls"]
    return weighted_mean(pd.concat([h_tot, a_tot]))

def referee_average(df, referee):
    g = df[df["Referee"] == referee].tail(N_JOGOS)
    return weighted_mean(g["TotalFouls"])

def calculate_lambda_components(df, home_team, away_team, referee, referee_avg_override=None):
    league_avg = df["TotalFouls"].mean()

    hc, hnc = home_stats(df, home_team)
    ac, anc = away_stats(df, away_team)
    ht, hnt = overall_stats(df, home_team)
    at, ant = overall_stats(df, away_team)

    home_comm = hc * W_HA + ht * W_TOT
    home_conc = hnc * W_HA + hnt * W_TOT
    away_comm = ac * W_HA + at * W_TOT
    away_conc = anc * W_HA + ant * W_TOT

    def_home = home_conc / (league_avg / 2)
    def_away = away_conc / (league_avg / 2)

    hp = team_pace(df, home_team)
    ap = team_pace(df, away_team)
    pace_factor = ((hp + ap) / 2) / league_avg

    if referee_avg_override is not None:
        ref_avg = referee_avg_override
    else:
        ref_avg = referee_average(df, referee)
        if pd.isna(ref_avg):
            ref_avg = league_avg

    ref_factor = ref_avg / league_avg

    lh = home_comm * def_away * pace_factor * ref_factor
    la = away_comm * def_home * pace_factor * ref_factor

    return {
        "Lambda_Home": lh, "Lambda_Away": la, "Lambda_Total": lh + la,
        "Referee_Avg": ref_avg,
    }

# ─── CALIBRAÇÃO TOTAL ─────────────────────────────────────────────────────────

def calibrar_total(historico, lambda_original):
    """Walk-forward: calibra Lambda_Total com regressão linear."""
    lambdas_t, reais_t = [], []

    for j in range(len(historico)):
        jt     = historico.iloc[j]
        hist_j = historico.iloc[:j]
        if len(hist_j) == 0:
            continue
        h, a, r = jt["HomeTeam"], jt["AwayTeam"], jt["Referee"]

        h_games = hist_j[(hist_j["HomeTeam"] == h) | (hist_j["AwayTeam"] == h)]
        a_games = hist_j[(hist_j["HomeTeam"] == a) | (hist_j["AwayTeam"] == a)]
        if len(h_games) < MIN_JOGOS or len(a_games) < MIN_JOGOS:
            continue

        try:
            comp = calculate_lambda_components(hist_j, h, a, r)
            lt   = comp["Lambda_Total"]
            real = jt["TotalFouls"]
            if pd.notna(lt) and pd.notna(real):
                lambdas_t.append(lt)
                reais_t.append(real)
        except Exception:
            continue

    if len(lambdas_t) >= MIN_CAL:
        cal = LinearRegression().fit(np.array(lambdas_t).reshape(-1, 1), np.array(reais_t))
        lc  = cal.predict([[lambda_original]])[0]
        return round(float(lc), 4)

    return np.nan

# ─── CALIBRAÇÃO INDIVIDUAL ────────────────────────────────────────────────────

def calibrar_individuais(historico, lh_orig, la_orig):
    """Walk-forward: calibra Lambda_Home e Lambda_Away separadamente."""
    lambdas_h, reais_h = [], []
    lambdas_a, reais_a = [], []

    for j in range(len(historico)):
        jt     = historico.iloc[j]
        hist_j = historico.iloc[:j]
        if hist_j.empty:
            continue
        h, a, r = jt["HomeTeam"], jt["AwayTeam"], jt["Referee"]

        h_games = hist_j[(hist_j["HomeTeam"] == h) | (hist_j["AwayTeam"] == h)]
        a_games = hist_j[(hist_j["HomeTeam"] == a) | (hist_j["AwayTeam"] == a)]
        if len(h_games) < MIN_JOGOS or len(a_games) < MIN_JOGOS:
            continue

        try:
            comp    = calculate_lambda_components(hist_j, h, a, r)
            lh_tr   = comp["Lambda_Home"]
            la_tr   = comp["Lambda_Away"]
            real_h  = jt["HomeFouls"]
            real_a  = jt["AwayFouls"]

            if pd.notna(lh_tr) and pd.notna(real_h):
                lambdas_h.append(lh_tr); reais_h.append(real_h)
            if pd.notna(la_tr) and pd.notna(real_a):
                lambdas_a.append(la_tr); reais_a.append(real_a)
        except Exception:
            continue

    lh_cal = np.nan
    la_cal = np.nan

    if len(lambdas_h) >= MIN_CAL:
        m = LinearRegression().fit(np.array(lambdas_h).reshape(-1, 1), np.array(reais_h))
        lh_cal = round(float(m.predict([[lh_orig]])[0]), 4)

    if len(lambdas_a) >= MIN_CAL:
        m = LinearRegression().fit(np.array(lambdas_a).reshape(-1, 1), np.array(reais_a))
        la_cal = round(float(m.predict([[la_orig]])[0]), 4)

    return lh_cal, la_cal

# ─── EDGE POISSON ─────────────────────────────────────────────────────────────

def calcular_edge_poisson(lambda_value, line, odd_over, odd_under):
    k  = int(line)
    pu = poisson.cdf(k, lambda_value)
    po = 1 - pu
    return {
        "prob_over":  po,  "prob_under": pu,
        "fair_over":  1/po if po > 0 else 0,
        "fair_under": 1/pu if pu > 0 else 0,
        "edge_over":  po * odd_over  - 1,
        "edge_under": pu * odd_under - 1,
    }

def stake_por_diff(diff):
    if pd.isna(diff) or diff < 1:   return 0
    elif diff < 2: return 1
    elif diff < 3: return 1
    elif diff < 4: return 1.25
    elif diff < 5: return 1.5
    elif diff < 6: return 1.75
    else:          return 2

# ─── HELPERS UI ───────────────────────────────────────────────────────────────

def badge(val):
    pct = f"{val*100:+.1f}%"
    if val > 0.03:    return f'<span class="badge-pos">{pct}</span>'
    elif val < -0.03: return f'<span class="badge-neg">{pct}</span>'
    else:             return f'<span class="badge-neu">{pct}</span>'

def result_card(titulo, lam, lam_orig, linha, ed):
    if not pd.isna(lam_orig):
        sub_str = f"λ calibrado · original {lam_orig:.2f} · linha {linha}"
    else:
        sub_str = f"λ original · linha {linha}"
    return f"""
    <div class="result-card">
        <div class="result-title">{titulo}</div>
        <div class="lambda-val">{lam:.2f}</div>
        <div class="lambda-sub">{sub_str}</div>
        <div class="edge-row">
            <span class="edge-label">Over</span>{badge(ed['edge_over'])}
            <span class="edge-label">Under</span>{badge(ed['edge_under'])}
        </div>
        <div class="prob-line">
            P(over) {ed['prob_over']*100:.1f}% · P(under) {ed['prob_under']*100:.1f}%
            &nbsp;|&nbsp; fair {ed['fair_over']:.2f} / {ed['fair_under']:.2f}
        </div>
    </div>"""

def stat_box(val, label):
    return f'<div class="stat-box"><div class="val">{val}</div><div class="lbl">{label}</div></div>'

def snap_half(val):
    return round(int(val) + 0.5, 1) if val % 1 != 0.5 else val

# ─── SESSION STATE ────────────────────────────────────────────────────────────

for k, v in [
    ("ligas", {}),
    ("ultimo_calculo", {"mandante": None, "visitante": None, "referee": None}),
    ("confirmacao_pendente", False),
    ("avisos_pendentes", []),
    ("jogos_ocultos", set()),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚽ Faltas BR")
    st.markdown("---")
    st.markdown("### Ligas")

    with st.expander("➕  Adicionar liga", expanded=len(st.session_state.ligas) == 0):
        nome_liga    = st.text_input("Nome da liga", placeholder="Ex: Brasileirão...", key="inp_nome")
        f_faltas_up  = st.file_uploader("CSV histórico de faltas", type=["csv"], key="up_faltas")
        f_jogos_up   = st.file_uploader("CSV jogos da rodada", type=["csv"], key="up_jogos",
                                         help="Colunas: HomeTeam, AwayTeam (+ Referee, Rodada, Date opcionais)")
        f_model_up   = st.file_uploader("CSV model — opcional", type=["csv"], key="up_model")

        if st.button("Adicionar", key="btn_add"):
            if not nome_liga:
                st.warning("Digite um nome.")
            elif not f_faltas_up:
                st.warning("Faça upload do CSV de faltas.")
            elif nome_liga in st.session_state.ligas:
                st.warning("Liga já existe.")
            else:
                df_f = pd.read_csv(f_faltas_up)
                df_f = normalizar_colunas(df_f)
                if "TotalFouls" not in df_f.columns:
                    df_f["TotalFouls"] = df_f["HomeFouls"] + df_f["AwayFouls"]
                if "Rodada" in df_f.columns and "id" in df_f.columns:
                    df_f = df_f.sort_values(["Rodada", "id"]).reset_index(drop=True)
                elif "Rodada" in df_f.columns:
                    df_f = df_f.sort_values("Rodada").reset_index(drop=True)

                df_j = pd.DataFrame()
                if f_jogos_up:
                    df_j = normalizar_jogos(pd.read_csv(f_jogos_up))

                df_h = pd.read_csv(f_model_up) if f_model_up else pd.DataFrame(columns=COLUNAS_MODELO)

                tem_arb_jogos  = "Referee" in df_j.columns if not df_j.empty else False
                tem_arb_faltas = "Referee" in df_f.columns

                st.session_state.ligas[nome_liga] = {
                    "df": df_f,
                    "df_jogos": df_j,
                    "historico": df_h,
                    "historico_original": df_h.copy(),
                    "tem_arb_jogos": tem_arb_jogos,
                    "tem_arb_faltas": tem_arb_faltas,
                }
                st.success(f"'{nome_liga}' adicionada!")
                st.rerun()

    if st.session_state.ligas:
        st.markdown("**Ligas carregadas:**")
        for nome in list(st.session_state.ligas.keys()):
            c1, c2 = st.columns([4, 1])
            with c1:
                n = len(st.session_state.ligas[nome]["historico"])
                st.markdown(f"<span style='font-size:13px'>⚽ {nome}</span> <span style='font-size:11px;color:#6e7681'>({n} reg.)</span>",
                            unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"rm_{nome}"):
                    del st.session_state.ligas[nome]
                    st.rerun()

# ─── SEM LIGAS ────────────────────────────────────────────────────────────────

if not st.session_state.ligas:
    st.markdown('<h1 style="color:#e6edf3;font-size:22px;font-weight:600;">Faltas BR Analyzer</h1>', unsafe_allow_html=True)
    st.info("⬅️  Adicione pelo menos uma liga na sidebar para começar.")
    st.stop()

# ─── SELETOR DE LIGA ──────────────────────────────────────────────────────────

ligas_disponiveis = list(st.session_state.ligas.keys())
st.markdown('<h1 style="color:#e6edf3;font-size:22px;font-weight:600;margin-bottom:6px;">Faltas BR Analyzer</h1>', unsafe_allow_html=True)

col_liga, col_info = st.columns([3, 7])
with col_liga:
    liga_ativa = st.selectbox("Liga ativa", ligas_disponiveis, key="liga_ativa")

lig      = st.session_state.ligas[liga_ativa]
df       = lig["df"]
df_jogos = lig["df_jogos"]
historico = lig["historico"]
times     = sorted(pd.concat([df["HomeTeam"], df["AwayTeam"]]).dropna().unique().tolist())
arbitros  = sorted(df["Referee"].dropna().unique().tolist()) if "Referee" in df.columns else []

with col_info:
    st.markdown(
        f'<p style="color:#6e7681;font-size:13px;margin-top:32px;">'
        f'{len(df)} partidas no histórico &nbsp;·&nbsp; {len(historico)} registros no model'
        f'{f" · {len(df_jogos)} jogos na rodada" if not df_jogos.empty else ""}</p>',
        unsafe_allow_html=True
    )

# ─── ABAS ─────────────────────────────────────────────────────────────────────

aba_analise, aba_apostas, aba_historico = st.tabs(["⚡  Análise", "🎯  Boas Apostas", "📋  Histórico"])

# ══════════════════════════════════════════════════════════════════════════════
# ABA ANÁLISE — FILA DE JOGOS
# ══════════════════════════════════════════════════════════════════════════════
with aba_analise:

    ESTREANTE = "⭐ Árbitro estreante (inserir média manual)"

    # Montar fila de jogos pendentes
    if not df_jogos.empty and "HomeTeam" in df_jogos.columns and "AwayTeam" in df_jogos.columns:
        jogos_calc = set(zip(historico["HomeTeam"].astype(str), historico["AwayTeam"].astype(str))) \
                     if not historico.empty else set()

        fila = []
        for _, row in df_jogos.iterrows():
            home = str(row["HomeTeam"])
            away = str(row["AwayTeam"])
            chave = f"{liga_ativa}|{home}|{away}"
            if (home, away) in jogos_calc:
                continue
            if chave in st.session_state.jogos_ocultos:
                continue
            fila.append(row)

        if fila:
            btns = st.columns([3, 1])
            with btns[1]:
                if st.button("🔄  Restaurar ocultos", key="rest_ocultos"):
                    st.session_state.jogos_ocultos = {
                        k for k in st.session_state.jogos_ocultos
                        if not k.startswith(f"{liga_ativa}|")
                    }
                    st.rerun()

            for idx_f, row_jogo in enumerate(fila):
                home_j = str(row_jogo["HomeTeam"])
                away_j = str(row_jogo["AwayTeam"])
                pfx    = f"{liga_ativa}_{home_j}_{away_j}_{idx_f}".replace(" ", "_")

                # Árbitro pré-preenchido do CSV de jogos se disponível
                arb_default = str(row_jogo["Referee"]) if "Referee" in row_jogo.index and pd.notna(row_jogo.get("Referee")) else None
                rodada_jogo = row_jogo.get("Rodada", 1) if "Rodada" in row_jogo.index else 1
                # Data do CSV de jogos (correta), fallback hoje
                date_j = str(row_jogo.get("Date", "")) if "Date" in row_jogo.index else ""
                data_jogo = date_j[:10] if date_j and date_j.strip() not in ("", "nan") else date.today().strftime("%Y-%m-%d")

                with st.container():
                    st.markdown(f'<div class="jogo-card"><div class="jogo-card-header">🏟️ {home_j} vs {away_j}</div></div>', unsafe_allow_html=True)

                    hdr1, hdr2 = st.columns([8, 1])
                    with hdr2:
                        if st.button("🗑️", key=f"{pfx}_hide", help="Ocultar este jogo"):
                            st.session_state.jogos_ocultos.add(f"{liga_ativa}|{home_j}|{away_j}")
                            st.rerun()

                    # Árbitro
                    arb_cols = st.columns([3, 3, 4])
                    with arb_cols[0]:
                        rodada_sel = st.number_input("Rodada", min_value=1, max_value=50,
                                                      value=int(rodada_jogo) if str(rodada_jogo).isdigit() else 1,
                                                      step=1, key=f"{pfx}_rod")
                    with arb_cols[1]:
                        opcoes_arb = arbitros + [ESTREANTE] if arbitros else [ESTREANTE]
                        idx_default = 0
                        if arb_default and arb_default in opcoes_arb:
                            idx_default = opcoes_arb.index(arb_default)
                        sel_arb = st.selectbox("Árbitro", opcoes_arb, index=idx_default, key=f"{pfx}_arb")

                    referee_manual_avg = None
                    if sel_arb == ESTREANTE:
                        with arb_cols[2]:
                            league_avg_ref = df["TotalFouls"].mean()
                            referee_manual_avg = st.number_input(
                                "Média do árbitro", min_value=0.0,
                                value=round(float(league_avg_ref), 2),
                                step=0.1, format="%.2f", key=f"{pfx}_arb_avg"
                            )
                        referee_nome_inp = st.text_input("Nome (opcional)", placeholder="Ex: João Silva", key=f"{pfx}_arb_nome")
                        referee = referee_nome_inp if referee_nome_inp else "Estreante"
                    else:
                        referee = sel_arb

                    # Linhas e odds
                    st.markdown('<div class="sec">Linhas e Odds</div>', unsafe_allow_html=True)
                    t_jogo, t_casa, t_fora = st.tabs([
                        "🏟️  Total", f"🏠  {home_j[:12]}", f"✈️  {away_j[:12]}"
                    ])

                    with t_jogo:
                        jc1, jc2, jc3 = st.columns(3)
                        with jc1:
                            _lj = st.number_input("Linha total", value=18.5, step=0.5, format="%.1f", key=f"{pfx}_lt")
                            line_jogo = snap_half(_lj)
                            if line_jogo != _lj: st.caption(f"↳ {line_jogo}")
                        with jc2: odd_jogo_over  = st.number_input("Odd Over",  value=1.90, step=0.01, format="%.2f", key=f"{pfx}_oto")
                        with jc3: odd_jogo_under = st.number_input("Odd Under", value=1.83, step=0.01, format="%.2f", key=f"{pfx}_otu")

                    with t_casa:
                        cc1, cc2, cc3 = st.columns(3)
                        with cc1:
                            _lc = st.number_input("Linha casa", value=9.5, step=0.5, format="%.1f", key=f"{pfx}_lh")
                            line_casa = snap_half(_lc)
                            if line_casa != _lc: st.caption(f"↳ {line_casa}")
                        with cc2: odd_over_home  = st.number_input("Odd Over",  value=2.00, step=0.01, format="%.2f", key=f"{pfx}_oho")
                        with cc3: odd_under_home = st.number_input("Odd Under", value=1.75, step=0.01, format="%.2f", key=f"{pfx}_ohu")

                    with t_fora:
                        fc1, fc2, fc3 = st.columns(3)
                        with fc1:
                            _lf = st.number_input("Linha fora", value=9.5, step=0.5, format="%.1f", key=f"{pfx}_la")
                            line_fora = snap_half(_lf)
                            if line_fora != _lf: st.caption(f"↳ {line_fora}")
                        with fc2: odd_over_away  = st.number_input("Odd Over",  value=2.10, step=0.01, format="%.2f", key=f"{pfx}_oao")
                        with fc3: odd_under_away = st.number_input("Odd Under", value=1.67, step=0.01, format="%.2f", key=f"{pfx}_oau")

                    if st.button(f"⚡  Calcular — {home_j} vs {away_j}", type="primary",
                                  use_container_width=True, key=f"{pfx}_calc"):

                        with st.spinner("Calculando calibração walk-forward..."):
                            try:
                                comp = calculate_lambda_components(df, home_j, away_j, referee,
                                                                    referee_avg_override=referee_manual_avg)
                                lh_orig = comp["Lambda_Home"]
                                la_orig = comp["Lambda_Away"]
                                lt_orig = comp["Lambda_Total"]
                                ref_avg = comp["Referee_Avg"]

                                # Calibração total
                                lt_cal = calibrar_total(df, lt_orig)
                                # Calibração individual
                                lh_cal, la_cal = calibrar_individuais(df, lh_orig, la_orig)

                                # Usar calibrado se disponível
                                lt_uso = lt_cal if not pd.isna(lt_cal) else lt_orig
                                lh_uso = lh_cal if not pd.isna(lh_cal) else lh_orig
                                la_uso = la_cal if not pd.isna(la_cal) else la_orig

                                te = calcular_edge_poisson(lt_uso, line_jogo, odd_jogo_over, odd_jogo_under)
                                he = calcular_edge_poisson(lh_uso, line_casa, odd_over_home, odd_under_home)
                                ae = calcular_edge_poisson(la_uso, line_fora, odd_over_away, odd_under_away)

                                diff_total = abs(lt_uso - line_jogo)
                                diff_home  = abs(lh_uso - line_casa)
                                diff_away  = abs(la_uso - line_fora)
                                diff_max   = max(diff_total, diff_home, diff_away)
                                stake      = stake_por_diff(diff_max)

                                # Exibir calibração
                                st.markdown('<div class="sec">Resultado</div>', unsafe_allow_html=True)
                                if not pd.isna(lt_cal):
                                    st.markdown(
                                        f'<div class="cal-box">Total: λ {lt_orig:.2f} → {lt_cal:.2f} · '
                                        f'diff {abs(lt_orig-line_jogo):.2f} → {diff_total:.2f}</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.caption(f"⚠️ Calibração indisponível (mín. {MIN_CAL} jogos). Usando λ original.")

                                r1, r2, r3 = st.columns(3)
                                with r1:
                                    st.markdown(result_card(
                                        f"🏟️ Total · diff {diff_total:.2f} · stake {stake_por_diff(diff_total)}u",
                                        lt_uso, lt_orig if not pd.isna(lt_cal) else np.nan, line_jogo, te
                                    ), unsafe_allow_html=True)
                                with r2:
                                    st.markdown(result_card(
                                        f"🏠 {home_j[:14]} · diff {diff_home:.2f}",
                                        lh_uso, lh_orig if not pd.isna(lh_cal) else np.nan, line_casa, he
                                    ), unsafe_allow_html=True)
                                with r3:
                                    st.markdown(result_card(
                                        f"✈️ {away_j[:14]} · diff {diff_away:.2f}",
                                        la_uso, la_orig if not pd.isna(la_cal) else np.nan, line_fora, ae
                                    ), unsafe_allow_html=True)

                                # Montar registro com data correta do CSV de jogos
                                registro = {
                                    "Date":       data_jogo,
                                    "Rodada":     rodada_sel,
                                    "HomeTeam":   home_j,
                                    "AwayTeam":   away_j,
                                    "Referee":    referee,
                                    "Arbitro":    referee,
                                    "Referee_Avg": round(float(ref_avg), 4),
                                    "Lambda_Home":  round(float(lh_uso), 4),
                                    "Lambda_Away":  round(float(la_uso), 4),
                                    "Lambda_Total": round(float(lt_uso), 4),
                                    "Line_Home":    line_casa,
                                    "Line_Away":    line_fora,
                                    "Total_Line":   line_jogo,
                                    "Odd_Over_Total":  odd_jogo_over,   "Odd_Under_Total": odd_jogo_under,
                                    "Odd_Over_Home":   odd_over_home,   "Odd_Under_Home":  odd_under_home,
                                    "Odd_Over_Away":   odd_over_away,   "Odd_Under_Away":  odd_under_away,
                                    "Prob_Over_Total":  round(te["prob_over"], 4), "Prob_Under_Total": round(te["prob_under"], 4),
                                    "Edge_Over_Total":  round(te["edge_over"], 4), "Edge_Under_Total": round(te["edge_under"], 4),
                                    "Prob_Over_Home":   round(he["prob_over"], 4), "Prob_Under_Home":  round(he["prob_under"], 4),
                                    "Edge_Over_Home":   round(he["edge_over"], 4), "Edge_Under_Home":  round(he["edge_under"], 4),
                                    "Prob_Over_Away":   round(ae["prob_over"], 4), "Prob_Under_Away":  round(ae["prob_under"], 4),
                                    "Edge_Over_Away":   round(ae["edge_over"], 4), "Edge_Under_Away":  round(ae["edge_under"], 4),
                                    "diff":   round(diff_max, 4),
                                    "stake":  stake,
                                    "Real_Fouls_Home": np.nan, "Real_Fouls_Away": np.nan, "Real_Fouls_Total": np.nan,
                                }

                                novo_hist = pd.concat(
                                    [st.session_state.ligas[liga_ativa]["historico"], pd.DataFrame([registro])],
                                    ignore_index=True
                                )
                                st.session_state.ligas[liga_ativa]["historico"] = novo_hist
                                st.session_state.ultimo_calculo = {"mandante": home_j, "visitante": away_j, "referee": referee}
                                st.success(f"✓  {home_j} vs {away_j} salvo!")

                            except Exception as e:
                                st.error(f"Erro: {e}")

                    st.markdown("---")

        else:
            st.info("✅ Todos os jogos da rodada já foram calculados ou ocultos.")

    else:
        # Modo manual (sem CSV de jogos)
        st.markdown('<div class="sec">Times & Árbitro</div>', unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            mandante  = st.selectbox("Mandante (Casa)", times, key="mandante")
        with mc2:
            visitante = st.selectbox("Visitante (Fora)", [t for t in times if t != mandante], key="visitante")
        with mc3:
            opcoes_arbitro = arbitros + [ESTREANTE] if arbitros else [ESTREANTE]
            sel_referee = st.selectbox("Árbitro", opcoes_arbitro, key="referee_sel")

        referee_manual_avg = None
        if sel_referee == ESTREANTE:
            ec1, ec2 = st.columns([2, 2])
            with ec1:
                referee_nome = st.text_input("Nome do árbitro (opcional)", key="referee_nome")
            with ec2:
                league_avg_ref = df["TotalFouls"].mean()
                referee_manual_avg = st.number_input("Média do árbitro", min_value=0.0,
                    value=round(float(league_avg_ref), 2), step=0.1, format="%.2f", key="referee_avg_manual")
            referee = referee_nome if referee_nome else "Estreante"
        else:
            referee = sel_referee

        rodada_m = st.number_input("Rodada", min_value=1, max_value=50, value=1, step=1, key="rodada_manual")

        st.markdown('<div class="sec">Linhas e Odds</div>', unsafe_allow_html=True)
        t_jogo, t_casa, t_fora = st.tabs(["🏟️  Total", f"🏠  {mandante}", f"✈️  {visitante}"])
        with t_jogo:
            j1, j2, j3 = st.columns(3)
            with j1:
                _lj = st.number_input("Linha total", value=18.5, step=0.5, format="%.1f", key="lj_m")
                line_jogo = snap_half(_lj)
            with j2: odd_jogo_over  = st.number_input("Odd Over",  value=1.90, step=0.01, format="%.2f", key="ojo_m")
            with j3: odd_jogo_under = st.number_input("Odd Under", value=1.83, step=0.01, format="%.2f", key="oju_m")
        with t_casa:
            c1, c2, c3 = st.columns(3)
            with c1:
                _lc = st.number_input("Linha casa", value=9.5, step=0.5, format="%.1f", key="lc_m")
                line_casa = snap_half(_lc)
            with c2: odd_over_home  = st.number_input("Odd Over",  value=2.00, step=0.01, format="%.2f", key="oho_m")
            with c3: odd_under_home = st.number_input("Odd Under", value=1.75, step=0.01, format="%.2f", key="ohu_m")
        with t_fora:
            f1, f2, f3 = st.columns(3)
            with f1:
                _lf = st.number_input("Linha fora", value=9.5, step=0.5, format="%.1f", key="lf_m")
                line_fora = snap_half(_lf)
            with f2: odd_over_away  = st.number_input("Odd Over",  value=2.10, step=0.01, format="%.2f", key="oao_m")
            with f3: odd_under_away = st.number_input("Odd Under", value=1.67, step=0.01, format="%.2f", key="oau_m")

        ult = st.session_state.ultimo_calculo

        if st.button("⚡  Calcular Edge", type="primary", use_container_width=True, key="calc_manual"):
            avisos = []
            if ult["mandante"] is not None:
                if ult["mandante"] == mandante and ult["visitante"] == visitante and ult["referee"] == referee:
                    avisos.append("⚠️ **Times e árbitro** são os mesmos do último cálculo.")
                elif ult["mandante"] == mandante and ult["visitante"] == visitante:
                    avisos.append(f"⚠️ **Times iguais** ao último cálculo. Árbitro: {referee}.")
                elif ult["referee"] == referee:
                    avisos.append(f"⚠️ **Árbitro igual** ao último cálculo ({referee}). Confirme para {mandante} vs {visitante}.")

            if avisos:
                st.session_state.confirmacao_pendente = True
                st.session_state.avisos_pendentes = avisos
                st.rerun()
            else:
                st.session_state.confirmacao_pendente = False
                with st.spinner("Calculando..."):
                    try:
                        comp = calculate_lambda_components(df, mandante, visitante, referee,
                                                            referee_avg_override=referee_manual_avg)
                        lh_orig = comp["Lambda_Home"]; la_orig = comp["Lambda_Away"]; lt_orig = comp["Lambda_Total"]
                        ref_avg = comp["Referee_Avg"]
                        lt_cal = calibrar_total(df, lt_orig)
                        lh_cal, la_cal = calibrar_individuais(df, lh_orig, la_orig)

                        lt_uso = lt_cal if not pd.isna(lt_cal) else lt_orig
                        lh_uso = lh_cal if not pd.isna(lh_cal) else lh_orig
                        la_uso = la_cal if not pd.isna(la_cal) else la_orig

                        te = calcular_edge_poisson(lt_uso, line_jogo, odd_jogo_over, odd_jogo_under)
                        he = calcular_edge_poisson(lh_uso, line_casa, odd_over_home, odd_under_home)
                        ae = calcular_edge_poisson(la_uso, line_fora, odd_over_away, odd_under_away)

                        diff_total = abs(lt_uso - line_jogo)
                        diff_home  = abs(lh_uso - line_casa)
                        diff_away  = abs(la_uso - line_fora)
                        diff_max   = max(diff_total, diff_home, diff_away)
                        stake      = stake_por_diff(diff_max)

                        st.markdown('<div class="sec">Resultado</div>', unsafe_allow_html=True)
                        if not pd.isna(lt_cal):
                            st.markdown(f'<div class="cal-box">Total: λ {lt_orig:.2f} → {lt_cal:.2f} · diff {abs(lt_orig-line_jogo):.2f} → {diff_total:.2f}</div>',
                                        unsafe_allow_html=True)
                        else:
                            st.caption(f"⚠️ Calibração indisponível (mín. {MIN_CAL} jogos). Usando λ original.")

                        r1, r2, r3 = st.columns(3)
                        with r1: st.markdown(result_card(f"🏟️ Total · diff {diff_total:.2f} · stake {stake_por_diff(diff_total)}u",
                                             lt_uso, lt_orig if not pd.isna(lt_cal) else np.nan, line_jogo, te), unsafe_allow_html=True)
                        with r2: st.markdown(result_card(f"🏠 {mandante[:14]} · diff {diff_home:.2f}",
                                             lh_uso, lh_orig if not pd.isna(lh_cal) else np.nan, line_casa, he), unsafe_allow_html=True)
                        with r3: st.markdown(result_card(f"✈️ {visitante[:14]} · diff {diff_away:.2f}",
                                             la_uso, la_orig if not pd.isna(la_cal) else np.nan, line_fora, ae), unsafe_allow_html=True)

                        # Modo manual: data de hoje (sem CSV de jogos para extrair a data real)
                        registro = {
                            "Date": datetime.now().strftime("%Y-%m-%d"),
                            "Rodada": rodada_m, "HomeTeam": mandante, "AwayTeam": visitante,
                            "Referee": referee, "Arbitro": referee, "Referee_Avg": round(float(ref_avg), 4),
                            "Lambda_Home": round(float(lh_uso), 4), "Lambda_Away": round(float(la_uso), 4), "Lambda_Total": round(float(lt_uso), 4),
                            "Line_Home": line_casa, "Line_Away": line_fora, "Total_Line": line_jogo,
                            "Odd_Over_Total": odd_jogo_over, "Odd_Under_Total": odd_jogo_under,
                            "Odd_Over_Home": odd_over_home, "Odd_Under_Home": odd_under_home,
                            "Odd_Over_Away": odd_over_away, "Odd_Under_Away": odd_under_away,
                            "Prob_Over_Total": round(te["prob_over"], 4), "Prob_Under_Total": round(te["prob_under"], 4),
                            "Edge_Over_Total": round(te["edge_over"], 4), "Edge_Under_Total": round(te["edge_under"], 4),
                            "Prob_Over_Home": round(he["prob_over"], 4),  "Prob_Under_Home":  round(he["prob_under"], 4),
                            "Edge_Over_Home": round(he["edge_over"], 4),  "Edge_Under_Home":  round(he["edge_under"], 4),
                            "Prob_Over_Away": round(ae["prob_over"], 4),  "Prob_Under_Away":  round(ae["prob_under"], 4),
                            "Edge_Over_Away": round(ae["edge_over"], 4),  "Edge_Under_Away":  round(ae["edge_under"], 4),
                            "diff": round(diff_max, 4), "stake": stake,
                            "Real_Fouls_Home": np.nan, "Real_Fouls_Away": np.nan, "Real_Fouls_Total": np.nan,
                        }
                        novo_hist = pd.concat([st.session_state.ligas[liga_ativa]["historico"], pd.DataFrame([registro])], ignore_index=True)
                        st.session_state.ligas[liga_ativa]["historico"] = novo_hist
                        st.session_state.ultimo_calculo = {"mandante": mandante, "visitante": visitante, "referee": referee}
                        st.success(f"✓  {mandante} vs {visitante} salvo em **{liga_ativa}**.")
                    except Exception as e:
                        st.error(f"Erro: {e}")

        if st.session_state.confirmacao_pendente:
            for av in st.session_state.avisos_pendentes:
                st.warning(av)
            cs, cn = st.columns(2)
            with cs:
                if st.button("✅  Sim, calcular", use_container_width=True, key="confirm_sim"):
                    st.session_state.confirmacao_pendente = False
                    st.session_state.avisos_pendentes = []
                    st.rerun()
            with cn:
                if st.button("❌  Cancelar", use_container_width=True, key="confirm_nao"):
                    st.session_state.confirmacao_pendente = False
                    st.session_state.avisos_pendentes = []
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ABA BOAS APOSTAS
# ══════════════════════════════════════════════════════════════════════════════
with aba_apostas:

    hist = st.session_state.ligas[liga_ativa]["historico"]

    if hist.empty or "diff" not in hist.columns:
        st.info("Calcule pelo menos um jogo na aba Análise para ver as apostas.")
    else:
        # Seletor de rodada
        rodadas = sorted(hist["Rodada"].dropna().unique().tolist(), key=lambda x: str(x), reverse=True)
        rod_sel = st.selectbox("Rodada", rodadas, key="rod_apostas") if rodadas else None

        df_rod = hist[hist["Rodada"] == rod_sel].copy() if rod_sel is not None else hist.copy()

        # ── Diff ≥ 2 ──────────────────────────────────────────────────────────
        mercados_2 = []
        for _, row in df_rod.iterrows():
            for mercado, linha_col, odd_o_col, odd_u_col, prob_o_col, edge_o_col, edge_u_col, lam_col in [
                ("Total",  "Total_Line", "Odd_Over_Total", "Odd_Under_Total", "Prob_Over_Total", "Edge_Over_Total", "Edge_Under_Total", "Lambda_Total"),
                ("Casa",   "Line_Home",  "Odd_Over_Home",  "Odd_Under_Home",  "Prob_Over_Home",  "Edge_Over_Home",  "Edge_Under_Home",  "Lambda_Home"),
                ("Fora",   "Line_Away",  "Odd_Over_Away",  "Odd_Under_Away",  "Prob_Over_Away",  "Edge_Over_Away",  "Edge_Under_Away",  "Lambda_Away"),
            ]:
                if linha_col not in row.index or lam_col not in row.index:
                    continue
                lam   = row[lam_col]
                linha = row[linha_col]
                if pd.isna(lam) or pd.isna(linha):
                    continue
                diff_merc = abs(lam - linha)
                if diff_merc < 2:
                    continue
                lado       = "Over" if lam > linha else "Under"
                odd_lado   = row[odd_o_col] if lado == "Over" else row[odd_u_col]
                edge_lado  = row[edge_o_col] if lado == "Over" else row[edge_u_col]
                stake_merc = stake_por_diff(diff_merc)
                mercados_2.append({
                    "jogo": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                    "mercado": mercado, "lado": lado, "linha": linha,
                    "odd": odd_lado, "edge": edge_lado, "lam": lam,
                    "diff": diff_merc, "stake": stake_merc,
                })

        if mercados_2:
            st.markdown(f'<div class="sec">Bets com diff ≥ 2 — Rodada {rod_sel}</div>', unsafe_allow_html=True)
            for b in mercados_2:
                st.markdown(f"""
                <div class="stake-card">
                    <div class="stake-header">⚽ {b['mercado']} — diff ≥ 2</div>
                    <div class="stake-match">{b['jogo']}</div>
                    <div class="stake-detail">
                        {b['lado']} {b['linha']} &nbsp;·&nbsp; odd {b['odd']:.2f}
                        &nbsp;·&nbsp; {badge(b['edge'])}
                        &nbsp;·&nbsp; λ {b['lam']:.2f} vs linha {b['linha']} &nbsp;·&nbsp; diff {b['diff']:.2f}
                    </div>
                    <div style="margin-top:8px;">
                        <span style="font-size:11px;color:#6e7681;text-transform:uppercase;letter-spacing:.08em;">Stake &nbsp;</span>
                        <span class="stake-val">{b['stake']}u</span>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info(f"Nenhuma bet com diff ≥ 2 na rodada {rod_sel}.")

        st.markdown("")

        # ── Diff ≥ 1 (1u) ─────────────────────────────────────────────────────
        mercados_1 = []
        for _, row in df_rod.iterrows():
            for mercado, linha_col, odd_o_col, odd_u_col, edge_o_col, edge_u_col, lam_col in [
                ("Total",  "Total_Line", "Odd_Over_Total", "Odd_Under_Total", "Edge_Over_Total", "Edge_Under_Total", "Lambda_Total"),
                ("Casa",   "Line_Home",  "Odd_Over_Home",  "Odd_Under_Home",  "Edge_Over_Home",  "Edge_Under_Home",  "Lambda_Home"),
                ("Fora",   "Line_Away",  "Odd_Over_Away",  "Odd_Under_Away",  "Edge_Over_Away",  "Edge_Under_Away",  "Lambda_Away"),
            ]:
                if linha_col not in row.index or lam_col not in row.index:
                    continue
                lam   = row[lam_col]
                linha = row[linha_col]
                if pd.isna(lam) or pd.isna(linha):
                    continue
                diff_merc = abs(lam - linha)
                if diff_merc < 1 or diff_merc >= 2:
                    continue
                lado      = "Over" if lam > linha else "Under"
                odd_lado  = row[odd_o_col] if lado == "Over" else row[odd_u_col]
                edge_lado = row[edge_o_col] if lado == "Over" else row[edge_u_col]
                mercados_1.append({
                    "jogo": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                    "mercado": mercado, "lado": lado, "linha": linha,
                    "odd": odd_lado, "edge": edge_lado, "lam": lam, "diff": diff_merc,
                })

        if mercados_1:
            st.markdown(f'<div class="sec">Bets com diff ≥ 1 (1u) — Rodada {rod_sel}</div>', unsafe_allow_html=True)
            for b in mercados_1:
                st.markdown(f"""
                <div class="stake-card" style="border-color:#388bfd;">
                    <div class="stake-header" style="color:#79c0ff;">📌 {b['mercado']} — 1u</div>
                    <div class="stake-match">{b['jogo']}</div>
                    <div class="stake-detail">
                        {b['lado']} {b['linha']} &nbsp;·&nbsp; odd {b['odd']:.2f}
                        &nbsp;·&nbsp; {badge(b['edge'])}
                        &nbsp;·&nbsp; λ {b['lam']:.2f} vs linha {b['linha']} &nbsp;·&nbsp; diff {b['diff']:.2f}
                    </div>
                    <div style="margin-top:8px;">
                        <span style="font-size:11px;color:#6e7681;text-transform:uppercase;letter-spacing:.08em;">Stake &nbsp;</span>
                        <span style="font-family:'DM Mono',monospace;font-size:22px;font-weight:500;color:#79c0ff;">1u</span>
                    </div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABA HISTÓRICO
# ══════════════════════════════════════════════════════════════════════════════
with aba_historico:

    hist = st.session_state.ligas[liga_ativa]["historico"]

    if hist.empty:
        st.info("Nenhum registro ainda para esta liga.")
    else:
        st.markdown(f'<div class="sec">Histórico — {liga_ativa} ({len(hist)} registros)</div>', unsafe_allow_html=True)

        cols_view = [
            "Date","Rodada","HomeTeam","AwayTeam","Referee",
            "Lambda_Home","Lambda_Away","Lambda_Total",
            "diff","stake",
            "Total_Line","Odd_Over_Total","Odd_Under_Total",
            "Line_Home","Odd_Over_Home","Odd_Under_Home",
            "Line_Away","Odd_Over_Away","Odd_Under_Away",
            "Real_Fouls_Home","Real_Fouls_Away","Real_Fouls_Total",
        ]
        cols_view = [c for c in cols_view if c in hist.columns]
        st.dataframe(hist[cols_view].round(4), use_container_width=True, hide_index=True)

        with st.expander("✏️  Preencher resultados reais"):
            idx_opcoes = [f"{i} · {r['HomeTeam']} vs {r['AwayTeam']} ({r.get('Date','')})" for i, r in hist.iterrows()]
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

        # Rodada para nome do arquivo
        rodadas_hist = sorted(hist["Rodada"].dropna().unique().tolist(), key=lambda x: str(x), reverse=True) if "Rodada" in hist.columns else []
        rod_export   = rodadas_hist[0] if rodadas_hist else "X"

        with a1:
            export_df  = hist.reindex(columns=[c for c in COLUNAS_MODELO if c in hist.columns])
            csv_bytes  = export_df.to_csv(index=False).encode("utf-8")
            nome_csv   = f"model_{liga_ativa.lower().replace(' ','_')}FALTAS.csv"
            st.download_button(
                f"⬇️  Exportar {liga_ativa}",
                data=csv_bytes, file_name=nome_csv, mime="text/csv",
                use_container_width=True
            )

        with a2:
            if _DRIVE_OK and drive_disponivel():
                if st.button("☁️  Salvar no Drive", use_container_width=True, key="btn_drive"):
                    nome_drive = f"model_{liga_ativa.lower().replace(' ','_')}FALTAS.csv"
                    ok = salvar_csv_drive(export_df, nome_drive, "faltas_br")
                    if ok:
                        st.success("✓ Salvo no Drive!")
                    else:
                        st.error("Erro ao salvar no Drive.")
            else:
                if st.button("🗑️  Apagar último jogo", use_container_width=True):
                    if len(hist) > 0:
                        st.session_state.ligas[liga_ativa]["historico"] = hist.iloc[:-1].reset_index(drop=True)
                        st.success("Último jogo removido.")
                        st.rerun()

        with a3:
            if _DRIVE_OK and drive_disponivel():
                if st.button("🗑️  Apagar último jogo", use_container_width=True, key="btn_del"):
                    if len(hist) > 0:
                        st.session_state.ligas[liga_ativa]["historico"] = hist.iloc[:-1].reset_index(drop=True)
                        st.success("Último jogo removido.")
                        st.rerun()
            else:
                if st.button("🗑️  Limpar sessão", use_container_width=True):
                    original = st.session_state.ligas[liga_ativa].get("historico_original")
                    st.session_state.ligas[liga_ativa]["historico"] = original.copy() if original is not None else pd.DataFrame(columns=COLUNAS_MODELO)
                    st.success("Sessão limpa. Registros originais mantidos.")
                    st.rerun()

        if _DRIVE_OK and drive_disponivel():
            if st.button("🗑️  Limpar sessão", use_container_width=True, key="btn_limpar"):
                original = st.session_state.ligas[liga_ativa].get("historico_original")
                st.session_state.ligas[liga_ativa]["historico"] = original.copy() if original is not None else pd.DataFrame(columns=COLUNAS_MODELO)
                st.success("Sessão limpa. Registros originais mantidos.")
                st.rerun()
