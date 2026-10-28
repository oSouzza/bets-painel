import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import nbinom
from datetime import date

st.set_page_config(page_title="Chutes Analyzer", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

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
.stat-box { background:#0d1117; border:1px solid #1c2128; border-radius:8px; padding:.8rem 1rem; text-align:center; }
.stat-box .val { font-family:'DM Mono',monospace; font-size:20px; color:#e6edf3; }
.stat-box .lbl { font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:#6e7681; margin-top:2px; }
.sec { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.12em; color:#6e7681; border-bottom:1px solid #1c2128; padding-bottom:6px; margin:1.2rem 0 .8rem; }
.input-label { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.1em; color:#6e7681; margin-bottom:3px; }
.warn-box { background:#1c1407; border:1px solid #9e6a03; border-radius:8px; padding:.8rem 1rem; font-size:13px; color:#d29922; margin-bottom:.8rem; }
</style>
""", unsafe_allow_html=True)

# ─── FUNÇÕES DO MODELO ────────────────────────────────────────────────────────

def weighted_mean(series):
    series = series.dropna()
    if len(series) == 0: return np.nan
    weights = np.arange(1, len(series) + 1)
    return np.average(series.values, weights=weights)

def away_attack(df, team):
    return weighted_mean(df[df["AwayTeam"] == team].tail(10)["AwayShots"])

def overall_pace(df, team):
    hg = df[df["HomeTeam"] == team].tail(10)
    ag = df[df["AwayTeam"] == team].tail(10)
    return weighted_mean(pd.concat([hg["HomeShots"] + hg["AwayShots"], ag["HomeShots"] + ag["AwayShots"]]))

def idx_att_home_5(df, team):
    games = df[df["HomeTeam"] == team].tail(5)
    if len(games) == 0: return np.nan
    ml = df["HomeShots"].mean()
    if pd.isna(ml) or ml == 0: return np.nan
    return games["HomeShots"].mean() / ml

def criar_variaveis_modelo(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values("Date").reset_index(drop=True)
    ih_list, aa_list, pace_list = [], [], []
    for pos in range(len(df)):
        hist = df.iloc[:pos]
        home, away = df.iloc[pos]["HomeTeam"], df.iloc[pos]["AwayTeam"]
        ih_list.append(idx_att_home_5(hist, home))
        aa_list.append(away_attack(hist, away))
        ph, pa = overall_pace(hist, home), overall_pace(hist, away)
        pace_list.append((ph + pa) / 2 if not (pd.isna(ph) or pd.isna(pa)) else np.nan)
    df["Idx_Att_Home_5"] = ih_list
    df["Atk_Away"]       = aa_list
    df["Pace"]           = pace_list
    return df

def remover_juice(oh, od, oa):
    ih, id_, ia = 1/oh, 1/od, 1/oa
    s = ih + id_ + ia
    return ih/s, id_/s, ia/s

def treinar_modelo_home(df):
    d = df.dropna(subset=["Idx_Att_Home_5","ProbHome","Pace","HomeShots"]).copy()
    X = sm.add_constant(d[["Idx_Att_Home_5","ProbHome","Pace"]], has_constant="add")
    return sm.NegativeBinomial(d["HomeShots"], X).fit(disp=False)

def treinar_modelo_away(df):
    d = df.dropna(subset=["Atk_Away","ProbAway","Pace","AwayShots"]).copy()
    X = sm.add_constant(d[["Atk_Away","ProbAway","Pace"]], has_constant="add")
    return sm.NegativeBinomial(d["AwayShots"], X).fit(disp=False)

def distribuicao_nb(lam, alpha, max_shots=70):
    r = 1 / alpha; p = r / (r + lam)
    return np.arange(max_shots + 1), nbinom.pmf(np.arange(max_shots + 1), r, p)

def probabilidades_linha(dist, linha):
    lim = int(np.floor(linha)); v = np.arange(len(dist))
    return dist[v > lim].sum(), dist[v <= lim].sum()

def odd_justa(prob):
    return np.nan if (pd.isna(prob) or prob <= 0) else 1 / prob

def calcular_edge(om, oj):
    return np.nan if (pd.isna(om) or pd.isna(oj)) else (om / oj) - 1

def calcular_lambdas(df, rh, ra, home, away, ph, pa):
    ih   = idx_att_home_5(df, home)
    aa   = away_attack(df, away)
    pph  = overall_pace(df, home)
    ppa  = overall_pace(df, away)
    pace = (pph + ppa) / 2 if not (pd.isna(pph) or pd.isna(ppa)) else np.nan
    Xh   = pd.DataFrame([{"const":1,"Idx_Att_Home_5":ih,"ProbHome":ph,"Pace":pace}])
    Xa   = pd.DataFrame([{"const":1,"Atk_Away":aa,"ProbAway":pa,"Pace":pace}])
    lh   = rh.predict(Xh).iloc[0]; la = ra.predict(Xa).iloc[0]
    return {"Idx_Att_Home_5":ih,"Atk_Away":aa,"Pace":pace,
            "Lambda_Home":lh,"Lambda_Away":la,"Lambda_Total":lh+la}

def calcular_mercado(lam, alpha, linha, odd_over, odd_under):
    _, dist = distribuicao_nb(lam, alpha)
    po, pu  = probabilidades_linha(dist, linha)
    ojo, oju = odd_justa(po), odd_justa(pu)
    return {"Prob_Over":po,"Prob_Under":pu,"OddJusta_Over":ojo,"OddJusta_Under":oju,
            "Edge_Over":calcular_edge(odd_over,ojo),"Edge_Under":calcular_edge(odd_under,oju)}

COLUNAS_MODELO = [
    "Date","HomeTeam","AwayTeam","HomeOdds","DrawOdds","AwayOdds","ProbHome","ProbDraw","ProbAway",
    "Idx_Att_Home_5","Atk_Away","Pace","Lambda_Home","Lambda_Away","Lambda_Total",
    "Line_Home","Odd_Home_Over","Odd_Home_Under",
    "Prob_Home_Over","Prob_Home_Under","OddJusta_Home_Over","OddJusta_Home_Under","Edge_Home_Over","Edge_Home_Under",
    "Line_Away","Odd_Away_Over","Odd_Away_Under",
    "Prob_Away_Over","Prob_Away_Under","OddJusta_Away_Over","OddJusta_Away_Under","Edge_Away_Over","Edge_Away_Under",
    "Line_Total","Odd_Total_Over","Odd_Total_Under",
    "Prob_Total_Over","Prob_Total_Under","OddJusta_Total_Over","OddJusta_Total_Under","Edge_Total_Over","Edge_Total_Under",
    "Real_HomeShots","Real_AwayShots","Real_TotalShots"
]

def snap_half(val):
    return round(int(val) + 0.5, 1) if val % 1 != 0.5 else val

def badge(val):
    if pd.isna(val): return "—"
    pct = f"{val*100:+.1f}%"
    if val > 0.03: return f'<span class="badge-pos">{pct}</span>'
    elif val < -0.03: return f'<span class="badge-neg">{pct}</span>'
    else: return f'<span class="badge-neu">{pct}</span>'

def result_card(titulo, lam, linha, merc):
    return f"""<div class="result-card">
        <div class="result-title">{titulo}</div>
        <div class="lambda-val">{lam:.2f}</div>
        <div class="lambda-sub">λ esperado · linha {linha}</div>
        <div class="edge-row"><span class="edge-label">Over</span>{badge(merc['Edge_Over'])}<span class="edge-label">Under</span>{badge(merc['Edge_Under'])}</div>
        <div class="prob-line">P(over) {merc['Prob_Over']*100:.1f}% · P(under) {merc['Prob_Under']*100:.1f}% &nbsp;|&nbsp; fair {merc['OddJusta_Over']:.2f} / {merc['OddJusta_Under']:.2f}</div>
    </div>"""

def stat_box(val, label):
    return f'<div class="stat-box"><div class="val">{val}</div><div class="lbl">{label}</div></div>'

# ─── SESSION STATE ────────────────────────────────────────────────────────────

if "ligas" not in st.session_state:
    st.session_state.ligas = {}
if "jogos_ocultos" not in st.session_state:
    st.session_state.jogos_ocultos = set()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎯 Chutes Analyzer")
    st.markdown("---")
    st.markdown("### Ligas")

    with st.expander("➕  Adicionar liga", expanded=len(st.session_state.ligas) == 0):
        nome_liga   = st.text_input("Nome da liga", placeholder="Ex: Alemanha...", key="inp_nome")
        f_hist_up   = st.file_uploader("CSV histórico de chutes", type=["csv"], key="up_hist",
                                        help="Colunas: Date, HomeTeam, AwayTeam, HomeShots, AwayShots, HomeOdds, DrawOdds, AwayOdds")
        f_jogos_up  = st.file_uploader("CSV de jogos da rodada", type=["csv"], key="up_jogos",
                                        help="Colunas: HomeTeam, AwayTeam (opcional: rodada, date)")
        f_modelo_up = st.file_uploader("CSV model — opcional", type=["csv"], key="up_modelo")

        if st.button("Adicionar", key="btn_add"):
            if not nome_liga:
                st.warning("Digite um nome.")
            elif not f_hist_up:
                st.warning("Faça upload do CSV histórico.")
            elif not f_jogos_up:
                st.warning("Faça upload do CSV de jogos.")
            elif nome_liga in st.session_state.ligas:
                st.warning("Liga já existe.")
            else:
                with st.spinner("Treinando modelos..."):
                    try:
                        df_h = pd.read_csv(f_hist_up)
                        if "ProbHome" not in df_h.columns:
                            probs = df_h.apply(lambda r: remover_juice(r["HomeOdds"], r["DrawOdds"], r["AwayOdds"]), axis=1)
                            df_h["ProbHome"] = probs.apply(lambda x: x[0])
                            df_h["ProbDraw"] = probs.apply(lambda x: x[1])
                            df_h["ProbAway"] = probs.apply(lambda x: x[2])
                        df_h = criar_variaveis_modelo(df_h)
                        res_home = treinar_modelo_home(df_h)
                        res_away = treinar_modelo_away(df_h)

                        df_j = pd.read_csv(f_jogos_up)
                        col_map = {"homeTeam":"HomeTeam","awayTeam":"AwayTeam","Rodada":"rodada"}
                        df_j = df_j.rename(columns=col_map)

                        df_modelo_init = pd.read_csv(f_modelo_up) if f_modelo_up else pd.DataFrame(columns=COLUNAS_MODELO)

                        st.session_state.ligas[nome_liga] = {
                            "df": df_h, "df_jogos": df_j,
                            "res_home": res_home, "res_away": res_away,
                            "historico": df_modelo_init,
                            "historico_original": df_modelo_init.copy(),
                        }
                        st.success(f"'{nome_liga}' adicionada!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao treinar: {e}")

    if st.session_state.ligas:
        st.markdown("**Ligas carregadas:**")
        for nome in list(st.session_state.ligas.keys()):
            c1, c2 = st.columns([4, 1])
            with c1:
                n = len(st.session_state.ligas[nome]["historico"])
                st.markdown(f"<span style='font-size:13px'>🎯 {nome}</span> <span style='font-size:11px;color:#6e7681'>({n} reg.)</span>", unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"rm_{nome}"):
                    del st.session_state.ligas[nome]
                    st.rerun()

# ─── SEM LIGAS ────────────────────────────────────────────────────────────────

if not st.session_state.ligas:
    st.markdown('<h1 style="color:#e6edf3;font-size:22px;font-weight:600;">Chutes Analyzer</h1>', unsafe_allow_html=True)
    st.info("⬅️  Adicione pelo menos uma liga na sidebar para começar.")
    st.stop()

# ─── SELETOR DE LIGA ──────────────────────────────────────────────────────────

st.markdown('<h1 style="color:#e6edf3;font-size:22px;font-weight:600;margin-bottom:6px;">Chutes Analyzer</h1>', unsafe_allow_html=True)

col_liga, col_info = st.columns([3, 7])
with col_liga:
    liga_ativa = st.selectbox("Liga ativa", list(st.session_state.ligas.keys()), key="liga_ativa")

liga_data = st.session_state.ligas[liga_ativa]
df        = liga_data["df"]
df_jogos  = liga_data["df_jogos"]
res_home  = liga_data["res_home"]
res_away  = liga_data["res_away"]
historico = liga_data["historico"]

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

    # Jogos já calculados
    if not historico.empty and "HomeTeam" in historico.columns:
        jogos_calculados = set(zip(historico["HomeTeam"].astype(str), historico["AwayTeam"].astype(str)))
    else:
        jogos_calculados = set()

    # Fila filtrada
    fila = []
    for _, row in df_jogos.iterrows():
        home = str(row["HomeTeam"]); away = str(row["AwayTeam"])
        rodada_j = str(row.get("rodada", ""))
        chave = f"{liga_ativa}|{home}|{away}|{rodada_j}"
        if (home, away) in jogos_calculados: continue
        if chave in st.session_state.jogos_ocultos: continue
        fila.append(row)

    col_fila, col_rest = st.columns([6, 4])
    with col_fila:
        st.markdown(f'<div class="sec">Fila de jogos — {len(fila)} pendentes · {len(jogos_calculados)} calculados</div>', unsafe_allow_html=True)
    with col_rest:
        if st.button("↺  Restaurar ocultos", use_container_width=True):
            st.session_state.jogos_ocultos = {k for k in st.session_state.jogos_ocultos if not k.startswith(f"{liga_ativa}|")}
            st.rerun()

    if not fila:
        st.info("Todos os jogos desta liga já foram calculados ou ocultados.")

    for idx_fila, row in enumerate(fila):
        home     = str(row["HomeTeam"]); away = str(row["AwayTeam"])
        rodada_j = str(row.get("rodada", "—"))
        date_j   = str(row.get("date", row.get("Date", "")))
        chave    = f"{liga_ativa}|{home}|{away}|{rodada_j}"
        prefix   = f"{liga_ativa}_{home}_{away}_{idx_fila}".replace(" ", "_")

        fk_home = df[df["HomeTeam"] == home]
        fk_away = df[df["AwayTeam"] == away]

        with st.container():
            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid #1c2128;border-radius:12px;padding:1.1rem 1.4rem;margin-bottom:.5rem;">
                <div style="font-size:15px;font-weight:600;color:#e6edf3;">🏠 {home} &nbsp;vs&nbsp; ✈️ {away}</div>
                <div style="font-size:11px;color:#6e7681;font-family:'DM Mono',monospace;margin-top:2px;">Rodada {rodada_j}{f" · {date_j[:10]}" if date_j else ""}</div>
            </div>""", unsafe_allow_html=True)

            # Stats rápidas
            s1, s2, s3, s4 = st.columns(4)
            mh = fk_home["HomeShots"].mean(); sh = fk_home["AwayShots"].mean()
            ma = fk_away["AwayShots"].mean(); sa = fk_away["HomeShots"].mean()
            with s1: st.markdown(stat_box(f"{mh:.1f}" if not pd.isna(mh) else "—", f"{home[:13]} chutes casa"), unsafe_allow_html=True)
            with s2: st.markdown(stat_box(f"{sh:.1f}" if not pd.isna(sh) else "—", f"Sofridos {home[:11]} casa"), unsafe_allow_html=True)
            with s3: st.markdown(stat_box(f"{ma:.1f}" if not pd.isna(ma) else "—", f"{away[:13]} chutes fora"), unsafe_allow_html=True)
            with s4: st.markdown(stat_box(f"{sa:.1f}" if not pd.isna(sa) else "—", f"Sofridos {away[:11]} fora"), unsafe_allow_html=True)

            st.markdown("")

            # Odds 1X2
            st.markdown('<div class="input-label">Odds 1X2</div>', unsafe_allow_html=True)
            o1, o2, o3 = st.columns(3)
            with o1: home_odds = st.number_input("Odd Casa",   value=1.70, step=0.01, format="%.2f", key=f"{prefix}_oh", label_visibility="collapsed")
            with o2: draw_odds = st.number_input("Odd Empate", value=4.00, step=0.01, format="%.2f", key=f"{prefix}_od", label_visibility="collapsed")
            with o3: away_odds = st.number_input("Odd Fora",   value=4.80, step=0.01, format="%.2f", key=f"{prefix}_oa", label_visibility="collapsed")
            ph_live, _, pa_live = remover_juice(home_odds, draw_odds, away_odds)
            st.markdown(f'<p style="font-size:11px;color:#6e7681;font-family:\'DM Mono\',monospace;margin:-4px 0 6px;">P(home) {ph_live*100:.1f}% · P(away) {pa_live*100:.1f}%</p>', unsafe_allow_html=True)

            # Linhas — Total | Casa | Fora
            c_tot, c_casa, c_fora = st.columns(3)

            with c_tot:
                st.markdown('<div class="input-label">🏟️ Total</div>', unsafe_allow_html=True)
                _lt = st.number_input("Linha total", value=25.5, step=0.5, format="%.1f", key=f"{prefix}_lt", label_visibility="collapsed")
                line_total = snap_half(_lt)
                if line_total != _lt: st.caption(f"↳ {line_total}")
                odd_total_over  = st.number_input("Over total",  value=1.85, step=0.01, format="%.2f", key=f"{prefix}_oto", label_visibility="collapsed")
                odd_total_under = st.number_input("Under total", value=1.85, step=0.01, format="%.2f", key=f"{prefix}_otu", label_visibility="collapsed")

            with c_casa:
                st.markdown('<div class="input-label">🏠 Casa</div>', unsafe_allow_html=True)
                _lh = st.number_input("Linha casa", value=14.5, step=0.5, format="%.1f", key=f"{prefix}_lh", label_visibility="collapsed")
                line_home = snap_half(_lh)
                if line_home != _lh: st.caption(f"↳ {line_home}")
                odd_home_over  = st.number_input("Over casa",  value=1.85, step=0.01, format="%.2f", key=f"{prefix}_oho", label_visibility="collapsed")
                odd_home_under = st.number_input("Under casa", value=1.85, step=0.01, format="%.2f", key=f"{prefix}_ohu", label_visibility="collapsed")

            with c_fora:
                st.markdown('<div class="input-label">✈️ Fora</div>', unsafe_allow_html=True)
                _la = st.number_input("Linha fora", value=10.5, step=0.5, format="%.1f", key=f"{prefix}_la", label_visibility="collapsed")
                line_away = snap_half(_la)
                if line_away != _la: st.caption(f"↳ {line_away}")
                odd_away_over  = st.number_input("Over fora",  value=1.85, step=0.01, format="%.2f", key=f"{prefix}_oao", label_visibility="collapsed")
                odd_away_under = st.number_input("Under fora", value=1.85, step=0.01, format="%.2f", key=f"{prefix}_oau", label_visibility="collapsed")

            st.markdown("")

            btn1, btn2 = st.columns([5, 1])
            with btn1:
                calcular = st.button("⚡  Calcular Edge", type="primary", use_container_width=True, key=f"{prefix}_calc")
            with btn2:
                ocultar = st.button("🗑️", use_container_width=True, key=f"{prefix}_ocultar", help="Remover da fila")

            if ocultar:
                st.session_state.jogos_ocultos.add(chave)
                st.rerun()

            if calcular:
                try:
                    prob_home, prob_draw, prob_away = remover_juice(home_odds, draw_odds, away_odds)
                    variaveis = calcular_lambdas(df, res_home, res_away, home, away, prob_home, prob_away)
                    lh = variaveis["Lambda_Home"]; la = variaveis["Lambda_Away"]; lt = variaveis["Lambda_Total"]
                    alpha_h = res_home.params["alpha"]; alpha_a = res_away.params["alpha"]

                    merc_home  = calcular_mercado(lh, alpha_h, line_home, odd_home_over, odd_home_under)
                    merc_away  = calcular_mercado(la, alpha_a, line_away, odd_away_over, odd_away_under)

                    _, dist_h = distribuicao_nb(lh, alpha_h)
                    _, dist_a = distribuicao_nb(la, alpha_a)
                    dist_total = np.convolve(dist_h, dist_a)
                    po_t, pu_t = probabilidades_linha(dist_total, line_total)
                    oj_to, oj_tu = odd_justa(po_t), odd_justa(pu_t)
                    merc_total = {"Prob_Over":po_t,"Prob_Under":pu_t,"OddJusta_Over":oj_to,"OddJusta_Under":oj_tu,
                                  "Edge_Over":calcular_edge(odd_total_over,oj_to),"Edge_Under":calcular_edge(odd_total_under,oj_tu)}

                    r1, r2, r3 = st.columns(3)
                    with r1: st.markdown(result_card(f"🏟️ Total", lt, line_total, merc_total), unsafe_allow_html=True)
                    with r2: st.markdown(result_card(f"🏠 {home}", lh, line_home, merc_home), unsafe_allow_html=True)
                    with r3: st.markdown(result_card(f"✈️ {away}", la, line_away, merc_away), unsafe_allow_html=True)

                    # Usar data do CSV de jogos se disponível, senão data de hoje
                    data_jogo = date_j[:10] if date_j and date_j.strip() not in ("","nan") else date.today().strftime("%Y-%m-%d")

                    registro = {
                        "Date": data_jogo,
                        "HomeTeam": home, "AwayTeam": away,
                        "HomeOdds": home_odds, "DrawOdds": draw_odds, "AwayOdds": away_odds,
                        "ProbHome": round(prob_home,4), "ProbDraw": round(prob_draw,4), "ProbAway": round(prob_away,4),
                        "Idx_Att_Home_5": round(variaveis["Idx_Att_Home_5"],4) if not pd.isna(variaveis["Idx_Att_Home_5"]) else np.nan,
                        "Atk_Away":  round(variaveis["Atk_Away"],4) if not pd.isna(variaveis["Atk_Away"]) else np.nan,
                        "Pace":      round(variaveis["Pace"],4) if not pd.isna(variaveis["Pace"]) else np.nan,
                        "Lambda_Home": round(lh,4), "Lambda_Away": round(la,4), "Lambda_Total": round(lt,4),
                        "Line_Home": line_home, "Odd_Home_Over": odd_home_over, "Odd_Home_Under": odd_home_under,
                        "Prob_Home_Over": round(merc_home["Prob_Over"],4), "Prob_Home_Under": round(merc_home["Prob_Under"],4),
                        "OddJusta_Home_Over": round(merc_home["OddJusta_Over"],4), "OddJusta_Home_Under": round(merc_home["OddJusta_Under"],4),
                        "Edge_Home_Over": round(merc_home["Edge_Over"],4), "Edge_Home_Under": round(merc_home["Edge_Under"],4),
                        "Line_Away": line_away, "Odd_Away_Over": odd_away_over, "Odd_Away_Under": odd_away_under,
                        "Prob_Away_Over": round(merc_away["Prob_Over"],4), "Prob_Away_Under": round(merc_away["Prob_Under"],4),
                        "OddJusta_Away_Over": round(merc_away["OddJusta_Over"],4), "OddJusta_Away_Under": round(merc_away["OddJusta_Under"],4),
                        "Edge_Away_Over": round(merc_away["Edge_Over"],4), "Edge_Away_Under": round(merc_away["Edge_Under"],4),
                        "Line_Total": line_total, "Odd_Total_Over": odd_total_over, "Odd_Total_Under": odd_total_under,
                        "Prob_Total_Over": round(po_t,4), "Prob_Total_Under": round(pu_t,4),
                        "OddJusta_Total_Over": round(oj_to,4) if not pd.isna(oj_to) else np.nan,
                        "OddJusta_Total_Under": round(oj_tu,4) if not pd.isna(oj_tu) else np.nan,
                        "Edge_Total_Over": round(merc_total["Edge_Over"],4), "Edge_Total_Under": round(merc_total["Edge_Under"],4),
                        "Real_HomeShots": np.nan, "Real_AwayShots": np.nan, "Real_TotalShots": np.nan,
                    }

                    novo_hist = pd.concat([st.session_state.ligas[liga_ativa]["historico"], pd.DataFrame([registro])], ignore_index=True)
                    st.session_state.ligas[liga_ativa]["historico"] = novo_hist
                    st.success(f"✓  {home} vs {away} salvo.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")

            st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ABA BOAS APOSTAS
# ══════════════════════════════════════════════════════════════════════════════
with aba_apostas:

    hist = st.session_state.ligas[liga_ativa]["historico"]

    if hist.empty:
        st.info("Calcule pelo menos um jogo na aba Análise para ver as apostas.")
    else:
        # Calcular diff e EV para cada mercado de cada jogo
        def montar_bets(hist):
            rows = []
            for i, r in hist.iterrows():
                for mercado, lam_col, line_col, oj_col, odd_col, edge_col in [
                    ("Total Over",  "Lambda_Total", "Line_Total",  "OddJusta_Total_Over",  "Odd_Total_Over",  "Edge_Total_Over"),
                    ("Total Under", "Lambda_Total", "Line_Total",  "OddJusta_Total_Under", "Odd_Total_Under", "Edge_Total_Under"),
                    ("Casa Over",   "Lambda_Home",  "Line_Home",   "OddJusta_Home_Over",   "Odd_Home_Over",   "Edge_Home_Over"),
                    ("Casa Under",  "Lambda_Home",  "Line_Home",   "OddJusta_Home_Under",  "Odd_Home_Under",  "Edge_Home_Under"),
                    ("Fora Over",   "Lambda_Away",  "Line_Away",   "OddJusta_Away_Over",   "Odd_Away_Over",   "Edge_Away_Over"),
                    ("Fora Under",  "Lambda_Away",  "Line_Away",   "OddJusta_Away_Under",  "Odd_Away_Under",  "Edge_Away_Under"),
                ]:
                    if lam_col not in r or line_col not in r: continue
                    lam  = r.get(lam_col, np.nan)
                    diff = abs(lam - r.get(line_col, np.nan)) if not pd.isna(lam) else np.nan
                    ev   = r.get(edge_col, np.nan)
                    if pd.isna(diff) or pd.isna(ev): continue
                    rows.append({
                        "_idx":    i,
                        "Jogo":    f"{r['HomeTeam']} vs {r['AwayTeam']}",
                        "Mercado": mercado,
                        "λ":       round(lam, 2),
                        "Linha":   r.get(line_col),
                        "Diff":    round(diff, 2),
                        "Odd":     r.get(odd_col),
                        "OddJusta": round(r.get(oj_col, np.nan), 2) if not pd.isna(r.get(oj_col, np.nan)) else np.nan,
                        "EV%":     round(ev * 100, 1),
                        "_line_col": line_col,
                        "_odd_col":  odd_col,
                        "_oj_col":   oj_col,
                        "_edge_col": edge_col,
                        "_lam_col":  lam_col,
                    })
            return pd.DataFrame(rows)

        df_bets = montar_bets(hist)

        # ── Filtros ───────────────────────────────────────────────────────────
        st.markdown('<div class="sec">Filtros</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        with f1:
            min_diff = st.number_input("Diff mínimo", min_value=0.0, value=2.0, step=0.5, format="%.1f", key="f_diff")
        with f2:
            min_ev = st.number_input("EV mínimo (%)", min_value=-100.0, value=3.0, step=0.5, format="%.1f", key="f_ev")

        df_filtrado = df_bets[(df_bets["Diff"] >= min_diff) & (df_bets["EV%"] >= min_ev)].copy()

        if df_filtrado.empty:
            st.info(f"Nenhuma bet com diff ≥ {min_diff} e EV ≥ {min_ev}%.")
        else:
            st.markdown(f'<div class="sec">{len(df_filtrado)} bets encontradas</div>', unsafe_allow_html=True)

            # Tabela resumo
            cols_show = ["Jogo","Mercado","λ","Linha","Diff","Odd","OddJusta","EV%"]
            st.dataframe(df_filtrado[cols_show].reset_index(drop=True), use_container_width=True, hide_index=True)

            # ── Reeditar odd ou linha ─────────────────────────────────────────
            st.markdown('<div class="sec">Atualizar odd ou linha</div>', unsafe_allow_html=True)
            st.caption("Selecione uma bet para atualizar a odd ou a linha e recalcular o EV.")

            opcoes_bets = [f"{r['Jogo']} · {r['Mercado']}" for _, r in df_filtrado.iterrows()]
            sel_bet = st.selectbox("Bet", opcoes_bets, key="sel_bet_apostas")
            bet_row = df_filtrado.iloc[opcoes_bets.index(sel_bet)]
            idx_hist = int(bet_row["_idx"])

            rb1, rb2, rb3 = st.columns(3)
            with rb1:
                nova_odd = st.number_input("Nova odd", value=float(bet_row["Odd"]), step=0.01, format="%.2f", key="nova_odd_ap")
            with rb2:
                nova_linha_raw = st.number_input("Nova linha", value=float(bet_row["Linha"]), step=0.5, format="%.1f", key="nova_linha_ap")
                nova_linha = snap_half(nova_linha_raw)
                if nova_linha != nova_linha_raw: st.caption(f"↳ {nova_linha}")
            with rb3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔄  Recalcular EV", use_container_width=True, key="btn_recalc"):
                    h_row = st.session_state.ligas[liga_ativa]["historico"].loc[idx_hist]
                    lam   = h_row[bet_row["_lam_col"]]

                    # Recalcular probabilidades com nova linha
                    alpha_key = "alpha"
                    is_home = "Home" in bet_row["Mercado"] or "Casa" in bet_row["Mercado"]
                    is_away = "Fora" in bet_row["Mercado"] or "Away" in bet_row["Mercado"]
                    is_total = "Total" in bet_row["Mercado"]

                    rh = st.session_state.ligas[liga_ativa]["res_home"]
                    ra = st.session_state.ligas[liga_ativa]["res_away"]
                    alpha = rh.params["alpha"] if (is_home or is_total) else ra.params["alpha"]

                    _, dist = distribuicao_nb(lam, alpha)
                    po, pu  = probabilidades_linha(dist, nova_linha)
                    oj      = odd_justa(po if "Over" in bet_row["Mercado"] else pu)
                    prob    = po if "Over" in bet_row["Mercado"] else pu
                    novo_ev = calcular_edge(nova_odd, oj)
                    novo_oj = oj

                    # Atualizar no histórico
                    h = st.session_state.ligas[liga_ativa]["historico"]
                    h.at[idx_hist, bet_row["_line_col"]] = nova_linha
                    h.at[idx_hist, bet_row["_odd_col"]]  = nova_odd
                    h.at[idx_hist, bet_row["_oj_col"]]   = round(novo_oj, 4) if not pd.isna(novo_oj) else np.nan
                    h.at[idx_hist, bet_row["_edge_col"]] = round(novo_ev, 4) if not pd.isna(novo_ev) else np.nan
                    st.session_state.ligas[liga_ativa]["historico"] = h

                    ev_pct = novo_ev * 100 if not pd.isna(novo_ev) else float("nan")
                    st.success(f"Atualizado — OddJusta {novo_oj:.2f} · EV {ev_pct:+.1f}%")
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ABA HISTÓRICO
# ══════════════════════════════════════════════════════════════════════════════
with aba_historico:

    hist = st.session_state.ligas[liga_ativa]["historico"]

    if hist.empty:
        st.info("Nenhum registro ainda para esta liga.")
    else:
        st.markdown(f'<div class="sec">Histórico — {liga_ativa} ({len(hist)} registros)</div>', unsafe_allow_html=True)

        cols_view = ["Date","HomeTeam","AwayTeam","Lambda_Home","Lambda_Away","Lambda_Total",
                     "Line_Home","Odd_Home_Over","Odd_Home_Under",
                     "Line_Away","Odd_Away_Over","Odd_Away_Under",
                     "Line_Total","Odd_Total_Over","Odd_Total_Under",
                     "Real_HomeShots","Real_AwayShots","Real_TotalShots"]
        cols_view = [c for c in cols_view if c in hist.columns]
        st.dataframe(hist[cols_view].round(4), use_container_width=True, hide_index=True)

        with st.expander("✏️  Preencher resultados reais"):
            idx_opcoes = [f"{i} · {r['HomeTeam']} vs {r['AwayTeam']} ({r.get('Date','')})" for i, r in hist.iterrows()]
            sel   = st.selectbox("Jogo", idx_opcoes, key="sel_jogo")
            idx_s = int(sel.split(" · ")[0])
            rc1, rc2, rc3 = st.columns(3)
            with rc1: real_h = st.number_input("Chutes Casa",  min_value=0, step=1, key="rh")
            with rc2: real_a = st.number_input("Chutes Fora",  min_value=0, step=1, key="ra")
            with rc3: real_t = st.number_input("Total Chutes", min_value=0, step=1, key="rt", value=int(real_h + real_a))
            if st.button("Salvar resultado real"):
                st.session_state.ligas[liga_ativa]["historico"].at[idx_s, "Real_HomeShots"]  = real_h
                st.session_state.ligas[liga_ativa]["historico"].at[idx_s, "Real_AwayShots"]  = real_a
                st.session_state.ligas[liga_ativa]["historico"].at[idx_s, "Real_TotalShots"] = real_t
                st.success("Resultado salvo!")
                st.rerun()

        st.markdown("")
        a1, a2, a3 = st.columns(3)

        with a1:
            csv_bytes = hist.to_csv(index=False).encode("utf-8")
            nome_csv  = f"model_{liga_ativa.lower().replace(' ','_')}SHOTS_{date.today().strftime('%Y%m%d')}.csv"
            st.download_button(f"⬇️  Exportar {liga_ativa}", data=csv_bytes, file_name=nome_csv, mime="text/csv", use_container_width=True)

        with a2:
            if st.button("🗑️  Apagar último jogo", use_container_width=True):
                st.session_state.ligas[liga_ativa]["historico"] = hist.iloc[:-1].reset_index(drop=True)
                st.success("Último jogo removido.")
                st.rerun()

        with a3:
            if st.button("🗑️  Limpar sessão", use_container_width=True):
                original = st.session_state.ligas[liga_ativa].get("historico_original")
                st.session_state.ligas[liga_ativa]["historico"] = original.copy() if original is not None else pd.DataFrame(columns=COLUNAS_MODELO)
                st.success("Sessão limpa.")
                st.rerun()
