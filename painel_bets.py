import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

st.set_page_config(page_title="Painel de Bets", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0a0c10; color: #c9d1d9; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1c2128; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] label { font-size:11px !important; text-transform:uppercase !important; letter-spacing:.08em !important; color:#6e7681 !important; font-weight:500 !important; }
.main { background: #0a0c10; }
.block-container { padding: 1.5rem 2rem; max-width: 1600px; }
div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] > div > div { background:#0d1117 !important; border:1px solid #1c2128 !important; color:#c9d1d9 !important; border-radius:6px !important; font-family:'DM Mono',monospace !important; font-size:14px !important; }
.stNumberInput label, .stSelectbox label, .stTextInput label, .stMultiSelect label { font-size:11px !important; text-transform:uppercase !important; letter-spacing:.08em !important; color:#6e7681 !important; font-weight:500 !important; }
.stButton > button { background:#161b22 !important; border:1px solid #1c2128 !important; color:#c9d1d9 !important; font-size:12px !important; border-radius:6px !important; }
.stButton > button:hover { border-color:#3fb950 !important; color:#3fb950 !important; }
.sec { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.12em; color:#6e7681; border-bottom:1px solid #1c2128; padding-bottom:6px; margin:1.2rem 0 .8rem; }
.badge-pos { font-family:'DM Mono',monospace; font-size:12px; font-weight:500; background:#0d2818; color:#3fb950; border:1px solid #2ea043; border-radius:5px; padding:1px 8px; }
.badge-neg { font-family:'DM Mono',monospace; font-size:12px; font-weight:500; background:#2d1111; color:#f85149; border:1px solid #da3633; border-radius:5px; padding:1px 8px; }
.badge-neu { font-family:'DM Mono',monospace; font-size:12px; font-weight:500; background:#131d2e; color:#79c0ff; border:1px solid #388bfd; border-radius:5px; padding:1px 8px; }

/* Cards com fundo levemente mais fraco e borda lateral mais grossa */
.bet-card {
    background: #0b0e14;
    border: 1px solid #161b22;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: .6rem;
    position: relative;
}
.bet-card.chutes    { border-left: 5px solid #388bfd; }
.bet-card.faltas    { border-left: 5px solid #f85149; }
.bet-card.removed   { opacity: 0.35; border-left-color: #333 !important; }

.bet-match { font-size:18px; font-weight:700; color:#e6edf3; letter-spacing:-.01em; }
.bet-meta  { font-size:11px; color:#6e7681; font-family:'DM Mono',monospace; margin-top:3px; }
.bet-market { font-size:12px; font-weight:600; color:#c9d1d9; margin-top:8px; }
.bet-numbers { display:flex; gap:12px; align-items:center; margin-top:6px; flex-wrap:wrap; }
.bet-num { font-family:'DM Mono',monospace; font-size:13px; color:#6e7681; }
.bet-num span { color:#e6edf3; }

.source-pill {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
    border-radius: 4px;
    padding: 1px 7px;
    margin-right: 6px;
}
.pill-chutes   { background: #131d2e; color: #79c0ff; border: 1px solid #388bfd; }
.pill-faltas-eu { background: #2d1111; color: #ff8080; border: 1px solid #f85149; }
.pill-faltas-br { background: #2d1111; color: #ff8080; border: 1px solid #f85149; }

.stat-pill { background:#0d1117; border:1px solid #1c2128; border-radius:8px; padding:.7rem 1rem; text-align:center; }
.stat-pill .val { font-family:'DM Mono',monospace; font-size:22px; color:#e6edf3; font-weight:500; }
.stat-pill .lbl { font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:#6e7681; margin-top:2px; }

.time-group { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.15em; color:#6e7681; border-bottom:1px solid #1c2128; padding-bottom:4px; margin:1.4rem 0 .8rem; }

/* Relógio / momento do dia */
.momento-box {
    background: #0d1117;
    border: 1px solid #1c2128;
    border-radius: 8px;
    padding: .5rem 1rem;
    display: inline-flex;
    align-items: center;
    gap: 10px;
}
.momento-hora { font-family:'DM Mono',monospace; font-size:20px; font-weight:500; color:#e6edf3; }
.momento-label { font-size:11px; color:#6e7681; text-transform:uppercase; letter-spacing:.08em; }

/* Separador de filtros */
.filter-section-title {
    font-size:10px; font-weight:700; text-transform:uppercase;
    letter-spacing:.12em; color:#6e7681;
    border-bottom:1px solid #1c2128; padding-bottom:4px; margin:12px 0 8px;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────

if "removidas" not in st.session_state:
    st.session_state.removidas = set()

if "dfs" not in st.session_state:
    st.session_state.dfs = {}

if "data_referencia" not in st.session_state:
    st.session_state.data_referencia = date.today()

# ─── FUNÇÕES AUXILIARES ────────────────────────────────────────────────────────

def badge_html(val):
    if pd.isna(val): return "—"
    pct = f"{val*100:+.1f}%"
    if val > 0.03:   return f'<span class="badge-pos">{pct}</span>'
    elif val < -0.03: return f'<span class="badge-neg">{pct}</span>'
    else:             return f'<span class="badge-neu">{pct}</span>'

def source_pill(fonte):
    cls = {"Chutes": "pill-chutes", "Faltas EU": "pill-faltas-eu", "Faltas BR": "pill-faltas-br"}.get(fonte, "pill-chutes")
    return f'<span class="source-pill {cls}">{fonte}</span>'

def card_class(fonte):
    return "chutes" if fonte == "Chutes" else "faltas"

def parse_date(v):
    if pd.isna(v) or str(v).strip() == "": return pd.NaT
    try: return pd.to_datetime(v)
    except: return pd.NaT

def momento_do_dia(hora):
    if 5 <= hora < 12:  return "🌅", "Manhã"
    if 12 <= hora < 14: return "☀️", "Almoço"
    if 14 <= hora < 18: return "🌤️", "Tarde"
    if 18 <= hora < 22: return "🌆", "Noite"
    return "🌙", "Madrugada"

# ─── PARSERS ────────────────────────────────────────────────────────────────────

def parse_chutes(df, nome_liga):
    rows = []
    mercados = [
        ("Total Over",  "Lambda_Total", "Line_Total",  "Odd_Total_Over",  "Edge_Total_Over"),
        ("Total Under", "Lambda_Total", "Line_Total",  "Odd_Total_Under", "Edge_Total_Under"),
        ("Casa Over",   "Lambda_Home",  "Line_Home",   "Odd_Home_Over",   "Edge_Home_Over"),
        ("Casa Under",  "Lambda_Home",  "Line_Home",   "Odd_Home_Under",  "Edge_Home_Under"),
        ("Fora Over",   "Lambda_Away",  "Line_Away",   "Odd_Away_Over",   "Edge_Away_Over"),
        ("Fora Under",  "Lambda_Away",  "Line_Away",   "Odd_Away_Under",  "Edge_Away_Under"),
    ]
    for _, r in df.iterrows():
        data = parse_date(r.get("Date",""))
        for mercado, lam_col, line_col, odd_col, edge_col in mercados:
            lam  = r.get(lam_col, np.nan)
            line = r.get(line_col, np.nan)
            odd  = r.get(odd_col, np.nan)
            edge = r.get(edge_col, np.nan)
            if pd.isna(edge): continue
            diff = abs(lam - line) if not (pd.isna(lam) or pd.isna(line)) else np.nan
            rows.append({
                "uid":      f"chutes|{nome_liga}|{r.get('HomeTeam','')}|{r.get('AwayTeam','')}|{mercado}",
                "Fonte":    "Chutes",
                "Liga":     nome_liga,
                "Data":     data,
                "HomeTeam": str(r.get("HomeTeam","")),
                "AwayTeam": str(r.get("AwayTeam","")),
                "Mercado":  mercado,
                "Lambda":   round(float(lam),2) if not pd.isna(lam) else np.nan,
                "Linha":    float(line) if not pd.isna(line) else np.nan,
                "Diff":     round(float(diff),2) if not pd.isna(diff) else np.nan,
                "Odd":      float(odd) if not pd.isna(odd) else np.nan,
                "Edge":     float(edge) if not pd.isna(edge) else np.nan,
                "Stake":    np.nan,
                "Rodada":   str(r.get("Rodada","")) if "Rodada" in r.index else "",
            })
    return rows

def parse_faltas(df, nome_liga, fonte):
    rows = []
    mercados = [
        ("Total Over",  "Lambda_Total", "Total_Line", "Odd_Over_Total",  "Edge_Over_Total"),
        ("Total Under", "Lambda_Total", "Total_Line", "Odd_Under_Total", "Edge_Under_Total"),
        ("Casa Over",   "Lambda_Home",  "Line_Home",  "Odd_Over_Home",   "Edge_Over_Home"),
        ("Casa Under",  "Lambda_Home",  "Line_Home",  "Odd_Under_Home",  "Edge_Under_Home"),
        ("Fora Over",   "Lambda_Away",  "Line_Away",  "Odd_Over_Away",   "Edge_Over_Away"),
        ("Fora Under",  "Lambda_Away",  "Line_Away",  "Odd_Under_Away",  "Edge_Under_Away"),
    ]
    for _, r in df.iterrows():
        # Faltas usa a data de CÁLCULO (quando foi gerado), não do jogo
        data = parse_date(r.get("Date",""))
        rodada = str(r.get("Rodada","")) if "Rodada" in df.columns else ""
        stake_jogo = r.get("stake", np.nan) if "stake" in df.columns else np.nan
        for mercado, lam_col, line_col, odd_col, edge_col in mercados:
            lam  = r.get(lam_col, np.nan)
            line = r.get(line_col, np.nan)
            odd  = r.get(odd_col, np.nan)
            edge = r.get(edge_col, np.nan)
            if pd.isna(edge): continue
            diff_val = r.get("diff", np.nan) if "diff" in df.columns else (
                abs(lam - line) if not (pd.isna(lam) or pd.isna(line)) else np.nan
            )
            rows.append({
                "uid":      f"{fonte}|{nome_liga}|{r.get('HomeTeam','')}|{r.get('AwayTeam','')}|{mercado}",
                "Fonte":    fonte,
                "Liga":     nome_liga,
                "Data":     data,
                "HomeTeam": str(r.get("HomeTeam","")),
                "AwayTeam": str(r.get("AwayTeam","")),
                "Mercado":  mercado,
                "Lambda":   round(float(lam),2) if not pd.isna(lam) else np.nan,
                "Linha":    float(line) if not pd.isna(line) else np.nan,
                "Diff":     round(float(diff_val),2) if not pd.isna(diff_val) else np.nan,
                "Odd":      float(odd) if not pd.isna(odd) else np.nan,
                "Edge":     float(edge) if not pd.isna(edge) else np.nan,
                "Stake":    float(stake_jogo) if not pd.isna(stake_jogo) else np.nan,
                "Rodada":   rodada,
            })
    return rows

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎯 Painel de Bets")
    st.markdown("---")

    # ── Relógio / data de referência ──
    agora = datetime.now()
    emoji_momento, label_momento = momento_do_dia(agora.hour)
    st.markdown(f"""
    <div class="momento-box">
        <div>
            <div class="momento-hora">{agora.strftime('%H:%M')}</div>
            <div class="momento-label">{emoji_momento} {label_momento} · {agora.strftime('%d/%m/%Y')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    # Data de referência manual (para filtrar jogos de hoje)
    data_ref = st.date_input(
        "Data de referência",
        value=st.session_state.data_referencia,
        key="data_ref_input",
        help="Data usada para destacar jogos de hoje. Por padrão é o dia atual."
    )
    st.session_state.data_referencia = data_ref

    st.markdown("---")
    st.markdown('<p style="font-size:11px;color:#6e7681;text-transform:uppercase;letter-spacing:.08em;font-weight:600;">Carregar modelos</p>', unsafe_allow_html=True)

    # CHUTES
    with st.expander("🎯 Chutes", expanded=False):
        chutes_files = st.file_uploader("CSVs de chutes", type=["csv"], accept_multiple_files=True, key="up_chutes",
                                         help="model_*SHOTS*.csv — um por liga")
        if chutes_files:
            for f in chutes_files:
                nome = f.name.replace(".csv","").replace("model_","").replace("SHOTS","").strip("_")
                if nome not in st.session_state.dfs:
                    try:
                        df = pd.read_csv(f)
                        st.session_state.dfs[nome] = ("chutes", df)
                        st.success(f"✓ {nome}")
                    except Exception as e:
                        st.error(f"Erro: {e}")

    # FALTAS EU
    with st.expander("⚽ Faltas EU", expanded=False):
        eu_files = st.file_uploader("CSVs de faltas EU", type=["csv"], accept_multiple_files=True, key="up_faltas_eu",
                                     help="model_*FALTAS*.csv — um por liga")
        if eu_files:
            for f in eu_files:
                nome = f.name.replace(".csv","").replace("model_","").replace("FALTAS","").strip("_")
                chave = f"eu_{nome}"
                if chave not in st.session_state.dfs:
                    try:
                        df = pd.read_csv(f)
                        st.session_state.dfs[chave] = ("faltas_eu", df)
                        st.success(f"✓ {nome}")
                    except Exception as e:
                        st.error(f"Erro: {e}")

    # FALTAS BR
    with st.expander("🇧🇷 Faltas BR", expanded=False):
        br_files = st.file_uploader("CSVs de faltas BR", type=["csv"], accept_multiple_files=True, key="up_faltas_br",
                                     help="model_brasilFALTAS*.csv")
        if br_files:
            for f in br_files:
                nome = f.name.replace(".csv","").replace("model_","").strip("_")
                chave = f"br_{nome}"
                if chave not in st.session_state.dfs:
                    try:
                        df = pd.read_csv(f)
                        st.session_state.dfs[chave] = ("faltas_br", df)
                        st.success(f"✓ {nome}")
                    except Exception as e:
                        st.error(f"Erro: {e}")

    if st.session_state.dfs:
        st.markdown("---")
        st.markdown('<p style="font-size:11px;color:#6e7681;text-transform:uppercase;letter-spacing:.08em;font-weight:600;">Arquivos carregados</p>', unsafe_allow_html=True)
        for chave, (tipo, _) in st.session_state.dfs.items():
            icone = {"chutes":"🎯","faltas_eu":"⚽","faltas_br":"🇧🇷"}.get(tipo,"📄")
            st.markdown(f'<p style="font-size:12px;">{icone} {chave}</p>', unsafe_allow_html=True)
        if st.button("🗑️  Limpar tudo", use_container_width=True):
            st.session_state.dfs = {}
            st.session_state.removidas = set()
            st.rerun()

# ─── SEM DADOS ─────────────────────────────────────────────────────────────────

if not st.session_state.dfs:
    st.markdown('<h1 style="color:#e6edf3;font-size:24px;font-weight:600;">Painel de Bets</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6e7681;">Carregue os CSVs dos modelos na sidebar para visualizar as bets.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0d1117;border:1px solid #1c2128;border-radius:12px;padding:1.4rem;margin-top:1rem;">
        <p style="font-size:12px;color:#6e7681;margin-bottom:.8rem;text-transform:uppercase;letter-spacing:.08em;font-weight:600;">Como usar</p>
        <p style="font-size:13px;color:#c9d1d9;">1. Na sidebar, expanda a seção do modelo desejado</p>
        <p style="font-size:13px;color:#c9d1d9;">2. Faça upload dos CSVs exportados de cada app</p>
        <p style="font-size:13px;color:#c9d1d9;">3. Aplique os filtros separados por modelo</p>
        <p style="font-size:13px;color:#c9d1d9;">4. As bets aparecem ordenadas por data dos jogos</p>
        <p style="font-size:13px;color:#c9d1d9;">5. Clique em ✓ para marcar uma bet como feita</p>
        <br>
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
            <span class="source-pill pill-chutes" style="font-size:11px;">🎯 Chutes</span>
            <span class="source-pill pill-faltas-eu" style="font-size:11px;">⚽ Faltas EU</span>
            <span class="source-pill pill-faltas-br" style="font-size:11px;">🇧🇷 Faltas BR</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── MONTAR BETS ───────────────────────────────────────────────────────────────

all_rows = []
for chave, (tipo, df) in st.session_state.dfs.items():
    nome_display = chave.replace("eu_","").replace("br_","")
    if tipo == "chutes":
        all_rows.extend(parse_chutes(df, nome_display))
    elif tipo == "faltas_eu":
        all_rows.extend(parse_faltas(df, nome_display, "Faltas EU"))
    elif tipo == "faltas_br":
        all_rows.extend(parse_faltas(df, nome_display, "Faltas BR"))

if not all_rows:
    st.warning("Nenhuma bet encontrada nos arquivos carregados.")
    st.stop()

df_bets = pd.DataFrame(all_rows)
df_bets["Data"] = pd.to_datetime(df_bets["Data"], errors="coerce")

# ─── SEPARAR EM CHUTES vs FALTAS ───────────────────────────────────────────────

df_chutes = df_bets[df_bets["Fonte"] == "Chutes"].copy()
df_faltas = df_bets[df_bets["Fonte"].isin(["Faltas EU","Faltas BR"])].copy()

# ─── FILTROS ───────────────────────────────────────────────────────────────────

st.markdown('<h1 style="color:#e6edf3;font-size:22px;font-weight:600;margin-bottom:4px;">Painel de Bets</h1>', unsafe_allow_html=True)

# ── Filtros CHUTES ──
st.markdown('<div class="filter-section-title">🎯 Filtros — Chutes</div>', unsafe_allow_html=True)
fc1, fc2, fc3, fc4 = st.columns([2, 2, 3, 2])
with fc1:
    edge_chutes = st.number_input("Edge mín (%)", value=3.0, step=0.5, format="%.1f", key="fc_edge")
with fc2:
    diff_chutes = st.number_input("Diff mín", value=0.0, step=0.5, format="%.1f", key="fc_diff")
with fc3:
    merc_chutes_disp = ["Total Over","Total Under","Casa Over","Casa Under","Fora Over","Fora Under"]
    merc_chutes = st.multiselect("Mercado", merc_chutes_disp, default=merc_chutes_disp, key="fc_merc")
with fc4:
    rem_chutes = st.checkbox("Mostrar removidas", value=False, key="fc_rem")

# ── Filtros FALTAS ──
st.markdown('<div class="filter-section-title">⚽ Filtros — Faltas</div>', unsafe_allow_html=True)
ff1, ff2, ff3, ff4, ff5 = st.columns([2, 2, 3, 3, 2])
with ff1:
    edge_faltas = st.number_input("Edge mín (%)", value=3.0, step=0.5, format="%.1f", key="ff_edge")
with ff2:
    diff_faltas = st.number_input("Diff mín", value=0.0, step=0.5, format="%.1f", key="ff_diff")
with ff3:
    merc_faltas_disp = ["Total Over","Total Under","Casa Over","Casa Under","Fora Over","Fora Under"]
    merc_faltas = st.multiselect("Mercado", merc_faltas_disp, default=merc_faltas_disp, key="ff_merc")
with ff4:
    fontes_faltas_disp = sorted(df_faltas["Fonte"].unique().tolist()) if not df_faltas.empty else ["Faltas EU","Faltas BR"]
    fontes_faltas = st.multiselect("Fonte", fontes_faltas_disp, default=fontes_faltas_disp, key="ff_fonte")
with ff5:
    rem_faltas = st.checkbox("Mostrar removidas", value=False, key="ff_rem")

# ─── APLICAR FILTROS ───────────────────────────────────────────────────────────

def filtrar(df, min_edge, min_diff, mercados, fontes=None, mostrar_rem=False):
    if df.empty: return df
    mask = (
        (df["Edge"] >= min_edge / 100) &
        (df["Diff"].fillna(0) >= min_diff) &
        (df["Mercado"].isin(mercados))
    )
    if fontes:
        mask = mask & df["Fonte"].isin(fontes)
    result = df[mask].copy()
    if not mostrar_rem:
        result = result[~result["uid"].isin(st.session_state.removidas)]
    return result.sort_values("Data", na_position="last").reset_index(drop=True)

df_c = filtrar(df_chutes, edge_chutes, diff_chutes, merc_chutes, mostrar_rem=rem_chutes)
df_f = filtrar(df_faltas, edge_faltas, diff_faltas, merc_faltas, fontes=fontes_faltas, mostrar_rem=rem_faltas)

df_all = pd.concat([df_c, df_f], ignore_index=True).sort_values("Data", na_position="last").reset_index(drop=True)

# ─── STATS ─────────────────────────────────────────────────────────────────────

st.markdown("")
pend  = len(df_all[~df_all["uid"].isin(st.session_state.removidas)])
feitas = len([u for u in df_all["uid"] if u in st.session_state.removidas])
pend_c = len(df_c[~df_c["uid"].isin(st.session_state.removidas)]) if not df_c.empty else 0
pend_f = len(df_f[~df_f["uid"].isin(st.session_state.removidas)]) if not df_f.empty else 0
avg_edge = df_all[~df_all["uid"].isin(st.session_state.removidas)]["Edge"].mean()
avg_odd  = df_all[~df_all["uid"].isin(st.session_state.removidas)]["Odd"].mean()

sc1,sc2,sc3,sc4,sc5,sc6 = st.columns(6)
with sc1: st.markdown(f'<div class="stat-pill"><div class="val">{pend}</div><div class="lbl">Pendentes</div></div>', unsafe_allow_html=True)
with sc2: st.markdown(f'<div class="stat-pill"><div class="val">{feitas}</div><div class="lbl">Apostadas</div></div>', unsafe_allow_html=True)
with sc3: st.markdown(f'<div class="stat-pill"><div class="val">{pend_c}</div><div class="lbl">Chutes pend.</div></div>', unsafe_allow_html=True)
with sc4: st.markdown(f'<div class="stat-pill"><div class="val">{pend_f}</div><div class="lbl">Faltas pend.</div></div>', unsafe_allow_html=True)
with sc5: st.markdown(f'<div class="stat-pill"><div class="val">{avg_edge*100:+.1f}%</div><div class="lbl">Edge médio</div></div>' if not pd.isna(avg_edge) else '<div class="stat-pill"><div class="val">—</div><div class="lbl">Edge médio</div></div>', unsafe_allow_html=True)
with sc6: st.markdown(f'<div class="stat-pill"><div class="val">{avg_odd:.2f}</div><div class="lbl">Odd média</div></div>' if not pd.isna(avg_odd) else '<div class="stat-pill"><div class="val">—</div><div class="lbl">Odd média</div></div>', unsafe_allow_html=True)

if df_all.empty:
    st.info("Nenhuma bet com os filtros aplicados.")
    st.stop()

# ─── CARDS (agrupados por data) ────────────────────────────────────────────────

st.markdown("")

data_ref_val = st.session_state.data_referencia

df_all["_date_key"] = df_all["Data"].dt.date.astype(str)
datas_unicas = df_all["_date_key"].unique()

for data_str in datas_unicas:
    grupo = df_all[df_all["_date_key"] == data_str]

    try:
        d = datetime.strptime(data_str, "%Y-%m-%d").date()
        label_data = datetime(d.year, d.month, d.day).strftime("%A, %d/%m/%Y").capitalize()
        is_hoje = (d == data_ref_val)
        hoje_badge = ' &nbsp;<span style="background:#0d2818;color:#3fb950;border:1px solid #2ea043;border-radius:4px;font-size:9px;padding:1px 6px;vertical-align:middle;">HOJE</span>' if is_hoje else ''
    except:
        label_data = data_str if data_str != "NaT" else "Sem data"
        hoje_badge = ""

    n_grupo = len(grupo[~grupo["uid"].isin(st.session_state.removidas)])
    st.markdown(f'<div class="time-group">📅 {label_data}{hoje_badge} &nbsp;·&nbsp; {n_grupo} bet{"s" if n_grupo!=1 else ""}</div>', unsafe_allow_html=True)

    for _, row in grupo.iterrows():
        uid = row["uid"]
        removida = uid in st.session_state.removidas
        cls_extra = "removed" if removida else ""
        card_cls  = card_class(row["Fonte"])

        hora_str   = row["Data"].strftime("%H:%M") if pd.notna(row["Data"]) and row["Data"].hour != 0 else ""
        rodada_str = f"Rodada {row['Rodada']}" if row["Rodada"] and row["Rodada"] not in ("nan","") else ""
        meta_parts = [p for p in [row["Liga"], rodada_str, hora_str] if p]
        meta_str   = " &nbsp;·&nbsp; ".join(meta_parts)

        diff_txt  = f"{row['Diff']:.2f}" if not pd.isna(row["Diff"]) else ""
        stake_txt = f"{row['Stake']}u" if not pd.isna(row["Stake"]) and row["Stake"] > 0 else ""

        col_card, col_btn = st.columns([11, 1])
        with col_card:
            st.markdown(f"""
            <div class="bet-card {card_cls} {cls_extra}">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;">
                    <div>
                        {source_pill(row['Fonte'])}
                        <span class="bet-match">{'✓ ' if removida else ''}{row['HomeTeam']} vs {row['AwayTeam']}</span>
                    </div>
                </div>
                <div class="bet-meta">{meta_str}</div>
                <div class="bet-market">{row['Mercado']}</div>
                <div class="bet-numbers">
                    <div class="bet-num">λ <span>{row['Lambda']:.2f}</span></div>
                    <div class="bet-num">linha <span>{row['Linha']:.1f}</span></div>
                    {'<div class="bet-num">diff <span>' + diff_txt + '</span></div>' if diff_txt else ''}
                    <div class="bet-num">odd <span>{row["Odd"]:.2f}</span></div>
                    <div>{badge_html(row['Edge'])}</div>
                    {'<div class="bet-num">stake <span>' + stake_txt + '</span></div>' if stake_txt else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_btn:
            st.markdown("<div style='padding-top:12px;'>", unsafe_allow_html=True)
            if removida:
                if st.button("↩", key=f"restore_{uid}", help="Restaurar"):
                    st.session_state.removidas.discard(uid)
                    st.rerun()
            else:
                if st.button("✓", key=f"rm_{uid}", help="Marcar como feita"):
                    st.session_state.removidas.add(uid)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# ─── RESTAURAR TODAS ───────────────────────────────────────────────────────────

if st.session_state.removidas:
    st.markdown("---")
    c1, c2, c3 = st.columns([4,2,4])
    with c2:
        if st.button("↩  Restaurar todas", use_container_width=True):
            st.session_state.removidas = set()
            st.rerun()
