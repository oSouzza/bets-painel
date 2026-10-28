"""
drive_utils.py — funções compartilhadas para ler/salvar dados no Google Sheets
Usado por app_chutes.py, app_faltas.py, app_faltas_br.py e painel_bets.py

Estrutura de planilhas:
  - "bets_chutes"    → abas: "hist_{liga}", "jogos_{liga}", "model_{liga}"
  - "bets_faltas_eu" → abas: "hist_{liga}", "jogos_{liga}", "model_{liga}"
  - "bets_faltas_br" → abas: "hist_{liga}", "jogos_{liga}", "model_{liga}"
"""

import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

PLANILHAS = {
    "chutes":    "bets_chutes",
    "faltas_eu": "bets_faltas_eu",
    "faltas_br": "bets_faltas_br",
}


@st.cache_resource
def get_drive_client():
    """Retorna cliente autenticado do gspread via secrets do Streamlit."""
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=SCOPES
        )
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Erro ao conectar ao Google Sheets: {e}")
        return None


def drive_disponivel() -> bool:
    """Verifica se as credenciais estão configuradas."""
    return "gcp_service_account" in st.secrets


def _get_sheet(subtipo: str):
    """Retorna a planilha do subtipo (deve existir e estar compartilhada)."""
    client = get_drive_client()
    if client is None:
        return None
    nome = PLANILHAS.get(subtipo, f"bets_{subtipo}")
    try:
        return client.open(nome)
    except gspread.SpreadsheetNotFound:
        st.error(f"❌ Planilha '{nome}' não encontrada. Crie-a no Google Sheets e compartilhe com a service account.")
        return None


def _get_or_create_worksheet(sh, titulo: str, rows=500, cols=50):
    """Retorna aba existente ou cria nova."""
    try:
        return sh.worksheet(titulo)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=titulo, rows=rows, cols=cols)


def _df_para_sheets(df: pd.DataFrame) -> list:
    """Converte DataFrame para lista de listas, limpando NaN."""
    df_clean = df.fillna("").astype(str).replace("nan", "").replace("None", "")
    return [df_clean.columns.tolist()] + df_clean.values.tolist()


def _sheets_para_df(ws) -> pd.DataFrame | None:
    """Lê aba do Sheets e retorna DataFrame, ou None se vazia."""
    dados = ws.get_all_records(default_blank="")
    if not dados:
        return None
    return pd.DataFrame(dados)


# ─── HISTÓRICO DO MODELO (jogos calculados) ───────────────────────────────────

def salvar_csv_drive(df: pd.DataFrame, nome_arquivo: str, subtipo: str) -> bool:
    """
    Salva o DataFrame do modelo numa aba da planilha.
    nome_arquivo ex: "model_italiaFALTAS.csv" → aba "model_italia"
    """
    try:
        sh = _get_sheet(subtipo)
        if sh is None:
            return False
        aba = nome_arquivo.replace(".csv", "")
        ws = _get_or_create_worksheet(sh, aba)
        ws.clear()
        ws.update(_df_para_sheets(df), value_input_option="RAW")
        return True
    except Exception as e:
        st.error(f"❌ Erro ao salvar no Drive: {e}")
        return False


def carregar_modelo_drive(liga: str, subtipo: str) -> pd.DataFrame | None:
    """
    Carrega o histórico do modelo para uma liga.
    liga ex: "italia" → aba "model_italia"
    """
    try:
        sh = _get_sheet(subtipo)
        if sh is None:
            return None
        sufixo = "FALTAS" if "faltas" in subtipo else "SHOTS"
        nome_aba = f"model_{liga.lower().replace(' ', '_')}{sufixo}"
        try:
            ws = sh.worksheet(nome_aba)
        except gspread.WorksheetNotFound:
            return None
        return _sheets_para_df(ws)
    except Exception as e:
        return None


# ─── HISTÓRICO BASE (partidas passadas para treinar o modelo) ─────────────────

def salvar_historico_drive(df: pd.DataFrame, liga: str, subtipo: str) -> bool:
    """
    Salva o CSV de histórico (partidas passadas) no Sheets.
    liga ex: "italia" → aba "hist_italia"
    """
    try:
        sh = _get_sheet(subtipo)
        if sh is None:
            return False
        aba = f"hist_{liga.lower().replace(' ', '_')}"
        ws = _get_or_create_worksheet(sh, aba)
        ws.clear()
        ws.update(_df_para_sheets(df), value_input_option="RAW")
        return True
    except Exception as e:
        st.error(f"❌ Erro ao salvar histórico: {e}")
        return False


def carregar_historico_drive(liga: str, subtipo: str) -> pd.DataFrame | None:
    """
    Carrega o CSV de histórico (partidas passadas) do Sheets.
    """
    try:
        sh = _get_sheet(subtipo)
        if sh is None:
            return None
        aba = f"hist_{liga.lower().replace(' ', '_')}"
        try:
            ws = sh.worksheet(aba)
        except gspread.WorksheetNotFound:
            return None
        return _sheets_para_df(ws)
    except Exception as e:
        return None


# ─── JOGOS DA RODADA ──────────────────────────────────────────────────────────

def salvar_jogos_drive(df: pd.DataFrame, liga: str, subtipo: str) -> bool:
    """
    Salva o CSV de jogos da rodada no Sheets.
    liga ex: "italia" → aba "jogos_italia"
    """
    try:
        sh = _get_sheet(subtipo)
        if sh is None:
            return False
        aba = f"jogos_{liga.lower().replace(' ', '_')}"
        ws = _get_or_create_worksheet(sh, aba)
        ws.clear()
        ws.update(_df_para_sheets(df), value_input_option="RAW")
        return True
    except Exception as e:
        return False


def carregar_jogos_drive(liga: str, subtipo: str) -> pd.DataFrame | None:
    """
    Carrega o CSV de jogos da rodada do Sheets.
    """
    try:
        sh = _get_sheet(subtipo)
        if sh is None:
            return None
        aba = f"jogos_{liga.lower().replace(' ', '_')}"
        try:
            ws = sh.worksheet(aba)
        except gspread.WorksheetNotFound:
            return None
        return _sheets_para_df(ws)
    except Exception as e:
        return None


# ─── LISTAR LIGAS DISPONÍVEIS NO DRIVE ───────────────────────────────────────

def ligar_ligas_drive(subtipo: str) -> list[str]:
    """
    Retorna lista de ligas que têm histórico salvo no Drive.
    """
    try:
        sh = _get_sheet(subtipo)
        if sh is None:
            return []
        abas = [ws.title for ws in sh.worksheets()]
        return [a.replace("hist_", "") for a in abas if a.startswith("hist_")]
    except Exception:
        return []
