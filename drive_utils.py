"""
drive_utils.py — funções compartilhadas para ler/salvar CSVs no Google Sheets
Usado por app_chutes.py, app_faltas.py, app_faltas_br.py e painel_bets.py

Estrutura de planilhas:
  - "bets_chutes"    → uma aba por liga (ex: "model_england")
  - "bets_faltas_eu" → uma aba por liga (ex: "model_italiaFALTAS")
  - "bets_faltas_br" → uma aba por liga
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


def salvar_csv_drive(df: pd.DataFrame, nome_arquivo: str, subtipo: str) -> bool:
    """
    Salva o DataFrame numa aba da planilha Google Sheets.
    O nome da aba é derivado do nome_arquivo (ex: "model_italiaFALTAS.csv" → "model_italiaFALTAS").
    Substitui os dados existentes na aba.
    """
    try:
        sh = _get_sheet(subtipo)
        if sh is None:
            return False

        aba = nome_arquivo.replace(".csv", "")

        try:
            ws = sh.worksheet(aba)
            ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=aba, rows=max(len(df) + 10, 100), cols=max(len(df.columns) + 5, 30))

        df_clean = df.fillna("").astype(str).replace("nan", "").replace("None", "")
        dados = [df_clean.columns.tolist()] + df_clean.values.tolist()
        ws.update(dados, value_input_option="RAW")
        return True

    except Exception as e:
        st.error(f"❌ Erro ao salvar no Drive: {e}")
        return False
