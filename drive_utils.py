"""
drive_utils.py — funções compartilhadas para ler/salvar dados no Google Sheets
Usado por app_chutes.py, app_faltas.py, app_faltas_br.py e painel_bets.py

Estrutura de planilhas:
  - "bets_chutes"    → uma aba por liga (ex: "england", "spain")
  - "bets_faltas_eu" → uma aba por liga
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

# Nome das planilhas no Google Sheets (serão criadas automaticamente)
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


def _get_or_create_sheet(client, subtipo: str):
    """Retorna a planilha do subtipo, criando se não existir."""
    nome = PLANILHAS.get(subtipo, f"bets_{subtipo}")
    try:
        return client.open(nome)
    except gspread.SpreadsheetNotFound:
        sh = client.create(nome)
        # Compartilhar com o dono (email do usuário nas secrets, se disponível)
        try:
            email = st.secrets["gcp_service_account"].get("user_email", "")
            if email:
                sh.share(email, perm_type="user", role="writer")
        except Exception:
            pass
        return sh


def salvar_csv_drive(df: pd.DataFrame, nome_arquivo: str, subtipo: str) -> bool:
    """
    Salva o DataFrame numa aba da planilha Google Sheets.
    O nome da aba é derivado do nome_arquivo (ex: "model_italiaFALTAS.csv" → "italia").
    Substitui os dados existentes na aba.
    """
    try:
        client = get_drive_client()
        if client is None:
            return False

        sh = _get_or_create_sheet(client, subtipo)

        # Derivar nome da aba a partir do nome do arquivo
        aba = nome_arquivo.replace(".csv", "")

        try:
            ws = sh.worksheet(aba)
            ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=aba, rows=max(len(df) + 10, 100), cols=max(len(df.columns) + 5, 30))

        # Escrever cabeçalho + dados (substituir NaN por string vazia)
        df_clean = df.fillna("").astype(str).replace("nan", "")
        dados = [df_clean.columns.tolist()] + df_clean.values.tolist()
        ws.update(dados, value_input_option="RAW")
        return True

    except Exception as e:
        st.error(f"❌ Erro ao salvar no Drive: {e}")
        return False


def listar_abas_drive(subtipo: str) -> list[str]:
    """Lista as abas disponíveis na planilha do subtipo."""
    try:
        client = get_drive_client()
        if client is None:
            return []
        sh = _get_or_create_sheet(client, subtipo)
        return [ws.title for ws in sh.worksheets() if ws.title != "Sheet1"]
    except Exception:
        return []


def carregar_csv_drive(nome_aba: str, subtipo: str) -> pd.DataFrame | None:
    """Carrega dados de uma aba da planilha como DataFrame."""
    try:
        client = get_drive_client()
        if client is None:
            return None
        sh = _get_or_create_sheet(client, subtipo)
        ws = sh.worksheet(nome_aba)
        dados = ws.get_all_records()
        if not dados:
            return None
        return pd.DataFrame(dados)
    except Exception as e:
        st.error(f"❌ Erro ao carregar do Drive: {e}")
        return None
