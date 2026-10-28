"""
drive_utils.py — funções compartilhadas para ler/salvar CSVs no Google Drive
Usado por app_chutes.py, app_faltas.py e painel_bets.py
"""

import io
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

# ID da pasta raiz no Drive (compartilhada com a service account)
PASTA_RAIZ_ID = "12fvtXVYLI1rsB_UhACAIVW9GIfShD8G3"

# Subpastas por tipo de modelo
SUBPASTAS = {
    "chutes":    "chutes",
    "faltas_eu": "faltas_eu",
    "faltas_br": "faltas_br",
}


@st.cache_resource
def get_drive_client():
    """Retorna cliente autenticado do Google Drive via secrets do Streamlit."""
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=SCOPES
        )
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"❌ Erro ao conectar ao Google Drive: {e}")
        return None


def _get_drive_service():
    """Retorna o serviço HTTP do Drive (para upload/download de arquivos CSV)."""
    from googleapiclient.discovery import build
    from google.oauth2.service_account import Credentials as SACredentials

    creds = SACredentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


@st.cache_data(ttl=60)
def listar_csvs_drive(subtipo: str) -> list[dict]:
    """
    Lista todos os CSVs na subpasta do Drive para um tipo de modelo.
    Retorna lista de dicts: [{name, id, modifiedTime}]
    subtipo: 'chutes' | 'faltas_eu' | 'faltas_br'
    """
    try:
        service = _get_drive_service()
        # Usar ID direto da pasta raiz
        raiz_id = PASTA_RAIZ_ID

        # Encontrar subpasta
        nome_sub = SUBPASTAS.get(subtipo, subtipo)
        res2 = service.files().list(
            q=f"name='{nome_sub}' and '{raiz_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id,name)"
        ).execute()
        sub_files = res2.get("files", [])
        if not sub_files:
            return []
        sub_id = sub_files[0]["id"]

        # Listar CSVs na subpasta
        res3 = service.files().list(
            q=f"'{sub_id}' in parents and mimeType='text/csv' and trashed=false",
            fields="files(id,name,modifiedTime)",
            orderBy="modifiedTime desc"
        ).execute()
        return res3.get("files", [])

    except Exception as e:
        st.warning(f"⚠️ Não foi possível listar arquivos no Drive: {e}")
        return []


def baixar_csv_drive(file_id: str) -> pd.DataFrame | None:
    """Baixa um CSV do Drive pelo file_id e retorna como DataFrame."""
    try:
        service = _get_drive_service()
        content = service.files().get_media(fileId=file_id).execute()
        return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        st.error(f"❌ Erro ao baixar arquivo do Drive: {e}")
        return None


def salvar_csv_drive(df: pd.DataFrame, nome_arquivo: str, subtipo: str) -> bool:
    """
    Salva (ou atualiza) um CSV no Drive na subpasta correta.
    Se já existir um arquivo com o mesmo nome, substitui.
    Retorna True se sucesso.
    subtipo: 'chutes' | 'faltas_eu' | 'faltas_br'
    """
    try:
        from googleapiclient.http import MediaIoBaseUpload
        service = _get_drive_service()

        # Usar ID direto da pasta raiz
        raiz_id = PASTA_RAIZ_ID

        nome_sub = SUBPASTAS.get(subtipo, subtipo)
        res2 = service.files().list(
            q=f"name='{nome_sub}' and '{raiz_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id)"
        ).execute()
        sub_files = res2.get("files", [])
        if not sub_files:
            # Criar subpasta automaticamente
            pasta_meta = {"name": nome_sub, "mimeType": "application/vnd.google-apps.folder", "parents": [raiz_id]}
            sub_id = service.files().create(body=pasta_meta, fields="id").execute()["id"]
        else:
            sub_id = sub_files[0]["id"]

        # Verificar se já existe arquivo com esse nome
        res3 = service.files().list(
            q=f"name='{nome_arquivo}' and '{sub_id}' in parents and trashed=false",
            fields="files(id)"
        ).execute()
        existentes = res3.get("files", [])

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        media = MediaIoBaseUpload(io.BytesIO(csv_bytes), mimetype="text/csv", resumable=False)

        if existentes:
            # Atualizar arquivo existente
            file_id = existentes[0]["id"]
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            # Criar novo arquivo
            meta = {"name": nome_arquivo, "parents": [sub_id]}
            service.files().create(body=meta, media_body=media, fields="id").execute()

        return True

    except Exception as e:
        st.error(f"❌ Erro ao salvar no Drive: {e}")
        return False


def drive_disponivel() -> bool:
    """Verifica se as credenciais do Drive estão configuradas."""
    return "gcp_service_account" in st.secrets
