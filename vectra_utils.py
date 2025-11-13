import json
from math import sqrt

import gspread
from openai import OpenAI
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


def parse_serp_result(value):
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def extract_references_from_df(df):
    serp_col = "serpapi_results"
    if df is None or serp_col not in df.columns:
        return pd.DataFrame()

    serp_data = df[serp_col].apply(parse_serp_result)
    rows = []
    queries = df.get("query", pd.Series())
    lookups = df.get("lookup_query", pd.Series())
    types = df.get("type", pd.Series())

    for idx, query in enumerate(queries.tolist()):
        payload = serp_data.iloc[idx] if idx < len(serp_data) else None
        if isinstance(payload, dict):
            overview = payload.get("ai_overview") if isinstance(payload.get("ai_overview"), dict) else None
            references = overview.get("references") if isinstance(overview, dict) else None
        else:
            references = None
        if isinstance(references, list):
            for ref in references:
                if isinstance(ref, dict):
                    rows.append({
                        "lookup_query": lookups.iloc[idx] if idx < len(lookups) else None,
                        "query": query,
                        "type": types.iloc[idx] if idx < len(types) else None,
                        "title": ref.get("title"),
                        "link": ref.get("link"),
                        "source": ref.get("source"),
                        "snippet": ref.get("snippet")
                    })
    return pd.DataFrame(rows)


def load_csvs_from_files(files):
    frames = []
    for file in files:
        try:
            frames.append(pd.read_csv(file))
        except Exception:
            continue
    if frames:
        return pd.concat(frames, ignore_index=True)
    return None


def get_page_text(url, fallback=""):
    cache = st.session_state.setdefault("page_text_cache", {})
    if not url:
        return fallback or ""
    if url in cache:
        return cache[url]
    headers = {"User-Agent": "Vectra-Embedding-Analyzer/1.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "h1", "h2", "h3"]) if p.get_text(strip=True)]
        text = "\n".join(paragraphs[:40])
    except Exception:
        text = ""
    if not text:
        text = fallback or ""
    cache[url] = text
    print(text)
    return text


def embed_texts(texts, api_key, model="text-embedding-ada-002"):
    if not texts:
        return []
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def sync_to_google_sheets(creds_json, spreadsheet_id, dataframes):
    creds_dict = json.loads(creds_json)
    client = gspread.service_account_from_dict(creds_dict)
    sh = client.open_by_key(spreadsheet_id)

    existing_sheets = {ws.title: ws for ws in sh.worksheets()}

    for sheet_name, df in dataframes.items():
        if df is None or df.empty:
            continue
        worksheet = existing_sheets.get(sheet_name)
        if worksheet is None:
            worksheet = sh.add_worksheet(title=sheet_name, rows=str(len(df) + 10), cols=str(len(df.columns) + 5))
        worksheet.clear()
        values = [df.columns.tolist()] + df.fillna("").astype(str).values.tolist()
        worksheet.update(values)
