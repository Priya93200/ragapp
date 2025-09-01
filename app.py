"""
Streamlit RAG app with FAISS + local sentence-transformers embeddings + Gemini for final generation.

Enhanced UI with modern layout, styled containers, and better feedback.
"""

import os
import json
import logging
from typing import List, Dict, Tuple
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import google.generativeai as genai

# ---------------- CONFIG ----------------
API_KEY = "AIzaSyDfkpTE45HQ0cdVpdQeWEwXjW43hp1KiMc"   # replace with your Gemini API key
genai.configure(api_key=API_KEY)

INDEX_DIR = "./rag_index"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 6

os.makedirs(INDEX_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Embedding model ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

embedder = load_embedder()

# ---------------- Utility functions ----------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.replace("\r", " ").split())

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    L = len(words)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == L:
            break
        start = max(end - overlap, start + 1)
    return chunks

def extract_text_from_pdf(file) -> str:
    try:
        pdf = PdfReader(file)
        return "\n".join([p.extract_text() or "" for p in pdf.pages])
    except Exception as e:
        logger.error("PDF read error: %s", e)
        return ""

def extract_text_from_docx(file) -> str:
    try:
        doc = DocxDocument(file)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        logger.error("DOCX read error: %s", e)
        return ""

def extract_text_from_csv(file) -> str:
    try:
        df = pd.read_csv(file)
        rows = df.astype(str).apply(lambda r: " | ".join(r.values), axis=1).tolist()
        return "\n".join(rows)
    except Exception as e:
        logger.error("CSV read error: %s", e)
        return ""

# ---------------- Persistence helpers ----------------
def save_metadata(metadata: List[Dict]):
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_metadata() -> List[Dict]:
    if not os.path.exists(METADATA_PATH):
        return []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_faiss_index(index: faiss.IndexFlatL2):
    faiss.write_index(index, FAISS_INDEX_PATH)

def load_faiss_index(dim: int) -> faiss.IndexFlatL2:
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    else:
        return faiss.IndexFlatL2(dim)

def clear_index():
    """Delete FAISS index and metadata files."""
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)
    return True

# ---------------- Build / Update Index ----------------
def embed_texts_local(texts: List[str]) -> np.ndarray:
    emb = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.array(emb, dtype="float32")

def build_or_update_index(chunks_with_meta: List[Tuple[str, Dict]]):
    logger.info("Updating FAISS index with %d chunks", len(chunks_with_meta))
    if not chunks_with_meta:
        return None, []

    texts = [c for c, m in chunks_with_meta]
    metas_new = [m for c, m in chunks_with_meta]

    vectors = embed_texts_local(texts)
    dim = vectors.shape[1]

    metadata_all = load_metadata()
    if os.path.exists(FAISS_INDEX_PATH):
        idx = faiss.read_index(FAISS_INDEX_PATH)
        idx.add(vectors)
        metadata_all.extend(metas_new)
    else:
        idx = faiss.IndexFlatL2(dim)
        idx.add(vectors)
        metadata_all = metas_new.copy()

    save_faiss_index(idx)
    save_metadata(metadata_all)
    logger.info("Index updated. total_chunks=%d", len(metadata_all))
    return idx, metadata_all

# ---------------- Retrieval ----------------
def retrieve_top_k(idx: faiss.IndexFlatL2, metadata: List[Dict], query: str, k: int = TOP_K) -> List[Dict]:
    qvec = embed_texts_local([query])
    D, I = idx.search(qvec, k)
    results = []
    for j, idx_pos in enumerate(I[0]):
        if idx_pos < 0 or idx_pos >= len(metadata):
            continue
        meta = metadata[idx_pos].copy()
        meta["score"] = float(D[0][j])
        results.append(meta)
    return results

# ---------------- Gemini call ----------------
def call_gemini_once(prompt: str, model_choice: str = "models/gemini-1.5-flash"):
    try:
        model = genai.GenerativeModel(model_choice)
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        logger.error("Gemini call error: %s", e)
        return f"[Error calling Gemini: {e}]"

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="📊 RAG - Supply Chain Risk Chatbot", layout="wide")
st.title("📦 Supply Chain Risk Chatbot")
st.markdown("**RAG with FAISS + Local Embeddings + Gemini**")

# Sidebar - Upload & Index
with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded = st.file_uploader("Choose files", accept_multiple_files=True, type=['csv','pdf','docx'])

    if st.button("⚡ Index uploaded files"):
        all_chunks_meta = []
        for f in uploaded:
            name = f.name
            ext = name.lower().split('.')[-1]
            if ext == "pdf":
                raw = extract_text_from_pdf(f)
            elif ext == "docx":
                raw = extract_text_from_docx(f)
            elif ext == "csv":
                raw = extract_text_from_csv(f)
            else:
                raw = ""

            raw = clean_text(raw)
            if not raw:
                st.warning(f"No text extracted from {name}")
                continue

            chunks = chunk_text(raw)
            for i, chunk in enumerate(chunks):
                meta = {"source": name, "chunk_id": i, "text_preview": chunk[:300]}
                all_chunks_meta.append((chunk, meta))

        if all_chunks_meta:
            with st.spinner("🔎 Embedding & indexing..."):
                idx, metadata_all = build_or_update_index(all_chunks_meta)
            st.success(f"✅ Indexed {len(all_chunks_meta)} new chunks. Total = {len(metadata_all)}")
        else:
            st.info("No chunks to index.")

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        metadata_existing = load_metadata()
        st.markdown(f"**📌 Indexed chunks:** {len(metadata_existing)}")

        if st.button("🗑 Clear Index"):
            cleared = clear_index()
            if cleared:
                st.success("✅ Index cleared successfully.")

# Main - Query
st.header("💡 Ask a Question")
query = st.text_area("Type your question here:", height=100, placeholder="e.g. What are the biggest supplier risks?")
num_context = st.slider("🔍 Number of retrieved chunks", 1, 50, 10)

if st.button("🚀 Get Answer"):
    if not query.strip():
        st.warning("❗ Please type a question first.")
    elif not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        st.warning("❗ No index found. Upload and index documents first.")
    else:
        metadata_all = load_metadata()
        idx = faiss.read_index(FAISS_INDEX_PATH)

        retr = retrieve_top_k(idx, metadata_all, query, k=num_context)
        context_lines = [f"[{i}] {r.get('source')} — {r.get('text_preview','')}" for i, r in enumerate(retr, start=1)]
        context_block = "\n\n".join(context_lines)

        final_prompt = f"""
You are a Supply Chain Risk Analysis assistant. Use ONLY the provided context to answer. 
If information is missing, say so.

Context:
{context_block}

Question:
{query}

Return:
1) Short answer (3-5 sentences)
2) Risk rating 1-5 with one-line justification
"""
        with st.spinner("🤖 Generating answer with Gemini..."):
            answer = call_gemini_once(final_prompt)

        st.subheader("📘 Answer")
        st.markdown(answer)
