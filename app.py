"""
app.py — FinSight RAG Streamlit UI
===================================
Wires Phase 1 (cited RAG answers) and Phase 2 (FinBERT sentiment) together
into a shareable web application.

Run locally:  streamlit run app.py
Deploy:       streamlit community cloud (connect to GitHub repo)

Layout:
  Sidebar  — ticker selector, filing ingestion, retrieval settings
  Main     — query input → cited answer → FinBERT sentiment badge → source expander
"""

import os
import re
import io
import time
import json
import logging
import requests
import functools
from pathlib import Path
from collections import Counter

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="FinSight RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — refined dark-financial aesthetic ─────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Answer box */
  .answer-box {
    background: #0f1117;
    border: 1px solid #1e2530;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    font-size: 0.95rem;
    line-height: 1.75;
    color: #e2e8f0;
    margin: 1rem 0;
  }

  /* Sentiment badge */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.02em;
  }
  .badge-positive { background: #052e16; color: #4ade80; border: 1px solid #166534; }
  .badge-negative { background: #2d0808; color: #f87171; border: 1px solid #7f1d1d; }
  .badge-neutral  { background: #1c1f26; color: #94a3b8; border: 1px solid #334155; }

  /* Source card */
  .source-card {
    background: #0d1117;
    border: 1px solid #1e2530;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
    font-family: 'DM Mono', monospace;
    color: #64748b;
  }
  .source-card .citation { color: #3b82f6; font-weight: 500; }
  .source-card .snippet  { color: #475569; margin-top: 4px; font-size: 0.78rem; }

  /* Metric strip */
  .metric-strip {
    display: flex;
    gap: 1rem;
    margin: 0.75rem 0;
    flex-wrap: wrap;
  }
  .metric-item {
    background: #0d1117;
    border: 1px solid #1e2530;
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #64748b;
  }
  .metric-item span { color: #e2e8f0; font-weight: 500; }

  /* Hide Streamlit branding */
  #MainMenu, footer { visibility: hidden; }
  header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Lazy imports (expensive libraries loaded only when needed) ─────────────────

@st.cache_resource(show_spinner="Loading FinBERT model (~440MB on first run)...")
def _load_finbert():
    """Load ProsusAI/finbert — cached for the session lifetime."""
    from transformers import pipeline
    return pipeline(
        task="text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        return_all_scores=True,
        device=-1,
    )


@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def _load_vector_store():
    """Load ChromaDB from the persist directory."""
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma(
        collection_name="finsight_filings",
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )
    return store


# ── Ingestor helpers (condensed from ingestor_final.ipynb) ────────────────────

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", "FinSight-RAG dev@example.com")}
TICKER_TO_CIK = {"V": 1403161, "MA": 1141391, "PYPL": 1633917, "SQ": 1512673}


def _get(url: str) -> requests.Response:
    time.sleep(0.15)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp


def ingest_and_index(ticker: str, form_type: str = "10-K", max_filings: int = 1) -> int:
    """
    Full pipeline: fetch → parse → chunk → embed → index into ChromaDB.
    Returns the number of chunks indexed.
    """
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from bs4 import BeautifulSoup

    cik = TICKER_TO_CIK.get(ticker.upper())
    if not cik:
        raise ValueError(f"Unknown ticker '{ticker}'")

    # Fetch filing metadata
    data   = _get(EDGAR_SUBMISSIONS_URL.format(cik=cik)).json()
    recent = data.get("filings", {}).get("recent", {})
    forms, accessions, dates = (
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("filingDate", []),
    )

    all_docs = []
    count = 0
    for form, accession, filing_date in zip(forms, accessions, dates):
        if form != form_type or count >= max_filings:
            continue
        count += 1

        acc_clean = accession.replace("-", "")
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{accession}-index.htm"

        # Find main document via index.json
        try:
            files = _get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/index.json"
                         ).json().get("directory", {}).get("item", [])
            candidates = [f for f in files
                          if f["name"].endswith((".htm", ".pdf"))
                          and not f["name"].startswith("R")
                          and "index" not in f["name"].lower()
                          and int(f.get("size", 0)) > 50_000]
            if not candidates:
                continue
            best    = max(candidates, key=lambda f: int(f.get("size", 0)))
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{best['name']}"
        except Exception as e:
            logger.error("Failed to find doc URL: %s", e)
            continue

        raw_bytes = _get(doc_url).content
        sample    = raw_bytes[:500].decode("utf-8", errors="ignore").lower()

        if any(m in sample for m in ["<html", "<!doctype", "<document"]):
            soup = BeautifulSoup(raw_bytes, "html.parser")
            for tag in soup(["script", "style", "head", "nav", "footer"]):
                tag.decompose()
            text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n")).strip()
            sections = [text[i:i+3000] for i in range(0, len(text), 3000)]
            for i, sec in enumerate(sections, 1):
                if len(sec.strip()) > 100:
                    all_docs.append(Document(
                        page_content=sec,
                        metadata={"ticker": ticker, "form_type": form,
                                  "filing_date": filing_date, "source": index_url,
                                  "page": i,
                                  "citation": f"{ticker} {form} ({filing_date}) · section {i}"}
                    ))

    if not all_docs:
        return 0

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3200, chunk_overlap=600,
        separators=["\n\n", "\n", ". ", " ", ""], length_function=len)
    chunks = []
    for doc in all_docs:
        for idx, text in enumerate(splitter.split_text(doc.page_content)):
            if len(text.strip()) >= 80:
                alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
                if alpha >= 0.15:
                    chunks.append(Document(
                        page_content=text,
                        metadata={**doc.metadata, "chunk_index": idx}
                    ))

    # Index
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="finsight_filings",
        persist_directory=chroma_dir,
    )
    _load_vector_store.clear()  # bust cache so UI reloads fresh store
    return len(chunks)


# ── RAG chain ──────────────────────────────────────────────────────────────────

def ask(question: str, ticker_filter: str | None = None) -> dict:
    """Run the full RAG chain and return {answer, sources, question}."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever

    store = _load_vector_store()

    search_kwargs = {"k": 6}
    if ticker_filter and ticker_filter != "All":
        search_kwargs["filter"] = {"ticker": ticker_filter}

    base_retriever = store.as_retriever(
        search_type="similarity", search_kwargs=search_kwargs
    )
    rewrite_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever   = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=rewrite_llm
    )

    docs = retriever.invoke(question)

    # Format numbered excerpts
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        parts.append(
            f"[{i}] {m.get('ticker','?')} {m.get('form_type','?')} "
            f"({m.get('filing_date','?')}) · Section {m.get('page','?')}\n"
            f"{doc.page_content[:1200]}"
        )
    context = "\n\n---\n\n".join(parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are FinSight, an expert financial analyst specialising in SEC filings.\n"
         "Answer ONLY from the provided context. Cite every claim with [N].\n"
         "If context is insufficient, say so. Never fabricate figures.\n"
         "Format numbers: $B billions, $M millions, % percentages.\n"
         "Bold key metrics with **metric**."),
        ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"),
    ])
    chain  = prompt | ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1024) | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    sources = [
        {"citation":    doc.metadata.get("citation", ""),
         "ticker":      doc.metadata.get("ticker", ""),
         "form_type":   doc.metadata.get("form_type", ""),
         "filing_date": doc.metadata.get("filing_date", ""),
         "page":        doc.metadata.get("page", ""),
         "snippet":     doc.page_content[:280] + "…"}
        for doc in docs
    ]
    return {"question": question, "answer": answer, "sources": sources}


# ── FinBERT sentiment ──────────────────────────────────────────────────────────

def score_text(text: str) -> dict:
    """Score a text with FinBERT. Returns label, avg_score, signed_score."""
    pipe      = _load_finbert()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) >= 20]
    if not sentences:
        return {"label": "neutral", "avg_score": 0.0, "signed_score": 0.0, "count": 0}

    results = []
    for i in range(0, len(sentences), 32):
        batch        = sentences[i:i+32]
        batch_output = pipe(batch, truncation=True, max_length=512)
        for result_list in batch_output:
            # Handle both list of dicts and single dict output formats
            if isinstance(result_list, dict):
                result_list = [result_list]
            best   = max(result_list, key=lambda x: x["score"])
            label  = best["label"].lower()
            score  = best["score"]
            signed = score if label == "positive" else (-score if label == "negative" else 0.0)
            results.append({"label": label, "score": score, "signed": signed})

    avg_score  = sum(r["score"]  for r in results) / len(results)
    signed_avg = sum(r["signed"] for r in results) / len(results)
    counts     = Counter(r["label"] for r in results)
    label      = counts.most_common(1)[0][0]

    return {
        "label":        label,
        "avg_score":    avg_score,
        "signed_score": signed_avg,
        "count":        len(results),
        "breakdown":    dict(counts),
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ FinSight")
    st.caption("SEC Filing Intelligence · Phase 2")
    st.divider()

    st.markdown("### Ingest a filing")
    ingest_ticker   = st.selectbox("Ticker", list(TICKER_TO_CIK.keys()), key="ingest_ticker")
    ingest_form     = st.selectbox("Form type", ["10-K", "10-Q"])
    ingest_count    = st.slider("Max filings", 1, 3, 1)

    if st.button("⬇ Fetch & index", use_container_width=True):
        if not os.getenv("OPENAI_API_KEY", "").startswith("sk-"):
            st.error("Set OPENAI_API_KEY in your .env file first.")
        else:
            with st.spinner(f"Ingesting {ingest_form} for {ingest_ticker}..."):
                try:
                    n = ingest_and_index(ingest_ticker, ingest_form, ingest_count)
                    st.success(f"Indexed {n} chunks.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()
    st.markdown("### Settings")
    ticker_filter  = st.selectbox("Filter by company", ["All"] + list(TICKER_TO_CIK.keys()))
    run_sentiment  = st.toggle("FinBERT sentiment badge", value=True)
    show_sources   = st.toggle("Show source chunks",      value=True)
    st.divider()
    st.caption("Built by Ahmed Raza · [ahmeraza.github.io](https://ahmeraza.github.io)")


# ── Main area ──────────────────────────────────────────────────────────────────

st.markdown("# 📊 FinSight RAG")
st.markdown(
    "Ask plain-English questions about SEC 10-K and 10-Q filings. "
    "Every answer is grounded in the actual filing text with inline citations."
)
st.divider()

# Check if API key is set
if not os.getenv("OPENAI_API_KEY", "").startswith("sk-"):
    st.warning("Add your `OPENAI_API_KEY` to a `.env` file in the project root, then restart.")
    st.stop()

# Query input
col_q, col_btn = st.columns([5, 1])
with col_q:
    question = st.text_input(
        "Question",
        placeholder="e.g. What was Visa's net revenue in FY2025 and how did it compare to FY2024?",
        label_visibility="collapsed",
    )
with col_btn:
    ask_btn = st.button("Ask →", type="primary", use_container_width=True)

# Example questions
with st.expander("💡 Example questions"):
    examples = [
        "What was Visa's total net revenue in FY2025?",
        "How did cross-border transaction volume change year over year?",
        "What risk factors related to regulation did management highlight?",
        "What was the impact of the interchange litigation on operating expenses?",
        "Did management express confidence or concern about consumer spending?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            question = ex
            ask_btn  = True

# ── Run query ──────────────────────────────────────────────────────────────────

if ask_btn and question:

    # Retrieval
    with st.spinner("Retrieving relevant filing sections..."):
        try:
            result  = ask(question, ticker_filter=ticker_filter)
            answer  = result["answer"]
            sources = result["sources"]
        except Exception as e:
            st.error(f"Query failed: {e}")
            logger.exception("Query error")
            st.stop()

    # Answer
    st.markdown("### Answer")
    st.markdown(
        f'<div class="answer-box">{answer}</div>',
        unsafe_allow_html=True,
    )

    # FinBERT sentiment badge
    if run_sentiment and sources:
        with st.spinner("Running FinBERT sentiment..."):
            try:
                combined = " ".join(s["snippet"] for s in sources[:4])
                sent     = score_text(combined)

                badge_class = {
                    "positive": "badge-positive",
                    "negative": "badge-negative",
                    "neutral":  "badge-neutral",
                }.get(sent["label"], "badge-neutral")

                emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(sent["label"], "")
                label = sent["label"].capitalize()

                breakdown_str = " · ".join(
                    f"{k}: {v}" for k, v in sorted(sent.get("breakdown", {}).items())
                )

                st.markdown(
                    f"""
                    <div class="metric-strip">
                      <div class="badge {badge_class}">{emoji} FinBERT: {label}</div>
                      <div class="metric-item">Confidence: <span>{sent['avg_score']*100:.0f}%</span></div>
                      <div class="metric-item">Signed score: <span>{sent['signed_score']:+.4f}</span></div>
                      <div class="metric-item">Sentences: <span>{sent['count']}</span></div>
                      <div class="metric-item">{breakdown_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.caption(f"FinBERT unavailable: {e}")

    # Sources
    if show_sources and sources:
        with st.expander(f"📄 {len(sources)} source chunks"):
            for i, src in enumerate(sources, 1):
                st.markdown(
                    f"""
                    <div class="source-card">
                      <div class="citation">[{i}] {src['citation']}</div>
                      <div class="snippet">{src['snippet']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

elif ask_btn and not question:
    st.warning("Enter a question above.")
