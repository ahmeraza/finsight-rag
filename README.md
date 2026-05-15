# FinSight RAG 📊

> **Production-grade RAG pipeline for interrogating SEC earnings filings in natural language — combining multi-query semantic retrieval, noise-filtered chunking, and GPT-4o cited answer generation, built on real financial domain expertise.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-green)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-purple)](https://trychroma.com)
[![OpenAI](https://img.shields.io/badge/GPT--4o-generation-orange)](https://openai.com)
[![SEC EDGAR](https://img.shields.io/badge/SEC%20EDGAR-free%20data-red)](https://data.sec.gov)

---

## Overview

Analysts and investors spend hours manually searching 200-page SEC filings for specific metrics, risk factors, and management commentary. FinSight RAG eliminates this by letting you ask natural-language questions and receive grounded, cited answers traceable back to the exact filing section.

This is not a demo that wraps an LLM around a PDF. It is a full retrieval pipeline with:
- **Format-aware ingestion** that handles both modern HTML filings and legacy PDFs
- **Multi-query retrieval** that rewrites questions 3 ways to maximise recall
- **Noise-filtered chunking** that removes SEC boilerplate before embedding
- **Cited generation** where every factual claim references a specific filing section
- **Persistent caching** at both the raw filing and vector store layers

Built with payments industry expertise (Visa, Mastercard, PayPal, Block) — the questions asked and the edge cases handled reflect real analyst workflows.

---

## Live Demo

![FinSight RAG Demo](assets/demo_output.png)

**Example — revenue query with citations:**

```
Q: What was Visa total net revenue in FY2025 and how did it compare to FY2024?

A: Visa's total net revenue in fiscal year 2025 was $40.0 billion [2],
   representing an 11% increase compared to $35.9 billion in fiscal year 2024 [2, 5].
   Service revenue — the largest component — grew to $17.5B [2], driven by continued
   growth in nominal payments volume. Data processing revenue reached $17.6B [3].

Sources:
  [1] V 10-K (2025-11-06) · section 70
  [2] V 10-K (2025-11-06) · section 77  ← net revenue breakdown table
  [3] V 10-K (2025-11-06) · section 78
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                          │
│                                                                 │
│  SEC EDGAR API (free, no key required)                          │
│       │                                                         │
│       ▼                                                         │
│  index.json ──► picks largest .htm/.pdf (>50KB size filter)    │
│       │                                                         │
│       ▼                                                         │
│  Byte-level format detection (not file extension)               │
│       ├── HTML path: BeautifulSoup → strip scripts/nav/footer   │
│       └── PDF path:  pdfplumber → pypdf fallback                │
│       │                                                         │
│       ▼                                                         │
│  Drive cache (.bin, format-neutral) ◄── zero re-download       │
└─────────────────┬───────────────────────────────────────────────┘
                  │  LangChain Documents + metadata
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CHUNKING LAYER                           │
│                                                                 │
│  Noise filter (regex removes SEC boilerplate)                   │
│       │                                                         │
│       ▼                                                         │
│  RecursiveCharacterTextSplitter                                 │
│    chunk_size=800 tokens · overlap=150 tokens                   │
│    separators: paragraph → line → sentence → word               │
│       │                                                         │
│       ▼                                                         │
│  Meaningfulness filter (min length + alpha ratio ≥15%)          │
│       │                                                         │
│       ▼                                                         │
│  Citation metadata: "V 10-K (2025-11-06) · section 42"        │
└─────────────────┬───────────────────────────────────────────────┘
                  │  155 chunks with citation metadata
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RETRIEVAL LAYER                           │
│                                                                 │
│  OpenAI text-embedding-3-small (1536-dim, $0.002/10-K)         │
│       │                                                         │
│       ▼                                                         │
│  ChromaDB (persisted to Drive — zero re-embedding on restart)   │
│       │                                                         │
│       ▼                                                         │
│  MultiQueryRetriever                                            │
│    gpt-4o-mini rewrites question 3 ways                         │
│    runs 3 parallel similarity searches                          │
│    deduplicates → returns top-6 diverse chunks                  │
└─────────────────┬───────────────────────────────────────────────┘
                  │  6 most relevant chunks
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GENERATION LAYER                           │
│                                                                 │
│  Chunks formatted as [1][2][3]... numbered excerpts             │
│       │                                                         │
│       ▼                                                         │
│  GPT-4o (temperature=0, max_tokens=1024)                        │
│  System prompt enforces: cite [N], refuse to guess,             │
│  bold key metrics, stay under 300 words                         │
│       │                                                         │
│       ▼                                                         │
│  { answer with inline citations, sources list, question }       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Advanced AI/ML Techniques

| Technique | Implementation | Why it matters |
|-----------|---------------|----------------|
| **Multi-query retrieval** | `MultiQueryRetriever` + gpt-4o-mini | "Revenue grew" and "sales increased" embed differently but mean the same — multi-query finds both |
| **Recursive chunking** | `RecursiveCharacterTextSplitter` with paragraph → sentence hierarchy | Keeps financially related sentences together; never splits mid-table |
| **Noise filtering** | Regex patterns for SEC boilerplate + alpha-ratio filter | Prevents page headers and TOC entries from polluting the vector space |
| **Semantic deduplication** | Post-retrieval dedup across 3 query variants | Returns diverse context, not 6 copies of the same paragraph |
| **Cited generation** | Numbered excerpts in prompt + citation rules in system prompt | Every figure traceable to source — eliminates hallucination risk |
| **Format-agnostic parsing** | Byte-level HTML/PDF detection before parsing | Handles the real-world mix of HTML (modern) and PDF (legacy) SEC filings |
| **Two-tier caching** | Raw filing cache (Drive) + vector cache (ChromaDB) | Zero re-download and zero re-embedding on session restart |
| **Deterministic generation** | `temperature=0` for GPT-4o | Financial figures must not vary between runs — auditability requirement |

**Phase 2 (in progress):** Cross-encoder reranker · FinBERT sentiment on MD&A · sentiment vs. price correlation

---

## Business Use Cases

**Buy-side analysts**
Compare management commentary tone across quarters. Ask "Did management express concern about consumer spending?" across 4 consecutive 10-Qs to identify trend inflection points before they appear in price action.

**Compliance teams**
Rapidly surface risk factor changes between filings. "What new regulatory risks did Visa add in the most recent 10-K compared to last year?" — work that previously took hours now takes seconds.

**Fintech researchers**
Cross-company metric comparison. Index Visa, Mastercard, and PayPal simultaneously and ask "Which company showed the strongest cross-border volume growth in FY2025?"

**Portfolio managers**
Pre-earnings preparation. Ingest the most recent 10-Q before an earnings call to understand the baseline management is being measured against — and surface the specific guidance language that analyst consensus is modelling.

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Data source | SEC EDGAR API | Free official filing data, no API key required |
| HTML parsing | BeautifulSoup 4 | Modern SEC HTML filing extraction |
| PDF parsing | pdfplumber + pypdf | Legacy filing extraction with table support |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Semantics-aware text splitting |
| Embeddings | OpenAI `text-embedding-3-small` | 1536-dim dense vectors |
| Vector store | ChromaDB | Local persistent vector database |
| Retrieval | LangChain `MultiQueryRetriever` | Multi-phrasing semantic search |
| Rewriting LLM | GPT-4o-mini | Query reformulation (~$0.001/query) |
| Generation LLM | GPT-4o | Cited answer synthesis (~$0.01/query) |
| Retry logic | tenacity | Exponential backoff for SEC API calls |
| Orchestration | LangChain LCEL | Composable retrieval → generation pipeline |

---

## Project Structure

```
finsight-rag/
├── notebooks/
│   ├── ingestor_final.ipynb      # 1 of 4: SEC EDGAR fetcher + HTML/PDF parser
│   ├── chunker_final.ipynb       # 2 of 4: noise filter + 800-token chunks
│   ├── retriever_final.ipynb     # 3 of 4: ChromaDB indexing + MultiQueryRetriever
│   └── chain_final.ipynb         # 4 of 4: GPT-4o cited answer generation
├── assets/
│   └── demo_output.png           # Live output screenshot
├── requirements.txt
├── .env.example
└── README.md
```

Each notebook is **self-contained** — it includes all imports, helper functions, and a health check cell that verifies correct output before passing to the next stage. Full inline documentation explains every design decision.

---

## Quickstart

**Prerequisites:** Google account · OpenAI API key with billing enabled (~$5 minimum)

### 1 — Clone the repo

```bash
git clone https://github.com/ahmeraza/finsight-rag
```

Upload the `notebooks/` folder to Google Drive, then open each in Colab.

### 2 — Configure secrets

In Colab, click the 🔑 key icon (left sidebar) and add:

```
OPENAI_API_KEY  →  sk-...
SEC_USER_AGENT  →  FinSight YourName yourname@email.com
```

The `SEC_USER_AGENT` is required by SEC's fair-use policy — without it you receive `403 Forbidden`.

### 3 — Run notebooks in order

| Notebook | What it does | Output |
|----------|-------------|--------|
| `ingestor_final.ipynb` | Downloads Visa 10-K from SEC EDGAR | `.bin` cached to Drive |
| `chunker_final.ipynb` | Splits into retrieval-ready chunks | 155 Documents |
| `retriever_final.ipynb` | Indexes into ChromaDB | 155 vectors on Drive |
| `chain_final.ipynb` | Builds `ask()` function | Cited answers |

Each notebook ends with a `🎉` health check. Only proceed when it passes.

### 4 — Query

```python
result = ask("What was Visa total net revenue in FY2025?")
print(result['answer'])   # cited answer with [1][2] references
print(result['sources'])  # exact filing sections used
```

---

## Key Design Decisions

| Decision | Alternative considered | Why this approach |
|----------|----------------------|-------------------|
| Cache as `.bin` not `.pdf` | Save as `.pdf` | SEC files HTML as `.htm` — saving as `.pdf` caused both parsers to crash on format mismatch |
| `index.json` API for file discovery | Regex scrape of index HTML | JSON includes file sizes — essential for picking the 500KB main report over 2KB stub exhibits |
| Size filter `>50KB` | No filter | Cover pages and XBRL exhibits are 1–5KB; the main 10-K body is always 500KB+ |
| `temperature=0` for GPT-4o | Default 0.7 | Financial figures used in investment decisions must not vary between identical runs |
| Cache validity `>100KB` | Trust file existence | Failed downloads produce tiny files — size check prevents re-using bad cache |
| Byte-level format detection | Trust file extension | SEC serves HTML with no extension or wrong extension — byte sniffing is the only reliable approach |
| Alphabetic ratio filter `≥15%` | Length filter only | Raw number grids from broken PDF extraction pass length checks but fail alpha ratio — correctly rejected |

---

## Roadmap

### Phase 1 — Core RAG (complete ✅)
- SEC EDGAR ingestion with HTML + PDF auto-detection
- Two-tier caching: raw filing (Drive) + vector store (ChromaDB)
- 800-token chunks with noise filtering and citation metadata
- MultiQueryRetriever with GPT-4o cited answer generation
- 4 self-documented Colab notebooks with health checks at each stage

### Phase 2 — Advanced ML (in progress 🔄)
- **Cross-encoder reranker** — `ms-marco-MiniLM` re-scores top-20 chunks, keeps top-5 — cuts noise significantly
- **FinBERT sentiment** — `ProsusAI/finbert` scores MD&A section sentence-by-sentence (Positive/Neutral/Negative)
- **Sentiment vs. price** — correlate filing sentiment score with next-day stock movement using `yfinance`
- **Streamlit UI** — shareable web app with ticker selector, query input, cited answer panel, and FinBERT badge

### Phase 3 — Agentic Layer
- Tool-calling agent: `search_filings()`, `get_price_data()`, `calculate_metric()`
- Multi-company comparison: index Visa + Mastercard + PayPal simultaneously
- Conversational memory: multi-turn follow-up questions within a session
- HuggingFace Spaces deployment for a publicly accessible demo

---

## Cost

| Item | Cost |
|------|------|
| SEC EDGAR data | Free |
| ChromaDB | Free (local) |
| OpenAI embeddings — 155 chunks, one-time | ~$0.002 |
| GPT-4o-mini — query rewriting per question | ~$0.001 |
| GPT-4o — answer generation per question | ~$0.01 |
| **Total for full development** | **~$2–5** |

---

## What makes this different from a generic RAG demo

Most RAG demos: upload PDF → chunk by character count → ask question → get answer.

FinSight: SEC EDGAR API → byte-level format detection → noise-filtered chunking → multi-query retrieval → cited answer where every claim references a specific filing section — built on payments domain expertise so the questions and edge cases reflect real analyst workflows.

The gap between a demo and a tool someone would actually trust with financial decisions is: cited sources, deterministic output, graceful degradation when context is missing, and an ingestion pipeline that handles the messy reality of how SEC actually publishes filings in 2025 (spoiler: it's HTML, not PDF, and the file you want is buried in a 30-file package).

---

*Built by Ahmed Raza · [ahmeraza.github.io](https://ahmeraza.github.io) · Fintech & Payments background (Checkout.com, Fuze)*
