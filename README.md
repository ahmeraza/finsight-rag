# FinSight RAG 📊

> **A RAG system for interrogating SEC earnings filings in natural language — with multi-query retrieval, ChromaDB vector search, and GPT-4o cited answers.**

Built for fintech analysts who need to ask plain-English questions across 10-K and 10-Q filings from companies like Visa, Mastercard, and PayPal, without manually hunting through 200-page PDFs.

---

## Live demo output

```
Q: What was Visa total net revenue in fiscal year 2025 and how did it compare to 2024?

A: Visa's total net revenue in fiscal year 2025 was $40.0 billion [2],
   representing an 11% increase compared to $35.9 billion in fiscal year 2024 [2, 5].
   Service revenue grew to $17.5B [2], driven by continued growth in payments volume.

Sources:
  [1] V 10-K (2025-11-06) · section 70
  [2] V 10-K (2025-11-06) · section 77
```

# Live Demo

## Example Questions & Cited Responses

The system retrieves relevant SEC filing sections from EDGAR, performs semantic search using ChromaDB, and generates grounded answers with source citations.

### Example Queries

- What were the main risk factors Visa highlighted related to regulation or geopolitics?
- How did cross-border transaction volume change year over year?
- What was Visa operating income and operating margin in FY2025?

### Example Output

![FinSight RAG Demo](assets/demo_output.png)

---

## What it does

| Feature | How it works |
|---|---|
| **Cited answers** | Every factual claim is tagged with the filing, date, and section number |
| **Multi-query retrieval** | LLM rewrites your question 3 ways, runs all searches, deduplicates — better recall on ambiguous questions |
| **ChromaDB vector store** | Embeddings persisted locally — no re-indexing on restart |
| **HTML + PDF support** | Auto-detects filing format — works with modern HTML filings and older PDFs |
| **Free data source** | SEC EDGAR official API — no API key required |
| **Drive cache** | Downloaded filings cached as `.bin` — re-runs never re-download |

---

## Stack

```
SEC EDGAR API
  → pdfplumber / BeautifulSoup  (parse HTML + PDF filings)
  → RecursiveCharacterTextSplitter  (800-token chunks with overlap)
  → OpenAI text-embedding-3-small  (1536-dim vectors)
  → ChromaDB  (local vector store)
  → MultiQueryRetriever  (gpt-4o-mini rewrites query 3 ways)
  → GPT-4o  (cited answer generation, temperature=0)
```

---

## Project structure

```
finsight-rag/
├── notebooks/
│   ├── ingestor_final.ipynb    # 1 of 4: SEC EDGAR fetcher + HTML/PDF parser
│   ├── chunker_final.ipynb     # 2 of 4: noise filter + 800-token chunks
│   ├── retriever_final.ipynb   # 3 of 4: ChromaDB indexing + MultiQueryRetriever
│   └── chain_final.ipynb       # 4 of 4: GPT-4o cited answer generation
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart (Google Colab)

**Prerequisites:** Google account, OpenAI API key with billing enabled (~$5 minimum)

```
1. Clone or download this repo
2. Upload the notebooks/ folder to Google Drive
3. Open ingestor_final.ipynb in Colab
4. Run top to bottom — each notebook ends with a health check
5. Continue: chunker → retriever → chain
```

Each notebook is self-contained and self-explanatory. Full inline documentation explains every design decision.

**Environment setup** — add to Colab Secrets (🔑 icon in left sidebar):
```
OPENAI_API_KEY = sk-...
SEC_USER_AGENT = FinSight YourName yourname@email.com
```

---

## Example questions

Once you've run all 4 notebooks and `ask()` is available:

```python
result = ask("What was Visa's net revenue in the most recent fiscal year?")
result = ask("How did cross-border transaction volume change year-over-year?")
result = ask("What risks related to regulation did management highlight?")
result = ask("Did management express confidence or concern about consumer spending?")
```

---

## Key design decisions

| Decision | Reason |
|----------|--------|
| Cache filings as `.bin` not `.pdf` | Most modern SEC filings are HTML — format-neutral extension avoids parser mismatch |
| `index.json` API to find documents | Machine-readable file list with sizes; picks largest file = main report body |
| 800-token chunks with 150-token overlap | Captures a full financial paragraph; overlap prevents sentences being split at boundaries |
| `temperature=0` for GPT-4o | Financial figures must be deterministic — no variation between runs |
| Cited answers with `[1][2]` | Every claim is traceable to a specific filing section — eliminates hallucination risk |

---

## Roadmap

**Phase 1 (this repo — complete):**
- SEC EDGAR ingestion with HTML + PDF support
- ChromaDB vector store with persistent Drive cache
- MultiQueryRetriever with GPT-4o cited answers

**Phase 2 (next):**
- Cross-encoder reranker — better retrieval precision
- FinBERT sentiment scoring on MD&A section
- Sentiment vs. price correlation chart (yfinance)
- Streamlit web UI

**Phase 3:**
- Agentic tool-calling loop (`search_filings`, `get_price_data`, `calculate_metric`)
- Multi-company comparison queries

---

## Cost estimate

| Item | Cost |
|---|---|
| OpenAI embeddings (`text-embedding-3-small`, 155 chunks) | ~$0.002 one-time |
| GPT-4o-mini query rewriting | ~$0.001 per question |
| GPT-4o answer generation | ~$0.01 per question |
| SEC EDGAR data | Free |
| ChromaDB | Free (local) |
| **Total for development** | **~$2–5** |

---

*Built by Ahmed Raza · [ahmeraza.github.io](https://ahmeraza.github.io)*
