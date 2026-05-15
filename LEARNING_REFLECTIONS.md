# Learning Reflections — FinSight RAG

> A technical retrospective on the engineering decisions, AI/ML concepts, and production patterns applied across the 4-notebook pipeline.
> Written to document what was built, why each decision was made, and what it means in a real fintech engineering context.

---

## Table of Contents

1. [The Core Insight](#1-the-core-insight)
2. [Ingestion Engineering](#2-ingestion-engineering)
3. [Chunking Strategy](#3-chunking-strategy)
4. [Retrieval Architecture](#4-retrieval-architecture)
5. [Generation & Prompt Engineering](#5-generation--prompt-engineering)
6. [Production Patterns Applied](#6-production-patterns-applied)
7. [Financial AI Concepts Embedded](#7-financial-ai-concepts-embedded)
8. [What This Enables in Fintech](#8-what-this-enables-in-fintech)
9. [Key Lessons](#9-key-lessons)

---

## 1. The Core Insight

> **Real AI systems are mostly data engineering. The LLM is the last 10%.**

The most important realisation across building these notebooks is that the quality of GPT-4o's answers is almost entirely determined by what happens *before* the LLM is called — how the filing is fetched, how it is parsed, how it is chunked, and which chunks are retrieved.

The common misconception is that better prompting fixes bad answers. In practice, bad retrieval — caused by poor chunking, unfiltered noise, or format-naive parsing — produces answers that no amount of prompt engineering can fix. The LLM can only reason about what it is given.

```
Data quality → Chunk quality → Retrieval quality → Answer quality

Fix at the source. Don't paper over ingestion failures with prompt tricks.
```

This is what separates a production RAG system from a notebook demo: the demo starts at the LLM; the production system starts at the data source.

---

## 2. Ingestion Engineering

### 2.1 Why ingestion is harder than it looks

SEC EDGAR is a government API with no authentication, no rate limit dashboard, and no consistent file format. In practice this means:
- 2025 10-K filings are HTML, not PDF
- The "filing" is a package of 30+ files (images, XBRL, exhibits, the actual report)
- The index page that lists those files is itself an HTML document that must be scraped
- One wrong assumption about format crashes the entire pipeline

The ingestor solves each of these explicitly.

### 2.2 Techniques applied

**`index.json` API over HTML scraping**

The filing index page is HTML — scraping it with regex is brittle. SEC provides an underdocumented `index.json` endpoint at the same path that returns a machine-readable file list with sizes. Using this instead of HTML scraping is more reliable and gives us the file sizes needed to identify the main document.

```python
# Fragile: HTML regex scraping
match = re.search(r'href="(/Archives/edgar/data/[^"]+\.pdf)"', html)

# Robust: JSON API with size data
json_url = f'.../Archives/edgar/data/{cik}/{acc}/index.json'
files = _get(json_url).json()['directory']['item']
best = max(candidates, key=lambda f: int(f.get('size', 0)))
```

**Byte-level format detection**

File extensions are unreliable — SEC serves `.htm` files that some crawlers save as `.pdf`. Trusting the extension causes parsers to crash. Reading the first 500 bytes and checking for HTML markers is always correct:

```python
sample  = raw_bytes[:500].decode('utf-8', errors='ignore').lower()
is_html = any(marker in sample for marker in ['<html', '<!doctype', '<document'])
```

**Format-neutral cache extension `.bin`**

Caching HTML as `.pdf` was the specific bug that caused both pdfplumber and pypdf to crash (`PDFSyntaxError: No /Root object`). Using `.bin` is format-neutral — the parser always detects format from byte content, never from the filename. This is a small decision with a large reliability impact.

**Size filter `>50KB`**

A 10-K filing package contains the main report (~500KB–5MB) alongside cover pages (~2KB), XBRL viewer fragments (~1KB each, named `R1.htm` through `R200.htm`), and exhibit stubs (~5KB). Without a size filter, the regex picks the first `.htm` in the list — often a 2KB cover page. The size filter `>50KB` eliminates all stubs reliably because the main report is always an order of magnitude larger.

**Cache validity `>100KB`**

A failed or partial download produces a tiny file (sometimes 0 bytes, sometimes a small error page). Checking `file.stat().st_size > 100_000` before using the cache prevents re-using bad downloads — a subtle but production-critical guard.

**Two-parser fallback chain**

pdfplumber handles tables better than pypdf (it extracts the cell structure). But pdfplumber occasionally fails on malformed PDFs. The fallback chain — pdfplumber → pypdf → log error — means a single malformed page never crashes a full ingest run.

### 2.3 Retry logic with exponential backoff

```python
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    reraise=True,
)
```

`tenacity`'s `@retry` decorator implements the standard production pattern for unreliable external APIs:
- **Exponential backoff** (2s → 4s → 8s → 16s) — reduces load on the server during outages and avoids thundering-herd problems
- **Maximum cap of 30s** — prevents a single request from stalling indefinitely
- **`reraise=True`** — the original exception propagates after all retries fail, so callers get meaningful error messages
- **`retry_if_exception_type(RequestException)`** — only retries network errors, not 4xx client errors (a 403 from SEC means your User-Agent is wrong — retrying won't fix that)

The `time.sleep(0.15)` polite delay keeps throughput at ~6 requests/second, respecting SEC's fair-use guideline of ≤10 req/s.

### 2.4 Metadata design

Every `Document` carries metadata that travels unchanged from ingestion through chunking, embedding, storage, retrieval, and into the final cited answer:

```python
meta = {
    'ticker':      'V',
    'form_type':   '10-K',
    'filing_date': '2025-11-06',
    'source':      'https://www.sec.gov/Archives/.../index.htm',
    'page':        42,           # added during parsing
}
# Added during chunking:
# 'chunk_index': 3
# 'citation':    'V 10-K (2025-11-06) · section 42'
```

This is not an afterthought — it is the foundational design decision that makes citations possible. Without `filing_date` and `page` persisting through every transformation, the final answer has no source to reference. Metadata is the thread that connects raw bytes to a cited claim.

---

## 3. Chunking Strategy

### 3.1 Why chunking is the most underrated step

A vector store retrieves the **closest matching chunk** to a query. This means the unit of retrieval is the chunk — not the document. If the chunk is too large, it contains multiple topics and the similarity score is diluted. If the chunk is too small, it loses the context that makes a number meaningful.

Getting chunking wrong is the most common RAG failure mode in production, and the hardest to diagnose — the LLM produces confident-sounding wrong answers because it was given confident-sounding but irrelevant context.

### 3.2 `RecursiveCharacterTextSplitter` — why recursive matters

A simple character splitter cuts at a fixed character count regardless of content. The recursive splitter tries separators in order of semantic importance:

```
\n\n (paragraph break)
  → \n (line break)
    → ". " (sentence boundary)
      → " " (word boundary)
        → "" (character — last resort)
```

It only falls through to the next separator when the current chunk is still too large. This means two sentences in the same paragraph about the same financial metric will almost always end up in the same chunk — which is what you want for retrieval.

### 3.3 Token vs character size

The splitter operates in characters, but analysts think in tokens (roughly 4 chars = 1 token). The conversion is explicit in the code:

```python
chunk_size=800 * 4     # 800 tokens × 4 chars/token = 3200 chars
chunk_overlap=150 * 4  # 150 tokens × 4 chars/token = 600 chars
```

800 tokens is approximately one dense page of financial text — enough to capture a full revenue breakdown table with surrounding context, but small enough that the embedding vector represents a coherent financial concept rather than a grab-bag of unrelated sentences.

### 3.4 Overlap — preventing boundary failures

150-token overlap means the last 150 tokens of chunk N are repeated at the start of chunk N+1. Without overlap, a sentence at the exact boundary between two chunks might appear in neither chunk in a useful form — "Net revenue increased 11%" at the end of chunk N gets cut off before the comparison figure that appears at the start of chunk N+1. With overlap, the full context is preserved in at least one chunk.

### 3.5 Noise filtering — two-stage

**Stage 1: Text cleaning (`_clean_text`)**
Regular expressions remove SEC-specific boilerplate before splitting:
- Standalone page numbers (`Page 42`) — appear on every page, add zero signal
- Table of Contents headers — high-frequency terms that inflate similarity scores for irrelevant chunks
- Horizontal rules — formatting artefacts from HTML→text conversion
- SEC cover page boilerplate — identical in every filing, creates false similarity

**Stage 2: Meaningfulness filter (`_is_meaningful`)**
Two checks after splitting:
- **Minimum length `≥80 chars`** — rejects page headers and blank pages that survived cleaning
- **Alphabetic ratio `≥15%`** — rejects raw number grids where pdfplumber extracted table data without column headers. A chunk that is 90% digits and symbols is not useful for semantic retrieval.

```python
alpha_ratio = sum(c.isalpha() for c in text) / len(text)
# "Revenue 17,539 16,114 14,826 9 % 9 %" → ratio ≈ 0.08 → rejected
# "Service revenue grew 9% year-over-year, driven by..." → ratio ≈ 0.72 → kept
```

### 3.6 Citation metadata

The `citation` field added to every chunk's metadata is the bridge between the vector store and the user-visible source reference:

```python
'citation': "V 10-K (2025-11-06) · section 77"
```

This string is stored in ChromaDB alongside the embedding. When a chunk is retrieved, its `citation` is immediately available without any additional lookup — it travels from ingestor through the entire pipeline and surfaces verbatim in the final answer.

---

## 4. Retrieval Architecture

### 4.1 The retrieval problem in financial documents

Financial filings use language inconsistently. "Net revenue", "total revenues", "net revenues", and "top-line growth" all refer to the same concept. A single-query similarity search will match whichever phrasing happens to be closest to the query embedding — potentially missing the most relevant section entirely if it uses different terminology.

This is the precise problem MultiQueryRetriever solves.

### 4.2 MultiQueryRetriever — how it works

```
User query: "How did Visa revenue perform this year?"
     │
     ▼
gpt-4o-mini rewrites to 3 alternatives:
  1. "Visa total net revenue fiscal year 2025"
  2. "Visa annual revenue growth year over year"
  3. "Visa financial performance top line results"
     │
     ▼
3 parallel similarity searches in ChromaDB
     │
     ▼
Results merged and deduplicated
     │
     ▼
Top-6 diverse chunks returned
```

The rewriting step costs ~$0.001 (gpt-4o-mini is very cheap) and typically recovers 1–3 additional relevant chunks that the original query would have missed. This is a significant recall improvement for a negligible cost.

### 4.3 ChromaDB — what it stores

Each indexed chunk is stored as a triplet:

```
{
  id:        "unique-uuid",
  embedding: [0.023, -0.147, 0.891, ...]  # 1536 floats
  document:  "Service revenue grew 9% year-over-year...",
  metadata:  {ticker, form_type, filing_date, page, citation, ...}
}
```

Similarity search computes cosine distance between the query embedding and all stored embeddings, returning the k nearest neighbours. The metadata is returned alongside the document text — no additional lookup required.

### 4.4 Two-tier persistence

**Tier 1: Raw filing cache (Drive)**
The downloaded `.bin` file persists to Google Drive. On all subsequent runs, the ingestor hits the cache instead of SEC's servers — zero network calls, instant load.

**Tier 2: ChromaDB vector cache (Drive)**
The embedded vectors persist to Drive. The idempotency check (`existing_count = existing._collection.count()`) means the $0.002 embedding cost is paid exactly once, even across Colab session restarts.

```python
if existing_count > 0:
    vector_store = existing  # reuse — no API call
else:
    vector_store = Chroma.from_documents(...)  # embed and persist
```

This pattern — check before compute, persist immediately — is the standard production approach to any expensive idempotent operation.

### 4.5 Similarity scores in HTML-extracted text

ChromaDB returns cosine distance (0 = identical, higher = less similar). For clean PDF text, good retrieval scores are typically 0.2–0.4. For HTML-extracted text, scores of 0.5–0.8 are normal and still return highly relevant content.

The reason: BeautifulSoup's `get_text()` produces text with inconsistent whitespace, missing formatting markers, and occasional extraction artefacts. These lower the cosine similarity without affecting semantic relevance. The correct approach is to evaluate retrieval quality by reading the returned text, not by treating the score as an absolute quality metric.

---

## 5. Generation & Prompt Engineering

### 5.1 The system prompt as a contract

The system prompt is not boilerplate — it is a formal specification of the LLM's behaviour:

```
Rule 1: "Base your answer ONLY on the provided context."
→ Prevents GPT-4o from using training data, which may include outdated financials

Rule 2: "After every factual claim, add a citation like [1] or [2, 3]."
→ Makes every number traceable — essential for financial use cases

Rule 3: "If context doesn't contain enough, say so. Do NOT guess."
→ Graceful degradation is a feature. A system that says "I don't know" is more
  trustworthy than one that confidently fabricates figures.

Rule 4: "temperature=0"
→ Deterministic output — the same question on the same context must return the
  same answer every time. Non-determinism is unacceptable in financial analysis.
```

### 5.2 Numbered excerpts as context structure

The prompt structures context as numbered excerpts:

```
[1] V 10-K (2025-11-06) · Section 77
The following table presents the components of our net revenue:
Service revenue: $17,539M (2025), $16,114M (2024)...

[2] V 10-K (2025-11-06) · Section 78
...
```

This structure serves two purposes:
1. GPT-4o can reference excerpts by number (`[1]`, `[2]`) rather than quoting them, keeping answers concise
2. The number maps directly to the source list returned alongside the answer, enabling the UI to display clickable citations

### 5.3 LangChain LCEL — composable pipelines

LCEL (LangChain Expression Language) uses the `|` pipe operator to compose pipeline steps:

```python
chain = prompt | generation_llm | StrOutputParser()
answer = chain.invoke({'context': context, 'question': question})
```

Each component is a `Runnable` — it accepts an input dict and returns an output. The pipe operator chains them: `prompt` formats the template, `generation_llm` calls the API, `StrOutputParser` extracts the text string. This is composable, testable, and replaceable — swapping GPT-4o for Claude is one line change.

---

## 6. Production Patterns Applied

### 6.1 Fault tolerance

| Pattern | Applied where | Why |
|---------|--------------|-----|
| Exponential backoff retry | `_get()` HTTP helper | SEC API has transient failures during peak hours |
| Multi-parser fallback chain | `_bytes_to_documents()` | pdfplumber fails on ~5% of malformed PDFs |
| Graceful skip on missing documents | `ingest_ticker()` | One failed filing shouldn't abort a multi-company ingest |
| Cache validity check | `ingest_ticker()` | Failed downloads produce tiny files that would be re-used as valid cache |
| LLM refusal on missing context | System prompt rule 3 | Prevents hallucinated figures when the relevant section wasn't retrieved |

### 6.2 Idempotency

Every expensive operation is idempotent — running it twice has the same effect as running it once:
- **Ingestion**: checks Drive cache before downloading
- **Chunking**: deterministic function on deterministic input
- **Indexing**: checks ChromaDB count before embedding
- **Generation**: `temperature=0` ensures identical queries produce identical answers

This is a production requirement: scheduled jobs, retries after failures, and operator re-runs must not produce duplicate data or unexpected side effects.

### 6.3 Health checks as validation gates

Each notebook ends with a health check cell that validates the output before passing it to the next stage:

```python
# Ingestor health check
assert len(docs) > 50          # catches stub document fetches
assert expected_keys == actual  # catches metadata schema drift
assert cached_files             # confirms Drive persistence

# Chunker health check
assert len(chunks) > 100        # catches over-aggressive noise filtering
assert 'citation' in metadata   # confirms citation metadata present
assert empty_chunks == 0        # catches degenerate splitting

# Retriever health check
assert count > 100              # confirms indexing succeeded
assert score < 0.9              # confirms retrieval is finding relevant content
assert has_citation             # confirms metadata survived to ChromaDB
```

This pattern — validate at every stage boundary — is standard in data pipelines (dbt tests, Great Expectations, etc.). It catches problems at the source rather than letting them propagate silently through the system.

### 6.4 Separation of concerns

Each notebook has a single responsibility:
- **Ingestor**: raw bytes → LangChain Documents
- **Chunker**: Documents → retrieval-ready chunks
- **Retriever**: chunks → indexed vectors + retrieval function
- **Chain**: question → cited answer

This separation means each stage can be tested, debugged, and replaced independently. If a better HTML parser is released, only the ingestor changes. If a better chunking strategy is identified, only the chunker changes. The interfaces between stages (LangChain `Document` objects with standard metadata) remain stable.

### 6.5 Observability

`logging` is used throughout instead of `print()` for two reasons:
1. Log level can be changed globally (`INFO` → `WARNING`) to silence debug output in production without modifying function code
2. Log messages include the function name and context, making it possible to trace a failure to its exact source

The health check cells provide a second observability layer — structured assertions with human-readable output that clearly distinguish ✅ passing checks from ❌ failures.

---

## 7. Financial AI Concepts Embedded

| Financial Use Case | Enabled By | Specific Implementation |
|-------------------|-----------|------------------------|
| Earnings Q&A bot | Cited chunk retrieval | `ask()` function with `[N]` inline citations |
| SEC filing analysis | Format-aware ingestion | HTML + PDF dual-parser with byte detection |
| Risk intelligence | Metadata-filtered retrieval | `ticker_filter` in retriever limits to specific company |
| Compliance copilots | Citation trail | Every answer references exact filing section |
| Investment research | Semantic retrieval | Multi-query finds synonymous financial terminology |
| Competitive analysis | Multi-company indexing | ChromaDB collection accepts any ticker in `TICKER_TO_CIK` |
| Regulatory monitoring | Cross-filing comparison | Index multiple years to compare risk factor changes |
| Pre-earnings prep | Rapid document search | Instant retrieval vs manual PDF search |

The domain specificity of the questions FinSight handles — cross-border transaction volume, MD&A sentiment, operating margin trends — reflects real payments analyst workflows from experience at Checkout.com and Fuze. The technical choices (which SEC form types to support, which companies to seed, which questions to test) are not arbitrary.

---

## 8. What This Enables in Fintech

### 8.1 The trust problem in financial AI

The single most important property of a financial AI system is **trustworthiness** — can you verify the answer? A system that says "Visa's revenue was $40B" with no source is useless in a professional context. A system that says "Visa's revenue was $40B [2] — from V 10-K (2025-11-06) · section 77" is actionable because it can be verified.

Every design decision in FinSight is oriented toward this:
- Citations make every claim verifiable
- `temperature=0` makes answers reproducible
- The refusal rule makes the system's knowledge boundaries explicit
- Metadata on every chunk makes the source traceable

### 8.2 Why RAG beats fine-tuning for financial documents

Fine-tuning a model on financial filings has three problems:
1. **Staleness** — a model trained on 2023 filings doesn't know 2025 figures
2. **Attribution** — a fine-tuned model can't tell you which document its answer came from
3. **Cost** — fine-tuning GPT-4o costs thousands of dollars per run

RAG sidesteps all three: the knowledge base is updated by adding documents, every answer has a citation, and the only cost is embedding ($0.002 per filing) and retrieval (negligible).

For financial applications where documents change every quarter and every figure needs a source, RAG is the correct architecture.

### 8.3 Metadata as enterprise AI infrastructure

Enterprise AI without metadata fails at the governance layer. Financial institutions need:
- **Auditability** — which document was the answer sourced from?
- **Filtering** — retrieve only from filings for this specific company
- **Explainability** — why did the system answer this way?
- **Versioning** — answer based on the 2024 10-K, not the 2023 one

FinSight's metadata schema (`ticker`, `form_type`, `filing_date`, `page`, `citation`) addresses all four. It is the minimum viable metadata for a production financial AI system.

---

## 9. Key Lessons

**1. Retrieval quality determines answer quality**
Garbage chunks → garbage retrieval → hallucinations. No amount of prompt engineering fixes bad ingestion. Fix at the source.

**2. Format assumptions are the most common real-world failure**
The entire SEC parsing saga — HTML served as `.pdf`, zero-byte failed downloads, 2KB cover pages selected instead of the 2MB main report — was caused by assumptions about file format that didn't match reality. Byte-level detection and size filters solve this class of problem permanently.

**3. Metadata is the thread that makes AI explainable**
Without `filing_date`, `page`, and `citation` metadata travelling unchanged through every pipeline stage, the cited answer is impossible. The metadata schema is a design decision that should be made at the ingestion stage and never changed — changing it mid-pipeline breaks everything downstream.

**4. Production AI requires fault tolerance at every layer**
Retries, caching, validation, fallbacks, health checks — these are not optional polish. They are what separates a notebook that works once from a system that runs reliably. The `@retry` decorator, the two-tier cache, and the health check assertions are as important as the LLM calls.

**5. The LLM is the last 10%**
SEC EDGAR fetching, format detection, cache management, noise filtering, chunk sizing, overlap tuning, metadata design, retrieval strategy — all of this happens before GPT-4o is called. Getting these right is what makes the LLM's output trustworthy.

**6. Self-documenting pipelines are production assets**
Every design decision in these notebooks is documented inline — not as comments that say *what* the code does, but as explanations of *why* this approach was chosen over the alternative. This is the difference between code that can be maintained and code that can only be rewritten.

**7. Domain expertise multiplies technical skill**
Knowing that MD&A sentiment correlates with price movement, that risk factors change between filing periods, that cross-border volume is the key metric for payments networks — this domain knowledge determines which questions to ask, which edge cases to handle, and which results to trust. Technical skill without domain expertise builds systems that work but don't matter.

---

*FinSight RAG — Phase 1 complete. Built by Ahmed Raza.*
*Fintech background: Checkout.com, Fuze · [ahmeraza.github.io](https://ahmeraza.github.io)*
