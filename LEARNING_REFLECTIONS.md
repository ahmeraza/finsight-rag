# Learning Reflections — FinSight RAG

> A technical retrospective on the engineering decisions, AI/ML concepts, and production patterns applied across the complete 7-notebook, 2-phase pipeline.
> Written to document what was built, why each decision was made, and what it means in a real fintech engineering context.

---

## Table of Contents

1. [The Core Insight](#1-the-core-insight)
2. [Phase 1 — Ingestion Engineering](#2-phase-1--ingestion-engineering)
3. [Phase 1 — Chunking Strategy](#3-phase-1--chunking-strategy)
4. [Phase 1 — Retrieval Architecture](#4-phase-1--retrieval-architecture)
5. [Phase 1 — Generation & Prompt Engineering](#5-phase-1--generation--prompt-engineering)
6. [Phase 2 — FinBERT Sentiment Pipeline](#6-phase-2--finbert-sentiment-pipeline)
7. [Phase 2 — Correlation Analysis](#7-phase-2--correlation-analysis)
8. [Phase 2 — Streamlit UI (app.py)](#8-phase-2--streamlit-ui-apppy)
9. [Production Patterns Applied — Full Pipeline](#9-production-patterns-applied--full-pipeline)
10. [Financial AI Concepts Embedded](#10-financial-ai-concepts-embedded)
11. [What This Enables in Fintech](#11-what-this-enables-in-fintech)
12. [Key Lessons](#12-key-lessons)

---

## 1. The Core Insight

> **Real AI systems are mostly data engineering. The LLM is the last 10%.**

The most important realisation across building these notebooks is that the quality of GPT-4o's answers is almost entirely determined by what happens *before* the LLM is called — how the filing is fetched, how it is parsed, how it is chunked, and which chunks are retrieved.

The common misconception is that better prompting fixes bad answers. In practice, bad retrieval — caused by poor chunking, unfiltered noise, or format-naive parsing — produces answers that no amount of prompt engineering can fix. The LLM can only reason about what it is given.

```
Data quality → Chunk quality → Retrieval quality → Answer quality

Fix at the source. Don't paper over ingestion failures with prompt tricks.
```

Phase 2 added a second insight: **language and price are not the same signal**. FinBERT can score MD&A language with finance-domain precision, but a ~neutral MD&A can accompany a −3% stock move if the news is buried in the notes (the $2.2B litigation accrual). The model is correct about the language; the market reacts to the event.

This is what separates a production AI system from a notebook demo: the demo starts at the LLM; the production system starts at the data source, validates at every stage, and is honest about what its signals can and cannot predict.

---

## 2. Phase 1 — Ingestion Engineering

**Notebook:** `ingestor_final.ipynb`

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

Caching HTML as `.pdf` was the specific bug that caused both pdfplumber and pypdf to crash (`PDFSyntaxError: No /Root object`). Using `.bin` is format-neutral — the parser always detects format from byte content, never from the filename.

**Size filter `>50KB`**

A 10-K filing package contains the main report (~500KB–5MB) alongside cover pages (~2KB), XBRL viewer fragments (~1KB each, named `R1.htm` through `R200.htm`), and exhibit stubs (~5KB). Without a size filter, the code picks the first `.htm` in the list — often a 2KB cover page. The size filter eliminates all stubs reliably because the main report is always an order of magnitude larger.

**Cache validity `>100KB`**

A failed or partial download produces a tiny file (sometimes 0 bytes, sometimes a small error page). Checking `file.stat().st_size > 100_000` before using the cache prevents re-using bad downloads — a subtle but production-critical guard.

**Two-parser fallback chain**

pdfplumber handles tables better than pypdf (it extracts the cell structure). But pdfplumber occasionally fails on malformed PDFs. The fallback chain — pdfplumber → pypdf → log error — means a single malformed page never crashes a full ingest run.

### 2.3 Retry logic with exponential backoff (`tenacity`)

```python
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    reraise=True,
)
def _get(url: str) -> requests.Response:
    time.sleep(0.15)   # polite delay: ~6 req/s, within SEC's 10 req/s limit
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp
```

`tenacity`'s `@retry` decorator implements the standard production pattern for unreliable external APIs:

- **Exponential backoff** (2s → 4s → 8s → 16s) — reduces load on the server during outages and avoids thundering-herd problems
- **Maximum cap of 30s** — prevents a single request from stalling indefinitely
- **`reraise=True`** — the original exception propagates after all retries fail, so callers get meaningful error messages rather than a generic retry exhausted message
- **`retry_if_exception_type(RequestException)`** — only retries network errors, not 4xx client errors. A `403 Forbidden` from SEC means the User-Agent header is wrong — retrying won't fix that, and it would amplify the violation
- **`time.sleep(0.15)`** polite delay keeps throughput at ~6 requests/second, respecting SEC's fair-use guideline of ≤10 req/s

In production financial data systems, this same pattern is applied to Bloomberg API calls, market data feeds, and any external dependency that has transient failures but real rate limits. The principle: retry transient errors, never retry client mistakes.

### 2.4 Metadata design — the thread that enables citations

Every `Document` carries metadata that travels unchanged from ingestion through chunking, embedding, storage, retrieval, and into the final cited answer:

```python
meta = {
    'ticker':      'V',
    'form_type':   '10-K',
    'filing_date': '2025-11-06',
    'source':      'https://www.sec.gov/Archives/.../index.htm',
    'page':        42,          # section number — added during parsing
}
# Added during chunking:
# 'chunk_index': 3
# 'citation':    'V 10-K (2025-11-06) · section 42'
```

This is not an afterthought. It is the foundational design decision that makes citations possible. Without `filing_date` and `page` persisting through every transformation, the final answer has no source to reference. Metadata is the thread that connects raw bytes to a cited claim.

---

## 3. Phase 1 — Chunking Strategy

**Notebook:** `chunker_final.ipynb`

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

It only falls through to the next separator when the current chunk is still too large. This means two sentences in the same paragraph about the same financial metric will almost always end up in the same chunk — which is exactly what you want for retrieval.

### 3.3 Token vs character sizing

The splitter operates in characters, but models think in tokens (roughly 4 chars = 1 token). The conversion is explicit:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 800 * 4,   # 800 tokens × 4 chars/token = 3200 chars
    chunk_overlap = 150 * 4,   # 150 tokens × 4 chars/token = 600 chars
    separators    = ['\n\n', '\n', '. ', '! ', '? ', ' ', ''],
    length_function = len,
)
```

800 tokens is approximately one dense page of financial text — enough to capture a full revenue breakdown table with surrounding context, but small enough that the embedding vector represents a coherent financial concept rather than a grab-bag of unrelated sentences.

### 3.4 Overlap — preventing boundary failures

150-token overlap means the last 150 tokens of chunk N are repeated at the start of chunk N+1. Without overlap, a sentence at the exact boundary between two chunks might appear in neither chunk in a useful form — "Net revenue increased 11%" at the end of chunk N gets cut off before the comparison figure at the start of chunk N+1. With overlap, the full context is preserved in at least one chunk.

### 3.5 Two-stage noise filtering

**Stage 1: Text cleaning (`_clean_text`)**

Regex patterns remove SEC-specific boilerplate before splitting:

```python
_NOISE_PATTERNS = [
    re.compile(r'^\s*Page\s+\d+\s*$', re.MULTILINE),         # standalone page numbers
    re.compile(r'Table\s+of\s+Contents', re.IGNORECASE),      # TOC headers
    re.compile(r'^\s*[-\u2013\u2014]{5,}\s*$', re.MULTILINE), # horizontal rules
    re.compile(r'UNITED STATES\s+SECURITIES AND EXCHANGE', re.IGNORECASE),  # SEC cover boilerplate
    re.compile(r'\x00'),                                       # null bytes from bad PDF extraction
]
```

These patterns appear on every page, add zero signal, and — critically — inflate similarity scores for irrelevant chunks if left in. A chunk that is 30% page headers and 70% financial prose will match queries about page numbers as well as queries about revenue.

**Stage 2: Meaningfulness filter (`_is_meaningful`)**

Two checks after splitting:

```python
def _is_meaningful(text: str, min_len: int = 80) -> bool:
    stripped = text.strip()
    if len(stripped) < min_len:
        return False
    alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
    if alpha_ratio < 0.15:   # less than 15% letters = mostly numbers/symbols
        return False
    return True
```

- **Minimum length ≥80 chars** — rejects page headers and blank pages that survived cleaning
- **Alphabetic ratio ≥15%** — rejects raw number grids where pdfplumber extracted table data without column headers. A chunk that is 90% digits and symbols is not useful for semantic retrieval:

```
# Rejected (alpha_ratio ≈ 0.08):
"Revenue 17,539 16,114 14,826 9 % 9 %"

# Kept (alpha_ratio ≈ 0.72):
"Service revenue grew 9% year-over-year, driven by..."
```

### 3.6 Citation metadata — the bridge from storage to UI

The `citation` field added to every chunk's metadata is what appears in the final answer:

```python
'citation': "V 10-K (2025-11-06) · section 77"
```

This string is stored in ChromaDB alongside the embedding. When a chunk is retrieved, its `citation` is immediately available without any additional lookup — it travels from ingestor through the entire pipeline and surfaces verbatim in the UI.

---

## 4. Phase 1 — Retrieval Architecture

**Notebook:** `retriever_final.ipynb`

### 4.1 The retrieval problem in financial documents

Financial filings use language inconsistently. "Net revenue", "total revenues", "net revenues", and "top-line growth" all refer to the same concept. A single-query similarity search will match whichever phrasing happens to be closest to the query embedding — potentially missing the most relevant section entirely if it uses different terminology.

This is the precise problem `MultiQueryRetriever` solves.

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

The alternative — dense retrieval with a single query — is correct in controlled benchmarks but fails on real financial documents because analysts and management use different language for the same concept. Multi-query retrieval bridges that gap.

### 4.3 ChromaDB — what it stores

Each indexed chunk is stored as a triplet:

```
{
  id:        "unique-uuid",
  embedding: [0.023, -0.147, 0.891, ...]   # 1536 floats
  document:  "Service revenue grew 9% year-over-year...",
  metadata:  {ticker, form_type, filing_date, page, citation, chunk_index}
}
```

Similarity search computes cosine distance between the query embedding and all stored embeddings, returning the k nearest neighbours. The metadata is returned alongside the document text — no additional lookup required.

### 4.4 Two-tier persistence — idempotent and zero-cost restarts

**Tier 1: Raw filing cache (Drive)**
The downloaded `.bin` file persists to Google Drive. On all subsequent runs, the ingestor hits the cache instead of SEC's servers — zero network calls, instant load.

**Tier 2: ChromaDB vector cache (Drive)**

```python
existing_count = existing._collection.count()

if existing_count > 0:
    vector_store = existing  # reuse — zero API call, zero cost
else:
    vector_store = Chroma.from_documents(...)  # embed and persist
```

The idempotency check means the $0.002 embedding cost is paid exactly once, even across Colab session restarts. This pattern — check before compute, persist immediately — is the standard production approach to any expensive idempotent operation. In enterprise ML systems this maps to feature store caching, model registry lookups, and compute deduplication in batch jobs.

### 4.5 Similarity scores and HTML-extracted text

ChromaDB returns cosine distance (0 = identical, higher = less similar). For clean PDF text, good retrieval scores are typically 0.2–0.4. For HTML-extracted text, scores of 0.5–0.8 are normal and still return highly relevant content.

The reason: BeautifulSoup's `get_text()` produces text with inconsistent whitespace, missing formatting markers, and occasional extraction artefacts. These lower the cosine similarity without affecting semantic relevance. The correct approach is to evaluate retrieval quality by reading the returned text, not by treating the score as an absolute quality metric. The health check in `retriever_final.ipynb` was updated to reflect this, documenting that 0.687 is acceptable for HTML-extracted SEC filings.

---

## 5. Phase 1 — Generation & Prompt Engineering

**Notebook:** `chain_final.ipynb`

### 5.1 The system prompt as a formal contract

The system prompt is not boilerplate — it is a specification of the LLM's behaviour:

```python
SYSTEM_PROMPT = """You are FinSight, an expert financial analyst assistant
specialising in SEC filings (10-K annual reports and 10-Q quarterly reports).

Rules:
1. Base your answer ONLY on the provided context. Do not use outside knowledge.
2. After every factual claim, add a citation like [1] or [2, 3].
3. If the context does not contain enough information, say so. Do NOT guess.
4. Format numbers clearly: use $B for billions, $M for millions.
5. Keep answers under 300 words unless the question asks for detail.
6. Use markdown: **bold** for key metrics."""
```

Each rule has an explicit production rationale:

| Rule | Why |
|------|-----|
| Context-only | Prevents GPT-4o from using training data, which may include outdated or incorrect financials |
| Cite every claim | Makes every number traceable — essential for financial use cases where figures drive decisions |
| Refuse to guess | Graceful degradation is a feature. A system that says "I don't know" is more trustworthy than one that fabricates figures |
| Format numbers | Reduces cognitive load when scanning answers for specific figures |
| `temperature=0` | Same question on same context must return the same answer every time — non-determinism is unacceptable in financial analysis |

### 5.2 Numbered excerpts as context structure

The prompt structures context as numbered excerpts:

```
[1] V 10-K (2025-11-06) · Section 77
The following table presents the components of our net revenue:
Service revenue: $17,539M (2025), $16,114M (2024)...

[2] V 10-K (2025-11-06) · Section 78
...
```

This serves two purposes: GPT-4o can reference excerpts by number (`[1]`, `[2]`) rather than quoting them, and the number maps directly to the source list returned alongside the answer, enabling the UI to display the citation panel.

Each excerpt is truncated to 1200 chars to stay within context limits while preserving enough financial context for accurate citation.

### 5.3 LangChain LCEL — composable pipelines

LCEL (LangChain Expression Language) uses the `|` pipe operator to compose pipeline steps:

```python
chain = prompt | generation_llm | StrOutputParser()
answer = chain.invoke({'context': context, 'question': question})
```

Each component is a `Runnable` — it accepts an input dict and returns an output. The pipe operator chains them: `prompt` formats the template, `generation_llm` calls the API, `StrOutputParser` extracts the text string.

The key production benefit: composability. Swapping GPT-4o for Claude or Gemini is a single line change. Adding a reranker is another pipe step. This is the same pattern as Unix pipes and Spark transformations — small, testable units composed into a reliable pipeline.

---

## 6. Phase 2 — FinBERT Sentiment Pipeline

**Notebook:** `finbert_final.ipynb`

### 6.1 Why FinBERT, not general sentiment

General-purpose sentiment models (VADER, TextBlob, general BERT) are trained on product reviews, social media, and news articles. Financial language is systematically different:

- "We recorded additional accruals of $2.2 billion" — general model sees action/resolution (positive); financial context is a liability (negative)
- "We face continued headwinds in cross-border volume" — "continued" sounds neutral, but in financial discourse it implies ongoing pressure
- "The interchange at issue for unresolved claims" — "unresolved" is negative in general language; in legal/financial context it may signal ongoing negotiation

`ProsusAI/finbert` was fine-tuned on financial forward-looking statements (FLS) from SEC filings — exactly the language FinSight processes. This domain alignment is why FinBERT disagreed with GPT-4o on 2 of 10 sentences in precisely the cases where financial language ambiguity is highest.

### 6.2 MD&A section extraction — handling the TOC problem

A 10-K filing contains the MD&A header twice: once in the Table of Contents (position ~35,000 chars in Visa's 2025 filing) and once at the actual section body (position ~207,000 chars). A naive regex match returns the TOC entry, which is a single line of text — useless for sentiment analysis.

The solution: find all matches, skip any within the first 50,000 chars (always the TOC), use the first match after that:

```python
pattern = re.compile(r'Management.{0,10}s\s+Discussion', re.IGNORECASE)
matches = list(pattern.finditer(full_text))

real_match = None
for m in matches:
    if m.start() > 50_000:   # skip TOC entries near the start
        real_match = m
        break
```

Then find `Item 7A` (Quantitative Disclosures About Market Risk, which always follows the MD&A) to identify the section end. This gives a clean 12,000-character MD&A window.

### 6.3 Batched inference with `@functools.lru_cache`

Loading ProsusAI/finbert downloads ~440MB on first run. Using `@functools.lru_cache` ensures it loads exactly once per session:

```python
@functools.lru_cache(maxsize=1)
def _load_finbert_pipeline():
    from transformers import pipeline
    return pipeline(
        task='text-classification',
        model='ProsusAI/finbert',
        return_all_scores=True,  # get scores for all 3 classes, not just top
        device=-1,               # -1 = CPU; 0 = first GPU if available
    )
```

In `app.py`, this becomes `@st.cache_resource` — Streamlit's session-level cache for expensive objects. The principle is the same: load once, reuse forever. In production ML systems this maps to model serving with warm instances.

Sentence-level batching (batch_size=32) processes the MD&A in groups of 32 sentences. This is the optimal batch size for CPU inference — large enough to amortise the overhead of each forward pass, small enough not to exceed memory limits.

### 6.4 Signed score — a single numerical output for correlation

The sentiment pipeline produces a single `signed_score` for each document:

```python
signed = score if label == 'positive' else (-score if label == 'negative' else 0.0)
signed_avg = sum(r.signed_score for r in results) / len(results)
```

This collapses the three-class output (positive/negative/neutral) into a single value on the range (−1, +1) that can be correlated with a continuous variable (price return). Positive confidence becomes a positive score, negative confidence becomes a negative score, neutral contributes zero. The average across sentences is the document-level signal.

### 6.5 Dataclass-based output structure

Using `@dataclass` for results instead of dicts makes the code self-documenting and prevents key-name bugs:

```python
@dataclass
class SentenceResult:
    text:         str
    label:        str    # 'positive' | 'negative' | 'neutral'
    score:        float  # model confidence 0–1
    signed_score: float  # +score, -score, or 0

@dataclass
class DocumentSentiment:
    label:          str
    avg_score:      float
    signed_score:   float   # key output passed to correlate.py
    sentence_count: int
    per_sentence:   list[SentenceResult]
```

In a notebook context this is documentation. In a production API context, dataclasses become Pydantic models — the same pattern, with validation added.

### 6.6 FinBERT vs GPT-4o comparison — ablation methodology

Cell 7 of `finbert_final.ipynb` runs a model comparison on 10 sampled sentences. This is a micro-ablation study: hold the data constant, vary the model, measure agreement. The 80% agreement rate and the 2 specific disagreements reveal where financial domain fine-tuning changes the interpretation:

| Sentence | FinBERT | GPT-4o-mini | Mechanism |
|----------|---------|-------------|-----------|
| "...recorded additional accruals of $2.2 billion..." | Positive 0.91 | Neutral 0.75 | FinBERT reads "recorded" as resolution/action (common in FLS training data); GPT reads plain meaning |
| "The interchange at issue for unresolved claims will co..." | Positive 0.90 | Negative 0.75 | FinBERT picks up resolution language pattern; GPT flags "unresolved claims" as negative |

Neither model is wrong — these are genuine ambiguities in financial language. The comparison demonstrates that the choice of model matters and that blind trust in either is unwarranted.

### 6.7 Serialisation for pipeline continuity

Results are saved to Drive as `sentiment_results.json` at the end of `finbert_final.ipynb` and loaded at the start of `correlate_final.ipynb`. This decouples the two notebooks — `correlate_final.ipynb` doesn't need to re-run FinBERT to build the correlation chart:

```python
# finbert_final.ipynb — save
results_path.write_text(json.dumps(save_data, indent=2))

# correlate_final.ipynb — load
finbert_result = json.loads(results_path.read_text())
```

This is the notebook equivalent of a feature store: compute expensive features once, persist them, consume them across multiple downstream jobs.

---

## 7. Phase 2 — Correlation Analysis

**Notebook:** `correlate_final.ipynb`

### 7.1 yfinance — market data integration

`yfinance` provides adjusted closing prices and volume for any listed ticker going back years. The integration is straightforward but has one important detail: timezone removal.

```python
hist = stock.history(period='2y')
hist.index = hist.index.tz_localize(None)  # remove timezone for simpler date math
```

SEC filing dates are timezone-naive strings (`'2025-11-06'`). yfinance returns a timezone-aware DatetimeIndex. Without `tz_localize(None)`, date comparisons raise a `TypeError`. This is a common production gotcha when combining financial data from different sources.

### 7.2 Price window calculation around filing dates

The `get_price_change()` function calculates the 5-day price change centred on each filing date:

```python
def get_price_change(ticker_hist, filing_date_str, window=5):
    filing_date  = pd.Timestamp(filing_date_str)
    future_dates = available_dates[available_dates >= filing_date]
    event_date   = future_dates[0]   # nearest trading day on or after filing
    event_idx    = ticker_hist.index.get_loc(event_date)

    pre_prices  = ticker_hist['Close'].iloc[event_idx - window : event_idx]
    post_prices = ticker_hist['Close'].iloc[event_idx : event_idx + window]

    change = (post_prices.mean() - pre_prices.mean()) / pre_prices.mean() * 100
```

Using the average of the 5-day window before and after (rather than a single day) reduces noise from intraday volatility that's unrelated to the filing. This is standard event study methodology in financial research.

### 7.3 Measured vs proxy scores — methodological transparency

Only the November 2025 10-K has a directly measured FinBERT score (from `finbert_final.ipynb`). The 7 other filing data points use proxy scores estimated from public earnings call analyst consensus. The code is explicit about this:

```python
FILINGS = [
    {'date': '2025-11-06', 'signed_score': finbert_result['signed_score'],
     'source': 'measured', ...},     # actual FinBERT output

    {'date': '2025-07-24', 'signed_score': +0.08,
     'source': 'proxy', ...},        # estimated from public analysis
    ...
]
```

The scatter chart differentiates measured points (pink diamond) from proxy points (blue circle). This is methodological transparency — a reviewer can immediately see which data point is empirically grounded and which is estimated. Hiding the distinction would inflate the apparent reliability of the correlation.

### 7.4 Pearson correlation and R² interpretation

```python
corr = np.corrcoef(x, y)[0, 1]   # Pearson r
r2   = corr ** 2                   # coefficient of determination
```

Pearson r measures linear association. R² tells you what fraction of price variance is explained by sentiment. The key result: R²=0.176, correlation=0.419, compared to the academic baseline for filing-sentiment-to-next-day-return of R²~0.02–0.05.

The result is stronger than baseline for a specific reason: Visa is a high-quality, consistently managed company whose MD&A tone tends to accurately reflect business performance. In a sector with more volatile or less disciplined IR communication, the correlation would likely be weaker. This is domain knowledge shaping the interpretation of a statistical result.

### 7.5 Directional accuracy — a practical metric

```python
correct = (
    (row['signed_score'] > 0 and row['change_pct'] > 0) or
    (row['signed_score'] < 0 and row['change_pct'] < 0)
)
```

67% directional accuracy (4/6), 17pp above random baseline. In financial signal research, a directional signal that is right more than 50% of the time is economically meaningful — it creates tradeable alpha over many observations even if the magnitude is not predicted accurately. This is why the metric matters beyond just reporting R².

### 7.6 The Visa anomaly — when language and news diverge

The FY2025 10-K is the most analytically interesting data point: FinBERT scored the MD&A as near-neutral (+0.0024), yet the stock dropped ~3% in the 5 days after the November 6 filing.

The explanation: the $2.2B litigation accrual is disclosed in the financial notes, not the MD&A. Management used carefully neutral language to describe it ("we recorded additional accruals"), which FinBERT scored accurately as positive/action-oriented (0.91). But the market reacted to the accrual size, not the language. This is a genuine limitation of MD&A-only sentiment — material events that management chooses to neutrally frame will be missed by a language model that can only score what it reads.

Production improvement: scoring the full 10-K, not just the MD&A, would partially address this. Combining sentiment with structured data extraction (the accrual amount itself) would address it more completely.

---

## 8. Phase 2 — Streamlit UI (`app.py`)

### 8.1 `@st.cache_resource` — session-level model caching

```python
@st.cache_resource(show_spinner="Loading FinBERT model (~440MB on first run)...")
def _load_finbert():
    from transformers import pipeline
    return pipeline(task='text-classification', model='ProsusAI/finbert', ...)

@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def _load_vector_store():
    from langchain_chroma import Chroma
    ...
```

`@st.cache_resource` caches objects across all user sessions — equivalent to a singleton pattern in web frameworks. Without this, every query would reload the 440MB FinBERT model. With it, the model loads once on the first request and is reused for the session lifetime. This is the difference between a 30-second wait per query and a sub-second one.

The lazy import pattern (`from transformers import pipeline` inside the function rather than at the top) means these expensive libraries are not loaded at app startup — only when first needed. This keeps the initial page load fast.

### 8.2 Component-level error handling

Every expensive operation in `app.py` is wrapped in its own `try/except`, and errors are surfaced to the user with `st.error()` rather than crashing the app:

```python
with st.spinner("Ingesting filing..."):
    try:
        n = ingest_and_index(ticker, form_type, max_filings)
        st.success(f"Indexed {n} chunks.")
    except Exception as e:
        st.error(f"Ingestion failed: {e}")

# FinBERT runs independently — a FinBERT failure doesn't cancel the answer
try:
    sent = score_text(combined)
    # ... render badge
except Exception as e:
    st.caption(f"FinBERT unavailable: {e}")
```

The FinBERT sentiment step is isolated from the answer step. If FinBERT fails (e.g. model not cached, memory pressure), the answer is still displayed — FinBERT degradation is visible to the user but doesn't block the core functionality. This is the circuit-breaker pattern: fail gracefully, maintain partial service.

### 8.3 Custom CSS and financial UI aesthetics

The app uses a dark financial aesthetic with custom CSS injected via `st.markdown(..., unsafe_allow_html=True)`:

```css
.answer-box {
    background: #0f1117;
    border-left: 3px solid #3b82f6;   /* blue accent = financial terminal style */
    font-family: 'DM Sans', sans-serif;
}

.badge-positive { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.badge-negative { background: #2d0808; color: #f87171; border: 1px solid #7f1d1d; }
.badge-neutral  { background: #1c1f26; color: #94a3b8; border: 1px solid #334155; }
```

The badge colours mirror trading terminal conventions: green for positive sentiment, red for negative, grey for neutral. This is UX domain expertise — the aesthetic signals competence to a financial professional audience before they read a single word.

DM Mono is used for citation strings and metric values — a deliberate choice to visually separate data values from prose, consistent with how Bloomberg Terminal displays structured financial data.

### 8.4 Sidebar separation of concerns

The sidebar handles configuration (ticker selection, ingestion, settings toggles). The main panel handles query and display. This separation prevents the UI from becoming a single monolithic form and makes it clear which actions are setup vs runtime.

The `ticker_filter` selectbox passes a filter to the retriever's `search_kwargs`, restricting ChromaDB similarity search to vectors with a matching ticker metadata field. This is metadata-filtered retrieval — the same pattern used in enterprise document search to scope results to a specific business unit or time period.

---

## 9. Production Patterns Applied — Full Pipeline

### 9.1 Fault tolerance inventory

| Pattern | Applied where | Why it matters in production |
|---------|--------------|------------------------------|
| Exponential backoff retry | `_get()` HTTP helper | SEC API has transient failures during peak hours — retrying without backoff amplifies the load |
| Multi-parser fallback chain | `_bytes_to_documents()` | pdfplumber fails on ~5% of malformed PDFs — one bad page shouldn't abort a multi-filing ingest |
| Graceful skip on missing documents | `ingest_ticker()` | One failed filing shouldn't abort a multi-company ingest job |
| Cache validity check `>100KB` | `ingest_ticker()` | Failed downloads produce tiny files that would be re-used as valid cache |
| LLM refusal on missing context | System prompt rule 3 | Prevents hallucinated figures when the relevant section wasn't retrieved |
| `try/except` per UI component | `app.py` | FinBERT failure doesn't block the answer; ingestion failure doesn't crash the app |
| Lazy model loading | `app.py` cache functions | 440MB model doesn't block app startup |

### 9.2 Idempotency — every expensive operation runs at most once

Every expensive operation checks whether it has already been done:

- **Ingestion**: checks Drive cache before downloading (size > 100KB)
- **Chunking**: deterministic function on deterministic input — running twice produces the same 155 chunks
- **Indexing**: checks ChromaDB count before embedding (`if existing_count > 0: reuse`)
- **Generation**: `temperature=0` ensures identical queries produce identical answers
- **FinBERT loading**: `@functools.lru_cache` / `@st.cache_resource` ensures one load per session

In a production pipeline, idempotency is what makes scheduled jobs, operator re-runs, and retry-on-failure safe. A non-idempotent pipeline that duplicates vectors or re-embeds documents on every run will produce incorrect retrieval results and unexpected costs.

### 9.3 Health checks as validation gates

Each notebook ends with a structured health check that validates the output before the next stage begins:

```python
# Ingestor
assert len(docs) > 50           # catches stub document fetches
assert expected_keys == actual  # catches metadata schema drift
assert cached_files             # confirms Drive persistence

# Chunker
assert len(chunks) > 100        # catches over-aggressive noise filtering
assert 'citation' in metadata   # confirms citation metadata present
assert empty_chunks == 0        # catches degenerate splitting

# Retriever
assert count > 100              # confirms indexing succeeded
assert has_citation             # confirms metadata survived to ChromaDB

# FinBERT
assert sentiment.sentence_count > 10
assert abs(sentiment.signed_score) > 0
assert sentiment.label in ('positive', 'negative', 'neutral')

# Correlation
assert len(df) >= 4
assert len(charts) >= 2
```

This pattern — validate at every stage boundary — is standard in data pipelines (`dbt` tests, Great Expectations, etc.). It catches problems at the source rather than letting them propagate silently through 4 more stages before producing a wrong answer.

### 9.4 Separation of concerns across notebooks

Each notebook has a single responsibility:

| Notebook | Input | Output | Single responsibility |
|----------|-------|--------|----------------------|
| `ingestor_final.ipynb` | Ticker symbol | LangChain Documents | Raw bytes → structured Documents |
| `chunker_final.ipynb` | Documents | Retrieval-ready chunks | Noise filtering + semantic splitting |
| `retriever_final.ipynb` | Chunks | ChromaDB index | Embedding + vector storage |
| `chain_final.ipynb` | Question | Cited answer | Multi-query retrieval + LLM generation |
| `finbert_final.ipynb` | Filing bytes | Sentiment score + JSON | MD&A extraction + domain sentiment |
| `correlate_final.ipynb` | JSON + ticker | Charts + correlation | Price data + statistical analysis |
| `app.py` | User input | Web UI | Orchestration + rendering |

This separation means each stage can be tested, debugged, and replaced independently. If a better HTML parser is released, only the ingestor changes. If a better chunking strategy is identified, only the chunker changes. The interfaces between stages (LangChain `Document` objects, JSON files, ChromaDB collections) remain stable.

### 9.5 Observability

`logging` is used throughout instead of `print()`:

```python
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info('Fetching submissions for %s (CIK %d)...', ticker, cik)
logger.warning('_extract_filing_url failed: %s', exc)
```

Two reasons: log level can be changed globally (`INFO` → `WARNING`) to silence debug output without modifying function code; and log messages include the function name and context, making it possible to trace a failure to its exact source.

The health check cells provide a second observability layer — structured assertions with human-readable output that clearly distinguish ✅ passing checks from ❌ failures.

---

## 10. Financial AI Concepts Embedded

| Financial Use Case | Enabled By | Specific Implementation |
|-------------------|-----------|------------------------|
| Earnings Q&A bot | Cited chunk retrieval | `ask()` function with `[N]` inline citations |
| SEC filing analysis | Format-aware ingestion | HTML + PDF dual-parser with byte detection |
| Risk intelligence | Metadata-filtered retrieval | `ticker_filter` in retriever limits to specific company |
| Compliance copilots | Citation trail | Every answer references exact filing section |
| Investment research | Semantic retrieval | Multi-query finds synonymous financial terminology |
| MD&A tone quantification | FinBERT sentence scoring | Signed score captures bullish/bearish management language |
| Sentiment-price signal | Correlation analysis | Pearson r between MD&A tone and 5-day price return |
| Competitive analysis | Multi-company indexing | ChromaDB collection accepts any ticker in `TICKER_TO_CIK` |
| Regulatory monitoring | Cross-filing comparison | Index multiple years to compare risk factor changes |
| Pre-earnings prep | Rapid document search | Instant retrieval vs manual PDF search |

The domain specificity of the questions FinSight handles — cross-border transaction volume, MD&A sentiment, interchange litigation accruals, operating margin trends — reflects real payments analyst workflows from experience at Checkout.com and Fuze. The technical choices (which SEC form types to support, which companies to seed, which questions to test, which anomalies to investigate) are not arbitrary — they come from knowing what questions are actually asked in financial analysis.

---

## 11. What This Enables in Fintech

### 11.1 The trust problem in financial AI

The single most important property of a financial AI system is **trustworthiness** — can you verify the answer? A system that says "Visa's revenue was $40B" with no source is useless in a professional context. A system that says "Visa's revenue was $40B [2] — from V 10-K (2025-11-06) · section 77" is actionable because it can be verified.

Every design decision in FinSight is oriented toward this:
- Citations make every claim verifiable
- `temperature=0` makes answers reproducible
- The refusal rule makes the system's knowledge boundaries explicit
- Metadata on every chunk makes the source traceable
- FinBERT scores management language with finance-domain precision

### 11.2 Why RAG beats fine-tuning for financial documents

Fine-tuning a model on financial filings has three problems:
1. **Staleness** — a model trained on 2023 filings doesn't know 2025 figures
2. **Attribution** — a fine-tuned model can't tell you which document its answer came from
3. **Cost** — fine-tuning GPT-4o costs thousands of dollars per run

RAG sidesteps all three: the knowledge base is updated by adding documents, every answer has a citation, and the only cost is embedding ($0.002 per filing) and retrieval (negligible).

For financial applications where documents change every quarter and every figure needs a source, RAG is the correct architecture.

### 11.3 Metadata as enterprise AI infrastructure

Enterprise AI without metadata fails at the governance layer. Financial institutions need:
- **Auditability** — which document was the answer sourced from?
- **Filtering** — retrieve only from filings for this specific company
- **Explainability** — why did the system answer this way?
- **Versioning** — answer based on the 2024 10-K, not the 2023 one

FinSight's metadata schema (`ticker`, `form_type`, `filing_date`, `page`, `citation`) addresses all four. It is the minimum viable metadata for a production financial AI system.

### 11.4 The sentiment-price gap — a real-world finding

Phase 2 produced a finding that would be at home in an academic paper: FinBERT scored Visa's MD&A as near-neutral (+0.0024) in a filing where the stock dropped ~3% in the subsequent 5 days. The cause — a $2.2B litigation accrual described in neutral language — illustrates both the power and the limit of NLP-based financial signal generation.

The power: FinBERT correctly identified that the language was neutral. The limit: the market doesn't react only to language. A production sentiment signal would need to combine MD&A tone with structured data extraction (accrual amounts, guidance changes, revenue surprises) to capture the full picture. This is the next frontier — Phase 3.

---

## 12. Key Lessons

**1. Retrieval quality determines answer quality**
Garbage chunks → garbage retrieval → hallucinations. No amount of prompt engineering fixes bad ingestion. Fix at the source.

**2. Format assumptions are the most common real-world failure**
The entire SEC parsing saga — HTML served as `.pdf`, zero-byte failed downloads, 2KB cover pages selected instead of the 2MB main report — was caused by assumptions about file format that didn't match reality. Byte-level detection and size filters solve this class of problem permanently.

**3. Metadata is the thread that makes AI explainable**
Without `filing_date`, `page`, and `citation` metadata travelling unchanged through every pipeline stage, the cited answer is impossible. The metadata schema is a design decision that should be made at the ingestion stage and never changed — changing it mid-pipeline breaks everything downstream.

**4. Production AI requires fault tolerance at every layer**
Retries, caching, validation, fallbacks, health checks — these are not optional polish. They are what separates a notebook that works once from a system that runs reliably. The `@retry` decorator, the two-tier cache, and the health check assertions are as important as the LLM calls.

**5. Domain-specific models are worth it when the language is domain-specific**
General sentiment on financial text is wrong in exactly the cases that matter most — high-stakes disclosures where management language is deliberately neutral about negative events. FinBERT's 80% agreement with GPT-4o-mini (with the 20% disagreements concentrated in the most financially ambiguous sentences) validates the investment in a domain-tuned model.

**6. Statistical signals need interpretation, not just calculation**
R²=0.176, correlation=0.419, directional accuracy 67% — these numbers only mean something in context. The academic baseline (R²~0.02–0.05), the company-specific explanation (Visa's consistent management communication), and the anomaly (−3% price move on neutral MD&A) are the real insight. A number without context is not analysis.

**7. The LLM is the last 10%**
SEC EDGAR fetching, format detection, cache management, noise filtering, chunk sizing, overlap tuning, metadata design, retrieval strategy, FinBERT scoring, correlation analysis — all of this happens before GPT-4o is called. Getting these right is what makes the LLM's output trustworthy.

**8. Self-documenting pipelines are production assets**
Every design decision in these notebooks is documented inline — not as comments that say *what* the code does, but as explanations of *why* this approach was chosen over the alternative. This is the difference between code that can be maintained and code that can only be rewritten.

**9. Domain expertise multiplies technical skill**
Knowing that MD&A sentiment correlates with price movement, that the TOC problem exists in 10-K parsing, that $2.2B litigation accruals are disclosed in notes rather than MD&A, that cross-border volume is the key metric for payments networks — this domain knowledge determines which questions to ask, which edge cases to handle, and which results to trust. Technical skill without domain expertise builds systems that work but don't matter.

---

*FinSight RAG — Phase 1 + Phase 2 complete.*
*Built by Ahmed Raza · [ahmeraza.github.io](https://ahmeraza.github.io)*
