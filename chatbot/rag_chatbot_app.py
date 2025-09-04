import os, logging, re, traceback
import html, unicodedata
import hashlib
import argparse
import sys
import time
import json
import threading
from pathlib import Path
from entities.document import Document
from chatbot.document_loader.format import Format
from chatbot.document_loader.text_splitter import create_recursive_text_splitter
from typing import List, Tuple
import streamlit as st
from chatbot.bot.client.lama_cpp_client import LamaCppClient
from chatbot.bot.conversation.chat_history import ChatHistory
from chatbot.bot.conversation.conversation_handler import answer_with_context, refine_question
from chatbot.bot.conversation.ctx_strategy import (
    BaseSynthesisStrategy,
    get_ctx_synthesis_strategies,
    get_ctx_synthesis_strategy,
)
from chatbot.bot.memory.embedder import Embedder
from chatbot.bot.memory.vector_database.chroma import Chroma
from chatbot.bot.model.model_registry import get_model_settings, get_models
from chatbot.financial_fetcher import (
    resolve_ticker_sec,get_stock_price, get_financial_news, 
    get_financial_metric, compose_metric_header
)
from chatbot.ingest_pipeline import ingest_ticker
from claude_api import call_claude_fallback, SYS_PROMPT
from helpers.log import get_logger

# STEP 1 Drop-in: Disable parallel tokenization to avoid Streamlit warnings    
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Index path guard (early check) 
logging.basicConfig(level=logging.INFO)
INDEX_DIR = Path(os.getenv("INDEX_PATH", "vector_store/docs_index"))
logging.info(f"[INIT] Using index path: {INDEX_DIR.resolve()}")
if not INDEX_DIR.exists():
    raise RuntimeError(f"[ERR] Index directory missing: {INDEX_DIR.resolve()} "
                       f"— run memory_builder to create it.")

logger = get_logger(__name__)

def escape_markdown(text: str) -> str:
    """Escape Markdown control chars so Streamlit won't render italics/monospace accidentally."""
    if not isinstance(text, str):
        return text
    return re.sub(r'([\\`*_{}[\]()#+\-!.>])', r'\\\1', text)

# Set Streamlit configuration for financial chatbot
st.set_page_config(page_title="Financial RAG Chatbot", page_icon="💰", initial_sidebar_state="collapsed")

# Step 2: Enable JIT (Just-In-Time) mode for financial RAG chatbot
# JIT config (single source of truth) 
JIT_ENABLED = os.getenv("RAG_JIT_ENABLED", "1") == "1"
PRELOAD = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "TSLA", "UNH", "JNJ"}
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# Optional: explanation intent 
EXPLAIN_RE = re.compile(r"\b(why|explain|because|reason|driver|drivers?|factor|factors?|cause|caused|impact|contribute|contributed)\b",re.I,)
def is_explanation_query(q: str) -> bool:
    return bool(EXPLAIN_RE.search(q or ""))

# One entry to get both meta parser & reranker for a given ticker 
def build_ticker_meta_tools(ticker: str):
    T = (ticker or "").upper()

    def parse_md_meta(p: Path) -> dict:
        name = p.stem
        parts = name.split("_")
        return {
            "ticker": T,
            "report_type": parts[1] if len(parts) > 1 else "",
            "fiscal_date": parts[2] if len(parts) > 2 else "",
            "source": str(p),
        }

    def rerank(docs: List[Document], top_k: int = 8, keywords=None):
        keywords = [kw.lower() for kw in (keywords or [])]

        def score(d: Document):
            md, txt = d.metadata or {}, d.page_content.lower()
            src = md.get("source", "").lower()
            s = 0
            if md.get("ticker") == T: s += 20
            if "10-q" in src or "10-k" in src: s += 8
            elif "8-k" in src: s += 3
            s += sum(1 for kw in keywords if kw in txt)
            return s

        return sorted(docs, key=score, reverse=True)[:top_k]

    return parse_md_meta, rerank

# Local docs guard 
def _has_enough_local_docs(ticker: str, cap: int = 3) -> bool:
    tkr = (ticker or "").upper()
    tags = ("8-K", "10-Q", "Exhibit", "press", "earnings", "shareholder")

    subdir = DOCS_DIR / tkr
    if subdir.exists():
        hits = [p for p in subdir.glob("*.md") if any(tag in p.name for tag in tags)]
        logging.debug(f"[JIT] Local subdir check for {tkr}: {len(hits)} hits")
        if len(hits) >= cap:
            return True

    flat = [p for p in DOCS_DIR.glob(f"{tkr}_*_*.md") if any(tag in p.name for tag in tags)]
    logging.debug(f"[JIT] Local flat check for {tkr}: {len(flat)} hits")
    return len(flat) >= cap


# Fire-and-forget SEC ingestion 
def ensure_company_indexed(ticker: str, cap: int = 3, timeout: int = 10) -> None:
    if _has_enough_local_docs(ticker, cap=cap):
        logging.info(f"[JIT] Skip ingestion for {ticker}, already has enough local docs")
        return

    def _job():
        try:
            ingest_ticker(ticker, forms=["8-K", "10-Q"], max_per_ticker=cap, since="2023-01-01")
            logging.info(f"[JIT] ingest_ticker finished for {ticker}")
        except Exception as e:
            logging.error(f"[JIT] ingest_ticker failed for {ticker}: {e!r}\n{traceback.format_exc()}")

    t = threading.Thread(target=_job, daemon=True)
    t.start()
    t.join(timeout=timeout)
    logging.warning(f"[JIT] {ticker.upper()} ingestion triggered (timeout={timeout}s). "
                    "Run memory_builder to persist into vector index.")

# Hot-insert freshly saved docs 
def hot_insert_ticker_docs(index, ticker: str, *, docs_dir: Path = DOCS_DIR,
    chunk_size: int = 512, chunk_overlap: int = 25,) -> int:
    tkr = (ticker or "").upper()
    parse_md_meta, _ = build_ticker_meta_tools(tkr)

    # Collect markdown files (flat + subdir)
    md_files = list(docs_dir.glob(f"{tkr}_*_*.md"))
    subdir = docs_dir / tkr
    if subdir.exists():
        md_files += list(subdir.glob("*.md"))

    logging.debug(f"[JIT] Found {len(md_files)} md files for {tkr}")
    if not md_files:
        logging.info(f"[JIT] No markdown files found for {tkr}")
        return 0

    # Minimal HTML → text sanitizer
    def _to_text(raw: str) -> str:
        if "<" in raw and ">" in raw:
            raw = re.sub(r"<[^>]+>", " ", raw)  # drop tags
            raw = re.sub(r"&nbsp;|&amp;|&lt;|&gt;|&quot;|&#160;", " ", raw)  # basic entities
            raw = re.sub(r"\s{2,}", " ", raw).strip()  # collapse spaces
        return raw

    # Load and sanitize
    docs: List[Document] = []
    for p in md_files:
        try:
            txt = _to_text(p.read_text(encoding="utf-8"))
            if txt:
                docs.append(Document(page_content=txt, metadata=parse_md_meta(p)))
        except Exception as e:
            logging.warning(f"[JIT] Read/sanitize failed for {p}: {e!r}")

    if not docs:
        return 0
    # Chunk & insert
    splitter = create_recursive_text_splitter(
        format=Format.MARKDOWN.value, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    try:
        chunks = splitter.split_documents(docs)
    except Exception as e:
        logging.warning(f"[JIT] split_documents failed: {e!r}")
        return 0
    if not chunks:
        return 0
    try:
        index.from_chunks(chunks)  # supported by your vector store
        logging.info(f"[JIT] Hot-inserted {len(chunks)} chunks for {tkr}")
        return len(chunks)
    except Exception as e:
        logging.warning(f"[JIT] Hot insert failed for {tkr}: {e!r}")
        return 0

# Step 3: Load the fine-tuned financial LLM client
@st.cache_resource()
def load_llm_client(model_folder: Path, model_name: str) -> LamaCppClient:
    model_settings = get_model_settings(model_name)
    llm = LamaCppClient(model_folder=model_folder, model_settings=model_settings)
    return llm

# Initialize short-term conversation memory
@st.cache_resource()
def init_chat_history(total_length: int = 2) -> ChatHistory:
    return ChatHistory(total_length=total_length)

# Load financial-specific RAG strat
@st.cache_resource()
def load_ctx_synthesis_strategy(ctx_synthesis_strategy_name: str, _llm: LamaCppClient) -> BaseSynthesisStrategy:
    return get_ctx_synthesis_strategy(ctx_synthesis_strategy_name, llm=_llm)

# Load financial documents vector index
@st.cache_resource()
def load_index(vector_store_path: Path) -> Chroma:
    """
    Loads a Vector Database index based on the specified vector store path.
    Args:
        vector_store_path (Path): The path to the vector store.
    Returns:
        Chroma: An instance of the Vector Database.
    """
    embedding = Embedder()
    index = Chroma(persist_directory=str(vector_store_path), embedding=embedding)
    return index

# Strict ticker extraction & resolution 
_TICKER_BLACKLIST = {
    "I","ME","MY","US","USA","EPS","PE","ROE","YOY","QOQ","FY","Q","THE","A","AN","AND",
    "WHAT","IS","PRICE","STOCK","ON","IN","FOR","OF"
}

_COMMON_NAME_MAP = {
    "apple": "AAPL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
}

# Keep original case; only pick explicit uppercase tokens or cashtags
def extract_possible_tickers(query: str) -> list[str]:
    cashtags = [m.group(1).upper() for m in re.finditer(r'(?<!\w)\$([A-Za-z]{1,5})\b', query)]
    uppercase_tokens = re.findall(r'\b[A-Z]{1,5}\b', query)
    tickers = {t for t in (cashtags + uppercase_tokens) if t not in _TICKER_BLACKLIST}
    return list(tickers)

def resolve_ticker_from_query(query: str) -> str | None:
    # 1) Try cashtags/explicit tickers
    candidates = extract_possible_tickers(query)
    if candidates:
        # If multiple, pick the first that isn't “A”, “I”, etc. (already filtered)
        return sorted(candidates, key=len, reverse=True)[0]
    # 2) Try company-name map
    qlow = query.lower()
    for name, tkr in _COMMON_NAME_MAP.items():
        if name in qlow:
            return tkr
    return None

# Fetch latest stock price via financial_fetcher.get_stock_price
def fallback_stock_lookup(query: str) -> List[Document]:
    """
    Only if the query looks like a price question and ticker is resolvable.
    Adds metadata.source_type="tool" and a structured payload for API fast-path.
    """
    if not re.search(r'\b(price|stock|share|quote|trading|last|close)\b', query, re.I):
        return []

    ticker = resolve_ticker_from_query(query)
    if not ticker:
        return []

    data = get_stock_price(ticker)  # expected: {ticker, price(or value), as_of, source}
    if not data:
        return []

    # Normalize value key and ensure it's present
    value = data.get("value", data.get("price"))
    if value is None:
        return []

    payload = {
        "kind": "price",
        "ticker": data.get("ticker", ticker),
        "value": value,
        "as_of": data.get("as_of"),
        "source": data.get("source", "financial_fetcher"),
        "raw": data,  # keep full raw for debugging
    }

    return [Document(
        page_content=json.dumps(data),
        metadata={
            "source": "financial_fetcher",
            "source_type": "tool",   # required for API fast-path
            "organization": data.get("ticker", ticker),
            "report_type": "API",
            "payload": payload,      # fast-path reads this
        }
    )]

# Fetch latest financial news for a resolved ticker if possible; otherwise use the raw query term.
def fallback_news_lookup(query: str) -> List[Document]:
    """
    Adds metadata.source_type="tool" and a structured payload (kind="news", items=[...]).
    """
    if not re.search(r"\b(news|headline|headlines|latest news|top news)\b", query, re.I):
        return []  
    ticker = resolve_ticker_from_query(query)
    term = ticker if ticker else query

    items = get_financial_news(term)  # list of dicts: {title, url, published_at, source?}
    if not items:
        return []

    # Try to propagate the true source if present on items; else default label
    news_source = None
    for it in items:
        if isinstance(it, dict) and it.get("source"):
            news_source = it["source"]
            break
    news_source = news_source or "financial_fetcher"

    payload = {
        "kind": "news",
        "items": items[:20],          # cap to a reasonable count
        "source": news_source,
        "query": term,
    }

    return [Document(
        page_content=json.dumps({"items": items}),
        metadata={
            "source": "financial_fetcher",
            "source_type": "tool",    # required for API fast-path
            "organization": ticker or "",
            "report_type": "API",
            "payload": payload,       # fast-path reads this
        }
    )]

# Simple metric resolver for Tools fast-path.
# Supports FY/CY/bare year-quarter patterns; backend still queried on fiscal (FY) data.
def fallback_financial_metric_lookup(query: str) -> List[Document]:
    if not re.search(r"\b(eps|earnings per share|revenue|net income|operating (income|margin))\b", query, re.I):
        return []
    text = query.strip()
    upper = text.upper()

    # Parse year & quarter and decide basis
    basis = "FY"          # default FY unless we clearly see CY or bare-year-quarter
    year = quarter = None
    cal_year = cal_quarter = None

    # FY2024 Q4 / Q4 FY2024
    m = re.search(r"\bFY\s*(\d{2,4})\s*Q([1-4])\b", upper) or re.search(r"\bQ([1-4])\s*FY\s*(\d{2,4})\b", upper)
    if m:
        if m.re.pattern.startswith(r"\bFY"):
            y_raw, q_raw = m.group(1), m.group(2)
        else:
            q_raw, y_raw = m.group(1), m.group(2)
        y = int(y_raw)
        year = (2000 + y) if y < 100 else y
        quarter = int(q_raw)
        basis = "FY"
    else:
        # CY2024 Q4 / Q4 CY2024
        m = re.search(r"\bCY\s*(\d{2,4})\s*Q([1-4])\b", upper) or re.search(r"\bQ([1-4])\s*CY\s*(\d{2,4})\b", upper)
        if m:
            if m.re.pattern.startswith(r"\bCY"):
                y_raw, q_raw = m.group(1), m.group(2)
            else:
                q_raw, y_raw = m.group(1), m.group(2)
            y = int(y_raw)
            year = (2000 + y) if y < 100 else y
            quarter = int(q_raw)
            basis = "CY"
            cal_year, cal_quarter = year, quarter
        else: # bare "2024 Q4" / "Q4 2024" → treat as CY
            m = re.search(r"\b(20\d{2})\s*Q([1-4])\b", upper) or re.search(r"\bQ([1-4])\s*(20\d{2})\b", upper)
            if m:
                if m.re.pattern.startswith(r"\b(20"):
                    y_raw, q_raw = m.group(1), m.group(2)
                else:
                    q_raw, y_raw = m.group(1), m.group(2)
                year = int(y_raw)
                quarter = int(q_raw)
                basis = "CY"
                cal_year, cal_quarter = year, quarter
            else: # just a year → default FY
                y = re.search(r"\b(20\d{2})\b", upper)
                year = int(y.group(1)) if y else None
                quarter = None # basis remains FY
    if not year:
        return []

    # Resolve ticker (sanitize keywords to avoid false positives like GAAP) ---
    query_for_ticker = re.sub(
        r"(?i)\b(NON[-\s]?GAAP|GAAP|EPS|EARNINGS|FISCAL|CALENDAR|FY|CY|Q[1-4]|REVENUE|NET\s+INCOME|OPERATING\s+(INCOME|MARGIN))\b",
        " ",
        text,
    )
    ticker = resolve_ticker_from_query(query_for_ticker)
    if not ticker:
        return []

    # Map metric keyword
    metric = None
    for kw, key in {
        "earnings per share": "eps",
        "eps": "eps",
        "revenue": "revenue",
        "net income": "net_income",
        "operating income": "operating_income",
        "operating margin": "operating_margin",
    }.items():
        if re.search(rf"\b{re.escape(kw)}\b", text, re.I):
            metric = key
            break
    if not metric:
        return []

    # Fetch (FY-based backend)
    data = get_financial_metric(ticker, year, metric, quarter)
    if not data:
        return []

    # If backend returns no quarter but user asked a quarter (common for Q4 in 10-K), show requested one
    effective_quarter = data.get("quarter", quarter)

    note = ""
    if basis == "CY":
        note = "requested calendar period; answered from fiscal data (SEC companyfacts)"

    payload = {
        "kind": "metric",
        "basis": basis,                         # CRUCIAL: CY or FY
        "ticker": data.get("ticker", ticker),
        "metric": data.get("metric", metric),
        "value": data.get("value"),
        "year": data.get("year", year),
        "quarter": effective_quarter,
        "as_of": data.get("as_of"),
        "source": data.get("source"),
        "raw": data,
        "calendar_year": cal_year,
        "calendar_quarter": cal_quarter,
        "note": note,
    }

    meta = {
        "source": "financial_fetcher",
        "source_type": "tool",
        "organization": ticker,
        "fiscal_year": str(year),
        "report_type": "API",
        "payload": payload,
        "fiscal_basis": basis,
    }
    if effective_quarter:
        meta["fiscal_quarter"] = f"Q{effective_quarter}"
    if basis == "CY":
        if cal_year: meta["calendar_year"] = str(cal_year)
        if cal_quarter: meta["calendar_quarter"] = f"Q{cal_quarter}"

    return [Document(page_content=json.dumps(data), metadata=meta)]

# Step 4: UI branding and initialization
# Initializes the page configuration for the application.
def init_page(root_folder: Path) -> None:
    left_column, central_column, right_column = st.columns([2, 1, 2])

    with left_column:
        st.write(" ")

    with central_column:
        # Display centered finance bot image
        st.image(str(root_folder / "images" / "finance-bot.png"), width=120)

     # Centered title and subtitle using HTML + inline style
        st.markdown(
            """
            <div style='text-align: center; margin-top: 0.5em;'>
               <span style='font-size: 28px; font-weight: bold;'>Your Financial Assistant</span><br/>
                <span style='color: gray;'>Got a financial question? I’m here to help!</span>
            </div>
            """,
           unsafe_allow_html=True,
        )

    with right_column:
        st.write(" ")

    st.sidebar.title("Financial Assistant Options")

# Display a finance-specific welcome message
@st.cache_resource
def init_welcome_message() -> None:
    with st.chat_message("assistant"):
        st.write("Welcome to your financial assistant. What would you like to analyze today?")

# Initializes the chat history, allowing users to clear the conversation.
def reset_chat_history(chat_history: ChatHistory) -> None:
    clear_button = st.sidebar.button("🗑️ Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []
        chat_history.clear()

# Display past user/assistant interactions
def display_messages_from_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("render") == "html" and msg["role"] == "assistant":
                st.markdown(msg.get("html") or msg["content"], unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

# Step 5: Routing & UI helpers 
# Produce a single, human-friendly path line.
def _route_line(kind: str, *, rag_docs: int | None = None, fact: str | None = None, ticker: str | None = None) -> str:
    if kind in ("HYBRID", "HYBRID JIT"):
        fact_label = "metric" if fact == "metric" else "price"
        tail = f" (RAG docs={rag_docs})" if rag_docs is not None else ""
        tick = f" [ticker={ticker}]" if ticker else ""
        return f"{kind} — {fact_label} fact + explanation{tail}{tick}"
    if kind == "JIT→RAG":
        tick = f" [ticker={ticker}]" if ticker else ""
        return f"JIT→RAG docs={rag_docs or 0}{tick}"
    if kind == "TOOLS":
        return f"TOOLS({fact})" if fact in ("metric", "price") else "TOOLS"
    if kind == "RAG":
        return f"RAG docs={rag_docs or 0}"
    if kind == "CLAUDE":
        return "CLAUDE (no retrieval)"
    return kind

# Return tool payload from a doc (only when source_type == tool).
def tool_payload(doc):
    md = getattr(doc, "metadata", {}) or {}
    return md.get("payload") if str(md.get("source_type", "")).lower() == "tool" else None

# Call this INSIDE a `with st.chat_message("assistant"):` block
# Render the sources list inside the provided container (same chat bubble)
def show_sources(container, srcs: list[dict]):  
    with container:
        st.markdown("📚 Found relevant context via **RAG**.")
        with st.expander("Show retrieval sources"):
            for s in srcs:
                st.markdown(
                    f"- **Score** {s.get('score', 0.0):.3f} • "
                    f"{s.get('organization','')} {s.get('report_type','')} {s.get('fiscal_year','')}  \n"
                    f"<{s.get('source','')}>  \n"
                    f"{(s.get('title') or s.get('content_preview') or '')}"
                )

# Wrap tool fact as a high-priority synthetic document to see it first in the RAG context window.
def make_tool_fact_doc(facts_md: str) -> Document:
    return Document(
        page_content=facts_md,
        metadata={"source": "TOOL_FACT", "priority": "high", "title": "Numeric fact from tools"}
    )

CITATION_RULES = (
    "Use ONLY the provided CONTEXT.\n"
    "If TOOL FACT conflicts with retrieved text, prefer TOOL FACT and state the discrepancy in one sentence"
    "Quote at least TWO numeric sentences from filings (10-K/10-Q/8-K/IR) "
    "showing exact figures/percent changes, then explain.\n"
    "Do NOT use news or third-party sites; if filings lack the answer, say so briefly."
)

# Put Tool fact and citation rules at the top, and Keeping instructions first materially improves adherence.
def build_task_prompt(user_input: str, facts_md: str | None, use_citation: bool) -> str:
    header_parts: list[str] = []
    if facts_md:
        header_parts.append("NUMERIC FACT (from tools):\n" + facts_md.strip())
    if use_citation:
        header_parts.append("[CITATION RULES]\n" + CITATION_RULES)
    header_parts.append("QUESTION:\n" + user_input.strip())
    return "\n\n".join(header_parts)

# Step 6: Main logic for launching financial RAG chatbot
def main(parameters) -> None:            
    root_folder = Path(__file__).resolve().parent.parent
    model_folder = root_folder / "models"
    vector_store_path = root_folder / "vector_store" / "docs_index"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    model_name = parameters.model
    synthesis_strategy_name = parameters.synthesis_strategy

    init_page(root_folder)
    llm = load_llm_client(model_folder, model_name)
    chat_history = init_chat_history(2)
    ctx_synthesis_strategy = load_ctx_synthesis_strategy(synthesis_strategy_name, _llm=llm)
    index = load_index(vector_store_path)
    reset_chat_history(chat_history)
    init_welcome_message()
    display_messages_from_history()

    # Supervise user input
    if user_input := st.chat_input("Ask a financial question (e.g. P/E ratio, ESG risk, portfolio)..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # One-line tracer setup (terminal)
        _TRACE = os.environ.get("TRACE_ROUTING") == "1"
        _LOG   = logging.getLogger("ROUTING")
        route  = "INIT"
    
        # Record final route & optionally print a single line to terminal.
        def set_route(kind: str, *, rag_docs: int | None = None, fact: str | None = None, ticker: str | None = None):   
            nonlocal route
            route = kind
            st.session_state.last_route = route
            if _TRACE:
                _LOG.info("[PATH] " + _route_line(kind, rag_docs=rag_docs, fact=fact, ticker=ticker))
            
        with st.spinner("📊 Refining your financial query and retrieving relevant documents…"):
            # Resolve ticker from query, refine question, and detect intent
            orig_query    = user_input
            refined_query = refine_question(llm, user_input, chat_history=chat_history)
            # tkr = (resolve_ticker_from_query(orig_query) or "").upper()
            # Robust ticker resolution (sec -> legacy fallback) 
            tk_sec = resolve_ticker_sec(orig_query)
            tkr    = (tk_sec or "").upper()
            _LOG.info("[TICKER] resolved=%s via=%s", (tkr or "-"), ("sec" if tk_sec else "none"))
            # Intent (compact)
            PRICE_RE   = re.compile(r"\b(price|quote|share price|last|close)\b", re.I)
            METRIC_RE  = re.compile(r"\b(eps|earnings per share|revenue|net income|operating (margin|income)|gross margin|guidance)\b", re.I)
            EXPLAIN_RE = re.compile(r"\b(why|explain|because|driver|impact|summari[sz]e|highlights|compare|vs\.?|versus|earnings call|conference call|prepared remarks)\b", re.I)
            q_price   = bool(PRICE_RE.search(refined_query or ""))
            q_metric  = bool(METRIC_RE.search(refined_query or ""))
            q_explain = bool(EXPLAIN_RE.search(refined_query or ""))
            
            if _TRACE:
                _LOG.info(f"[DBG] intent: price={q_price}, metric={q_metric}, explain={q_explain}, ticker={tkr or '-'}")

            # 1) Run API first: stock -> metric -> news (single loop)
            facts_md, fact_kind = "", None  # keep one tool fact for HYBRID
            answered = False

            # tiny locals for prior-period + number cast
            def _prior(y, q):
                if y is None: return None, None
                if q is None: return y - 1, None
                return (y, q - 1) if q > 1 else (y - 1, 4)

            def _num(x):
                try:
                    return float(str(x).replace(",", "").strip())
                except Exception:
                    return None

            def _fmt_metric_hdr(tkr, metric, val, y, q, asof=None, basis="FY"):
                try:
                    payload = {"ticker": tkr, "metric": metric, "value": val, "year": y, "quarter": q,
                            "as_of": asof, "basis": basis}
                    return compose_metric_header(payload)
                except Exception:
                    qtxt = f" Q{q}" if q else ""
                    s = f"**{tkr} {metric} {y or ''}{qtxt}: {val}**"
                    if asof: s += f" (as of {asof})"
                    return s
            
            # Be more permissive for "explain/compare" intent to trigger HYBRID
            _compare_re = re.compile(
                r"\b(vs|versus|compare|comparison|prior quarter|previous quarter|qoq|quarter[- ]over[- ]quarter|yoy|year[- ]over[- ]year|y/y|change[d]?)\b",
                re.I)
            want_hybrid = bool(q_explain or _compare_re.search(refined_query or ""))

            for tool in [fallback_stock_lookup, fallback_financial_metric_lookup, fallback_news_lookup]:
                fallback_docs = tool(refined_query) or []
                if not fallback_docs:
                    continue

                # Scan the tool payloads and pick the first usable one
                for d in fallback_docs:
                    md = getattr(d, "metadata", {}) or {}
                    if str(md.get("source_type", "")).lower() != "tool":
                        continue
                    p = md.get("payload") or {}
                    kind = p.get("kind")

                    # price 
                    if kind == "price" and p.get("ticker") and p.get("value") is not None:
                        t, v   = p["ticker"], p["value"]
                        asof   = p.get("as_of", "")
                        src    = p.get("source", "financial_fetcher")
                        header = f"**{t}** latest price: **{v}**" + (f" (as of {asof})" if asof else "")

                        if want_hybrid:
                            # Keep the fact and continue to JIT/RAG for explanation
                            facts_md  = f"**Fact (tool):**\n\n{header}\n\n_Source: {src}_\n\n"
                            fact_kind = "price"
                            if _TRACE: _LOG.info(f"[DBG] captured fact: kind=price, ticker={t}, asof={asof}")
                        else:
                            set_route("TOOLS", fact="price")
                            with st.chat_message("assistant"):
                                st.markdown(f"**Route:** TOOLS\n\n{header}\n\n_Source: {src}_")
                            answered = True
                        break  # used a price payload

                    # metric (EPS / revenue / etc.) 
                    if kind == "metric" and p.get("ticker") and p.get("metric") and p.get("value") is not None:
                        tkr   = p["ticker"]
                        mname = p["metric"]
                        val   = p["value"]
                        y, qv, asof = p.get("year"), p.get("quarter"), p.get("as_of", "")
                        basis = p.get("basis", "FY")
                        src   = p.get("source", "financial_fetcher")

                        cur_hdr  = _fmt_metric_hdr(tkr, mname, val, y, qv, asof=asof, basis=basis)
                        cur_num  = _num(val)
                        prior_hdr, prior_num = "", None

                        # Only fetch prior period when we intend to run HYBRID
                        if want_hybrid:
                            try:
                                py, pq = _prior(y, qv)
                                if py is not None:
                                    prev = get_financial_metric(tkr, py, metric=mname, quarter=pq)
                                    if prev and prev.get("value") is not None:
                                        prior_hdr = _fmt_metric_hdr(
                                            tkr, mname, prev["value"], py, pq,
                                            asof=prev.get("as_of"),
                                            basis=("FY" if pq is None else basis)
                                        )
                                        prior_num = _num(prev["value"])
                            except Exception as e:
                                if _TRACE: _LOG.info(f"[DBG] prior fetch failed: {e}")

                        if want_hybrid:
                            parts = [f"**Fact (tool):**\n\n{cur_hdr}\n\n_Source: {src}_"]
                            if prior_hdr:
                                parts.append(f"**Prior period (tool):**\n\n{prior_hdr}\n\n_Source: SEC EDGAR companyfacts_")
                            if cur_num is not None and prior_num not in (None, 0.0):
                                yoy = (cur_num - prior_num) / prior_num * 100.0
                                parts.append(f"**YoY change (auto-calc): {yoy:.1f}%**")
                            facts_md  = "\n\n".join(parts) + "\n\n"
                            fact_kind = "metric"
                            if _TRACE: _LOG.info(f"[DBG] captured fact: metric={mname}, prior_found={bool(prior_hdr)}")
                        else:
                            set_route("TOOLS", fact="metric")
                            with st.chat_message("assistant"):
                                st.markdown(f"**Route:** TOOLS\n\n{cur_hdr}\n\n_Source: {src}_")
                            answered = True
                        break

                    # news 
                    if kind == "news" and (items := p.get("items")):
                        # Only non-explanatory queries should return news
                        if not want_hybrid:
                            src  = p.get("source", "financial_fetcher")
                            top  = items[:3]
                            lines = [f"- {it.get('title','(untitled)')} <{it.get('url','')}>" for it in top]
                            set_route("TOOLS")
                            with st.chat_message("assistant"):
                                st.markdown(f"**Route:** TOOLS\n\n**Latest headlines:**\n" + "\n".join(lines) + f"\n\n_Source: {src}_")
                            answered = True
                        break

                if answered:
                    return  # API answered a non-explanatory query; we're done

            if _TRACE:
                _LOG.info(f"[DBG] after API: fact_kind={fact_kind}, facts_md_len={len(facts_md or '')}, want_hybrid={want_hybrid}, q_explain={q_explain}")

            # 2) JIT (if needed) + RAG retrieval
            # --- A) Resolve ticker from the original query (SEC-backed) ---
            tkr = resolve_ticker_sec(orig_query)  # <-- use the new resolver; do NOT use legacy name
            if _TRACE:
                _LOG.info(f"[TICKER] resolved={tkr or '-'} via=sec")

            used_jit = False

            # --- B) Trigger JIT only for tickers NOT in PRELOAD and only if JIT is enabled ---
            if tkr and (tkr not in PRELOAD) and JIT_ENABLED:
                try:
                    # Guard: skip ingestion if we already have enough local docs for this ticker
                    has_enough = _has_enough_local_docs(tkr, cap=3)
                    if _TRACE:
                        _LOG.info(f"[JIT] has_enough_local_docs(ticker={tkr}) -> {has_enough}")

                    if not has_enough:
                        # Kick off a short live ingest (8-K / 10-Q); returns after `timeout` even if still downloading
                        ensure_company_indexed(tkr, cap=3, timeout=10)
                        used_jit = True
                        if _TRACE:
                            _LOG.info(f"[JIT] ingestion fired for ticker={tkr}")

                        # --- C) Hot-insert the newly saved markdown into the in-memory index (no rebuild) ---
                        try:
                            inserted = hot_insert_ticker_docs(
                                index=index,
                                ticker=tkr,
                                docs_dir=DOCS_DIR,
                                chunk_size=getattr(parameters, "chunk_size", 512),
                                chunk_overlap=getattr(parameters, "chunk_overlap", 25),
                            )
                            if _TRACE:
                                _LOG.info(f"[JIT] hot_insert inserted={inserted} chunks for {tkr}")
                        except Exception as e:
                            _LOG.warning(f"[JIT] hot_insert error: {e!r}")

                except Exception as e:
                    # Keep running even if JIT fails; we'll still try RAG with whatever we have
                    _LOG.warning(f"[JIT] ensure_company_indexed error: {e!r}")

            # --- D) Log current index stats for visibility (helps you see live index size drift) ---
            try:
                stats = getattr(getattr(index, "_collection", None), "count", lambda: "?")()
            except Exception:
                stats = "?"
            _LOG.info(f"[INDEX] docs_in_memory={stats}, used_jit={used_jit}")

            # --- E) First retrieval (after possible JIT/hot-insert) ---
            # Anchor query with ticker + finance hints to reduce cross-company hits.
            anchor_terms   = "operating income operating margin results of operations md&a exhibit 99"
            anchored_query = f"{refined_query} {tkr} {anchor_terms}" if tkr else refined_query

            retrieved_contents, sources = index.similarity_search_with_threshold(
                query=anchored_query,
                k=parameters.k,
                threshold=0.05,
                exclude_tools=True,
            )
            if _TRACE:
               _LOG.info(f"[DBG] RAG.retrieved={len(retrieved_contents)} (used_jit={used_jit})")

            # Same-company-first rerank (uses your build_ticker_meta_tools.rerank)
            def _realign_sources(_docs: list[Document], _sources: list[dict]) -> list[dict]:
                """Realign sources to docs by matching source/filepath key."""
                if not _docs or not _sources:
                    return _sources

                def key(meta): 
                    return (meta or {}).get("source") or (meta or {}).get("filepath") or ""

                # map: key → list of indices
                mp = {}
                for i, s in enumerate(_sources):
                    mp.setdefault(key(s), []).append(i)

                aligned = []
                for d in _docs:
                    k = key(d.metadata)
                    if k in mp and mp[k]:
                        i = mp[k].pop(0)  # ✅ pop the index, not the whole list
                        aligned.append(_sources[i])
                    else:
                        aligned.append({"source": k, "score": 0.0})

                return aligned


            if tkr and retrieved_contents:
                _, rerank = build_ticker_meta_tools(tkr)
                retrieved_contents = rerank(
                    retrieved_contents,
                    top_k=parameters.k,
                    keywords=["operating income", "operating margin", "md&a", "exhibit 99"],
                )
                sources = _realign_sources(retrieved_contents, sources)

            # --- F) Cheap retry once if JIT used and still nothing ---
            if not retrieved_contents and used_jit:
                if _TRACE:
                    _LOG.info("[DBG] RAG.empty_after_jit → retry once with anchored query")
                retrieved_contents, sources = index.similarity_search_with_threshold(
                    query=anchored_query,
                    k=parameters.k,
                    threshold=0.05,
                    exclude_tools=True,
                )
                if tkr and retrieved_contents:
                    _, rerank = build_ticker_meta_tools(tkr)
                    retrieved_contents = rerank(
                        retrieved_contents,
                        top_k=parameters.k,
                        keywords=["operating income", "operating margin", "md&a", "exhibit 99"],
                    )
                    sources = _realign_sources(retrieved_contents, sources)
                if _TRACE:
                    _LOG.info(f"[DBG] RAG.retry_retrieved={len(retrieved_contents)}")


            # --- G) Fallback to Claude only when retrieval truly fails ---
            if not retrieved_contents:
                set_route("CLAUDE")
                with st.chat_message("assistant"):
                    out = call_claude_fallback(refined_query, model="anthropic/claude-3.7-sonnet") or "No response."
                    st.write("🤖 Claude response:\n\n" + out)
                return

            # 3) Final route tag + header bubble (route + sources)
            if q_explain:
                final_kind = (
                    "HYBRID JIT" if (used_jit and facts_md) else
                    ("JIT→RAG"   if (used_jit and not facts_md) else
                    ("HYBRID"    if facts_md else "RAG"))
                )
                set_route(final_kind, rag_docs=len(retrieved_contents), fact=(fact_kind if facts_md else None), ticker=(tkr if used_jit else None))
            else:
                final_kind = "RAG"
                set_route("RAG", rag_docs=len(retrieved_contents))

            with st.chat_message("assistant"):
                header = st.empty()
                box    = st.container()
                if facts_md:
                    header.markdown(f"**Route:** {final_kind}\n\n{facts_md}")
                else:
                    header.markdown(f"**Route:** {final_kind}\n")
                show_sources(box, sources)

        # 4) Streaming answer bubble
        start_time = time.time()

        # --- Helper: are all retrieved docs 8-K? (case-insensitive, checks source/filepath) ---
        def _all_are_8k(docs: list) -> bool:
            if not docs:
                return False
            def _src(d):
                md = (d.metadata or {})
                return (md.get("source") or md.get("filepath") or "").lower()
            return all("8-k" in _src(d) for d in docs)

        # Apply citation rules only when retrieval is used
        CITATION_ROUTES = {"RAG", "HYBRID", "HYBRID JIT", "JIT→RAG"}
        use_citation = st.session_state.get("last_route") in CITATION_ROUTES
        if _TRACE:
            _LOG.info(f"[DBG] streaming: citation={use_citation}, route={st.session_state.get('last_route')}")
            _LOG.info(f"[DBG] facts_md: len={len(facts_md or '')}, sha1={hashlib.sha1((facts_md or '').encode('utf-8')).hexdigest()[:10]}")

        # Extra guardrails when only 8-K is available
        extra_rules = ""
        if use_citation and _all_are_8k(retrieved_contents):
            extra_rules = (
                "\nIf 10-Q is not available, answer using 8-K (e.g., Exhibit 99.x press release) "
                "and cite it explicitly. Do not add disclaimers about lacking 10-Q."
            )

        # Build task text; inject TOOL FACT so the model must see it
        task_for_model = (
            f"{user_input}\n\n[TOOL FACT]\n{facts_md}\n\n[CITATION RULES]\n{CITATION_RULES}{extra_rules}"
            if use_citation else user_input
        )

        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # single, reusable area

            # Helpers (HTML-safe rendering + glue fixes + YoY guard)
            def _fix_glue(s: str) -> str:
                s = unicodedata.normalize("NFKC", s).replace("\u00A0", " ")
                s = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", s)     # 5.62The -> 5.62 The
                s = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", s)     # EPS38.8 -> EPS 38.8
                s = re.sub(r"([.,;:])(?!\s|$)", r"\1 ", s)     # space after punctuation
                s = re.sub(r"\s{2,}", " ", s)
                s = re.sub(r"\n{3,}", "\n\n", s)
                return s.strip()

            def _to_html(s: str) -> str:
                # keep bold, escape the rest, preserve <br>, disable ligatures/italics
                s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
                s = html.escape(s, quote=False) \
                        .replace("&lt;strong&gt;", "<strong>") \
                        .replace("&lt;/strong&gt;", "</strong>") \
                        .replace("\n", "<br>")
                return (
                    "<div style=\"white-space:pre-wrap;font-variant-ligatures:none;"
                    "-webkit-font-smoothing:antialiased;font-feature-settings:'liga' 0,'clig' 0;"
                    "font-style:normal;line-height:1.55;font-size:1rem;\">"
                    f"{s}"
                    "</div>"
                )

            YOY_RE = re.compile(r"(?i)\b(yoy|year[- ]over[- ]year)\b")
            NUM_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
            def _yoy_warn(s: str) -> str:
                if YOY_RE.search(s) and len(NUM_RE.findall(s)) < 2:
                    s += ("\n\n⚠️ Note: Mentions YoY but missing the prior-period value. "
                            "Ask me to fetch both periods and I’ll include them clearly.")
                return s

            # Build model context (TOOL FACT as highest-priority virtual doc)
            context_docs: list[Document] = []
            if facts_md:
                context_docs.append(
                    Document(
                        page_content=facts_md,
                        metadata={"source": "TOOL_FACT", "priority": "must_use", "weight": 2.0},
                    )
                )
            context_docs.extend(retrieved_contents or [])

            if _TRACE:
                head_src = context_docs[0].metadata.get("source") if context_docs else "NONE"
                _LOG.info(f"[DBG] context_docs: count={len(context_docs)}, head_source={head_src}")
                _LOG.info(f"[DBG] task_for_model[:300]={task_for_model[:300]!r}")

            # Generate with sentence-buffered streaming (single placeholder)
            with st.spinner("Generating financial insight from documents and chat history…"):
                streamer, fmt_prompts = answer_with_context(
                    llm, ctx_synthesis_strategy, task_for_model, chat_history, context_docs
                )
                _ = fmt_prompts  # keep for future debugging

                buffer, rendered = "", ""
                last_len = 0
                BOUNDARY_RE = re.compile(r"([.!?])+(\s|\n|$)")
                token_i = 0

                for token in streamer:
                    piece = llm.parse_token(token)
                    if _TRACE and token_i < 200:
                        _LOG.info(f"[TOK#{token_i:03d}] {piece!r}")
                    token_i += 1

                    buffer += piece

                    # Flush only at sentence/paragraph boundaries
                    if BOUNDARY_RE.search(buffer) or "\n\n" in buffer:
                        rendered += buffer
                        buffer = ""
                        preview = _fix_glue(rendered)  # no YoY note mid-stream
                        if len(preview) > last_len:    # update only if longer
                            message_placeholder.markdown(_to_html(preview) + " ▌", unsafe_allow_html=True)
                            last_len = len(preview)

                # Final flush: glue fix + YoY note (once) + HTML render
                rendered += buffer
                final_text = _yoy_warn(_fix_glue(rendered))
                html_out   = _to_html(final_text)
                message_placeholder.markdown(html_out, unsafe_allow_html=True)

                # Persist: store plain text for analysis + html for safe replay
                chat_history.append(f"question: {user_input}, answer: {final_text}")

        # Store assistant message for this turn (keep render mode for future replay)
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_text,  # plain text
            "render": "html",
            "html": html_out,       # exact HTML used above
        })
        logging.getLogger(__name__).info(f"\n--- Took {time.time() - start_time:.2f} seconds ---")

# Step 7: Command-line interface (CLI) for financial RAG chatbot
# CLI arguments to select model & strategy
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Financial RAG Chatbot")

    model_list = get_models()
    default_model = model_list[0]

    synthesis_strategy_list = get_ctx_synthesis_strategies()
    default_synthesis_strategy = synthesis_strategy_list[0]

    parser.add_argument(
        "--model",
        type=str,
        choices=model_list,
        help=f"Model to be used. Defaults to {default_model}.",
        required=False,
        const=default_model,
        nargs="?",
        default=default_model,
    )
    parser.add_argument(
        "--synthesis-strategy",
        type=str,
        choices=synthesis_strategy_list,
        help=f"Model to be used. Defaults to {default_synthesis_strategy}.",
        required=False,
        const=default_synthesis_strategy,
        nargs="?",
        default=default_synthesis_strategy,
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Number of chunks to return from the similarity search. Defaults to 2.",
        required=False,
        default=2,
    )
    return parser.parse_args()

# streamlit run rag_chatbot_app.py
if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)