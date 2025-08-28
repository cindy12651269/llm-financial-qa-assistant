import os, logging, re
import argparse
import sys
import time
import re
import json
import threading
from pathlib import Path
import glob
from entities.document import Document
from typing import List, Optional
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
    resolve_ticker_from_query,get_stock_price, get_financial_news, 
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
# Enable/disable via env var; set RAG_JIT_ENABLED=0 to turn off
JIT_ENABLED = os.getenv("RAG_JIT_ENABLED", "1") == "1"

# Preloaded companies (skip JIT if found here)
PRELOAD = {"AAPL", "MSFT", "TSLA", "NVDA", "AMZN"}

# Pattern to detect explanation/analysis questions
EXPLAIN_RE = re.compile(r"\b(why|explain|reason|driver|cause|impact)\b", re.I)

# Base docs folder (adjust to match your project)
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# JIT helpers: Return True if the query likely asks for explanations/analysis.
def is_explanation_query(q: str) -> bool:
    return bool(EXPLAIN_RE.search(q or ""))

# Check if we already have enough local docs for this ticker.
def _has_enough_local_docs(ticker: str, cap: int = 3) -> bool:
    """
    Check if we already have >=cap local markdown files for this ticker
    that look like 8-K or 10-Q forms. This is a fast local guard to skip redundant ingestion.
    """
    tkr = ticker.upper()
    # Case 1: check docs/<TICKER>/*.md 
    subdir = DOCS_DIR / tkr
    if subdir.exists():
        hits = [p for p in subdir.glob("*.md") if ("8-K" in p.name or "10-Q" in p.name)]
        if len(hits) >= cap:
            return True

    # Case 2: Check flat files docs/<TICKER>_FORM_YYYY-MM-DD.md
    flat = [p for p in DOCS_DIR.glob(f"{tkr}_*_*.md") if ("8-K" in p.name or "10-Q" in p.name)]
    return len(flat) >= cap

# Run ingestion in a short-lived thread; wait up to `timeout` seconds, and return without blocking the main response.
# Fire-and-forget ingestion for a single ticker, Skip if local doc count >=cap.
def ensure_company_indexed(ticker: str, cap: int = 3, timeout: int = 10) -> None:
    if _has_enough_local_docs(ticker, cap=cap):
        return
    def _job():
        try:
            ingest_ticker(ticker, forms=["8-K", "10-Q"], max_per_ticker=cap, since="2023-01-01")
        except Exception:
            pass  # Silent failure
    t = threading.Thread(target=_job, daemon=True)
    t.start()
    t.join(timeout=timeout)  # Return after timeout regardless of completion
    logging.warning(  # Always warn user to rebuild persistent vector index
        "[JIT] %s ingestion triggered. If new .md files were written to docs/, "
        "run memory_builder to include them in the persistent index.",
        (ticker or "").upper(),
    )
    
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

# Allow users to clear financial conversation memory
def reset_chat_history(chat_history: ChatHistory) -> None:
    """
    Initializes the chat history, allowing users to clear the conversation.
    """
    clear_button = st.sidebar.button("🗑️ Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []
        chat_history.clear()

# Display past user/assistant interactions
def display_messages_from_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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
            tkr = (resolve_ticker_from_query(orig_query) or "").upper()

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
            used_jit = False
            if q_explain and tkr and JIT_ENABLED:
                try:
                    PRELOAD = {"AAPL","MSFT","GOOGL","AMZN","NVDA","META","BRK-B","TSLA","UNH","JNJ"}
                    # Only trigger JIT when the ticker is NOT in PRELOAD and we don't already have enough local docs.
                    if (tkr not in PRELOAD) and (not _has_enough_local_docs(tkr, cap=3)):
                        ensure_company_indexed(tkr, cap=3, timeout=10)  # fire-and-forget; returns after `timeout`
                        used_jit = True
                        if _TRACE:
                            _LOG.info(f"[DBG] JIT.ensure_company_indexed fired for ticker={tkr}")
                except Exception as e:
                    if _TRACE:
                        _LOG.warning(f"[DBG] JIT.ensure_company_indexed error: {e!r}")         
            # Log current index stats to confirm how many docs/chunks are loaded in memory
            try:
                stats = getattr(getattr(index, "_collection", None), "count", lambda: "?")()
            except Exception:
                stats = "?"
            _LOG.info(f"[INDEX] Current docs in memory: {stats}")

            retrieved_contents, sources = index.similarity_search_with_threshold(
                query=refined_query, k=parameters.k, threshold=0.05, exclude_tools=True
            )
            if _TRACE:
                _LOG.info(f"[DBG] RAG.retrieved={len(retrieved_contents)}")

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

        # Minimal post-processor: remove leaked markers, tidy spacing, warn on YoY without two numbers
        def _post_sanitize_answer(text: str) -> str:
            """Minimal sanitizer: remove markers, clean spacing, warn if YoY lacks numbers."""
            if not text:
                return text
            out = re.sub(r"\[(TOOL FACT|FACT|CONTEXT|CITATION RULES|RULES)\]\s*", "", text, flags=re.I)
            out = re.sub(r"\s{2,}", " ", out)       # collapse multiple spaces
            out = re.sub(r"(?m)^[ \t]+", "", out)   # left-trim each line
            if re.search(r"(?i)\b(yoy|year[- ]over[- ]year)\b", out):
                if len(re.findall(r"\b\d+(?:\.\d+)?%?\b", out)) < 2:
                    out += ("\n\n_⚠️ Note: Mentions YoY but missing the prior-period value. "
                            "Ask me to fetch both periods and I’ll include them clearly._")
            return out

        # Only enforce citation rules for routes that used retrieval
        CITATION_ROUTES = {"RAG", "HYBRID", "HYBRID JIT", "JIT→RAG"}
        use_citation = st.session_state.get("last_route") in CITATION_ROUTES
        if _TRACE:
            _LOG.info(f"[DBG] streaming: use_citation_rules={use_citation}, route={st.session_state.get('last_route')}")

        # Put the TOOL FACT directly into the model prompt (forces the model to see it)
        task_for_model = (
            f"{user_input}\n\n[TOOL FACT]\n{facts_md}\n\n[CITATION RULES]\n{CITATION_RULES}"
            if use_citation else user_input
        )

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("Generating financial insight from documents and chat history…"):
                # Assemble context: keep TOOL FACT as a high-priority virtual doc
                context_docs: list[Document] = []
                if facts_md:
                    # <-- as requested: ensure TOOL FACT is injected into context
                    context_docs.append(
                        Document(
                            page_content=facts_md,
                            metadata={"source": "TOOL_FACT", "priority": "must_use", "weight": 2.0}
                        )
                    )
                context_docs.extend(retrieved_contents or [])

                # Debug: verify TOOL FACT sits at the head of the context
                if _TRACE:
                    head_src = context_docs[0].metadata.get("source") if context_docs else "NONE"
                    _LOG.info(f"[DBG] context_docs: count={len(context_docs)}, head_source={head_src}")
                    _LOG.info(f"[DBG] task_for_model prefix:\n{task_for_model[:300]}...")

                # Use task_for_model (with FACT + RULES), not raw user_input
                streamer, fmt_prompts = answer_with_context(
                    llm, ctx_synthesis_strategy, task_for_model, chat_history, context_docs
                )
                _ = fmt_prompts  # keep for debugging; silence "not accessed"

                # Token streaming
                for token in streamer:
                    full_response += llm.parse_token(token)
                    message_placeholder.markdown(full_response + "▌")

                # Sanitize once at the end (removes leaked headers, fixes spacing, adds YoY note if needed)
                final_text = _post_sanitize_answer(full_response)
                message_placeholder.markdown(final_text)

                chat_history.append(f"question: {user_input}, answer: {final_text}")

        st.session_state.messages.append({"role": "assistant", "content": final_text})
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