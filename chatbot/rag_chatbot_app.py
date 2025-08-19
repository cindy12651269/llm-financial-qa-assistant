import os, logging, re
import argparse
import sys
import time
import re
import json
import threading
from pathlib import Path
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
    # First check docs/<TICKER>/*.md
    subdir = DOCS_DIR / tkr
    if subdir.exists():
        hits = [p for p in subdir.glob("*.md") if ("8-K" in p.name or "10-Q" in p.name)]
        if len(hits) >= cap:
            return True

    # Then check flat files docs/<TICKER>_FORM_YYYY-MM-DD.md
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
def _route_line(kind: str, *, rag_docs: int | None = None, fact: str | None = None) -> str:
    """
    Examples:
      HYBRID — metric fact + explanation (RAG docs=4)
      TOOLS(metric)
      TOOLS(price)
      RAG docs=3
      CLAUDE (no retrieval)
    """
    if kind == "HYBRID":
        part = f"HYBRID — {('metric' if fact=='metric' else 'price')} fact + explanation"
        return part + (f" (RAG docs={rag_docs})" if rag_docs is not None else "")
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
    return md.get("payload") if str(md.get("source_type","")).lower() == "tool" else None

# Render a compact RAG sources expander.
def show_sources(message_placeholder, srcs: list[dict]):
    message_placeholder.markdown("📚 Found relevant context via **RAG**. (expand below for sources)")
    with st.expander("Show retrieval sources"):
        for s in srcs:
            st.markdown(
                f"- **Score** {s.get('score',0.0):.3f} • "
                f"{s.get('organization','')} {s.get('report_type','')} {s.get('fiscal_year','')}  \n"
                f"<{s.get('source','')}>  \n"
                f"{(s.get('title') or s.get('content_preview') or '')}"
            )

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

        # 1) Retrieve related finance documents (with previews) + stream final answer
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            retrieved_contents: list[Document] = []
            sources: list[dict] = []
            route = "INIT"

            # One-line route tracer (prints ONE final, company-agnostic line)
            _TRACE = os.environ.get("TRACE_ROUTING") == "1"
            _LOG   = logging.getLogger("ROUTING")
            
            # Set final route and optionally print a single terminal line.
            def set_route(kind: str, *, rag_docs: int | None = None, fact: str | None = None):
                nonlocal route
                route = kind
                if _TRACE:
                    _LOG.info("[PATH] " + _route_line(kind, rag_docs=rag_docs, fact=fact))

            with st.spinner("📊 Refining your financial query and retrieving relevant documents…"):
                # Resolve ticker from query, refine question, and detect intent
                orig_query    = user_input
                refined_query = refine_question(llm, user_input, chat_history=chat_history)
                tkr = (resolve_ticker_from_query(orig_query) or "").upper()

                # Intent (compact)
                PRICE_RE   = re.compile(r"\b(price|quote|share price|last|close)\b", re.I)
                METRIC_RE  = re.compile(r"\b(eps|earnings per share|revenue|net income|operating (margin|income)|gross margin|guidance)\b", re.I)
                EXPLAIN_RE = re.compile(r"\b(why|explain|because|driver|impact|summari[sz]e|highlights|compare|vs\.?|versus|earnings call|conference call|prepared remarks)\b", re.I)
                q_price, q_metric, q_explain = map(
                    bool,
                    (PRICE_RE.search(refined_query or ""),
                     METRIC_RE.search(refined_query or ""),
                     EXPLAIN_RE.search(refined_query or ""))
                )

                # HYBRID: explain-like + (metric or price)
                if q_explain and (q_metric or q_price):
                    facts_md = ""
                    fact_kind = None  # "metric" | "price"

                    # (1) metric fact preferred
                    if q_metric:
                        for d in (fallback_financial_metric_lookup(refined_query) or []):
                            p = tool_payload(d)
                            if p and p.get("kind") == "metric" and p.get("value") is not None:
                                try:
                                    header = compose_metric_header(p)
                                except Exception:
                                    t,m,v,y,q = p.get("ticker"),p.get("metric"),p.get("value"),p.get("year"),p.get("quarter")
                                    header = f"**{t} {m} {y or ''}{(' Q'+str(q)) if q else ''}: {v}**"
                                facts_md  = f"**Fact (tool):**\n\n{header}\n\n_Source: {p.get('source','financial_fetcher')}_\n\n"
                                fact_kind = "metric"
                                break

                    # (2) price fact if no metric captured
                    if not facts_md and q_price:
                        for d in (fallback_stock_lookup(refined_query) or []):
                            p = tool_payload(d)
                            if p and p.get("kind") == "price" and p.get("value") is not None:
                                asof    = f" (as of {p.get('as_of','')})" if p.get("as_of") else ""
                                facts_md  = f"**Fact (tool):** **{p['ticker']}** latest price: **{p['value']}**{asof}\n\n_Source: {p.get('source','financial_fetcher')}_\n\n"
                                fact_kind = "price"
                                break

                    # JIT (non-blocking), then RAG for explanations/citations
                    if JIT_ENABLED and tkr:
                        try: ensure_company_indexed(tkr, cap=3, timeout=10)
                        except Exception: pass

                    retrieved_contents, sources = index.similarity_search_with_threshold(
                        query=refined_query, k=parameters.k, threshold=0.05, exclude_tools=True
                        )
                    if not retrieved_contents:
                        set_route("CLAUDE")
                        out = call_claude_fallback(refined_query, model="anthropic/claude-3.7-sonnet") or "No response."
                        message_placeholder.write("🤖 Claude response:\n\n" + out)
                        st.session_state.last_route = route
                        return

                    # Final single-line path in terminal, and UI route badge
                    set_route("HYBRID" if facts_md else "RAG", rag_docs=len(retrieved_contents), fact=fact_kind)
                    message_placeholder.markdown(f"**Route:** {'HYBRID' if facts_md else 'RAG'}\n\n" + (facts_md or ""))
                    show_sources(message_placeholder, sources)
                    st.session_state.last_route = route

                # Non-explanation: API Tools 
                else: # metric → tools
                    # 1) Tools (stock -> metric -> news) — API first, binary routing
                    if not retrieved_contents:
                        for tool in [fallback_stock_lookup, fallback_financial_metric_lookup, fallback_news_lookup]:
                            fallback_docs = tool(refined_query)  # ← 用 refined_query
                            if not fallback_docs:
                                continue

                            # Fast-path: scan all docs for a usable tool payload; answer & return
                            for d in fallback_docs:
                                md = getattr(d, "metadata", {}) or {}
                                if str(md.get("source_type", "")).lower() != "tool":
                                    continue
                                payload = md.get("payload") or {}
                                kind = payload.get("kind")

                                # price
                                if kind == "price" and payload.get("ticker") and payload.get("value") is not None:
                                    t, v = payload["ticker"], payload["value"]
                                    asof = payload.get("as_of", "")
                                    src  = payload.get("source", "financial_fetcher")
                                    route = "TOOLS"
                                    message_placeholder.markdown(
                                        f"**Route:** {route}\n\n"
                                        f"**{t}** latest price: **{v}**"
                                        + (f" (as of {asof})" if asof else "")
                                        + f"\n\n_Source: {src}_"
                                    )
                                    st.session_state.messages.append({"role": "assistant", "content": f"{t} latest price: {v} (src: {src})"})
                                    st.session_state.last_route = route
                                    return

                                # metric (EPS / revenue / etc.)
                                if kind == "metric" and payload.get("ticker") and payload.get("metric") and payload.get("value") is not None:
                                    t, m, val = payload["ticker"], payload["metric"], payload["value"]
                                    y, q, asof = payload.get("year"), payload.get("quarter"), payload.get("as_of", "")
                                    basis = payload.get("basis", "FY")
                                    cal_y = payload.get("calendar_year")
                                    cal_q = payload.get("calendar_quarter")
                                    note  = payload.get("note")
                                    qtxt  = f" Q{q}" if q else ""
                                    src   = payload.get("source", "financial_fetcher")
                                    route = "TOOLS"  # 或 set_route("TOOLS", fact="metric")

                                    # Compose an informative header that shows the basis clearly
                                    header  = f"**{t} {m} {y or ''}{qtxt}: {val}**"
                                    if asof: header += f" (as of {asof})"
                                    header += f"  \n_Basis: **{basis}**_"
                                    if basis == "CY" and (cal_y or cal_q):
                                        header += f" — Calendar: **{cal_y or ''}{' Q'+str(cal_q) if cal_q else ''}**"
                                    if note:
                                        header += f"  \n_Note: {note}_"

                                    message_placeholder.markdown(f"**Route:** {route}\n\n{header}\n\n_Source: {src}_")
                                    st.session_state.last_route = route
                                    return

                                # news
                                if kind == "news" and (items := payload.get("items")):
                                    src = payload.get("source", "financial_fetcher")
                                    top = items[:3]
                                    lines = [f"- {it.get('title','(untitled)')} <{it.get('url','')}>" for it in top]
                                    route = "TOOLS"  # 或 set_route("TOOLS", fact="news")
                                    message_placeholder.markdown(
                                        f"**Route:** {route}\n\n"
                                        f"**Latest headlines:**\n" + "\n".join(lines) + f"\n\n_Source: {src}_"
                                    )
                                    st.session_state.messages.append({"role": "assistant", "content": "Latest headlines shown (Tools)."})
                                    st.session_state.last_route = route
                                    return

                    # API tools didn’t answer → RAG → Claude
                    retrieved_contents, sources = index.similarity_search_with_threshold(
                        query=refined_query, k=parameters.k, threshold=0.05, exclude_tools=True
                    )
                    if not retrieved_contents:
                        set_route("CLAUDE")  # prints: [PATH] CLAUDE (no retrieval)
                        out = call_claude_fallback(refined_query, model="anthropic/claude-3.7-sonnet") or "No response."
                        message_placeholder.write("🤖 Claude response:\n\n" + out)
                        st.session_state.last_route = route
                        return
                    # Prints: [PATH] RAG docs=N
                    set_route("RAG", rag_docs=len(retrieved_contents))  
                    message_placeholder.markdown("**Route:** RAG\n")
                    show_sources(message_placeholder, sources)
                    st.session_state.last_route = route

        # 2. Stream financial analysis response
        start_time = time.time()

        # Add once, near this block: citation rules for filing-grounded answers 
        CITATION_RULES = (
            "Use ONLY the provided documents.\n"
            "Quote at least TWO numeric sentences from filings (10-K/10-Q/8-K/IR), "
            "showing exact figures/percent changes, then explain.\n"
            "Do NOT use news or third-party sites; if filings lack the answer, say so briefly."
        )

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Build the final task prompt:
            # - For RAG/HYBRID routes, enforce filing citations via rules.
            # - For other routes (TOOLS/CLAUDE), keep original user_input.
            if st.session_state.get("last_route") in {"RAG", "HYBRID"}:
                # Enforce numeric filing quotes + no news.
                task_with_rules = f"{user_input}\n\n[CITATION RULES]\n{CITATION_RULES}"
                task_for_model = task_with_rules
            else:
                task_for_model = user_input

            with st.spinner(text="Generating financial insight from documents and chat history – hang tight! "):
                # Keep your existing synthesis strategy and retrieval context
                streamer, fmt_prompts = answer_with_context(
                    llm,
                    ctx_synthesis_strategy,
                    task_for_model,            # Pass the task WITH citation rules when RAG/HYBRID
                    chat_history,
                    retrieved_contents         # Same doc list as before
                )
                # Token streaming (unchanged)
                for token in streamer:
                    full_response += llm.parse_token(token)
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                chat_history.append(f"question: {user_input}, answer: {full_response}")

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        took = time.time() - start_time
        logger.info(f"\n--- Took {took:.2f} seconds ---")

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