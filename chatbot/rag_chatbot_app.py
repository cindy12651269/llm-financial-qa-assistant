import os, logging, re, traceback
import html
import argparse
import sys
import time
import threading
from pathlib import Path
from entities.document import Document
from chatbot.document_loader.format import Format, to_html, all_are_8k, stream_tokens
from chatbot.document_loader.text_splitter import create_recursive_text_splitter
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
from chatbot.financial_fetcher import(
compose_metric_header,fallback_news_lookup, fallback_stock_lookup, fallback_financial_metric_lookup, 
resolve_ticker_guarded, guess_ticker_from_path, is_ticker_doc, rebuild_sources)
from chatbot.ingest_pipeline import ingest_ticker
from chatbot.memory_builder import purge_ticker_from_index,infer_ticker
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
logging.basicConfig(level=logging.INFO)
logging.getLogger("TICKER_RESOLVE").setLevel(logging.INFO)

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

# Step 5: Routing logic helpers
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


# intent helpers 
INTENT_PRICE_RE   = re.compile(r"\b(price|quote|share price|last|close)\b", re.I)
INTENT_METRIC_RE  = re.compile(r"\b(eps|earnings per share|revenue|net income|operating (margin|income)|gross margin|guidance)\b", re.I)
INTENT_EXPLAIN_RE = re.compile(r"\b(why|explain|because|driver|impact|summari[sz]e|highlights|compare|vs\.?|versus|earnings call|conference call|prepared remarks)\b", re.I)
INTENT_COMPARE_RE = re.compile(r"\b(vs|versus|compare|comparison|prior quarter|previous quarter|qoq|quarter[- ]over[- ]quarter|yoy|year[- ]over[- ]year|y/y|change[d]?)\b", re.I)

def detect_intent(refined_query: str) -> dict:
    """Return compact intent flags."""
    return {
        "price":   bool(INTENT_PRICE_RE.search(refined_query or "")),
        "metric":  bool(INTENT_METRIC_RE.search(refined_query or "")),
        "explain": bool(INTENT_EXPLAIN_RE.search(refined_query or "")),
        "compare": bool(INTENT_COMPARE_RE.search(refined_query or "")),
    }

# Scan tool payloads once in order (price -> metric -> news)
def run_tools_fastpath(refined_query: str, want_hybrid: bool, trace_log=None):
    """
    If want_hybrid=True, capture a single 'Fact (tool)' block but DO NOT return early.
    If want_hybrid=False and a usable payload is found, render and return answered=True.
    Returns: (facts_md: str, fact_kind: Optional['price'|'metric'], answered: bool)
    """
    facts_md, fact_kind, answered = "", None, False
    for tool in (fallback_stock_lookup, fallback_financial_metric_lookup, fallback_news_lookup):
        docs = tool(refined_query) or []
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            if str(md.get("source_type", "")).lower() != "tool":
                continue
            p = (md.get("payload") or {})
            kind = p.get("kind")

            # price
            if kind == "price" and p.get("ticker") and p.get("value") is not None:
                header = f"**{p['ticker']}** latest price: **{p['value']}**"
                if p.get("as_of"): header += f" (as of {p['as_of']})"
                src = p.get("source", "financial_fetcher")

                if want_hybrid:
                    facts_md  = f"**Fact (tool):**\n\n{header}\n\n_Source: {src}_\n\n"
                    fact_kind = "price"
                else:
                    st.markdown(f"**Route:** TOOLS\n\n{header}\n\n_Source: {src}_")
                    answered = True
                break

            # metric
            if kind == "metric" and p.get("ticker") and p.get("metric") and p.get("value") is not None:
                payload = {
                    "ticker": p["ticker"], "metric": p["metric"], "value": p["value"],
                    "year": p.get("year"), "quarter": p.get("quarter"),
                    "as_of": p.get("as_of"), "basis": p.get("basis", "FY"),
                }
                try:
                    header = compose_metric_header(payload)
                except Exception:
                    qtxt = f" Q{payload['quarter']}" if payload["quarter"] else ""
                    header = f"**{payload['ticker']} {payload['metric']} {payload['year'] or ''}{qtxt}: {payload['value']}**"
                    if payload["as_of"]: header += f" (as of {payload['as_of']})"
                src = p.get("source", "financial_fetcher")

                if want_hybrid:
                    facts_md  = f"**Fact (tool):**\n\n{header}\n\n_Source: {src}_\n\n"
                    fact_kind = "metric"
                else:
                    st.markdown(f"**Route:** TOOLS\n\n{header}\n\n_Source: {src}_")
                    answered = True
                break

            # news (only for non-explanatory)
            if kind == "news" and (items := p.get("items")) and not want_hybrid:
                src = p.get("source", "financial_fetcher")
                top = items[:3]
                lines = "\n".join(f"- {it.get('title','(untitled)')} <{it.get('url','')}>" for it in top)
                st.markdown(f"**Route:** TOOLS\n\n**Latest headlines:**\n{lines}\n\n_Source: {src}_")
                answered = True
                break

        if answered or facts_md:
            break

    return facts_md, fact_kind, answered

# One-line tracer
_TRACE = os.environ.get("TRACE_ROUTING") == "1"
_LOG   = logging.getLogger("ROUTING")

def jit_and_retrieve(index, refined_query: str, tkr: str | None,
    k: int, parameters=None, used_jit_out: dict | None = None):

    T = (tkr or "").upper()
    used_jit = False
    if used_jit_out:
        used_jit_out["flag"] = False

    # Optional: purge residual tickers via env var, e.g. PURGE_TICKERS="CRM,NFLX"
    try:
        purge_env = os.getenv("PURGE_TICKERS", "")
        if purge_env:
            for tk in [s.strip().upper() for s in purge_env.split(",") if s.strip()]:
                removed = purge_ticker_from_index(index, tk)
                if _TRACE: _LOG.info(f"[PURGE] ticker={tk} removed_vectors={removed}")
    except Exception as e:
        _LOG.warning(f"[PURGE] failed: {e!r}")

    # JIT ingest (only for non-PRELOAD tickers) 
    if T and (T not in PRELOAD) and JIT_ENABLED:
        if not _has_enough_local_docs(T, cap=3):
            try:
                ensure_company_indexed(T, cap=3, timeout=10)
                hot_insert_ticker_docs(
                    index, T, DOCS_DIR,
                    getattr(parameters, "chunk_size", 512),
                    getattr(parameters, "chunk_overlap", 25)
                )
                used_jit = True
                if _TRACE: _LOG.info(f"[JIT] ingestion fired for {T}")
            except Exception as e:
                _LOG.warning(f"[JIT] error: {e!r}")

    # Initial retrieval 
    q = f"\"{T}\" {refined_query}" if T else refined_query
    docs, _ = index.similarity_search_with_threshold(q, k=k, threshold=0.05, exclude_tools=True)
    if _TRACE:
        _LOG.info(f"[RAG] initial query='{q}' got {len(docs)} docs")
        for d in docs[:3]:
            _LOG.info(f"[RAG] initial doc source={d.metadata.get('source')} ticker={d.metadata.get('ticker')}")

    # Rerank + hard filter 
    if T and docs:
        try:
            _, rerank = build_ticker_meta_tools(T)
            docs = rerank(docs, top_k=k, keywords=None)
            if _TRACE: _LOG.info(f"[RERANK] after rerank={len(docs)} docs")
        except Exception as e:
            if _TRACE: _LOG.info(f"[RERANK] skipped error={e!r}")
        before = len(docs)
        docs = [d for d in docs if is_ticker_doc(d, T)]
        if _TRACE:
            _LOG.info(f"[FILTER] kept {len(docs)}/{before} for {T}")
            for d in docs[:3]:
                _LOG.info(f"[FILTER] doc source={d.metadata.get('source')} ticker={d.metadata.get('ticker')}")
    srcs = rebuild_sources(docs)

    # Retry if empty 
    if T and not docs:
        rq = f"\"{T}\" {refined_query}"
        docs, _ = index.similarity_search_with_threshold(rq, k=k, threshold=0.05, exclude_tools=True)
        if _TRACE: _LOG.info(f"[RETRY] query='{rq}' got {len(docs)} docs")
        if docs:
            try:
                _, rerank = build_ticker_meta_tools(T)
                docs = rerank(docs, top_k=k, keywords=None)
                if _TRACE: _LOG.info(f"[RETRY] after rerank={len(docs)} docs")
            except Exception as e:
                if _TRACE: _LOG.info(f"[RETRY] rerank skipped error={e!r}")
            before = len(docs)
            docs = [d for d in docs if is_ticker_doc(d, T)]
            if _TRACE: _LOG.info(f"[RETRY] kept {len(docs)}/{before} for {T}")
        srcs = rebuild_sources(docs)

    # Fallback: still empty → keep unfiltered docs, prefer inferred T 
    if T and not docs:
        docs, _ = index.similarity_search_with_threshold(q, k=max(k, 5), threshold=0.08, exclude_tools=True)
        # Prefer docs whose inferred ticker == T; if none, keep original unfiltered result
        inferred = [d for d in docs if infer_ticker(getattr(d, "metadata", {}) or {}) == T]
        if inferred:
            docs = inferred
        srcs = rebuild_sources(docs)
        if _TRACE:
            _LOG.info(f"[FALLBACK] kept {len(docs)} docs for {T} (prefer-inferred)")

    if used_jit_out:
        used_jit_out["flag"] = used_jit
    return docs, srcs


# Answer with the currently loaded base LLM when no retrieved documents exist.
# If that fails, fall back to Claude and render consistently in HTML.
def run_llm_or_claude(user_input: str, llm, ctx_synthesis_strategy,
    chat_history: list, model_name: str, _LOG, set_route,) -> None:
    try: 
        # Route: base LLM (e.g., llama, qwen, phi…)
        set_route(f"LLM:{model_name}")
        with st.chat_message("assistant"):
            ph = st.empty()
            with st.spinner("Generating answer…"):
                # No context docs for pure LLM fallback
                streamer, _ = answer_with_context(
                    llm, ctx_synthesis_strategy, user_input, chat_history, []
                )
                # Reuse the shared streaming utility (sentence-buffered + HTML)
                plain, html_out = stream_tokens(llm, streamer, ph, yoy_check=False)
        # Persist to history/state
        chat_history.append(f"question: {user_input}, answer: {plain}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": plain,
            "render": "html",
            "html": html_out,
        })
        return

    except Exception as e:
        # If the chosen model fails entirely, fall back to Claude
        _LOG.warning(f"[LLM fallback error] {e!r}")
        set_route("CLAUDE")
        with st.chat_message("assistant"):
            try:
                out = call_claude_fallback(
                    user_input, model="anthropic/claude-3.7-sonnet"
                ) or "No response."
            except Exception as e2:
                out = f"Sorry—both the base model and Claude failed. Details: {e2!r}"

            # Render Claude's reply with the same HTML style
            html_out = to_html(out.strip())
            st.markdown(html_out, unsafe_allow_html=True)

        # Persist Claude response
        st.session_state.messages.append({
            "role": "assistant",
            "content": out.strip(),
            "render": "html",
            "html": html_out,
            "via": "Claude",
        })
        chat_history.append(f"question: {user_input}, answer: {out.strip()} (via Claude)")

# One-stop header renderer: route badge (HTML), optional TOOL FACT (HTML via _to_html), and collapsible RAG sources list (HTML)
# Render the sources list inside the provided container (same chat bubble)
def render_header_html(route_kind: str, facts_md: str | None, sources: list[dict]) -> None:
    # Top line + optional fact
    header = st.empty()
    route_html = (
        "<div style='white-space:pre-wrap;font-variant-ligatures:none;"
        "-webkit-font-smoothing:antialiased;font-feature-settings:\"liga\" 0,\"clig\" 0;"
        "font-style:normal;line-height:1.55;font-size:1rem;'>"
        f"<strong>Route:</strong> {html.escape(route_kind)}</div>"
    )
    header.markdown(route_html + (to_html(facts_md) if facts_md else ""), unsafe_allow_html=True)
    # Sources (collapsible), skip if none
    if not sources:
        return

    with st.container():
        exp = st.expander("📚 Show retrieval sources")
        with exp:
            for s in sources:
                score = f"{s.get('score', 0.0):.3f}"
                org   = s.get("organization", "")
                rtype = s.get("report_type", "")
                fy    = s.get("fiscal_year", "")
                src   = s.get("source", "")
                title = s.get("title") or s.get("content_preview") or ""

                block = (
                    "<div style='white-space:pre-wrap;font-variant-ligatures:none;"
                    "-webkit-font-smoothing:antialiased;font-feature-settings:\"liga\" 0,\"clig\" 0;"
                    "font-style:normal;line-height:1.55;font-size:0.95rem;margin-bottom:0.8em;'>"
                    f"<strong>Score:</strong> {score} • {html.escape(org)} {html.escape(rtype)} {html.escape(str(fy))}<br>"
                    f"<a href='{html.escape(src)}' target='_blank'>{html.escape(src)}</a><br>"
                    f"{html.escape(title)}"
                    "</div>"
                )
                st.markdown(block, unsafe_allow_html=True)
                
# Stream the final answer into UI.
def stream_answer(llm, ctx_synthesis_strategy, user_input, facts_md, retrieved_docs, chat_history):
    route = st.session_state.get("last_route")
    CITATION_ROUTES = {"RAG", "HYBRID", "HYBRID JIT", "JIT→RAG"}
    # citation rule handling
    extra_rules = ""
    if route in CITATION_ROUTES and all_are_8k(retrieved_docs):
        extra_rules = ("\nIf 10-Q is not available, answer using 8-K "
                       "and cite it explicitly.")

    rules_text = globals().get("CITATION_RULES", "Use ONLY the provided CONTEXT. "
                     "Prefer TOOL FACT if conflict. "
                     "Quote at least TWO numeric sentences with citations.")

    task_for_model = (
        f"{user_input}\n\n[TOOL FACT]\n{facts_md}\n\n[CITATION RULES]\n{rules_text}{extra_rules}"
        if route in CITATION_ROUTES else user_input
    )

    # build context
    context_docs = []
    if facts_md:
        context_docs.append(Document(
            page_content=facts_md,
            metadata={"source": "TOOL_FACT", "priority": "must_use", "weight": 2.0}
        ))
    context_docs.extend(retrieved_docs or [])

    # streaming with helper
    with st.chat_message("assistant"):
        ph = st.empty()
        with st.spinner("Generating financial insight…"):
            streamer, _ = answer_with_context(llm, ctx_synthesis_strategy, task_for_model, chat_history, context_docs)
            plain, html_out = stream_tokens(llm, streamer, ph, yoy_check=True)

    # persist
    chat_history.append(f"question: {user_input}, answer: {plain}")
    st.session_state.messages.append({
        "role": "assistant",
        "content": plain,
        "render": "html",
        "html": html_out,
    })

# Step 6: Main logic for launching financial RAG chatbot
def main(parameters) -> None:
    # 1) init UI/LLM/index/history
    root_folder        = Path(__file__).resolve().parent.parent
    model_folder       = root_folder / "models"
    vector_store_path  = root_folder / "vector_store" / "docs_index"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    model_name                 = parameters.model
    synthesis_strategy_name    = parameters.synthesis_strategy

    init_page(root_folder)
    llm = load_llm_client(model_folder, model_name)
    chat_history = init_chat_history(2)
    ctx_synthesis_strategy = load_ctx_synthesis_strategy(synthesis_strategy_name, _llm=llm)
    index = load_index(vector_store_path)
    reset_chat_history(chat_history)
    init_welcome_message()
    display_messages_from_history()

    # 2) read user input
    user_input = st.chat_input("Ask a financial question (e.g. P/E ratio, ESG risk, portfolio)…")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    # 3) intent + guarded ticker
    def set_route(kind: str, *, rag_docs: int | None = None, fact: str | None = None, ticker: str | None = None):
        st.session_state.last_route = kind
        if _TRACE:
            _LOG.info("[PATH] " + _route_line(kind, rag_docs=rag_docs, fact=fact, ticker=ticker))
    
    # refine + intent + ticker (guarded; no forced usage) 
    with st.spinner("📊 Refining your question and preparing retrieval…"):
        refined_query = refine_question(llm, user_input, chat_history=chat_history)
        intent        = detect_intent(refined_query)           # returns dict: {"price":bool,"metric":bool,"explain":bool,"compare":bool}
        tkr           = resolve_ticker_guarded(user_input)     # returns None for generic concept questions

        # LLM-only: no ticker and no numeric intent → answer directly with current model
        if not tkr and not (intent["price"] or intent["metric"]):
            run_llm_or_claude(
                user_input=user_input,
                llm=llm,
                ctx_synthesis_strategy=ctx_synthesis_strategy,
                chat_history=chat_history,
                model_name=model_name,
                _LOG=_LOG,
                set_route=set_route,
            )
            return
        # 4) tools fast-path (capture fact for HYBRID if explanatory/compare)
        # Tools fast-path (stock → metric → news). May capture facts for HYBRID. 
        want_hybrid = bool(intent["explain"] or intent["compare"])
        facts_md, fact_kind, answered = run_tools_fastpath(refined_query, want_hybrid)
        if answered:
            # tools already rendered final answer (non-explanatory path)
            return
        # 5) JIT ingest + RAG retrieval
        # JIT + RAG retrieval (anchored by ticker if present) 
        used_jit_box = {"flag": False}
        retrieved_docs, sources = jit_and_retrieve(
            index=index,
            refined_query=refined_query,
            tkr=tkr,                 # ← use 'tkr' (matches function)
            k=parameters.k,
            used_jit_out=used_jit_box,  # ← use 'used_jit_out' (matches function)
        )

        # 6) fallback to base LLM; Claude only if that fails
        if not (retrieved_docs or facts_md):
           run_llm_or_claude(
               user_input=user_input,
               llm=llm,
               ctx_synthesis_strategy=ctx_synthesis_strategy,
               chat_history=chat_history,
               model_name=model_name,
               _LOG=_LOG,
               set_route=set_route,
           )
           return

        # 7) render header (route + sources) in HTML
        final_kind = (
            "HYBRID JIT" if (used_jit_box["flag"] and facts_md) else
            ("JIT→RAG"   if (used_jit_box["flag"] and not facts_md) else
            ("HYBRID"    if facts_md else "RAG"))
        )
        set_route(
            final_kind,
            rag_docs=len(retrieved_docs),
            fact=(fact_kind if facts_md else None),
            ticker=(tkr if used_jit_box["flag"] else None),
        )
        with st.chat_message("assistant"):
            render_header_html(final_kind, facts_md, sources)

    # 8) stream final answer (inject TOOL FACT + docs when present)
    stream_answer(
        llm=llm,
        ctx_synthesis_strategy=ctx_synthesis_strategy,
        user_input=user_input,
        facts_md=facts_md,
        retrieved_docs=retrieved_docs,
        chat_history=chat_history,
    )

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