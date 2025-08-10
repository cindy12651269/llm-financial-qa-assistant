import os
import argparse
import sys
import time
import re
import json
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
from chatbot.financial_fetcher import get_stock_price, get_financial_news, get_financial_metric, get_cik_from_ticker
from claude_api import call_claude_fallback, SYS_PROMPT
from helpers.log import get_logger
from helpers.prettier import prettify_source

# --- helpers for Step 1 ------------------------------------------------------
def _as_text(x, fallback: str = "") -> str:
    """Coerce any value to a non-empty string; fall back if None/empty."""
    if x is None:
        return fallback
    s = str(x)
    return s if s.strip() else fallback

def _preview_str(s: str, limit: int = 800) -> str:
    """Pretty-print JSON if possible; otherwise return a trimmed plaintext preview."""
    if not isinstance(s, str):
        s = _as_text(s, "")
    txt = s.strip()
    if not txt:
        return ""
    # try to pretty JSON
    if txt.startswith("{") or txt.startswith("["):
        try:
            import json
            return "```json\n" + json.dumps(json.loads(txt), indent=2)[:limit] + "\n```"
        except Exception:
            pass
    return txt[:limit]

# --------------------------- STEP 1 (drop-in) --------------------------------

    
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)

def escape_markdown(text: str) -> str:
    """Escape Markdown control chars so Streamlit won't render italics/monospace accidentally."""
    if not isinstance(text, str):
        return text
    return re.sub(r'([\\`*_{}[\]()#+\-!.>])', r'\\\1', text)

# Set Streamlit configuration for financial chatbot
st.set_page_config(page_title="Financial RAG Chatbot", page_icon="💰", initial_sidebar_state="collapsed")


# Load the fine-tuned financial LLM client
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

def extract_possible_tickers(query: str) -> list[str]:
    # Keep original case; only pick explicit uppercase tokens or cashtags
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


# ----------------- fallbacks -----------------

def fallback_stock_lookup(query: str) -> List[Document]:
    """
    Fetch latest stock price via financial_fetcher.get_stock_price
    only if the query looks like a price question and ticker is resolvable.

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

def fallback_news_lookup(query: str) -> List[Document]:
    """
    Fetch latest financial news for a resolved ticker if possible; otherwise use the raw query term.

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


def fallback_financial_metric_lookup(query: str) -> List[Document]:
    """
    Simple metric resolver for Tools fast-path.
    Supports FY/CY/bare year-quarter patterns; backend still queried on fiscal (FY) data.
    """
    if not re.search(r"\b(eps|earnings per share|revenue|net income|operating (income|margin))\b", query, re.I):
        return []

    text = query.strip()
    upper = text.upper()

    # --- Parse year & quarter and decide basis ---
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
        else:
            # bare "2024 Q4" / "Q4 2024" → treat as CY
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
            else:
                # just a year → default FY
                y = re.search(r"\b(20\d{2})\b", upper)
                year = int(y.group(1)) if y else None
                quarter = None
                # basis remains FY

    if not year:
        return []

    # --- Resolve ticker (sanitize keywords to avoid false positives like GAAP) ---
    query_for_ticker = re.sub(
        r"(?i)\b(NON[-\s]?GAAP|GAAP|EPS|EARNINGS|FISCAL|CALENDAR|FY|CY|Q[1-4]|REVENUE|NET\s+INCOME|OPERATING\s+(INCOME|MARGIN))\b",
        " ",
        text,
    )
    ticker = resolve_ticker_from_query(query_for_ticker)
    if not ticker:
        return []

    # --- Map metric keyword ---
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

    # --- Fetch (FY-based backend) ---
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
        "basis": basis,                         # <-- CRUCIAL: CY or FY
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

# Initialize UI branding for financial assistant
def init_page(root_folder: Path) -> None:
    """
    Initializes the page configuration for the application.
    """
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

# Main logic for launching financial RAG chatbot
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

        # Step 1: Retrieve related finance documents (with previews)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            retrieved_contents = []
            sources = []
            # optional: simple routing flag for debugging
            route = "INIT"

            with st.spinner(
                text="📊 Refining your financial query and retrieving relevant documents– hang tight! "
                     "This should take seconds."
            ):
                refined_user_input = refine_question(llm, user_input, chat_history=chat_history)
                # 1) Tools (stock -> metric -> news) — API first, binary routing
                if not retrieved_contents:
                    for tool in [fallback_stock_lookup, fallback_financial_metric_lookup, fallback_news_lookup]:
                        fallback_docs = tool(refined_user_input)
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
                                t, v, asof = payload["ticker"], payload["value"], payload.get("as_of", "")
                                src = payload.get("source", "financial_fetcher")
                                route = "TOOLS"
                                message_placeholder.markdown(
                                    f"**Route:** {route}\n\n"
                                    f"**{t}** latest price: **{v}** (as of {asof})\n\n"
                                    f"_Source: {src}_"
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
                                note = payload.get("note")
                                qtxt = f" Q{q}" if q else ""
                                src = payload.get("source", "financial_fetcher")
                                route = "TOOLS"

                                # Compose an informative header that shows the basis clearly
                                header = f"**{t} {m} {y or ''}{qtxt}: {val}** (as of {asof})  \n"
                                header += f"_Basis: **{basis}**_"
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
                                route = "TOOLS"
                                message_placeholder.markdown(
                                    f"**Route:** {route}\n\n"
                                    f"**Latest headlines:**\n" + "\n".join(lines) + f"\n\n_Source: {src}_"
                                )
                                st.session_state.messages.append({"role": "assistant", "content": "Latest headlines shown (Tools)."})
                                st.session_state.last_route = route
                                return

                        # No fast-path payload from this tool → try next tool (do NOT inject; keep binary clean)
                        # (do nothing here; loop continues to next tool)

                # 2) RAG — only if API could not answer
                if not retrieved_contents:
                    retrieved_contents, sources = index.similarity_search_with_threshold(
                        query=refined_user_input, k=parameters.k, exclude_tools=True  # ensure tool docs are hidden in RAG list
                    )
                    if not retrieved_contents:
                        # 3) Claude fallback — only when both Tools and RAG failed
                        route = "CLAUDE"
                        claude_response = call_claude_fallback(
                            refined_user_input,
                            model="anthropic/claude-3.7-sonnet"
                        )
                        full_response += "🤖 Claude response:\n\n"
                        full_response += (claude_response or "No response.")
                        message_placeholder.write(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        st.session_state.last_route = route
                        return
                
                    # 4) Compact UI for RAG only
                    route = "RAG"
                    message_placeholder.markdown("📚 Found relevant context via **RAG**. (expand below for sources)")
                    with st.expander("Show retrieval sources"):
                        for s in sources:
                            st.markdown(
                                f"- **Score** {s.get('score', 0.0):.3f} • "
                                f"{s.get('organization','')} {s.get('report_type','')} {s.get('fiscal_year','')}  \n"
                                f"`{s.get('source','')}`  \n"
                                f"{s.get('content_preview','')}"
                            )
                    st.session_state.messages.append({"role": "assistant", "content": "📚 Found relevant context via RAG."})
                    st.session_state.last_route = route

        # Step 2: Stream financial analysis response
        start_time = time.time()
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner(text="Generating financial insight from documents and chat history – hang tight! "):
                streamer, fmt_prompts = answer_with_context(
                    llm, ctx_synthesis_strategy, user_input, chat_history, retrieved_contents
                )
                for token in streamer:
                    full_response += llm.parse_token(token)
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                chat_history.append(f"question: {user_input}, answer: {full_response}")

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        took = time.time() - start_time
        logger.info(f"\n--- Took {took:.2f} seconds ---")


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