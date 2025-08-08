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
    Attempts to fetch stock price via financial_fetcher.get_stock_price, only if the
    query actually looks like a stock price question, and only for validated tickers.
    """
    if not re.search(r'\b(price|stock|share|quote|trading|last|close)\b', query, re.I):
        return []

    ticker = resolve_ticker_from_query(query)
    if not ticker:
        return []

    data = get_stock_price(ticker)
    if not data:
        return []

    return [Document(
        page_content=json.dumps(data),
        metadata={
            "source": "financial_fetcher",
            "organization": data.get("ticker", ticker),
            "document": "tool:get_stock_price",
        }
    )]

def fallback_news_lookup(query: str) -> List[Document]:
    """
    Fetch latest financial news for a resolved ticker if possible; otherwise fallback to the raw query.
    """
    ticker = resolve_ticker_from_query(query)
    term = ticker if ticker else query
    items = get_financial_news(term)
    if not items:
        return []
    return [Document(
        page_content=json.dumps(item),
        metadata={"source": "financial_fetcher", "document": "tool:get_financial_news"}
    ) for item in items]

def fallback_financial_metric_lookup(query: str) -> List[Document]:
    """
    Fetch a specific financial metric (EPS, revenue, net income, operating income/margin)
    via financial_fetcher.get_financial_metric, using a safe resolver and strict intent checks.
    """
    # Must look like a metric question
    if not re.search(r"\b(eps|earnings per share|revenue|net income|operating (income|margin))\b", query, re.I):
        return []

    ym = re.search(r"\b(20\d{2})\b", query)
    if not ym:
        return []
    year = int(ym.group(1))

    qm = re.search(r"\bQ([1-4])\b", query, re.I)
    quarter = int(qm.group(1)) if qm else None

    ticker = resolve_ticker_from_query(query)
    if not ticker:
        return []

    metric = None
    for kw, key in {
        "earnings per share": "eps",
        "eps": "eps",
        "revenue": "revenue",
        "net income": "net_income",
        "operating income": "operating_income",
        "operating margin": "operating_margin",
    }.items():
        if kw in query.lower():
            metric = key
            break
    if not metric:
        return []

    data = get_financial_metric(ticker, year, metric, quarter)
    if not data:
        return []

    return [Document(
        page_content=json.dumps(data),
        metadata={
            "source": "financial_fetcher",
            "organization": ticker,
            "document": f"tool:get_{metric}",
            "fiscal_year": str(year),
            "report_type": ("10-Q" if quarter else "10-K"),
        }
    )]

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
            with st.spinner(
                text="📊 Refining your financial query and retrieving relevant documents– hang tight! "
                     "This should take seconds."
            ):
                refined_user_input = refine_question(llm, user_input, chat_history=chat_history)

                # 1) RAG
                retrieved_contents, sources = index.similarity_search_with_threshold(
                    query=refined_user_input, k=parameters.k
                )

                # 2) Tools (stock -> metric -> news)
                if not retrieved_contents:
                    fallback_docs = []
                    for tool in [fallback_stock_lookup, fallback_financial_metric_lookup, fallback_news_lookup]:
                        fallback_docs = tool(refined_user_input)
                        if fallback_docs:
                            index.from_chunks(fallback_docs)
                            retrieved_contents, sources = index.similarity_search_with_threshold(
                                query=refined_user_input, k=parameters.k
                            )
                            break

                # 3) Claude fallback — moved OUT of the loop
                if not retrieved_contents:
                    claude_response = call_claude_fallback(
                        refined_user_input,
                        model="anthropic/claude-3.7-sonnet"
                    )
                    full_response += "🤖 Claude response:\n\n"
                    full_response += (claude_response or "No response.")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    return

                # 4) Show sources preview
                full_response += "📚 Relevant financial excerpts:\n\n"
                message_placeholder.markdown(full_response)
                for source in sources:
                    full_response += prettify_source(source) + "\n\n"
                    message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

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