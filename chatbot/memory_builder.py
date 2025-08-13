import argparse
import sys
from pathlib import Path
import json
import re
from typing import List, Optional
from chatbot.bot.memory.embedder import Embedder
from chatbot.bot.memory.vector_database.chroma import Chroma
from chatbot.financial_fetcher import get_stock_price, get_financial_news, get_financial_metric, get_cik_from_ticker
from document_loader.format import Format
from document_loader.loader import DirectoryLoader
from document_loader.text_splitter import create_recursive_text_splitter
from entities.document import Document
from helpers.log import get_logger

logger = get_logger(__name__)


def load_documents(docs_path: Path) -> list[Document]:
    """
    Load all Markdown files under docs_path (recursive).

    Args:
        docs_path (Path): The path to the documents.

    Returns:
        List[Document]: A list of loaded documents.
    """
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.md",  
        show_progress=True,
    )
    return loader.load()


def split_chunks(sources: list, chunk_size: int = 512, chunk_overlap: int = 25) -> list:
    """
    Splits a list of financial documents into smaller chunks.

    Args:
        sources (List): The list of source documents.
        chunk_size (int): Max size of each chunk.
        chunk_overlap (int): Overlap between chunks to preserve context.

    Returns:
        List: Chunks of financial content.
    """
    chunks = []
    splitter = create_recursive_text_splitter(
        format=Format.MARKDOWN.value, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

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

def build_memory_index(docs_path: Path, vector_store_path: str, chunk_size: int, chunk_overlap: int):
    """
    Loads financial documents, splits them, embeds, and stores in vector DB.

    Args:
        docs_path (Path): Path to the source docs (e.g. 'docs/demo.md')
        vector_store_path (str): Path to persist Chroma DB.
        chunk_size (int): Chunk size for document splitting.
        chunk_overlap (int): Overlap between chunks.
    """
    logger.info(f"Loading documents from: {docs_path}")
    sources = load_documents(docs_path)
    logger.info(f"Number of loaded documents: {len(sources)}")

    logger.info("Chunking documents...")
    chunks = split_chunks(sources, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info(f"Number of generated chunks: {len(chunks)}")

    logger.info("Creating memory index...")
    embedding = Embedder()
    vector_database = Chroma(persist_directory=str(vector_store_path), embedding=embedding)
    vector_database.from_chunks(chunks)
    logger.info("Memory Index has been created successfully!")


def get_args() -> argparse.Namespace:
    """
    Parse CLI arguments for chunking.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Memory Builder for Financial Documents")
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="The maximum size of each chunk. Defaults to 512.",
        required=False,
        default=512,
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="The amount of overlap between consecutive chunks. Defaults to 25.",
        required=False,
        default=25,
    )

    return parser.parse_args()


def main(parameters):
    """
    Main entry point: builds memory index from docs.
    """
    root_folder = Path(__file__).resolve().parent.parent
    doc_path = root_folder / "docs"
    vector_store_path = root_folder / "vector_store" / "docs_index"

    build_memory_index(
        doc_path,
        str(vector_store_path),
        parameters.chunk_size,
        parameters.chunk_overlap,
    )


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
