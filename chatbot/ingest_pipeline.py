import os
import time
import requests
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from chatbot.financial_fetcher import (  # Directly reuse existing utilities
    get_cik_from_ticker,
    SEC_HEADERS,
)

# Load environment variables from .env 
load_dotenv()
OUTPUT_DIR = Path(os.getenv("SEC_OUTPUT_DIR", "./docs")) # Where SEC filings are saved (.md files)
INDEX_DIR = Path(os.getenv("INDEX_PATH", "vector_store/docs_index"))# Where vector index is stored
REQUEST_TIMEOUT = 30

# Ensure output directory exists 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_recent_filings_basic(cik: str) -> List[dict]:
    """
    Fetch recent filings metadata from SEC's submissions API for a given CIK.
    Returns a list of dicts with form type, filing date, document name,
    description, and accession number.
    """
    if not cik:
        return []
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    recent = j.get("filings", {}).get("recent", {})
    out = []
    for form, date, doc, desc, acc in zip(
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("primaryDocument", []),
        recent.get("primaryDocDescription", []),
        recent.get("accessionNumber", []),
    ):
        out.append({
            "form": form,
            "date": date,
            "doc": doc,
            "desc": desc,
            "acc": (acc or "").replace("-", ""),
        })
    return out

# Download text content from a given SEC filing document URL. Handles both HTML and plain text.
def download_text_from_url(url: str) -> str:
    r = requests.get(url, headers=SEC_HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text

def ingest_ticker(ticker: str, forms: List[str], max_per_ticker: int, since: Optional[str] = None) -> int:
    """
    Ingest filings for a given stock ticker.
    - Resolves the CIK via financial_fetcher
    - Fetches recent filings metadata
    - Downloads & stores up to `max_per_ticker` filings matching given forms & date filter
    """
    ticker_up = ticker.upper()
    cik = get_cik_from_ticker(ticker_up)
    if not cik:
        print(f"[skip] CIK not found for {ticker_up}")
        return 0

    filings = fetch_recent_filings_basic(cik)
    count = 0
    for filing in filings:
        if filing["form"] not in forms:
            continue
        if since and filing["date"] < since:
            continue

        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{filing['acc']}/{filing['doc']}"
        try:
            text = download_text_from_url(doc_url)
        except Exception as e:
            print(f"[error] Failed to fetch {doc_url}: {e}")
            continue

        # Save filing content as .md (minimal header + raw body)
        header = (
            f"# {ticker_up} — {filing['form']}\n\n"
            f"- **Date**: {filing['date']}\n"
            f"- **Source**: <{doc_url}>\n"
            f"- **Title**: {filing.get('desc','') or '(no title)'}\n\n"
            f"---\n\n"
        )
        out_path = OUTPUT_DIR / f"{ticker_up}_{filing['form']}_{filing['date']}.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(text)  # keep raw HTML/text; no conversion

        count += 1
        print(f"[ok] Saved {out_path}")

        if count >= max_per_ticker:   
            break

        time.sleep(0.5)               
    return count                       # Don't return None


# RAG-focused CLI smoke test 
def rag_cli_smoke_test() -> None:
    """
    RAG smoke test for three retrieval questions：
      - Target only the filings most likely to contain explanations/citations:
        * AAPL: FY2023 Q4 reasons → 10-K/10-Q/8-K around FY2023
        * MSFT: latest earnings call highlights → earnings press usually in 8-K + 10-Q
        * TSLA: auto gross margin 2022 vs 2023 → 10-K (yearly comparison), plus 10-Q/8-K
      - Keep it small & polite: cap max_per_ticker to reduce requests.
    """
    # AAPL — Explain why EPS changed in FY2023 Q4
    print("\n[TEST] AAPL — FY2023 Q4 EPS change (10-K/10-Q/8-K since 2023-07-01)")
    ingest_ticker("AAPL", forms=["10-K", "10-Q", "8-K"], max_per_ticker=3, since="2023-07-01")

    # MSFT — Summarize latest earnings call (8-K + 10-Q since 2024-01-01)
    print("\n[TEST] MSFT — latest earnings highlights (8-K/10-Q since 2024-01-01)")
    ingest_ticker("MSFT", forms=["8-K", "10-Q"], max_per_ticker=3, since="2024-01-01")

    # TSLA — Compare automotive gross margin between 2022 and 2023
    print("\n[TEST] TSLA — auto gross margin 2022 vs 2023 (10-K/10-Q/8-K since 2022-01-01)")
    ingest_ticker("TSLA", forms=["10-K", "10-Q", "8-K"], max_per_ticker=4, since="2022-01-01")

    print("\n[HINT] Files saved in:", OUTPUT_DIR)
    print("[HINT] Next step:")
    print("  1) Rebuild index (memory_builder.py) to load these docs")
    print("  2) In the app, ask:")
    print("     - Explain why AAPL EPS changed in FY2023 Q4 (quote sources).")
    print("     - Summarize Microsoft’s latest earnings call highlights with citations.")
    print("     - Compare TSLA automotive gross margin between 2022 and 2023, with sources.")

def main():
    """
    Default: preload top tickers.
    If RAG_CLI_TEST=1, run targeted RAG smoke test instead.
    """
    if os.getenv("RAG_CLI_TEST") == "1":
        # Run RAG-focused minimal ingestion for the three test questions
        rag_cli_smoke_test()
        return

    # Original default preload (keep your existing setup below) ---
    top_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                   "META", "BRK-B", "TSLA", "UNH", "JNJ"]
    forms = ["8-K", "10-Q", "10-K"]      # ensure earnings press + quarterly + annual
    since = "2022-01-01"
    max_docs = 3

    for ticker in top_tickers:
        print(f"[start] Ingesting {ticker}")
        count = ingest_ticker(ticker, forms, max_docs, since)
        print(f"[done] {ticker}: {count} docs saved")


if __name__ == "__main__":
    main()

