import yfinance as yf
import requests
import json
import logging
import os
import functools
import re
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from edgar import Company

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT")

# Global SEC request headers
headers = {
    "User-Agent": EDGAR_USER_AGENT
}

# Load SEC mapping once 
@functools.lru_cache(maxsize=1)
def _load_sec_ticker_map() -> list[dict]:
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers={"User-Agent": EDGAR_USER_AGENT or "finance-bot/1.0"})
    resp.raise_for_status()
    raw = resp.json()
    # normalize to list of dicts: {"ticker": "AAPL", "title": "Apple Inc.", "cik": "0000320193"}
    out = []
    for _, v in raw.items():
        out.append({
            "ticker": v["ticker"].upper(),
            "title": v["title"].lower(),
            "cik": str(v["cik_str"]).zfill(10),
        })
    return out

# Resolve_ticker from query safely 
STOP_TICKERS = {"ME","I","US","THE","AND","FOR","EPS","PE","ROE","FCF","SEC","IR"}

def resolve_ticker_from_query(query: str) -> Optional[str]:
    q = query.lower()

    # 1) explicit patterns: $AAPL, NASDAQ:AAPL, (AAPL)
    m = re.search(r'(?:\$|nasdaq:|nyse:|amex:)?\b([A-Z]{1,5})\b', query)
    if m:
        cand = m.group(1).upper()
        if cand not in STOP_TICKERS and any(row["ticker"] == cand for row in _load_sec_ticker_map()):
            return cand

    # 2) try fuzzy company-name contains
    rows = _load_sec_ticker_map()
    for row in rows:
        name = row["title"]
        if len(name) > 4 and name in q:
            return row["ticker"]

    # 3) last resort: ALL-CAPS tokens but validate against SEC map
    tokens = [t for t in re.findall(r"\b[A-Z]{1,5}\b", query) if t not in STOP_TICKERS]
    valid = {row["ticker"] for row in rows}
    for t in tokens:
        if t in valid:
            return t
    return None

# Get stock price from Yahoo Finance
def get_stock_price_yahoo(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetch latest stock price from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if not data.empty:
            return {
                "ticker": ticker.upper(),
                "price": round(data["Close"].iloc[-1], 2),
                "source": "Yahoo Finance"
            }
    except Exception as e:
        logger.error(f"Yahoo Finance fetch error: {e}")
    return None

# Get stock price from Alpha Vantage
def get_stock_price_alpha_vantage(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetch real-time stock price using Alpha Vantage API."""
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json().get("Global Quote", {})
        if data and "05. price" in data:
            return {
                "ticker": ticker.upper(),
                "price": round(float(data["05. price"]), 2),
                "source": "Alpha Vantage"
            }
    except Exception as e:
        logger.error(f"Alpha Vantage fetch error: {e}")
    return None

# Unified stock price getter
def get_stock_price(ticker: str) -> Optional[Dict[str, Any]]:
    """Try Alpha Vantage first, fallback to Yahoo."""
    return get_stock_price_alpha_vantage(ticker) or get_stock_price_yahoo(ticker)

# Get news from NewsAPI
def get_financial_news(query: str) -> List[Dict[str, str]]:
    """Search recent financial news using NewsAPI."""
    try:
        url = (
            f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWSAPI_KEY}"
            f"&language=en&sortBy=publishedAt&pageSize=3"
        )
        response = requests.get(url)
        articles = response.json().get("articles", [])
        return [
            {"title": article["title"], "url": article["url"]}
            for article in articles if article.get("title") and article.get("url")
        ]
    except Exception as e:
        logger.error(f"NewsAPI fetch error: {e}")
    return []

# Get CIK from SEC's official mapping file (with simple in-memory cache)
_CIK_CACHE: Dict[str, str] = {}

def get_cik_from_ticker(ticker: str) -> Optional[str]:
    """
    Resolve CIK from a stock ticker using SEC's official mapping JSON.
    - Caches results in memory to avoid repeated downloads.
    - Pads CIK to 10 digits (required by SEC endpoints).
    """
    try:
        if not ticker:
            return None
        symbol = ticker.strip().upper()

        if symbol in _CIK_CACHE:
            return _CIK_CACHE[symbol]

        url = "https://www.sec.gov/files/company_tickers.json"
        # Ensure you have a global `headers` with a proper User-Agent registered to the SEC guidelines
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        for entry in data.values():
            if entry.get("ticker", "").upper() == symbol:
                cik = str(entry.get("cik_str", "")).zfill(10)
                _CIK_CACHE[symbol] = cik
                return cik

        logger.warning(f"CIK not found in SEC mapping for ticker={symbol}")
        return None
    except Exception as e:
        logger.error(f"CIK fetch error for {ticker}: {e}")
        return None


def get_financial_metric(
    ticker: str, year: int, metric: str, quarter: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetch a financial metric (EPS, revenue, net income, operating income, etc.) from SEC EDGAR XBRL 'companyfacts'.

    - Uses https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
    - Maps friendly metric names to US GAAP taxonomy keys.
    - EPS uses 'USD/shares'; others use 'USD'.
    - Filters by fiscal year (fy == year) and fiscal period (fp == 'Q{n}' or 'FY').
    - Prefers 10-Q for quarters and 10-K for FY.
    """
    try:
        cik = get_cik_from_ticker(ticker)
        if not cik:
            logger.warning(f"CIK not found for {ticker}")
            return None

        gaap_map = {
            "eps": "EarningsPerShareDiluted",
            "revenue": "Revenues",
            "net_income": "NetIncomeLoss",
            "operating_income": "OperatingIncomeLoss",
        }
        key = gaap_map.get((metric or "").lower())
        if not key:
            logger.warning(f"Unsupported metric: {metric}")
            return None

        facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(facts_url, headers=headers, timeout=30)  # headers must include proper UA
        resp.raise_for_status()
        facts = resp.json().get("facts", {})
        us_gaap = facts.get("us-gaap", {})
        if key not in us_gaap:
            logger.warning(f"US-GAAP key not found for {ticker}: {key}")
            return None

        preferred_units = ["USD/shares"] if key == "EarningsPerShareDiluted" else ["USD"]
        units_dict = us_gaap[key].get("units", {})
        chosen_unit = next((u for u in preferred_units if u in units_dict), None) or (next(iter(units_dict.keys())) if units_dict else None)
        if not chosen_unit:
            logger.warning(f"No units for {key} on {ticker}")
            return None

        observations = units_dict.get(chosen_unit, []) or []
        if not observations:
            logger.warning(f"No observations for {key} with unit {chosen_unit} on {ticker}")
            return None

        target_fp = f"Q{quarter}" if quarter else "FY"
        matches = [v for v in observations if v.get("fy") == year and v.get("fp") == target_fp]

        if not matches and quarter:
            # Q4 sometimes lives under FY in 10-K
            matches = [v for v in observations if v.get("fy") == year and v.get("fp") in (f"Q{quarter}", "FY")]
        elif not matches and not quarter:
            matches = [v for v in observations if v.get("fy") == year]

        if not matches:
            logger.warning(f"No facts matched year/period for {ticker} {key} {year} q={quarter}")
            return None

        preferred_form = "10-Q" if quarter else "10-K"
        def score(item: Dict[str, Any]) -> tuple:
            form_ok = 1 if item.get("form") == preferred_form else 0
            return (form_ok, item.get("end", ""))  # ISO string is OK to compare lexicographically

        best = max(matches, key=score)
        value = best.get("val")

        return {
            "ticker": ticker.upper(),
            "cik": cik,
            "year": year,
            "quarter": quarter,
            "metric": metric,
            "gaap_key": f"us-gaap_{key}",
            "value": value,
            "unit": chosen_unit,
            "form": best.get("form"),
            "as_of": best.get("end"),
            "source": "SEC EDGAR companyfacts",
        }

    except Exception as e:
        logger.error(f"EDGAR metric fetch error: {e}")
        return None


# CLI test (for local testing)
if __name__ == "__main__":
    print("--- Yahoo Finance ---")
    try:
        print(get_stock_price_yahoo("AAPL"))
    except Exception as e:
        print(f"[Yahoo Finance error] {e}")

    print("--- Alpha Vantage ---")
    try:
        print(get_stock_price_alpha_vantage("AAPL"))
    except Exception as e:
        print(f"[Alpha Vantage error] {e}")

    print("--- Unified Stock Price ---")
    try:
        print(get_stock_price("AAPL"))
    except Exception as e:
        print(f"[Unified price error] {e}")

    print("--- Financial News (Apple sample) ---")
    try:
        news_items = get_financial_news("Apple earnings")
        if news_items:
            for item in news_items[:3]:
                print(f"- {item.get('title','(no title)')} ({item.get('url','')})")
        else:
            print("No news returned.")
    except Exception as e:
        print(f"[News API error] {e}")

    print("--- SEC Financial Metric (TSLA EPS Q4 2023) ---")
    try:
        print(get_financial_metric("TSLA", 2023, metric="eps", quarter=4))
    except Exception as e:
        print(f"[SEC metric error] {e}")

    print("--- SEC Financial Metric (NVDA Revenue FY2024 Q4) ---")
    # NVDA fiscal calendar differs from calendar year; this checks our SEC path.
    try:
        print(get_financial_metric("NVDA", 2024, metric="revenue", quarter=4))
    except Exception as e:
        print(f"[SEC NVDA revenue error] {e}")

    print("--- Financial News (MSFT top 3 headlines) ---")
    # Using a simple query string. Adjust if your function supports ticker-based search.
    try:
        msft_news = get_financial_news("MSFT earnings")
        if msft_news:
            for item in msft_news[:3]:
                print(f"- {item.get('title','(no title)')} ({item.get('url','')})")
        else:
            print("No MSFT news returned.")
    except Exception as e:
        print(f"[News API MSFT error] {e}")
