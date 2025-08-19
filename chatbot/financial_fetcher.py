# chatbot/financial_fetcher.py
import os
import re
import glob
import functools
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import requests
import yfinance as yf
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT") or "finance-bot/1.0"

SEC_HEADERS = {"User-Agent": EDGAR_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# ──────────────────────────────────────────────────────────────────────────────
# SEC mapping & ticker/CIK utils
# ──────────────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _load_sec_ticker_map() -> list[dict]:
    """Return [{'ticker': 'AAPL', 'title': 'apple inc.', 'cik': '0000320193'}, ...]."""
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=20)
    resp.raise_for_status()
    raw = resp.json()
    return [
        {"ticker": v["ticker"].upper(), "title": (v["title"] or "").lower(), "cik": str(v["cik_str"]).zfill(10)}
        for v in raw.values()
    ]

_STOP = {"ME", "I", "US", "THE", "AND", "FOR", "EPS", "PE", "ROE", "FCF", "SEC", "IR"}
_MEDIA = {"CNBC", "BLOOMBERG", "REUTERS", "WSJ", "YAHOO", "FINANCE", "MARKETWATCH", "SEEKINGALPHA"}

def resolve_ticker_from_query(query: str) -> Optional[str]:
    """Safe ticker resolve: explicit → ALL-CAPS (2–5) → whole-word company name; all validated by SEC list."""
    if not query:
        return None
    rows = _load_sec_ticker_map()
    valid = {r["ticker"] for r in rows}
    qlow = (query or "").lower()

    # 1) explicit: $AAPL / NASDAQ:AAPL / (AAPL)
    m = re.search(r'(?:(?:\$)|(?:nasdaq:)|(?:nyse:)|(?:amex:)|(?:arca:))?(\b[A-Za-z]{2,5}\b)', query, re.I)
    if m:
        cand = m.group(1).upper()
        if cand in valid and cand not in _STOP and cand not in _MEDIA:
            return cand

    # 2) ALL-CAPS tokens (2–5)
    for t in re.findall(r"\b[A-Z]{2,5}\b", query or ""):
        if t in valid and t not in _STOP and t not in _MEDIA:
            return t

    # 3) whole-word company name
    for r in rows:
        name = r["title"]
        main = name.split(",")[0].strip()
        if (re.search(rf"\b{re.escape(name)}\b", qlow) or (main and re.search(rf"\b{re.escape(main)}\b", qlow))):
            return r["ticker"]
    return None

_CIK_CACHE: Dict[str, str] = {}

def get_cik_from_ticker(ticker: str) -> Optional[str]:
    """Resolve CIK (10-digit) via SEC mapping, with a small in-memory cache."""
    if not ticker:
        return None
    sym = ticker.upper().strip()
    if sym in _CIK_CACHE:
        return _CIK_CACHE[sym]
    for r in _load_sec_ticker_map():
        if r["ticker"] == sym:
            _CIK_CACHE[sym] = r["cik"]
            return r["cik"]
    logger.warning("CIK not found in SEC mapping for ticker=%s", sym)
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Stock price / News (Tools)
# ──────────────────────────────────────────────────────────────────────────────
def get_stock_price_alpha_vantage(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        if not ALPHA_VANTAGE_API_KEY:
            return None
        url = "https://www.alphavantage.co/query"
        resp = requests.get(url, params={"function":"GLOBAL_QUOTE","symbol":ticker,"apikey":ALPHA_VANTAGE_API_KEY}, timeout=20)
        data = resp.json().get("Global Quote", {})
        if "05. price" in data:
            return {"ticker": ticker.upper(), "price": round(float(data["05. price"]), 2), "source": "Alpha Vantage"}
    except Exception as e:
        logger.error("Alpha Vantage fetch error: %s", e)
    return None

def get_stock_price_yahoo(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        if not hist.empty:
            return {"ticker": ticker.upper(), "price": round(hist["Close"].iloc[-1], 2), "source": "Yahoo Finance"}
    except Exception as e:
        logger.error("Yahoo Finance fetch error: %s", e)
    return None

def get_stock_price(ticker: str) -> Optional[Dict[str, Any]]:
    """Unified price: Alpha Vantage → Yahoo."""
    return get_stock_price_alpha_vantage(ticker) or get_stock_price_yahoo(ticker)

def get_financial_news(query: str) -> List[Dict[str, str]]:
    """Top 3 headlines via NewsAPI (if configured)."""
    if not NEWSAPI_KEY:
        return []
    try:
        url = "https://newsapi.org/v2/everything"
        resp = requests.get(url, params={"q":query, "apiKey":NEWSAPI_KEY, "language":"en", "sortBy":"publishedAt", "pageSize":3}, timeout=20)
        arts = resp.json().get("articles", []) or []
        return [{"title":a.get("title",""), "url":a.get("url","")} for a in arts if a.get("title") and a.get("url")]
    except Exception as e:
        logger.error("NewsAPI fetch error: %s", e)
        return []

# ──────────────────────────────────────────────────────────────────────────────
# SEC metric (Tools)
# ──────────────────────────────────────────────────────────────────────────────
_GAAP = {
    "eps": "EarningsPerShareDiluted",
    "revenue": "Revenues",
    "net_income": "NetIncomeLoss",
    "operating_income": "OperatingIncomeLoss",
    "operating_margin": "OperatingIncomeLoss",  # keep mapping minimal; margin may be derived upstream
    "gross_margin": "GrossProfit",
}

def get_financial_metric(ticker: str, year: int, metric: str, quarter: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    SEC 'companyfacts' lookup for a metric (eps/revenue/net_income/operating_income/...).
    Prefers 10-Q for quarters, 10-K for FY. Units auto-picked: EPS -> USD/shares, others -> USD.
    """
    try:
        cik = get_cik_from_ticker(ticker)
        if not cik:
            return None

        key = _GAAP.get((metric or "").lower())
        if not key:
            logger.warning("Unsupported metric: %s", metric)
            return None

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        facts = requests.get(url, headers=SEC_HEADERS, timeout=30).json().get("facts", {}).get("us-gaap", {})
        if key not in facts:
            logger.warning("US-GAAP key not found: %s for %s", key, ticker)
            return None

        units = facts[key].get("units", {})
        preferred_units = ["USD/shares"] if key == "EarningsPerShareDiluted" else ["USD"]
        unit = next((u for u in preferred_units if u in units), None) or (next(iter(units.keys())) if units else None)
        if not unit:
            return None

        obs = units.get(unit, [])
        target_fp = f"Q{quarter}" if quarter else "FY"
        matches = [v for v in obs if v.get("fy") == year and v.get("fp") == target_fp]
        if not matches and quarter:
            matches = [v for v in obs if v.get("fy") == year and v.get("fp") in (f"Q{quarter}", "FY")]
        if not matches and not quarter:
            matches = [v for v in obs if v.get("fy") == year]
        if not matches:
            return None

        preferred_form = "10-Q" if quarter else "10-K"
        best = max(matches, key=lambda v: (1 if v.get("form")==preferred_form else 0, v.get("end","")))
        return {
            "ticker": ticker.upper(),
            "cik": cik,
            "year": year,
            "quarter": quarter,
            "metric": metric,
            "gaap_key": f"us-gaap_{key}",
            "value": best.get("val"),
            "unit": unit,
            "form": best.get("form"),
            "as_of": best.get("end"),
            "source": "SEC EDGAR companyfacts",
        }
    except Exception as e:
        logger.error("EDGAR metric fetch error: %s", e)
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Filing citation snippets (local .md from ingest_pipeline)
# ──────────────────────────────────────────────────────────────────────────────
_SECTION_HINTS = ("md&a", "management’s discussion", "management's discussion", "results of operations", "outlook", "guidance")
_NUMBER_HINT = re.compile(r"(\d{1,3}(,\d{3})+|\d+\.\d+|\d+)%|\$\s?\d+(\.\d+)?\s?(b|bn|m|mm|thousand)?", re.I)

def _parse_ingest_header(md_text: str) -> dict:
    """Pick Date/Source/Title from the first ~40 lines of an ingest .md header."""
    head = md_text.split("\n", 40)[:40]
    def _pick(prefix: str) -> str:
        for ln in head:
            if ln.startswith(prefix):
                return ln.split(":", 1)[1].strip()
        return ""
    return {
        "date":   _pick("- **Date**:"),
        "source": _pick("- **Source**:").strip("<>").strip(),
        "title":  _pick("- **Title**:"),
    }

def get_filing_citation_snippets(ticker: str, limit: int = 6) -> list[dict]:
    """Return up to `limit` numeric, citation-ready lines from docs/{TICKER}_*.md (priority: MD&A/Results/Outlook)."""
    tkr = (ticker or "").upper()
    if not tkr:
        return []
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    paths = sorted(glob.glob(str(docs_dir / f"{tkr}_*.md")))
    out: list[dict] = []

    for p in paths:
        try:
            raw = Path(p).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        meta = _parse_ingest_header(raw)
        body = raw.split("\n---\n", 1)[1] if "\n---\n" in raw else raw

        for line in body.splitlines():
            s = line.strip()
            if 40 <= len(s) <= 600 and _NUMBER_HINT.search(s) and any(k in s.lower() for k in _SECTION_HINTS):
                out.append({"quote": s, "source": meta["source"], "title": meta["title"], "date": meta["date"]})
                if len(out) >= limit:
                    return out
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Tiny routing helpers (kept for app Step-5 brevity)
# ──────────────────────────────────────────────────────────────────────────────
_PRICE_RE   = re.compile(r"\b(price|quote|share price|trading|last|close)\b", re.I)
_METRIC_RE  = re.compile(r"\b(eps|earnings per share|revenue|net income|operating (margin|income)|gross margin)\b", re.I)
_EXPLAIN_RE = re.compile(r"\b(why|explain|reason|driver|cause|impact|背景|原因|解讀|說明)\b", re.I)
_COMPARE_RE = re.compile(r"\b(compare|vs\.?|versus|對比)\b", re.I)
_CALL_RE    = re.compile(r"\b(earnings call|conference call|prepared remarks)\b", re.I)

def parse_year(q: str) -> Optional[str]:
    m = re.search(r"\b(20\d{2})\b", q or "")
    return m.group(1) if m else None

def parse_quarter(q: str) -> Optional[str]:
    m = re.search(r"\bQ([1-4])\b", q or "", re.I)
    return m.group(1) if m else None

def classify_intent(q: str) -> Dict[str, bool]:
    """One-pass intent flags used by the app."""
    return {
        "price":   bool(_PRICE_RE.search(q or "")),
        "metric":  bool(_METRIC_RE.search(q or "")),
        "explain": bool(_EXPLAIN_RE.search(q or "") or _COMPARE_RE.search(q or "") or _CALL_RE.search(q or "")),
    }

def safe_entities(orig_query: str, resolver) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse entities from ORIGINAL user text to avoid refined-text noise (e.g., CNBC)."""
    return resolver(orig_query), parse_year(orig_query), parse_quarter(orig_query)

def should_skip_news(orig_query: str, refined_query: str, resolver) -> bool:
    """Skip News for explanation-like tasks when a real ticker is present."""
    tkr = resolver(orig_query)
    intents = classify_intent(refined_query)
    return bool(tkr and intents["explain"])

def compose_metric_header(payload: Dict[str, Any]) -> str:
    """Compact, consistent header for metric answers from the Tools layer."""
    t = payload.get("ticker"); m = payload.get("metric"); v = payload.get("value")
    y = payload.get("year"); q = payload.get("quarter")
    asof = payload.get("as_of"); basis = payload.get("basis", "FY")
    cal_y = payload.get("calendar_year"); cal_q = payload.get("calendar_quarter"); note = payload.get("note")
    header = f"**{t} {m} {str(y) if y else ''}{(' Q'+str(q)) if q else ''}: {v}**"
    if asof: header += f" (as of {asof})"
    header += f"  \n_Basis: **{basis}**_"
    if basis == "CY" and (cal_y or cal_q):
        header += f" — Calendar: **{cal_y or ''}{(' Q'+str(cal_q)) if cal_q else ''}**"
    if note: header += f"  \n_Note: {note}_"
    return header

# ──────────────────────────────────────────────────────────────────────────────
# Simple CLI smoke (optional)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Price:", get_stock_price("AAPL"))
    print("News (MSFT):", get_financial_news("MSFT earnings"))
    print("Metric (TSLA EPS Q4 2023):", get_financial_metric("TSLA", 2023, "eps", 4))
    

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

    print("Snippets (AAPL):", get_filing_citation_snippets("AAPL", limit=2))

