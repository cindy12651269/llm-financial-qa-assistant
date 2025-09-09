import os
import re
import json
import glob
import functools
import logging
from pathlib import Path
from entities.document import Document
from typing import Optional, List, Tuple, Dict, Any
import unicodedata
import requests
import yfinance as yf
from dotenv import load_dotenv

# Setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT") or "finance-bot/1.0"
SEC_HEADERS = {"User-Agent": EDGAR_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# SEC mapping & ticker/CIK utils
# Return [{'ticker': 'AAPL', 'title': 'apple inc.', 'cik': '0000320193'}, ...].
@functools.lru_cache(maxsize=1)
def _load_sec_ticker_map() -> list[dict]:
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=20)
    resp.raise_for_status()
    raw = resp.json()
    return [
        {"ticker": v["ticker"].upper(), "title": (v["title"] or "").lower(), "cik": str(v["cik_str"]).zfill(10)}
        for v in raw.values()
    ]

STOP  = {"ME","I","US","IT","THE","AND","FOR","SEC","IR","GAAP","NONGAAP","NON-GAAP","YOY","Q","FY","FQ",
         "EPS","PE","P/E","ESG","ROE","FCF","EV/EBITDA","D/E","VS"}
MEDIA = {"CNBC","BLOOMBERG","REUTERS","WSJ","YAHOO","FINANCE","MARKETWATCH","SEEKINGALPHA","NYSE","NASDAQ","AMEX","ARCA"}

# Small, safe demo overrides (keeps behavior stable even if SEC map is imperfect)
BRAND_OVERRIDES = {
    "apple": "AAPL", "alphabet": "GOOGL", "google": "GOOGL",
    "amazon": "AMZN", "microsoft": "MSFT", "tesla": "TSLA",
    "nvidia": "NVDA", "meta": "META", "salesforce": "CRM",
    "adobe": "ADBE", "netflix": "NFLX", "berkshire": "BRK-B",
}

log = logging.getLogger("TICKER_RESOLVE")

def resolve_ticker_sec(query: str) -> Optional[str]:
    if not query:
        log.info("[resolve] empty query")
        return None

    # Normalize and neutralize “vs” so VS isn’t mistaken as a ticker
    q  = unicodedata.normalize("NFKC", query).replace("’","'").replace("`","'")
    q  = re.sub(r"'\s*s\b", "", q)                     # Apple's -> Apple
    q  = re.sub(r"(?i)\bvs\.?\b", "versus", q)         # “vs” -> “versus”
    q  = re.sub(r"(?i)\bP\/E\b", "PE ratio", q)        # ratio tokens
    q  = re.sub(r"(?i)\bD\/E\b", "debt equity", q)
    q  = re.sub(r"(?i)\bEV\/EBITDA\b", "EV EBITDA", q)
    ql = q.lower()

    log.info(f"[resolve] raw='{query}' | sanitized='{q}'")

    rows  = _load_sec_ticker_map()                     # your existing loader
    valid = {r["ticker"].upper() for r in rows}

    # (A) Brand override — strict word boundary, first so demo stays stable
    for brand, tk in BRAND_OVERRIDES.items():
        if re.search(rf"\b{re.escape(brand)}\b", ql):
            log.info(f"[resolve] brand override '{brand}' -> {tk}")
            return tk

    # (B) Explicit tickers ($AAPL, NASDAQ:CRM, etc.)
    for m in re.finditer(r'(?i)\b(?:\$|nasdaq:|nyse:|amex:|arca:)([A-Za-z]{1,5}(?:[-.][A-Za-z]{1,2})?)\b', q):
        t = m.group(1).upper()
        log.info(f"[resolve] explicit cand={t} valid={t in valid}")
        if t in valid and t not in STOP and t not in MEDIA:
            log.info(f"[resolve] explicit ACCEPT -> {t}")
            return t

    # (C) Parentheses (GOOGL)
    for m in re.finditer(r'\(([A-Za-z]{3,5}(?:[-.][A-Za-z]{1,2})?)\)', q):
        t = m.group(1).upper()
        log.info(f"[resolve] paren cand={t} valid={t in valid}")
        if t in valid and t not in STOP and t not in MEDIA:
            log.info(f"[resolve] paren ACCEPT -> {t}")
            return t

    # (D) Unprefixed ALL-CAPS (len≥3 blocks VS)
    for t in re.findall(r"\b[A-Z]{3,5}(?:[-.][A-Z]{1,2})?\b", q):
        log.info(f"[resolve] allcaps cand={t} valid={t in valid}")
        if t in valid and t not in STOP and t not in MEDIA:
            log.info(f"[resolve] allcaps ACCEPT -> {t}")
            return t

    # (E) SEC name index: full title + stripped main name
    def strip_sfx(s: str) -> str:
        s = re.sub(r",?\s+(incorporated|inc|corp|corporation|co|company|ltd|plc|nv|sa|ag|holdings|group)\.?\b","",s,flags=re.I)
        s = re.sub(r"^\s*the\s+","",s,flags=re.I)
        return re.sub(r"\s{2,}"," ",s).strip()

    name_idx, mains = {}, []
    for r in rows:
        title = (r.get("title") or "").strip()
        if not title: continue
        main = strip_sfx(title)
        tk   = r["ticker"].upper()
        name_idx[title.lower()] = tk
        name_idx[main.lower()]  = tk
        mains.append((main.lower(), tk))

    # Longest name first
    for name in sorted(name_idx.keys(), key=len, reverse=True):
        if len(name) >= 4 and re.search(rf"\b{re.escape(name)}\b", ql):
            tk = name_idx[name]
            if tk in valid:
                log.info(f"[resolve] name match '{name}' -> {tk}")
                return tk

    # (F) Unique capitalized token → main name
    for tok in re.findall(r"\b([A-Z][a-z]{3,})\b", q):
        hits = [(m, tk) for (m, tk) in mains if m.startswith(tok.lower())]
        log.info(f"[resolve] cap token '{tok}' hits={hits}")
        if len(hits) == 1:
            log.info(f"[resolve] cap token ACCEPT -> {hits[0][1]}")
            return hits[0][1]

    log.info("[resolve] no ticker found")
    return None

def resolve_ticker_guarded(query: str) -> Optional[str]:
    t = resolve_ticker_sec(query)
    if t:
        return t
    # minimal fallback: try brand overrides again with a strict word boundary
    low = unicodedata.normalize("NFKC", query).lower()
    for brand, tk in BRAND_OVERRIDES.items():
        if re.search(rf"\b{re.escape(brand)}\b", low):
            return tk
    return None

# CIK cache 
_CIK_CACHE: Dict[str,str] = {}

# Resolve CIK (10-digit) via SEC mapping, with a small in-memory cache.
def get_cik_from_ticker(ticker: str) -> Optional[str]:
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

# Stock price / News (Tools)
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

# SEC metric (Tools)
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

# Filing citation snippets (local .md from ingest_pipeline)
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

# Tiny routing helpers (kept for app Step-5 brevity)
_PRICE_RE   = re.compile(r"\b(price|quote|share price|trading|last|close)\b", re.I)
_METRIC_RE  = re.compile(r"\b(eps|earnings per share|revenue|net income|operating (margin|income)|gross margin)\b", re.I)
_EXPLAIN_RE = re.compile(r"\b(why|explain|reason|driver|cause|impact)\b", re.I)
_COMPARE_RE = re.compile(r"\b(compare|vs\.?|versus)\b", re.I)
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

# Fetch latest stock price via financial_fetcher.get_stock_price
def fallback_stock_lookup(query: str) -> List[Document]:
    """
    Only if the query looks like a price question and ticker is resolvable.
    Adds metadata.source_type="tool" and a structured payload for API fast-path.
    """
    if not re.search(r'\b(price|stock|share|quote|trading|last|close)\b', query, re.I):
        return []

    ticker = resolve_ticker_sec(query)
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
    ticker = resolve_ticker_sec(query)
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
def fallback_financial_metric_lookup(query: str) -> list[Document]:
    """
    Tools fast-path with precise debug logs.
    - Parses FY/CY/bare year-quarter
    - Resolves ticker via resolve_ticker_sec()
    - Detects metric (prefers operating income/margin first)
    - Calls get_financial_metric()
    - Returns a single tool Document with payload if found; otherwise [].
    """
    import re, json, logging

    q_raw = (query or "").strip()
    if not q_raw:
        logging.info("[TOOL metric] empty query")
        return []

    # 0) Quick gate: only run if we see any relevant metric keywords 
    METRIC_GATE = re.compile(
        r"\b(eps|earnings\s+per\s+share|revenue|net\s+income|operating\s+(income|margin))\b",
        re.I,
    )
    if not METRIC_GATE.search(q_raw):
        logging.info("[TOOL metric] gate: no metric keyword in query=%r", q_raw)
        return []

    upper = q_raw.upper()
    basis = "FY"        # default, unless CY/bare-year-quarter detected
    year = quarter = None
    cal_year = cal_quarter = None

    # 1) Parse period: FY2024 Q4 / Q4 FY2024
    m = (re.search(r"\bFY\s*(\d{2,4})\s*Q([1-4])\b", upper)
         or re.search(r"\bQ([1-4])\s*FY\s*(\d{2,4})\b", upper))
    if m:
        if m.re.pattern.startswith(r"\bFY"):
            y_raw, q_raw_num = m.group(1), m.group(2)
        else:
            q_raw_num, y_raw = m.group(1), m.group(2)
        y = int(y_raw)
        year = (2000 + y) if y < 100 else y
        quarter = int(q_raw_num)
        basis = "FY"
    else:
        # 2) CY2024 Q4 / Q4 CY2024 
        m = (re.search(r"\bCY\s*(\d{2,4})\s*Q([1-4])\b", upper)
             or re.search(r"\bQ([1-4])\s*CY\s*(\d{2,4})\b", upper))
        if m:
            if m.re.pattern.startswith(r"\bCY"):
                y_raw, q_raw_num = m.group(1), m.group(2)
            else:
                q_raw_num, y_raw = m.group(1), m.group(2)
            y = int(y_raw)
            year = (2000 + y) if y < 100 else y
            quarter = int(q_raw_num)
            basis = "CY"
            cal_year, cal_quarter = year, quarter
        else:
            # 3) bare 2024 Q4 / Q4 2024 → treat as CY
            m = (re.search(r"\b(20\d{2})\s*Q([1-4])\b", upper)
                 or re.search(r"\bQ([1-4])\s*(20\d{2})\b", upper))
            if m:
                if m.re.pattern.startswith(r"\b(20"):
                    y_raw, q_raw_num = m.group(1), m.group(2)
                else:
                    q_raw_num, y_raw = m.group(1), m.group(2)
                year = int(y_raw); quarter = int(q_raw_num)
                basis = "CY"
                cal_year, cal_quarter = year, quarter
            else:
                # 4) just a year → default FY 
                y = re.search(r"\b(20\d{2})\b", upper)
                year = int(y.group(1)) if y else None
                quarter = None  # FY
    if not year:
        logging.info("[TOOL metric] period parse failed; no year found | query=%r", q_raw)
        return []

    # 5) Resolve ticker (sanitize to avoid GAAP/EPS noise) 
    sanitized_for_ticker = re.sub(
        r"(?i)\b(NON[-\s]?GAAP|GAAP|EPS|EARNINGS|FISCAL|CALENDAR|FY|CY|Q[1-4]|REVENUE|NET\s+INCOME|OPERATING\s+(INCOME|MARGIN))\b",
        " ",
        q_raw,
    )
    ticker = resolve_ticker_sec(sanitized_for_ticker)
    if not ticker:
        logging.info(
            "[TOOL metric] ticker not resolved | raw=%r | sanitized=%r",
            q_raw, sanitized_for_ticker
        )
        return []
    logging.info("[TOOL metric] ticker=%s parsed_ok basis=%s year=%s quarter=%s",
                 ticker, basis, year, quarter)

    # 6) Detect metric (priority: operating income → operating margin → EPS → revenue → net income)
    metric = None
    # operating income
    if re.search(r"\boperating\s+income\b", q_raw, re.I):
        metric = "operating_income"; why = "matched operating_income"
    # operating margin
    elif re.search(r"\boperating\s+margin\b", q_raw, re.I):
        metric = "operating_margin"; why = "matched operating_margin"
    # EPS
    elif re.search(r"\b(eps|earnings\s+per\s+share)\b", q_raw, re.I):
        metric = "eps"; why = "matched eps"
    # revenue
    elif re.search(r"\brevenue\b", q_raw, re.I):
        metric = "revenue"; why = "matched revenue"
    # net income
    elif re.search(r"\bnet\s+income\b", q_raw, re.I):
        metric = "net_income"; why = "matched net_income"

    if not metric:
        logging.info("[TOOL metric] metric keyword not found | query=%r", q_raw)
        return []
    logging.info("[TOOL metric] metric=%s (%s)", metric, why)

    # 7) Fetch from backend (FY-based data source) 
    try:
        data = get_financial_metric(ticker, year, metric, quarter)
    except Exception as e:
        logging.info("[TOOL metric] backend error for ticker=%s metric=%s y=%s q=%s | %r",
                     ticker, metric, year, quarter, e)
        return []

    if not data or data.get("value") is None:
        logging.info("[TOOL metric] backend returned no data | ticker=%s metric=%s y=%s q=%s",
                     ticker, metric, year, quarter)
        return []

    effective_quarter = data.get("quarter", quarter)
    note = ""
    if basis == "CY":
        note = "requested calendar period; answered from fiscal data (SEC companyfacts)"

    payload = {
        "kind": "metric",
        "basis": basis,
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

    logging.info("[TOOL metric] OK | ticker=%s metric=%s basis=%s y=%s q=%s value=%r",
                 ticker, metric, basis, year, effective_quarter, data.get("value"))
    return [Document(page_content=json.dumps(data), metadata=meta)]

def preview(txt: str, n: int = 240) -> str:
    s = (txt or "").replace("\n", " ").strip()
    return (s[:n] + "…") if len(s) > n else s

def rebuild_sources(final_docs: list):
    out = []
    for d in (final_docs or []):
        md = getattr(d, "metadata", {}) or {}
        # score is a Document attribute, not in metadata
        try:
            score_val = float(getattr(d, "score", 0.0))
        except Exception:
            score_val = 0.0

        out.append({
            "source": md.get("source") or md.get("filepath") or "(unknown)",
            "score":  score_val,
            "organization": md.get("organization", "") or "",
            "report_type":  md.get("report_type", "")  or "",
            "fiscal_year":  md.get("fiscal_year", "")  or "",
            "title":        md.get("title", "")        or "",
            "content_preview": preview(getattr(d, "page_content", "") or ""),
        })
    return out

# Simple CLI smoke (optional)
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

