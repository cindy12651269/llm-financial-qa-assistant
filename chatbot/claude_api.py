import re
import unicodedata
import os
import logging
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

# System prompt to bias answers when tools didn't return data
SYS_PROMPT = (
    "You are a financial analyst. If the user asks for a current price or recent EPS and tool "
    "Use plain ASCII characters only. No mathematical/italic Unicode letters."
    "Provide concise answers and cite primary sources (SEC EDGAR, company IR)."
    "Results are missing, give a concise best-effort summary and point to exact primary sources "
    "(SEC EDGAR, company IR, Alpha Vantage). Do not say you cannot browse; be actionable."
)

# Read key from either OPENROUTER_API_KEY or CLAUDE_API_KEY (both supported)
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("CLAUDE_API_KEY")

# Preferred / fallback model ids on OpenRouter
PRIMARY_MODEL = "anthropic/claude-3.7-sonnet"
FALLBACK_MODEL = "anthropic/claude-3.5-sonnet-20240620"

# Optional headers for OpenRouter analytics (safe to keep empty or customize)
_DEFAULT_HEADERS = {
    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8501"),
    "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Financial RAG Chatbot"),
}

def _make_client() -> OpenAI:
    """
    Build an OpenAI-compatible client that talks to OpenRouter.
    Requires OPENROUTER_API_KEY (or CLAUDE_API_KEY) in the environment.
    """
    if not _OPENROUTER_API_KEY:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY (or CLAUDE_API_KEY) in environment."
        )
    return OpenAI(
        api_key=_OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers=_DEFAULT_HEADERS or None,
    )

# Extract text from a chat.completions.create response
def _extract_text(resp) -> str:
    """
    Safely extract text from a chat.completions.create response.
    Handles both string and list-of-parts content formats used by some routers.
    """
    msg = resp.choices[0].message
    # Message may be a dict-like or pydantic object
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part.strip())
            elif isinstance(part, dict):
                txt = (part.get("text") or part.get("content") or "").strip()
                if txt:
                    pieces.append(txt)
        text = " ".join(pieces)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    return str(content).strip()

# URL Handling
_URL_RE = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)

# Replace URLs with placeholders to avoid spacing/escaping changes.
def _protect_urls(text: str):
    urls = []
    def _sub(m):
        urls.append(m.group(0))
        return f"__URL_{len(urls)-1}__"
    protected = _URL_RE.sub(_sub, text)
    return protected, urls

# Restore URL placeholders; wrap with angle brackets for Markdown.
def _restore_urls(text: str, urls: list[str]) -> str:
    for i, u in enumerate(urls):
        text = text.replace(f"__URL_{i}__", f"<{u}>")
    return text

# Lightly escape Markdown control characters to avoid accidental italics/bold in UI.
def _escape_markdown(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace("*", r"\*")
        .replace("_", r"\_")
        .replace("`", r"\`")
    )

# Unicode guards
def _fix_unicode_spaces(s: str) -> str:
    """Convert all Unicode space separators and no-break spaces to regular spaces, then collapse."""
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if ch in ("\u00A0", "\u202F") or cat.startswith("Z"):  # NBSP, NNBSP, and all space separators
            out.append(" ")
        else:
            out.append(ch)
    # collapse consecutive spaces
    return re.sub(r" {2,}", " ", "".join(out)).strip()

_DIGIT_WORD_TO_CHAR = {
    "ZERO": "0","ONE":"1","TWO":"2","THREE":"3","FOUR":"4",
    "FIVE":"5","SIX":"6","SEVEN":"7","EIGHT":"8","NINE":"9",
}

def _demath_alphanum(s: str) -> str:
    """Map MATHEMATICAL italic/bold/doublestruck letters/digits to plain ASCII letters/digits."""
    out = []
    for ch in s:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            out.append(ch); continue

        if "MATHEMATICAL" not in name:
            out.append(ch); continue

        # letters: e.g. 'MATHEMATICAL ITALIC SMALL F', 'MATHEMATICAL BOLD CAPITAL Q'
        m = re.search(r"MATHEMATICAL [A-Z \-]* (SMALL|CAPITAL) ([A-Z])$", name)
        if m:
            case, letter = m.group(1), m.group(2)
            out.append(letter.lower() if case == "SMALL" else letter)
            continue

        # digits: e.g. 'MATHEMATICAL DOUBLE-STRUCK DIGIT FOUR'
        m = re.search(r"MATHEMATICAL [A-Z \-]* DIGIT ([A-Z]+)$", name)
        if m:
            word = m.group(1)
            out.append(_DIGIT_WORD_TO_CHAR.get(word, ch))
            continue

        out.append(ch)  # fallback: keep as-is
    return "".join(out)

def _normalize_llm_text(s: str, escape_md: bool = True) -> str:
    """Full sanitation pipeline: fix spaces, demath, normalize, (optional) escape markdown."""
    s = _fix_unicode_spaces(s)
    s = _demath_alphanum(s)
    s = unicodedata.normalize("NFKC", s)  # fold compatibility glyphs
    if escape_md:
        s = (s.replace("\\", "\\\\")
               .replace("*", r"\*")
               .replace("_", r"\_")
               .replace("`", r"\`"))
    return s

def call_claude_fallback(
    prompt: str,
    system_prompt: Optional[str] = SYS_PROMPT,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    escape_markdown: bool = True,
) -> Optional[str]:
    """
    Call Claude via OpenRouter using the OpenAI-compatible Chat Completions API.
    Tries 3.7 Sonnet first; on failure, falls back to 3.5 Sonnet.

    Returns the response text (optionally Markdown-escaped) or None on failure.
    """
    client = _make_client()
    chosen_model = model or PRIMARY_MODEL
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Try primary model
    try:
        logger.info(f"Calling Claude model: {chosen_model}")
        resp = client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = _extract_text(resp)
        if not text:
            raise ValueError("Empty content from primary model.")
        text = _normalize_llm_text(text, escape_md=True)
        return _escape_markdown(text) if escape_markdown else text
    except Exception as e:
        logger.error(f"Primary Claude call failed ({chosen_model}): {e}")

    # Fallback model
    try:
        logger.info(f"Falling back to Claude model: {FALLBACK_MODEL}")
        resp = client.chat.completions.create(
            model=FALLBACK_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = _extract_text(resp)
        if not text:
            return None
        text = _normalize_llm_text(text, escape_md=True)
        return _escape_markdown(text) if escape_markdown else text
    except Exception as e:
        logger.error(f"Fallback Claude call failed ({FALLBACK_MODEL}): {e}")
        return None

if __name__ == "__main__":
    # Simple smoke test
    ans = call_claude_fallback("What was Apple’s net income in 2022?")
    print("Claude Answer:", ans)

