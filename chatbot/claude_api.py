import os
import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# System prompt to bias answers when tools didn't return data
SYS_PROMPT = (
    "You are a financial analyst. If the user asks for a current price or recent EPS and tool "
    "results are missing, give a concise best-effort summary and point to exact primary sources "
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


def _extract_text(resp) -> str:
    """
    Safely extract text from a chat.completions.create response.
    Handles both string and list-of-parts content formats used by some routers.
    """
    try:
        msg = resp.choices[0].message
        # Message may be a dict-like or pydantic object
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")

        if content is None:
            return ""

        if isinstance(content, str):
            return content.strip()

        # Some providers return a list of parts ({type,text} or plain strings)
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    txt = part.get("text") or part.get("content")
                    if isinstance(txt, str):
                        parts.append(txt)
            return "\n".join(parts).strip()

        return str(content).strip()
    except Exception as e:
        logger.warning(f"Failed to parse Claude response content: {e}")
        return ""


def _escape_markdown(s: str) -> str:
    """
    Lightly escape Markdown control characters to avoid accidental italics/bold in UI.
    """
    return (
        s.replace("\\", "\\\\")
        .replace("*", r"\*")
        .replace("_", r"\_")
        .replace("`", r"\`")
    )


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
        return _escape_markdown(text) if escape_markdown else text
    except Exception as e:
        logger.error(f"Fallback Claude call failed ({FALLBACK_MODEL}): {e}")
        return None


if __name__ == "__main__":
    # Simple smoke test
    ans = call_claude_fallback("What was Apple’s net income in 2022?")
    print("Claude Answer:", ans)

