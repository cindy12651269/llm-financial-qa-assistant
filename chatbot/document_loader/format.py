from enum import Enum
import html
import re
import unicodedata


class Format(Enum):
    MARKDOWN = "markdown"

# Financial-aware text separators for financial documents
SUPPORTED_FORMATS = {
    Format.MARKDOWN.value: [
        # First, try to split along Markdown headings (H1–H6, starting with level 2)
        r"\n#{1,6} ",
        # Note the alternative syntax for headings (below) is not handled here
        # Heading level 2
        # ---------------
        # End of code block(common in earnings code sections)
        r"```\n",
        # Horizontal lines
        r"\n\*\*\*+\n",
        r"\n---+\n",
        r"\n___+\n",

        # Split before quarterly indicators (Q1, Q2, Q3, Q4)
        r"(?<=\s)(Q[1-4]\s\d{4})(?=\s)",
        # Split before financial keywords such as Revenue, Net Income, etc.
        r"(?<=\n)(?=Revenue:|Net Income:|Earnings per Share:|EBITDA:)",

        # Note that this splitter doesn't handle horizontal lines defined
        # by *three or more* of ***, ---, or ___, but this is not handled

        # Split by section breaks (double newline)
        r"\n\n",
        # Fallback single newline (sentence or line end)
        r"\n",
        # Split by spaces (least preferred)
        r" ",
        # Final fallback: empty string (character level)
        "",
    ]
}


def get_separators(format: str):
    """
    Retrieve a prioritized list of regex-based separators for chunking
    financial documents in the specified format.

    Args:
        format (str): The format key (e.g., "markdown").

    Returns:
        List[str]: List of string or regex separators in order of preference.

    Raises:
        KeyError: If the format is not defined in SUPPORTED_FORMATS.
    """
    separators = SUPPORTED_FORMATS.get(format)

    # validate input
    if separators is None:
        raise KeyError(f"{format} is not a supported format")

    return separators

def all_are_8k(docs):
    """Check if all retrieved docs are 8-K."""
    if not docs:
        return False
    def _src_lower(d):
        md = (d.metadata or {})
        return (md.get("source") or md.get("filepath") or "").lower()
    return all("8-k" in _src_lower(d) for d in docs)

def fix_glue(text: str) -> str:
    """Normalize spaces and avoid glued tokens like EPS38.8."""
    text = unicodedata.normalize("NFKC", text).replace("\u00A0", " ")
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"([.,;:])(?!\s|$)", r"\1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def to_html(text: str) -> str:
    """Convert markdown-like text to safe HTML with bold support."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = html.escape(text, quote=False) \
               .replace("&lt;strong&gt;", "<strong>") \
               .replace("&lt;/strong&gt;", "</strong>") \
               .replace("\n", "<br>")
    return (
        "<div style=\"white-space:pre-wrap;font-variant-ligatures:none;"
        "-webkit-font-smoothing:antialiased;font-feature-settings:'liga' 0,'clig' 0;"
        "font-style:normal;line-height:1.55;font-size:1rem;\">"
        f"{text}"
        "</div>"
    )


def yoy_warn(text: str) -> str:
    """Warn if YoY mentioned but missing prior-period values."""
    YOY_RE = re.compile(r"(?i)\b(yoy|year[- ]over[- ]year)\b")
    NUM_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
    if YOY_RE.search(text) and len(NUM_RE.findall(text)) < 2:
        text += ("\n\n⚠️ Note: Mentions YoY but missing the prior-period value. "
                 "Ask me to fetch both periods and I’ll include them clearly.")
    return text


def render_final_text(raw: str, yoy_check=True) -> tuple[str, str]:
    """
    Normalize → optional YoY warning → wrap as HTML.
    Returns (plain_text, html_text).
    """
    fixed = fix_glue(raw)
    if yoy_check:
        fixed = yoy_warn(fixed)
    return fixed, to_html(fixed)

# Stream tokens with sentence/paragraph buffering and HTML rendering.
def stream_tokens(llm, streamer, placeholder, yoy_check=True):
    """
    Args:
        llm: LLM client
        streamer: generator from answer_with_context
        placeholder: st.empty() slot for progressive updates
        yoy_check: whether to append YoY warning on final flush

    Returns:
        (plain_text, html_text)
    """
    buf, rendered, last_len = "", "", 0
    BOUNDARY_RE = re.compile(r"([.!?])+(\s|\n|$)")

    for token in streamer:
        buf += llm.parse_token(token)
        if BOUNDARY_RE.search(buf) or "\n\n" in buf:
            rendered += buf
            buf = ""
            plain, html_out = render_final_text(rendered, yoy_check=False)
            if len(plain) > last_len:
                placeholder.markdown(html_out + " ▌", unsafe_allow_html=True)
                last_len = len(plain)

    # Final flush
    rendered += buf
    plain, html_out = render_final_text(rendered, yoy_check=yoy_check)
    placeholder.markdown(html_out, unsafe_allow_html=True)
    return plain, html_out
