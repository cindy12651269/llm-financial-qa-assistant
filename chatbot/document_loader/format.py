from enum import Enum


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
