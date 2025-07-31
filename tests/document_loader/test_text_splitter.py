from chatbot.document_loader.format import Format
from chatbot.document_loader.text_splitter import RecursiveCharacterTextSplitter, create_recursive_text_splitter


def test_recursive_character_text_splitter_keep_separators() -> None:
    split_tags = ["$", "%"]
    query = "Revenue increased by $100 million, a 25% growth."

    # keep_separator True
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=0,
        separators=split_tags,
        keep_separator=True,
    )
    result = splitter.split_text(query)
    assert result == ["Revenue increased by ", "$100 million, a 25", "% growth."]

    # keep_separator False
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=0,
        separators=split_tags,
        keep_separator=False,
    )
    result = splitter.split_text(query)
    assert result == ["Revenue increased by ", "100 million, a 25", "growth."]


def test_iterative_text_splitter() -> None:
    """Test iterative text splitter with financial phrases."""

    text = """EPS up.\n\nQ1 2023: Net income surged.\n\nRevenue from AWS exceeded.
\nGross margin improved.\nCapEx held steady.
\n\nEnd\n\n-H."""

    splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=2)
    output = splitter.split_text(text)
    assert any("Q1 2023" in chunk for chunk in output)
    assert any("Net income" in chunk for chunk in output)
    assert any("AWS" in chunk for chunk in output)
    assert any("Gross margin" in chunk for chunk in output)
    assert any("CapEx" in chunk for chunk in output)
    assert any("End" in chunk for chunk in output)
    assert any("-H." in chunk for chunk in output)


def test_markdown_splitter() -> None:
    splitter = create_recursive_text_splitter(format=Format.MARKDOWN.value, chunk_size=20, chunk_overlap=0)
    code = """
# Q4 2023 Financial Report

## Overview

Revenue: $120M
Net Income: $30M
EPS: $1.25

## Business Units

- Cloud: Strong growth
- Devices: Flat
- Advertising: +10%

## Notes

***********
____________
-------------------

#### Code blocks
```
Gross Margin = Revenue - COGS
# sample code
a = 1
b = 2
```
    """
    chunks = splitter.split_text(code)
    assert any("Revenue" in c for c in chunks)
    assert any("Net Income" in c for c in chunks)
    assert any("EPS" in c for c in chunks)

    # Special test for horizontal split markers in financial markdown
    code = "Earnings Call\n***\nGuidance maintained"
    chunks = splitter.split_text(code)
    # Assert that each expected concept appears in at least one chunk
    assert any("Earnings" in c for c in chunks)
    assert any("***" in c for c in chunks)
    assert any("Guidance" in c for c in chunks)
    assert any("maintained" in c for c in chunks)
