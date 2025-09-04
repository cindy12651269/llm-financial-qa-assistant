import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from chatbot.bot.memory.embedder import Embedder
from chatbot.bot.memory.vector_database.chroma import Chroma
from chatbot.document_loader.format import Format
from chatbot.document_loader.loader import DirectoryLoader
from chatbot.document_loader.text_splitter import create_recursive_text_splitter
from chatbot.entities.document import Document
from chatbot.helpers.log import get_logger

logger = get_logger(__name__)

# Init logging and show target index path (do NOT fail if it doesn't exist)
logging.basicConfig(level=logging.INFO)
INDEX_DIR = Path(os.getenv("INDEX_PATH", "vector_store/docs_index"))
logging.info(f"[INIT] Target index path (will be created if missing): {INDEX_DIR.resolve()}")

# Load all Markdown files under docs_path (recursive).
def load_documents(docs_path: Path) -> list[Document]:
    """
    Args:
        docs_path (Path): The path to the documents.
    Returns:
        List[Document]: A list of loaded documents.
    """
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.md", 
        recursive=True, 
        show_progress=True,
    )
    return loader.load()

# Lightweight loader for JIT: read *.md as-is and wrap into Document objects.
def load_markdown_docs(docs_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for md_file in docs_dir.rglob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8")
            documents.append(
                Document(page_content=text, metadata={"source": str(md_file), "filepath": str(md_file)})
            )
        except Exception as e:
            logger.warning(f"[load_markdown_docs] Failed to read {md_file}: {e!r}")
    logger.info(f"[load_markdown_docs] Loaded {len(documents)} markdown files from {docs_dir}")
    return documents

# Metadata helpers
# Infer ticker & form from filename like: AAPL_10-Q_2025-08-01.md  /  TSLA_8-K_2025-07-23.md
def infer_org_form(file_name: str) -> tuple[str, str]:
    stem = Path(file_name).stem
    parts = stem.split("_", 2)  # [TICKER, FORM, ...]
    org  = parts[0].upper() if parts else ""
    form = parts[1].upper() if len(parts) > 1 else ""
    return org, form

# Parse the ingest header from ingest_pipeline.
def parse_header(md_text: str) -> Dict[str, str]:
    """
    Parse the simple ingest header (first ~40 lines) expected from ingest_pipeline:
      - **Date**: YYYY-MM-DD
      - **Source**: <URL>
      - **Title**: ...
    """
    date = source = title = ""
    for i, line in enumerate(md_text.splitlines()):
        if i > 40:  # header is always at the top
            break
        line = line.strip()
        if line.startswith("- **Date**:"):
            date = line.split(":", 1)[1].strip()
        elif line.startswith("- **Source**:"):
            raw = line.split(":", 1)[1].strip()
            source = raw.strip("<>")  # remove <...>
        elif line.startswith("- **Title**:"):
            title = line.split(":", 1)[1].strip()
    return {"date": date, "source": source, "title": title}

# Build lightweight, consistent metadata for scoring/filters & citations.
def normalize_metadata(file_path: Path, md_text: str, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    org, form = infer_org_form(file_path.name)
    hdr = parse_header(md_text)
    fiscal_year = (hdr["date"][:4] if hdr["date"] else "")

    meta = dict(base or {})
    meta.update({
        "organization": org,          # e.g., AAPL
        "report_type": form,          # 10-K / 10-Q / 8-K
        "fiscal_year": fiscal_year,   # YYYY
        "source_type": "filing",      # later: ir / slides
        "source": hdr["source"],      # canonical URL
        "title": hdr["title"],        # human-friendly title
    })
    return meta

# Splits a list of financial documents into smaller chunks.
def split_chunks(sources: list, chunk_size: int = 512, chunk_overlap: int = 25) -> list:
    """
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

def build_memory_index(docs_path: Path, vector_store_path: str, chunk_size: int, chunk_overlap: int):
    """
    Loads financial documents, splits them, embeds, and stores in vector DB.

    Args:
        docs_path (Path): Path to the source docs (e.g. 'docs/demo.md')
        vector_store_path (str): Path to persist Chroma DB.
        chunk_size (int): Chunk size for document splitting.
        chunk_overlap (int): Overlap between chunks.
        Build the vector index from Markdown docs under `docs_path`.

    What it does (minimal, safe):
    1) Load all `**/*.md` recursively via DirectoryLoader (already configured in load_documents()).
    2) Normalize metadata per file (organization/report_type/fiscal_year/source_type/source/title).
    3) Strip the ingest header (above the first '---') and only index the BODY.
    4) Chunk the BODY and persist to Chroma.

    """
    logger.info(f"Loading documents from: {docs_path}")
    sources = load_documents(docs_path)
    logger.info(f"Number of loaded documents: {len(sources)}")
    
    if not sources:
        logger.warning(f"[INDEX BUILDER] No markdown files found under {docs_path}. Nothing to index.")
        return
    # Ensure the index directory exists (critical fix so first-time build doesn't fail)
    Path(vector_store_path).mkdir(parents=True, exist_ok=True)

    logger.info("Chunking documents...")
    splitter = create_recursive_text_splitter(
        format=Format.MARKDOWN.value,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunk_docs: list[Document] = []

    for src in sources:
        md_text = src.page_content or ""
        # Prefer local file path if available; fall back to 'source' only when it's not a URL.
        src_path_str = (
            src.metadata.get("filepath")
            or (
                src.metadata.get("source")
                if not str(src.metadata.get("source", "")).lower().startswith(("http://", "https://"))
                else ""
            )
            or ""
        )
        file_path = Path(src_path_str) if src_path_str else Path("unknown.md")
        # Normalize metadata using filename + header
        meta = normalize_metadata(file_path, md_text, base=src.metadata)
        # Body after header separator (avoid indexing the header itself)
        body = md_text.split("\n---\n", 1)[1] if "\n---\n" in md_text else md_text
        if not body.strip():
            continue
        # Split BODY and attach the same normalized metadata to each chunk
        for piece in splitter.split_text(body):
            if piece.strip():
                chunk_docs.append(Document(page_content=piece, metadata=meta))

    logger.info(f"Number of generated chunks: {len(chunk_docs)}")

    logger.info("Creating memory index...")
    embedding = Embedder()
    vector_database = Chroma(persist_directory=str(vector_store_path), embedding=embedding)
    vector_database.from_chunks(chunk_docs)
    logger.info(
    f"[INDEX BUILT] files={len(sources)} chunks={len(chunk_docs)} "
    f"chunk_size={chunk_size}, overlap={chunk_overlap} -> {vector_store_path}")
    logger.info("Memory Index has been created successfully!")

# CLI test
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

#  Main entry point: builds memory index from docs/.
def main(parameters):
    root_folder = Path(__file__).resolve().parent.parent  # project root (adjust if needed)
    docs_path = root_folder / "docs"
    vector_store_path = root_folder / "vector_store" / "docs_index"

    build_memory_index(
        docs_path,
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