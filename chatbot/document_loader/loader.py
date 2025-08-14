from pathlib import Path
from typing import Any
from chatbot.entities.document import Document
from chatbot.helpers.log import get_logger
from tqdm import tqdm
from unstructured.partition.auto import partition

logger = get_logger(__name__)


class DirectoryLoader:
    """
    Loader class for extracting financial documents from a specific directory.
    Optimized for docs where financial metadata
    such as report type, fiscal year, and organization can be pre-defined.
    """

    def __init__(
        self,
        path: Path,
        glob: str = "**/[!.]*",
        recursive: bool = False,
        show_progress: bool = False,
        use_multithreading: bool = False,
        max_concurrency: int = 4,
        **partition_kwargs: Any,
    ):
        """Initialize with a path to directory and how to glob over it.

        Args:
            Initialize the DirectoryLoader.

        Args:
            path (Path): Root path to the directory containing md files from docs
            glob (str): File match pattern (default: 'demo.md')
            recursive (bool): Whether to search subdirectories (not used)
            show_progress (bool): Whether to show tqdm progress bar
            use_multithreading (bool): Enable multithreaded parsing (not required here)
            max_concurrency (int): Maximum thread count for parsing
            partition_kwargs (dict): Extra arguments for `unstructured.partition` API
        """
        self.path = path
        self.glob = glob
        self.recursive = recursive
        self.show_progress = show_progress
        self.use_multithreading = use_multithreading
        self.max_concurrency = max_concurrency
        self.partition_kwargs = partition_kwargs

    def load(self) -> list[Document]:
        """
        Load financial documents.

        Returns:
            list[Document]: List of Document objects with financial metadata
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Directory not found: '{self.path}'")
        if not self.path.is_dir():
            raise ValueError(f"Expected directory, got file: '{self.path}'")

        docs: list[Document] = []
        items = list(self.path.rglob(self.glob))

        pbar = tqdm(total=len(items)) if self.show_progress else None

        for i in items:
            self.load_file(i, docs, pbar)

        if pbar:
            pbar.close()

        return docs

    def load_file(self, doc_path: Path, docs: list[Document], pbar: Any | None) -> None:
        """
         Load and parse a single file, apply financial metadata.

        Args:
            doc_path (Path): File path (expects demo.md)
            docs (list[Document]): Output document list
            pbar (tqdm or None): Optional progress bar instance
        """
        if doc_path.is_file():
            try:
                logger.debug(f"Processing file: {str(doc_path)}")
                # Use unstructured `partition()` to detect file type and extract content
                # Automatically routes to the correct parser using libmagic
                elements = partition(filename=str(doc_path), **self.partition_kwargs)
                # Note: The `partition` function returns a list of elements that we can filter by type based on the
                # specific format.
                text = "\n\n".join([str(el) for el in elements])
                # Inject static metadata relevant to finance domain
                metadata = {
                    "source": str(doc_path),
                    "report_type": "10-K",  # Common financial report type
                    "fiscal_year": "2023",
                    "organization": "DemoCorp",
                    "sector": "Technology",
                }

                docs.append(Document(page_content=text, metadata=metadata))
            finally:
                if pbar:
                    pbar.update(1)

if __name__ == "__main__":
    # Script entry point for local testing/demo
    root_folder = Path(__file__).resolve().parent.parent.parent
    docs_path = root_folder / "docs"
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.md",
        recursive=True,
        use_multithreading=True,
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
