import logging
import uuid
from typing import Any, Callable, Iterable

import chromadb
import chromadb.config
from chromadb.utils.batch_utils import create_batches
from cleantext import clean

from chatbot.bot.memory.embedder import Embedder
from chatbot.bot.memory.vector_database.distance_metric import DistanceMetric, get_relevance_score_fn
from chatbot.entities.document import Document

logger = logging.getLogger(__name__)

def _to_unit_interval(raw: float) -> float:
    """
    Map various similarities/distances to [0,1] where 1 = most relevant.
    Heuristics:
      - raw < 0            -> cosine similarity in [-1,1]  => (x+1)/2
      - 0 <= raw <= 1      -> already normalized           => clamp
      - 1 < raw <= 2       -> cosine distance in [0,2]     => 1 - x/2
      - raw  > 2           -> L2 distance                  => 1/(1+x)
    """
    x = float(raw)
    if x < 0:
        y = (x + 1.0) / 2.0
    elif 0.0 <= x <= 1.0:
        y = x
    elif x <= 2.0:
        y = 1.0 - (min(x, 2.0) / 2.0)
    else:
        y = 1.0 / (1.0 + max(x, 0.0))
    return max(0.0, min(1.0, y))


class Chroma:
    """
    Wrapper class for Chroma vector database integration.
    Supports storing and retrieving financial documents with embedding and metadata.
    """

    # Initializes the Chroma vector store client for financial document use cases.
    def __init__(
        self,
        client: chromadb.Client = None,  # client (chromadb.Client, optional): An existing Chroma client. If not provided, one will be created.
        embedding: Embedder
        | None = None,  # embedding (Embedder, optional): The embedding engine used to convert text into vectors.
        persist_directory: str
        | None = None,  # persist_directory (str, optional): Filesystem path where Chroma should persist its vector index.
        collection_name: str = "finance_docs",  # collection_name (str): The name of the vector collection to use. Defaults to "finance_docs".
        collection_metadata: dict
        | None = None,  # collection_metadata (dict, optional): Metadata associated with the collection (e.g., index type).
        is_persistent: bool = True,  # is_persistent (bool): Whether to persist the vector index to disk. Defaults to True.
    ) -> None:
        # Create a configuration object for Chroma to control persistence and location
        client_settings = chromadb.config.Settings(is_persistent=is_persistent)
        client_settings.persist_directory = persist_directory

        # Use an existing Chroma client if passed in, otherwise create a new one
        if client is not None:
            self.client = client
        else:
            self.client = chromadb.Client(client_settings)

        self.embedding = embedding  # Set the embedding model to use

        # Either get an existing collection or create one named `finance_docs`
        # This collection stores our embedded financial text data
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,  # Handle embeddings manually
            metadata=collection_metadata,
        )



    @property
    def embeddings(self) -> Embedder | None:
        """
        Property accessor for the embedder instance.

        Returns:
            Embedder | None: The current embedding model used to convert text to vector form.
            This can be a custom class wrapping OpenAI, HuggingFace, or any vectorizer.
        """
        return self.embedding

    def __query_collection(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 4,
        where: dict[str, str] | None = None,
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        """
        Low-level query method for retrieving financial documents from Chroma.

        Supports querying using either raw financial text (`query_texts`) or precomputed
        embedding vectors (`query_embeddings`), returning top-k similar document chunks
        with optional metadata filters.

        Args:
            query_texts (list[str] | None): Raw financial queries (e.g., "What is ROE?").
            query_embeddings (list[list[float]] | None): Precomputed embeddings for the queries.
            n_results (int): Number of most relevant results to return. Default is 4.
            where (dict | None): Metadata filter (e.g., {"fiscal_year": "2023"}).
            where_document (dict | None): Filter by document content (e.g., $contains keyword).
            **kwargs (Any): Additional options passed to Chroma's `query()` function.

        Returns:
            dict: Dictionary containing:
            - "documents": Matched financial text chunks
            - "metadatas": Metadata (e.g., report_type, sector)
            - "distances": Vector distances
            - "ids": Internal doc IDs

        Notes:
            - Provide either `query_texts` or `query_embeddings`, not both.
            - Preferred wrapper: `similarity_search_with_threshold()` for financial RAG pipelines.
            - See more: https://docs.trychroma.com/reference/py-collection#query
        """

        return self.collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            **kwargs,
        )

    # Add a batch of texts and their financial metadata into Chroma vectorstore
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Embed and store a batch of financial texts into the Chroma vector database,
        along with optional financial metadata and document IDs.

        Each text may correspond to a paragraph or section from financial reports,
        filings (e.g., 10-K), earnings call transcripts, or glossary entries. Metadata
        can include tags such as fiscal_year, report_type, sector, and organization.

        Args:
            texts (Iterable[str]): Raw financial text entries (e.g., ["EPS is..."]).
            metadatas (list[dict] | None): Optional list of metadata dicts, one per text.
                Each dict may contain financial keys like "fiscal_year", "organization", etc.
            ids (list[str] | None): Optional list of unique document IDs. If not provided, UUIDs are generated.

        Returns:
            list[str]: List of document IDs inserted into the vector database.
        """

        # If no IDs are provided, generate unique UUIDs
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = None
        texts = list(texts)

        # Generate embeddings for financial texts if embedding model is present
        if self.embedding is not None:
            embeddings = self.embedding.embed_documents(texts)

        if metadatas:
            # Fill any missing metadata entries with empty dicts
            # Did not specify metadata for all texts
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff

            # Separate texts with metadata vs. without
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)

            # Insert documents that have metadata (e.g., from financial reports)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                texts_with_metadatas = [texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self.collection.upsert(
                        metadatas=metadatas,
                        embeddings=embeddings_with_metadatas,
                        documents=texts_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    # If Chroma raises an error about metadata format, give a finance-specific hint
                    if "Expected metadata value to be" in str(e):
                        msg = "Try filtering complex metadata fields like nested financial structures."
                        raise ValueError(e.args[0] + "\n\n" + msg)
                    else:
                        raise e
            # Insert documents without metadata (e.g., glossary definitions)
            if empty_ids:
                texts_without_metadatas = [texts[j] for j in empty_ids]
                embeddings_without_metadatas = [embeddings[j] for j in empty_ids] if embeddings else None
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self.collection.upsert(
                    embeddings=embeddings_without_metadatas,
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:  # Insert all texts if no metadata provided
            self.collection.upsert(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
            )
        return ids

    def from_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """
        Adds a batch of raw financial texts to the Chroma vector store.

        This method is suitable for adding financial documents such as
        glossary definitions, KPI explanations, or pre-cleaned earnings call excerpts.
        Metadata fields can include financial dimensions like "fiscal_year", "organization", or "report_type".

        Args:
            texts (list[str]): Raw financial texts (e.g., "EPS stands for Earnings Per Share...").
            metadatas (list[dict], optional): One-to-one metadata entries aligned with `texts`.
                Each entry can include financial tags such as:
                - source: e.g., '10-K 2023 Tesla'
                - fiscal_year: '2023'
                - report_type: 'Earnings Call'
                - sector: 'Technology'
                - organization: 'Tesla'
            ids (list[str], optional): Unique document IDs for each entry. If not provided, UUIDs are auto-generated.

        Returns:
            None
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Split large dataset into smaller batches for efficiency
        for batch in create_batches(
            api=self.client,
            ids=ids,
            metadatas=metadatas,
            documents=texts,
        ):
            self.add_texts(
                texts=batch[3] if batch[3] else [],
                metadatas=batch[2] if batch[2] else None,
                ids=batch[0],
            )

    def from_chunks(self, chunks: list) -> None:
        """
        Adds a batch of preprocessed Document objects into the Chroma vector store.

        This is typically used after documents have already been parsed, cleaned, and enriched
        with financial metadata fields such as:
            - fiscal_year: '2022'
            - report_type: '10-K'
            - organization: 'Apple'
            - sector: 'Consumer Electronics'

        Args:
            chunks (list[Document]): List of parsed and enriched financial document chunks.
                These chunks usually come from splitters applied to longer filings or reports.

        Returns:
            None
        """
        texts = [clean(doc.page_content, no_emoji=True) for doc in chunks]  # Sanitize text for consistency
        metadatas = [doc.metadata for doc in chunks]  # Retain structured financial tags
        self.from_texts(
            texts=texts,
            metadatas=metadatas,
        )

    # Perform semantic search and return results with financial metadata fields
    def similarity_search_with_threshold(
        self,
        query: str,
        k: int = 4,
        threshold: float | None = 0.2,
        exclude_tools: bool = True,  # NEW: default to exclude tool-injected docs from RAG
    ) -> tuple[list[Document], list[dict[str, Any]]]:
        """
        Perform a semantic similarity search using financial context,
        and return results with metadata such as fiscal year, report type, etc.

        This is often used in retrieval-augmented generation (RAG) settings
        for answering financial questions from company filings or glossary entries.

        Args:
            query (str): A financial query (e.g. "What is Tesla's EPS in 2023?")
            k (int): Max number of candidate documents to consider (default = 4).
            threshold (float): Filter out results with relevance scores below this (range: 0 to 1).

        Returns:
        tuple:
            - List[Document]: Filtered and sorted document chunks relevant to the query.
            - List[dict[str, Any]]: Structured metadata for UI display or debugging, including:
                - score: Similarity score
                - source: Original filename or reference (e.g. "10-K_2023_Tesla")
                - organization: Company name
                - fiscal_year: Financial year (e.g., "2022")
                - report_type: Type of document (e.g., "Earnings Call", "10-K")
                - sector: Industry sector
                - content_preview: First 256 characters of matched text

        """
        # `similarity_search_with_relevance_scores` return docs and relevance scores in the range [0, 1].
        # 0 is dissimilar, 1 is most similar.
        docs_and_scores = self.similarity_search_with_relevance_scores(query, k)
        
        def _norm(x: float) -> float:
            if x is None: return 0.0
            if x <= 0.0: return 0.0
            if x >= 1.0: return 1.0
            return float(x)

        docs_and_scores = [(doc, _norm(score)) for doc, score in docs_and_scores]

        normalized: list[tuple[Document, float]] = []
        for doc, raw in docs_and_scores:
            if raw is None:
                continue
            score01 = _to_unit_interval(raw)
            # NEW: skip tool docs if requested
            if exclude_tools and str(doc.metadata.get("source_type", "")).lower() == "tool":
                continue
            normalized.append((doc, score01))

        if threshold is not None:
            # keep only results with score >= threshold
            normalized = [p for p in normalized if p[1] >= threshold]
            if not normalized:
                logger.warning(
                    "No relevant docs were retrieved using the relevance score threshold %s",
                    threshold,
                )

        normalized.sort(key=lambda p: p[1], reverse=True)

        retrieved_contents = [doc for doc, _ in normalized]
        sources: list[dict[str, Any]] = []
        for doc, score in normalized:
            sources.append(
                {
                    "score": round(score, 3),
                    "source": doc.metadata.get("source", "unknown"),
                    "document": doc.metadata.get("document"),
                    "organization": doc.metadata.get("organization", ""),
                    "fiscal_year": doc.metadata.get("fiscal_year", ""),
                    "report_type": doc.metadata.get("report_type", ""),
                    "sector": doc.metadata.get("sector", ""),
                    "content_preview": f"{doc.page_content[:256]}...",
                }
            )

        return retrieved_contents, sources

    # Simple search wrapper to get the most relevant financial documents
    def similarity_search(self, query: str, k: int = 4, filter: dict[str, str] | None = None) -> list[Document]:
        """
        Perform basic similarity search over financial documents.

        This method is suitable for direct lookups without needing relevance score filtering,
        useful for debugging or deterministic lookups (e.g., glossary queries).

        Args:
            query (str): Financial query or keyword (e.g., "ROE definition").
            k (int): Number of documents to retrieve (default = 4).
            filter (dict[str, str], optional): Metadata-based filters (e.g., {"report_type": "10-K"}).

        Returns:
            List[Document]: Ranked financial document chunks based on vector similarity.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, filter=filter)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        where_document: dict[str, str] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Perform low-level vector similarity search over embedded financial texts.
        Useful for retrieving raw Chroma vector distances (e.g., cosine, L2)
        between a user's query and stored financial documents (e.g., 10-K filings, earnings transcripts).

        Args:
            query (str): User query such as "Tesla's ROE in 2023" or "definition of operating margin".
            k (int): Number of top results to retrieve (default = 4).
            filter (dict, optional): Metadata-based filtering (e.g., {"organization": "Tesla", "report_type": "10-K"}).
            where_document (dict, optional): Document content filtering using operators like {"$contains": {"text": "EPS"}}.

        Returns:
            list of tuples:
                - Document: Retrieved document chunk.
                - float: Vector distance score (lower is better similarity).
        """
        if self.embedding is None:
            results = self.__query_collection(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=where_document,
            )
        else:
            query_embedding = self.embedding.embed_query(query)
            results = self.__query_collection(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                where_document=where_document,
            )
        return [
            (Document(page_content=result[0], metadata=result[1] or {}), result[2])
            for result in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def __select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        Choose an appropriate scoring function to convert vector distance into a relevance score [0, 1].

        This function is critical in financial search pipelines to normalize vector distances
        across various distance metrics (L2, cosine, etc.), enabling consistent interpretation
        of document relevance to user financial queries.

        Returns:
            Callable: A function that maps raw distance → normalized relevance score.
        """

        distance = DistanceMetric.L2
        distance_key = "hnsw:space"
        metadata = self.collection.metadata

        if metadata and distance_key in metadata:
            distance = metadata[distance_key]
        return get_relevance_score_fn(distance)

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """
        Perform similarity search and return normalized relevance scores in [0, 1].

        This is used for user-facing financial chatbots and retrieval-augmented generation (RAG)
        to fetch financial knowledge chunks such as:
          - Glossary definitions ("Sharpe Ratio", "EBITDA")
          - Metrics from reports ("EPS in Q4 2023")
          - Investment concepts ("mean-variance optimization")

        Args:
            query (str): Input question from the user.
            k (int): Number of most relevant documents to retrieve. Default is 4.

        Returns:
            list of tuples:
                - Document: Financial document object.
                - float: Normalized relevance score between 0.0 (poor match) and 1.0 (perfect match).
        """
        # relevance_score_fn is a function to calculate relevance score from distance.
        relevance_score_fn = self.__select_relevance_score_fn()

        docs_and_scores = self.similarity_search_with_score(query, k)
        docs_and_similarities = [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]
        # Check for out-of-bound values
        if any(similarity < 0.0 or similarity > 1.0 for _, similarity in docs_and_similarities):
            logger.warning("Relevance scores must be between" f" 0 and 1, got {docs_and_similarities}")
        return docs_and_similarities
