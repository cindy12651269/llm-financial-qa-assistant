import logging
import uuid
import re
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
        client: chromadb.Client = None,
        embedding: Embedder | None = None,
        persist_directory: str | None = None,
        collection_name: str = "finance_docs",
        collection_metadata: dict | None = None,
        is_persistent: bool = True,
    ) -> None:
        # Create a configuration object for Chroma to control persistence and location
        client_settings = chromadb.config.Settings(is_persistent=is_persistent)
        client_settings.persist_directory = persist_directory

        # Use an existing Chroma client if passed in, otherwise create a new one
        if client is not None:
            self.client = client
        else:
            self.client = chromadb.Client(client_settings)

        self.embedding = embedding

        # Either get an existing collection or create one named `finance_docs`
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,  # Handle embeddings manually
            metadata=collection_metadata,
        )

    @property
    def embeddings(self) -> Embedder | None:
        """Accessor for the embedder instance."""
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
        Low-level query to Chroma's collection.
        Either `query_texts` or `query_embeddings` should be supplied.
        """
        return self.collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            **kwargs,
        )

    # -------- Ingest APIs --------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Embed and upsert a batch of texts with optional metadata and IDs.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = None
        texts = list(texts)

        if self.embedding is not None:
            embeddings = self.embedding.embed_documents(texts)

        if metadatas:
            # Fill missing metadata dicts
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff

            # Split by non-empty/empty metadata
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)

            if non_empty_ids:
                metadatas_ne = [metadatas[idx] for idx in non_empty_ids]
                texts_ne = [texts[idx] for idx in non_empty_ids]
                embeddings_ne = [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                ids_ne = [ids[idx] for idx in non_empty_ids]
                try:
                    self.collection.upsert(
                        metadatas=metadatas_ne,
                        embeddings=embeddings_ne,
                        documents=texts_ne,
                        ids=ids_ne,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = "Try filtering complex metadata fields like nested financial structures."
                        raise ValueError(e.args[0] + "\n\n" + msg)
                    else:
                        raise e

            if empty_ids:
                texts_e = [texts[j] for j in empty_ids]
                embeddings_e = [embeddings[j] for j in empty_ids] if embeddings else None
                ids_e = [ids[j] for j in empty_ids]
                self.collection.upsert(
                    embeddings=embeddings_e,
                    documents=texts_e,
                    ids=ids_e,
                )
        else:
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
        Batched add_texts wrapper using Chroma's batch utils.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

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
        Add a list of Document chunks (already split & cleaned) into the store.
        """
        texts = [clean(doc.page_content, no_emoji=True) for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        self.from_texts(
            texts=texts,
            metadatas=metadatas,
        )

    # -------- Query APIs --------

    def similarity_search_with_threshold(
        self,
        query: str,
        k: int = 4,
        threshold: float | None = 0.2,
        exclude_tools: bool = True,  # default: exclude tool-injected docs for RAG
    ) -> tuple[list[Document], list[dict[str, Any]]]:
        """
        Financial RAG search with thresholding and light source-type heuristics.
        """
        KPI_RE = re.compile(r"\b(eps|earnings per share|revenue|sales|ebitda|guidance)\b", re.I)
        TRUSTED_TYPES = {"filing", "ir", "slides"}
        STRONG_SECTIONS = {"md&a", "mda", "outlook", "guidance", "results"}

        def _clamp01(x: float) -> float:
            try:
                return 0.0 if x is None else max(0.0, min(1.0, float(x)))
            except Exception:
                return 0.0

        # 1) retrieve a wider pool, then rescore & filter
        docs_and_scores = self.similarity_search_with_relevance_scores(query, k * 3)
        is_kpi = bool(KPI_RE.search(query or ""))

        rescored: list[tuple[Document, float, dict]] = []
        for doc, raw_score in docs_and_scores:
            base = _clamp01(raw_score)
            md = dict(getattr(doc, "metadata", {}) or {})
            st = str(md.get("source_type", "")).lower()

            if exclude_tools and st == "tool":
                continue
            if is_kpi and st == "news":
                continue

            bonus = 0.0
            if st in TRUSTED_TYPES:
                bonus += 0.02

            sec = str(md.get("section", "")).lower()
            if sec in STRONG_SECTIONS or "md&a" in sec or "management discussion" in sec:
                bonus += 0.02

            score = base + bonus
            md["score"] = round(score, 3)
            rescored.append((doc, score, md))

        thr = 0.0 if threshold is None else float(threshold)
        kept = [(d, s, m) for (d, s, m) in rescored if s >= thr]
        kept.sort(key=lambda x: x[1], reverse=True)
        kept = kept[:k]

        retrieved_contents = [d for d, _, _ in kept]
        sources: list[dict[str, Any]] = []
        for d, s, m in kept:
            sources.append(
                {
                    "score": round(s, 3),
                    "source": m.get("source", "unknown"),
                    "document": m.get("document"),
                    "organization": m.get("organization", ""),
                    "fiscal_year": m.get("fiscal_year", ""),
                    "report_type": m.get("report_type", ""),
                    "source_type": m.get("source_type", ""),
                    "section": m.get("section", ""),
                    "title": m.get("title", ""),
                    "content_preview": f"{(d.page_content or '')[:256]}...",
                }
            )

        return retrieved_contents, sources

    def similarity_search(self, query: str, k: int = 4, filter: dict[str, str] | None = None) -> list[Document]:
        """
        Simple search without score thresholding (for debugging).
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
        Return raw (doc, distance) from Chroma (distance = backend-specific).
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
        Select a distance→score mapping based on collection metadata.
        Falls back to L2 if unknown.
        """
        # Chroma commonly stores metric in metadata["hnsw:space"] as a string: "l2", "ip", "cosine"
        meta = self.collection.metadata or {}
        space = meta.get("hnsw:space", DistanceMetric.L2.value)  # default "l2"
        try:
            metric = space if isinstance(space, DistanceMetric) else DistanceMetric(str(space))
        except Exception:
            metric = DistanceMetric.L2
        return get_relevance_score_fn(metric)

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """
        Run search and return (doc, score∈[0,1]) where higher = more relevant.
        """
        relevance_score_fn = self.__select_relevance_score_fn()
        docs_and_distances = self.similarity_search_with_score(query, k)

        out: list[tuple[Document, float]] = []
        for doc, raw_distance in docs_and_distances:
            try:
                s = float(relevance_score_fn(float(raw_distance)))
            except Exception:
                # Fallback to heuristic mapping if score function explodes
                s = _to_unit_interval(raw_distance)
            # Final clamp for absolute safety
            s = 0.0 if s is None else max(0.0, min(1.0, float(s)))
            out.append((doc, s))
        return out

