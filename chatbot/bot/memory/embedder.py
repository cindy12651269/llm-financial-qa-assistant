from typing import Any

import sentence_transformers


class Embedder:
    """
    Embedder for financial documents and queries.
    
    This class transforms financial texts (e.g. earnings reports, trading terms, financial Q&A)
    into dense vector embeddings using a transformer-based model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: str | None = None, **kwargs: Any):
        """
        Initialize the Embedder with a financial-compatible sentence transformer model.

        Args:
            model_name (str): SentenceTransformer model name. Default is 'all-MiniLM-L6-v2',
                              which offers lightweight and general-purpose embeddings.
            cache_folder (str, optional): Directory to cache model files locally.
            **kwargs (Any): Additional arguments passed to SentenceTransformer.
        """
        self.client = sentence_transformers.SentenceTransformer(model_name, cache_folder=cache_folder, **kwargs)

    def embed_documents(self, texts: list[str], multi_process: bool = False, **encode_kwargs: Any) -> list[list[float]]:
        """
        Embed a list of financial texts (e.g. 10-K summaries, investment strategies).

        Args:
            texts (list[str]): The list of financial paragraphs to embed.
            multi_process (bool): Enable multiprocessing for large batches.
            **encode_kwargs (Any): Extra encoding options (e.g., batch_size, device).

        Returns:
            list[list[float]]: List of embeddings for each input text.
        """
        # Normalize line breaks to ensure clean embedding input
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        if multi_process:
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.client.encode(texts, show_progress_bar=True, **encode_kwargs)

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a financial query string for semantic search (e.g. "What is ROE?").

        Args:
            text (str): The question or financial term to embed.

        Returns:
            list[float]: Vector embedding for the input query.
        """
        return self.embed_documents([text])[0]
