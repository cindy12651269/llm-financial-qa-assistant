from pathlib import Path

import chromadb
from chatbot.bot.memory.embedder import Embedder
from chatbot.bot.memory.vector_database.chroma import Chroma
from chatbot.helpers.prettier import prettify_source

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent.parent

    # Declarative memory (RAG documents)
    # Contains an extract of documents uploaded to the RAG bot
    declarative_vector_store_path = root_folder / "vector_store" / "exp_docs_index"
    embedding = Embedder()
    index = Chroma(persist_directory=str(declarative_vector_store_path), embedding=embedding)

    query = "What was Tesla's EPS in Q4 2023?"
    matched_docs, sources = index.similarity_search_with_threshold(query)

    for source in sources:
        print("\nDeclarative memory match:")
        print(prettify_source(source))

    # Episodic memory: simulates past financial conversations
    # Contains an extract of things the user said in the past
    episodic_vector_store_path = root_folder / "vector_store" / "episodic_index"
    persistent_client = chromadb.PersistentClient(path=str(episodic_vector_store_path))
    collection = persistent_client.get_or_create_collection("episodic_memory")

    collection.add(
        ids=["1", "2", "3"],
        documents=[
            "Tesla reported strong revenue in Q4",
            "EPS exceeded expectations",
            "The stock price surged 10% after earnings"
        ],
    )

    chroma = Chroma(
        client=persistent_client,
        collection_name="episodic_memory",
        embedding=embedding,
    )

    docs = chroma.similarity_search("EPS")
    print("\nSimilarity Search:")
    for d in docs:
        print(d.page_content)

    docs_with_score = chroma.similarity_search_with_score("EPS")
    print("\nWith Score:")
    for doc, score in docs_with_score:
        print(f"Score: {score:.4f} - {doc.page_content}")

    docs_with_relevance_score = chroma.similarity_search_with_relevance_scores("EPS")
    print("\nWith Relevance Score:")
    for doc, rel in docs_with_relevance_score:
        print(f"Relevance: {rel:.4f} - {doc.page_content}")

    matched_doc = max(docs_with_relevance_score, key=lambda x: x[1])

    # Raw Chroma client API (low-level)
    # The returned distance score is cosine distance. Therefore, a lower score is better.
    results = collection.query(
        query_texts=["EPS"],  # Financial query term: Earnings Per Share
        n_results=3,
        where={"fiscal_year": "2023"},  # Filter by financial metadata (e.g., 2023 annual reports)
        where_document={"$contains": "Net Income"}  # Optionally filter documents containing this keyword
    )
    print("\nRaw Chroma query (financial context):")
    print(results)
