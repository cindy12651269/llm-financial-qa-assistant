import pytest

from chatbot.bot.memory.embedder import Embedder
from chatbot.bot.memory.vector_database.chroma import Chroma
from chatbot.entities.document import Document


@pytest.fixture
def chroma_instance(tmp_path):
    return Chroma(embedding=Embedder(), persist_directory=str(tmp_path))


def test_initialization(chroma_instance):
    assert chroma_instance.embedding is not None
    assert chroma_instance.client is not None
    assert chroma_instance.collection is not None


def test_add_texts(chroma_instance):
    texts = ["Apple reported a 12% increase in revenue for Q4 2023."]
    metadatas = [
        {
            "source": "Earnings Report",
            "fiscal_year": "2023",
            "report_type": "10-K",
            "organization": "Apple Inc.",
            "sector": "Technology",
        }
    ]
    ids = chroma_instance.add_texts(texts, metadatas)
    assert len(ids) == 1


def test_similarity_search(chroma_instance):
    texts = ["Apple reported a 12% increase in revenue for Q4 2023."]
    metadatas = [{"source": "Earnings Report"}]
    chroma_instance.add_texts(texts, metadatas)

    results = chroma_instance.similarity_search("Apple revenue", k=1)
    assert len(results) == 1
    assert isinstance(results[0], Document)


def test_similarity_search_with_threshold(chroma_instance):
    texts = ["Net income of Tesla in 2022 exceeded analyst expectations."]
    metadatas = [{"source": "MarketWatch"}]
    chroma_instance.add_texts(texts, metadatas)

    query = "Tesla income 2022"
    results, source = chroma_instance.similarity_search_with_threshold(query, k=1, threshold=0.0)

    assert len(results) == 1
    assert len(source) == 1
    assert isinstance(results[0], Document)
    assert isinstance(source[0].get("score"), float)
    assert 0.0 <= source[0]["score"] <= 1.0


def test_similarity_search_with_score(chroma_instance):
    texts = ["Alphabet's advertising segment remains the primary revenue driver."]
    metadatas = [{"source": "Financial Times"}]
    chroma_instance.add_texts(texts, metadatas)

    results = chroma_instance.similarity_search_with_score("advertising revenue", k=1)
    assert len(results) == 1
    assert isinstance(results[0][0], Document)
    assert isinstance(results[0][1], float)


def test_similarity_search_with_relevance_scores(chroma_instance):
    texts = ["Amazon's Q1 2024 report highlighted AWS growth and logistics efficiency."]
    metadatas = [{"source": "SEC Filing"}]
    chroma_instance.add_texts(texts, metadatas)

    results = chroma_instance.similarity_search_with_relevance_scores("AWS growth", k=1)
    assert len(results) == 1
    assert isinstance(results[0][0], Document)
    assert isinstance(results[0][1], float)
    assert 0.0 <= results[0][1] <= 1.0


print(Document.__module__)
