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


def test_similarity_search_with_threshold_basic(chroma_instance):
    texts = ["Net income of Tesla in 2022 exceeded analyst expectations."]
    metadatas = [{"source": "MarketWatch"}]  # no source_type provided
    chroma_instance.add_texts(texts, metadatas)

    query = "Tesla income 2022"
    results, sources = chroma_instance.similarity_search_with_threshold(
        query, k=1, threshold=0.0
    )

    assert len(results) == 1
    assert len(sources) == 1
    assert isinstance(results[0], Document)
    assert isinstance(sources[0].get("score"), float)
    assert 0.0 <= sources[0]["score"] <= 1.0
    # shape remains stable
    assert "source" in sources[0]
    assert "organization" in sources[0]  # may be empty if not provided
    assert "report_type" in sources[0]   # may be empty if not provided

def test_similarity_search_with_threshold_kpi_excludes_news(chroma_instance):
    texts = [
        "EPS up due to lower opex per management commentary.",
        "Media recap: Company raised guidance for FY."
    ]
    metadatas = [
        {"source": "10-Q", "source_type": "filing"},
        {"source": "Some News Site", "source_type": "news"},
    ]
    chroma_instance.add_texts(texts, metadatas)

    # KPI-like query (contains eps/guidance keywords)
    query = "What was the EPS and guidance change?"
    results, sources = chroma_instance.similarity_search_with_threshold(
        query, k=2, threshold=0.0
    )

    # ensure news is filtered out
    assert len(results) >= 1
    for s in sources:
        assert s.get("source_type", "").lower() != "news"

def test_similarity_search_with_threshold_trusted_boost_fields(chroma_instance):
    texts = [
        "Management Discussion and Analysis: revenue grew due to pricing.",
        "Random forum post: revenues might grow."
    ]
    metadatas = [
        {"source": "10-K", "source_type": "filing", "section": "MD&A"},
        {"source": "forum", "source_type": "other"},
    ]
    chroma_instance.add_texts(texts, metadatas)

    query = "Explain why revenue increased"
    results, sources = chroma_instance.similarity_search_with_threshold(
        query, k=2, threshold=0.0
    )

    assert len(results) >= 1
    # Shape & fields exist
    for s in sources:
        assert "score" in s
        assert "source_type" in s
        assert "section" in s

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
