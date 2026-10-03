from unittest.mock import patch

from src.embeddings import EmbeddingModel


TEST_TEXTS = [
    "Machine learning is a field of artificial intelligence.",
    "Regression models predict continuous numerical values.",
]


def test_document_embedding_fallback():
    """
    Simulate Gemini embedding quota exhaustion and verify
    automatic fallback to all-MiniLM-L6-v2.
    """

    model = EmbeddingModel()

    print("=" * 80)
    print("EMBEDDING FALLBACK TEST")
    print("=" * 80)

    print(f"Initial provider: {model.provider}")

    with patch(
        "src.embeddings.embed_texts",
        side_effect=RuntimeError(
            "Gemini document embedding failed: "
            "429 RESOURCE_EXHAUSTED"
        ),
    ):

        embeddings = model.embed_documents(TEST_TEXTS)

    print(f"Provider after Gemini failure: {model.provider}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(
        f"Local model loaded: "
        f"{model.local_model is not None}"
    )

    assert model.provider == "local"
    assert model.local_model is not None
    assert len(embeddings) == len(TEST_TEXTS)
    assert len(embeddings[0]) == 384
    assert all(
        len(embedding) == 384
        for embedding in embeddings
    )

    print()
    print("Document embedding fallback test: PASS")


def test_query_embedding_uses_local_after_fallback():
    """
    Verify that once the provider switches to local,
    subsequent query embeddings use MiniLM directly.
    """

    model = EmbeddingModel(
        force_local=True
    )

    embedding = model.embed_query(
        "What is machine learning?"
    )

    print()
    print("=" * 80)
    print("LOCAL QUERY EMBEDDING TEST")
    print("=" * 80)

    print(f"Provider: {model.provider}")
    print(f"Embedding dimension: {len(embedding)}")

    assert model.provider == "local"
    assert len(embedding) == 384

    print("Local query embedding test: PASS")


if __name__ == "__main__":
    test_document_embedding_fallback()
    test_query_embedding_uses_local_after_fallback()

    print()
    print("=" * 80)
    print("ALL EMBEDDING FALLBACK TESTS PASSED")
    print("=" * 80)

