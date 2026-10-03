from unittest.mock import patch

from langchain_core.documents import Document

from src.embeddings import EmbeddingModel
from src.rag_chain import (
    build_context,
    generate_rag_answer,
    retrieve_context,
    run_rag,
)
from src.text_splitter import clean_documents, split_documents
from src.vector_stores import FAISSVectorStore


# ================================================================
# TEST FIXTURES
# ================================================================

CORPUS = [
    (
        "Python is a programming language widely used for "
        "data science and machine learning."
    ),
    (
        "Regression models predict continuous numerical "
        "values such as house prices."
    ),
    (
        "AWS SageMaker is a managed cloud service used to "
        "build and deploy machine learning models."
    ),
]

QUERY = (
    "Which managed cloud service is used to build and "
    "deploy machine learning models?"
)

CANNED_ANSWER = (
    "AWS SageMaker is the managed cloud service used to "
    "build and deploy machine learning models."
)


def build_vector_store():
    """
    Build a real hybrid index using the local MiniLM model.

    MiniLM is used so this test requires no Gemini access and
    no network calls.
    """

    documents = [
        Document(
            page_content=content,
            metadata={
                "source": "corpus.txt",
                "document_name": "corpus.txt",
                "file_type": "txt",
                "page": index,
            },
        )
        for index, content in enumerate(CORPUS)
    ]

    cleaned_documents = clean_documents(documents)

    chunks = split_documents(cleaned_documents)

    embedding_model = EmbeddingModel(
        force_local=True
    )

    embeddings = embedding_model.embed_documents(
        [
            chunk.page_content
            for chunk in chunks
        ]
    )

    vector_store = FAISSVectorStore(
        dimension=len(embeddings[0])
    )

    vector_store.add_documents(
        chunks,
        embeddings,
    )

    return embedding_model, vector_store


# ================================================================
# CONTRACT TESTS
# ================================================================

def test_hybrid_search_returns_four_tuples(
    embedding_model,
    vector_store,
):
    """
    The producer contract is a 4-tuple.

    hybrid_search() must return:
        (document, final_score, semantic_score, keyword_score)
    """

    results = retrieve_context(
        query=QUERY,
        embedding_model=embedding_model,
        vector_store=vector_store,
        k=3,
    )

    print()
    print("=" * 70)
    print("PRODUCER CONTRACT: hybrid_search()")
    print("=" * 70)

    assert results, "Retrieval returned no results."

    for result in results:

        assert len(result) == 4, (
            "hybrid_search() must return 4-tuples, "
            f"received arity {len(result)}."
        )

        document, final_score, semantic_score, keyword_score = (
            result
        )

        assert isinstance(document, Document)
        assert isinstance(final_score, float)
        assert isinstance(semantic_score, float)
        assert isinstance(keyword_score, float)

    print(f"Results returned: {len(results)}")
    print("Tuple arity: 4 (document, final, semantic, keyword)")
    print("All score components are floats.")

    print("hybrid_search 4-tuple contract: PASS")


def test_build_context_consumes_four_tuples(
    embedding_model,
    vector_store,
):
    """
    build_context() must consume the 4-tuple producer contract
    without raising an unpacking ValueError, and the context
    must still contain the retrieved document content.
    """

    results = retrieve_context(
        query=QUERY,
        embedding_model=embedding_model,
        vector_store=vector_store,
        k=3,
    )

    context = build_context(results)

    print()
    print("=" * 70)
    print("CONSUMER CONTRACT: build_context()")
    print("=" * 70)

    assert context.strip(), "Context must not be empty."

    # The previous defect raised here:
    # ValueError: too many values to unpack (expected 2)

    for document, _, _, _ in results:

        assert document.page_content in context, (
            "Retrieved content is missing from the context."
        )

    assert "[Source 1 |" in context

    print(f"Results consumed: {len(results)}")
    print(f"Context length: {len(context)} characters")
    print("Retrieved content present in context.")

    print("build_context 4-tuple consumption: PASS")


def test_run_rag_consumes_four_tuples(
    embedding_model,
    vector_store,
):
    """
    run_rag() must consume the 4-tuple producer contract without
    an arity failure and expose every score component.
    """

    with patch(
        "src.rag_chain.generate_response",
        return_value=CANNED_ANSWER,
    ):
        result = run_rag(
            query=QUERY,
            embedding_model=embedding_model,
            vector_store=vector_store,
            k=3,
        )

    print()
    print("=" * 70)
    print("CONSUMER CONTRACT: run_rag()")
    print("=" * 70)

    sources = result["sources"]

    assert sources, "run_rag() returned no sources."

    for source in sources:

        assert "score" in source
        assert "semantic_score" in source
        assert "keyword_score" in source
        assert source["content"].strip()
        assert isinstance(source["score"], float)
        assert isinstance(
            source["semantic_score"], float
        )
        assert isinstance(
            source["keyword_score"], float
        )

    timing = result["timing"]

    for key in (
        "retrieval",
        "context",
        "generation",
        "total",
    ):
        assert key in timing
        assert isinstance(timing[key], float)

    print(f"Sources returned: {len(sources)}")

    for source in sources:
        print(
            f"  score={source['score']:.4f} "
            f"semantic={source['semantic_score']:.4f} "
            f"keyword={source['keyword_score']:.4f}"
        )

    print("run_rag 4-tuple consumption: PASS")


def test_generation_returns_answer_and_provider(
    embedding_model,
    vector_store,
):
    """
    The generation contract must remain (answer, provider).

    It must not be flattened into a single string.
    """

    results = retrieve_context(
        query=QUERY,
        embedding_model=embedding_model,
        vector_store=vector_store,
        k=3,
    )

    context = build_context(results)

    with patch(
        "src.rag_chain.generate_response",
        return_value=CANNED_ANSWER,
    ) as mocked_generate:
        returned = generate_rag_answer(
            query=QUERY,
            context=context,
        )

    print()
    print("=" * 70)
    print("GENERATION CONTRACT: (answer, provider)")
    print("=" * 70)

    assert isinstance(returned, tuple)
    assert len(returned) == 2

    answer, provider = returned

    assert answer == CANNED_ANSWER
    assert provider == "Gemini 2.5 Flash"
    assert isinstance(answer, str)

    # The query and built context must both reach the prompt.
    prompt = mocked_generate.call_args[0][0]

    assert QUERY in prompt
    assert context.strip().splitlines()[1] in prompt

    print(f"Provider: {provider}")
    print(f"Answer: {answer}")
    print("Context and query both present in prompt.")

    print("(answer, provider) generation contract: PASS")


# ================================================================
# TEST RUNNER
# ================================================================

def main():
    """Run the RAG chain contract tests."""

    print("=" * 70)
    print("RAG CHAIN RESULT-CONTRACT TEST")
    print("=" * 70)

    embedding_model, vector_store = build_vector_store()

    test_hybrid_search_returns_four_tuples(
        embedding_model,
        vector_store,
    )

    test_build_context_consumes_four_tuples(
        embedding_model,
        vector_store,
    )

    test_run_rag_consumes_four_tuples(
        embedding_model,
        vector_store,
    )

    test_generation_returns_answer_and_provider(
        embedding_model,
        vector_store,
    )

    print()
    print("=" * 70)
    print("ALL RAG CHAIN CONTRACT TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()