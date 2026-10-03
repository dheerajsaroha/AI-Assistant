from unittest.mock import patch

from langchain_core.documents import Document

from src.document_loader import load_pdf
from src.text_splitter import clean_documents, split_documents
from src.embeddings import EmbeddingModel
from src.vector_stores import FAISSVectorStore
from src.rag_chain import (
    build_context,
    generate_rag_answer,
    retrieve_context,
    run_rag,
)


# ================================================================
# TEST FIXTURES
# ================================================================

# Regression fixture only. The application must never depend on
# this specific document.
PDF_PATH = "data/documents/ML_Engineer_Learning_Playlist.pdf"

QUERY = "Which week covers regression models?"

EXPECTED_TOPIC = "Regression Models"

MOCK_ANSWER = "Test RAG answer."

TOP_K = 3


def build_index():
    """
    Build the rag_chain.py retrieval path.

    Local MiniLM embeddings are used explicitly so this test
    requires neither Gemini quota nor network access.
    """

    documents = load_pdf(PDF_PATH)

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

    dimension = len(embeddings[0])

    vector_store = FAISSVectorStore(
        dimension=dimension
    )

    vector_store.add_documents(
        chunks,
        embeddings,
    )

    return documents, chunks, embedding_model, vector_store


# ================================================================
# DOCUMENT PIPELINE TESTS
# ================================================================

def test_document_pipeline_produces_chunks(
    documents,
    chunks,
    vector_store,
):
    """Load, clean and split must produce retrievable chunks."""

    print()
    print("=" * 70)
    print("DOCUMENT PIPELINE")
    print("=" * 70)

    assert len(documents) == 7
    assert len(chunks) >= len(documents)

    for document in documents:
        assert document.metadata["document_name"]
        assert document.metadata["file_type"] == "pdf"
        assert isinstance(
            document.metadata["page"], int
        )

    assert [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ] == list(range(len(chunks)))

    assert vector_store.index.ntotal == len(chunks)

    print(f"Pages: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"FAISS vectors: {vector_store.index.ntotal}")

    print("Document pipeline: PASS")


# ================================================================
# RETRIEVAL CONTRACT TESTS
# ================================================================

def test_retrieval_returns_four_tuples(
    embedding_model,
    vector_store,
):
    """
    Retrieval must return the 4-tuple contract:

        (document, final_score, semantic_score, keyword_score)
    """

    results = retrieve_context(
        query=QUERY,
        embedding_model=embedding_model,
        vector_store=vector_store,
        k=TOP_K,
    )

    print()
    print("=" * 70)
    print("RETRIEVAL CONTRACT")
    print("=" * 70)

    assert len(results) == TOP_K

    for document, final_score, semantic_score, keyword_score in (
        results
    ):

        assert isinstance(document, Document)

        assert isinstance(final_score, float)
        assert isinstance(semantic_score, float)
        assert isinstance(keyword_score, float)

        assert 0.0 <= final_score <= 1.0
        assert keyword_score >= 0.0

    top_document = results[0][0]

    print(f"Query: {QUERY}")
    print(f"Results: {len(results)} (arity 4)")

    for rank, (
        document,
        final_score,
        semantic_score,
        keyword_score,
    ) in enumerate(results, start=1):

        print(
            f"  {rank}. final={final_score:.4f} "
            f"semantic={semantic_score:.4f} "
            f"keyword={keyword_score:.4f} "
            f"page={document.metadata.get('page')}"
        )

    print()
    print(f"Top chunk: {top_document.page_content[:70]}...")

    assert EXPECTED_TOPIC in top_document.page_content

    print(f"Top result contains '{EXPECTED_TOPIC}'.")

    print("Retrieval 4-tuple contract: PASS")


def test_build_context_contains_retrieved_content(
    embedding_model,
    vector_store,
):
    """
    build_context() must consume the 4-tuple results without
    raising an unpacking error, and the context must contain
    the retrieved document content.
    """

    results = retrieve_context(
        query=QUERY,
        embedding_model=embedding_model,
        vector_store=vector_store,
        k=TOP_K,
    )

    context = build_context(results)

    print()
    print("=" * 70)
    print("CONTEXT BUILDER")
    print("=" * 70)

    assert isinstance(context, str)
    assert context.strip()

    # Previously raised here:
    # ValueError: too many values to unpack (expected 2)

    for document, _, _, _ in results:
        assert document.page_content in context

    assert "[Source 1 |" in context
    assert "Chunk" in context

    print(f"Results consumed: {len(results)}")
    print(f"Context length: {len(context)}")
    print(f"First header: {context.splitlines()[0]}")

    print("build_context 4-tuple consumption: PASS")


# ================================================================
# GENERATION TESTS
# ================================================================

def test_generation_returns_answer_and_provider(
    embedding_model,
    vector_store,
):
    """
    generate_rag_answer() must return (answer, provider).

    The tuple contract must remain intact.
    """

    results = retrieve_context(
        query=QUERY,
        embedding_model=embedding_model,
        vector_store=vector_store,
        k=TOP_K,
    )

    context = build_context(results)

    with patch(
        "src.rag_chain.generate_response",
        return_value=MOCK_ANSWER,
    ) as mocked_generate:
        returned = generate_rag_answer(
            query=QUERY,
            context=context,
        )

    print()
    print("=" * 70)
    print("GENERATION CONTRACT")
    print("=" * 70)

    assert isinstance(returned, tuple)
    assert len(returned) == 2

    answer, provider = returned

    assert isinstance(answer, str)
    assert isinstance(provider, str)
    assert answer == MOCK_ANSWER
    assert provider == "Gemini 2.5 Flash"

    prompt = mocked_generate.call_args[0][0]

    assert QUERY in prompt
    assert EXPECTED_TOPIC in prompt
    assert "[Source 1 |" in prompt

    print(f"Answer: {answer}")
    print(f"Provider: {provider}")
    print("Query and context both present in prompt.")

    print("(answer, provider) generation contract: PASS")


def test_run_rag_consumes_four_tuples(
    embedding_model,
    vector_store,
):
    """
    run_rag() must complete without tuple-unpacking errors and
    expose every retrieval score component.
    """

    with patch(
        "src.rag_chain.generate_response",
        return_value=MOCK_ANSWER,
    ):
        result = run_rag(
            query=QUERY,
            embedding_model=embedding_model,
            vector_store=vector_store,
            k=TOP_K,
        )

    print()
    print("=" * 70)
    print("RUN_RAG")
    print("=" * 70)

    sources = result["sources"]

    assert len(sources) == TOP_K

    for source in sources:

        assert "score" in source
        assert "semantic_score" in source
        assert "keyword_score" in source

        assert isinstance(source["score"], float)
        assert isinstance(
            source["semantic_score"], float
        )
        assert isinstance(
            source["keyword_score"], float
        )
        assert source["content"].strip()

    timing = result["timing"]

    for key in (
        "retrieval",
        "context",
        "generation",
        "total",
    ):
        assert key in timing
        assert isinstance(timing[key], float)
        assert timing[key] >= 0.0

    assert timing["total"] >= timing["retrieval"]

    # run_rag() stores the generate_rag_answer() tuple directly.
    # This records the existing behaviour unchanged.
    assert result["answer"] == (
        MOCK_ANSWER,
        "Gemini 2.5 Flash",
    )

    print(f"Sources: {len(sources)}")

    for source in sources:
        print(
            f"  score={source['score']:.4f} "
            f"semantic={source['semantic_score']:.4f} "
            f"keyword={source['keyword_score']:.4f}"
        )

    print(f"Timing total: {timing['total']:.4f}s")

    print("run_rag 4-tuple consumption: PASS")


# ================================================================
# INPUT VALIDATION TESTS
# ================================================================

def test_invalid_input_is_rejected(
    embedding_model,
    vector_store,
):
    """Existing input validation must remain enforced."""

    print()
    print("=" * 70)
    print("INPUT VALIDATION")
    print("=" * 70)

    try:
        retrieve_context(
            query="",
            embedding_model=embedding_model,
            vector_store=vector_store,
            k=TOP_K,
        )
    except ValueError as error:
        assert "empty" in str(error).lower()
        print(f"Empty query rejected: {error}")
    else:
        raise AssertionError(
            "An empty query must raise ValueError."
        )

    try:
        vector_store.hybrid_search(
            query="   ",
            query_embedding=embedding_model.embed_query(
                QUERY
            ),
            k=TOP_K,
        )
    except ValueError as error:
        assert "Query cannot be empty" in str(error)
        print(f"Blank query rejected: {error}")
    else:
        raise AssertionError(
            "A blank query must raise ValueError."
        )

    assert build_context([]) == ""

    print("Empty results produce an empty context.")

    print("Input validation: PASS")


# ================================================================
# TEST RUNNER
# ================================================================

def main():
    """Run the deterministic RAG chain regression test."""

    print("=" * 70)
    print("RAG CHAIN REGRESSION TEST")
    print("=" * 70)

    documents, chunks, embedding_model, vector_store = (
        build_index()
    )

    test_document_pipeline_produces_chunks(
        documents,
        chunks,
        vector_store,
    )

    test_retrieval_returns_four_tuples(
        embedding_model,
        vector_store,
    )

    test_build_context_contains_retrieved_content(
        embedding_model,
        vector_store,
    )

    test_generation_returns_answer_and_provider(
        embedding_model,
        vector_store,
    )

    test_run_rag_consumes_four_tuples(
        embedding_model,
        vector_store,
    )

    test_invalid_input_is_rejected(
        embedding_model,
        vector_store,
    )

    print()
    print("=" * 70)
    print("ALL RAG CHAIN REGRESSION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()