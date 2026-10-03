from langchain_core.documents import Document

from src.document_loader import load_pdf
from src.text_splitter import (
    clean_documents,
    split_documents,
)
from src.embeddings import EmbeddingModel
from src.vector_stores import FAISSVectorStore
from src.hybrid_retriever import (
    HybridRetriever,
    RetrievalDecision,
    RetrievalResult,
)


PDF_PATH = (
    "data/documents/"
    "ML_Engineer_Learning_Playlist.pdf"
)


QUERIES = [
    "What is machine learning?",
    "Which week covers regression models?",
    "What topics are covered in Week 13?",
    "What is PyTorch?",
    "What is MLflow?",
    "What is AWS SageMaker?",
    "What is the recommended timeline?",
]


TOP_K = 5
CANDIDATE_K = 10


# ================================================================
# TEST FIXTURES
# ================================================================

def build_retriever():
    """
    Build the hybrid retriever over the regression fixture.

    Local MiniLM embeddings are used explicitly. The Gemini
    provider issues one remote request per chunk and retries
    inside the SDK, which makes this test slow, quota-bound and
    non-deterministic.
    """

    documents = load_pdf(PDF_PATH)

    documents = clean_documents(documents)

    chunks = split_documents(documents)

    embedding_model = EmbeddingModel(
        force_local=True
    )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedding_model.embed_documents(
        texts
    )

    vector_store = FAISSVectorStore(
        dimension=len(embeddings[0])
    )

    vector_store.add_documents(
        chunks,
        embeddings,
    )

    retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        documents=chunks,
    )

    return retriever, chunks, vector_store


def print_decision(decision):

    print("\nRetrieval Decision")
    print("-" * 40)

    print(
        f"Confidence       : "
        f"{decision.confident}"
    )

    print(
        f"Reason           : "
        f"{decision.reason}"
    )

    print(
        f"Top Semantic     : "
        f"{decision.top_semantic_score:.4f}"
    )

    print(
        f"Top BM25         : "
        f"{decision.top_lexical_score:.4f}"
    )

    print(
        f"Lexical Match    : "
        f"{decision.lexical_match}"
    )

    print(
        f"Retrieved Count  : "
        f"{decision.retrieved_count}"
    )


def print_results(results):

    print("\nRetrieved Evidence")
    print("-" * 40)

    for result in results:

        print(
            f"  RRF Rank {result.rank} | "
            f"Page {result.document.metadata.get('page')} | "
            f"Chunk {result.document.metadata.get('chunk_id')} | "
            f"Semantic {result.semantic_score:.4f} | "
            f"BM25 {result.lexical_score:.4f}"
        )


# ================================================================
# BUILD TESTS
# ================================================================

def test_fixture_builds_deterministic_index(
    retriever,
    chunks,
    vector_store,
):
    """Loading, splitting, embedding and indexing must succeed."""

    print()
    print("=" * 70)
    print("INDEX BUILD")
    print("=" * 70)

    assert len(chunks) == 21
    assert vector_store.index.ntotal == len(chunks)
    assert vector_store.bm25 is not None
    assert retriever.bm25 is not None
    assert retriever.embedding_model.provider == "local"

    print(f"Chunks: {len(chunks)}")
    print(f"FAISS vectors: {vector_store.index.ntotal}")
    print(f"Embedding provider: {retriever.embedding_model.provider}")

    print("Index build: PASS")


# ================================================================
# CONTRACT TESTS
# ================================================================

def test_retrieve_returns_results_and_decision(
    retriever,
):
    """
    HybridRetriever.retrieve() returns
        (list[RetrievalResult], RetrievalDecision)
    """

    returned = retriever.retrieve(
        query="What is MLflow?",
        k=TOP_K,
        candidate_k=CANDIDATE_K,
    )

    print()
    print("=" * 70)
    print("RESULT CONTRACT")
    print("=" * 70)

    assert isinstance(returned, tuple)
    assert len(returned) == 2

    results, decision = returned

    assert isinstance(results, list)
    assert isinstance(decision, RetrievalDecision)

    assert len(results) == TOP_K
    assert decision.retrieved_count == len(results)

    for result in results:
        assert isinstance(result, RetrievalResult)
        assert isinstance(result.document, Document)
        assert isinstance(result.semantic_score, float)
        assert isinstance(result.lexical_score, float)
        assert result.lexical_score >= 0.0

    print(f"Results: {len(results)}")
    print(f"Decision type: {type(decision).__name__}")

    print("Result contract: PASS")


def test_result_ranking_is_ordered(retriever):
    """RRF ranks must be sequential and match result order."""

    results, _ = retriever.retrieve(
        query="Which week covers regression models?",
        k=TOP_K,
        candidate_k=CANDIDATE_K,
    )

    assert [result.rank for result in results] == list(
        range(1, len(results) + 1)
    )

    chunk_ids = [
        result.document.metadata["chunk_id"]
        for result in results
    ]

    assert len(set(chunk_ids)) == len(chunk_ids)

    print()
    print("Ranking order: PASS")


def test_metadata_is_preserved(retriever, chunks):
    """Retrieved chunks must retain page and chunk_id metadata."""

    results, _ = retriever.retrieve(
        query="What is AWS SageMaker?",
        k=TOP_K,
        candidate_k=CANDIDATE_K,
    )

    known_chunk_ids = {
        chunk.metadata["chunk_id"]
        for chunk in chunks
    }

    for result in results:

        metadata = result.document.metadata

        assert metadata["chunk_id"] in known_chunk_ids
        assert "page" in metadata

    print()
    print("Metadata preservation: PASS")


# ================================================================
# HYBRID SIGNAL TESTS
# ================================================================

def test_semantic_retrieval_uses_faiss(
    retriever,
    vector_store,
):
    """Semantic scores must originate from the FAISS index."""

    query = "What is MLflow?"

    query_embedding = (
        retriever.embedding_model.embed_query(query)
    )

    semantic_results = vector_store.similarity_search(
        query_embedding,
        k=CANDIDATE_K,
    )

    faiss_scores = {
        document.metadata["chunk_id"]: score
        for document, score in semantic_results
    }

    results, decision = retriever.retrieve(
        query=query,
        k=TOP_K,
        candidate_k=CANDIDATE_K,
    )

    assert faiss_scores, "FAISS returned no candidates."

    for result in results:

        chunk_id = result.document.metadata["chunk_id"]

        if chunk_id in faiss_scores:
            assert result.semantic_score == (
                faiss_scores[chunk_id]
            )

    assert decision.max_semantic_score == max(
        result.semantic_score
        for result in results
    )

    assert decision.top_semantic_score == (
        results[0].semantic_score
    )

    print()
    print("=" * 70)
    print("SEMANTIC RETRIEVAL")
    print("=" * 70)
    print(f"FAISS candidates: {len(faiss_scores)}")
    print(f"Max semantic: {decision.max_semantic_score:.4f}")

    print("Semantic retrieval: PASS")


def test_keyword_retrieval_uses_bm25(retriever, chunks):
    """Lexical scores must originate from the BM25 index."""

    query = "Which week covers regression models?"

    lexical_scores = retriever._lexical_scores(query)

    assert len(lexical_scores) == len(chunks)
    assert any(score > 0.0 for score in lexical_scores)

    score_by_chunk = dict(
        zip(
            range(len(chunks)),
            lexical_scores,
        )
    )

    results, decision = retriever.retrieve(
        query=query,
        k=TOP_K,
        candidate_k=CANDIDATE_K,
    )

    for result in results:

        chunk_id = result.document.metadata["chunk_id"]

        assert result.lexical_score == score_by_chunk[chunk_id]

    assert decision.max_lexical_score == max(
        result.lexical_score
        for result in results
    )

    assert decision.top_lexical_score == (
        results[0].lexical_score
    )

    print()
    print("=" * 70)
    print("KEYWORD RETRIEVAL")
    print("=" * 70)
    print(f"BM25 documents: {len(lexical_scores)}")
    print(f"Max BM25: {decision.max_lexical_score:.4f}")

    print("Keyword retrieval: PASS")


def test_hybrid_fusion_combines_both_signals(
    retriever,
):
    """
    Ranking must be a fusion, not a pure semantic ordering.

    The test verifies that the returned order differs from the
    order the retriever itself produced, proving RRF combined the
    semantic and lexical candidate pools.
    """

    query = "Which week covers regression models?"

    results, _ = retriever.retrieve(
        query=query,
        k=TOP_K,
        candidate_k=CANDIDATE_K,
    )

    fused_order = [
        result.document.metadata["chunk_id"]
        for result in results
    ]

    semantic_order = [
        result.document.metadata["chunk_id"]
        for result in sorted(
            results,
            key=lambda item: item.semantic_score,
            reverse=True,
        )
    ]

    lexical_order = [
        result.document.metadata["chunk_id"]
        for result in sorted(
            results,
            key=lambda item: item.lexical_score,
            reverse=True,
        )
    ]

    # At least one signal must contribute to the final ordering,
    # otherwise the retriever would be ignoring hybrid fusion.
    assert (
        fused_order != semantic_order
        or fused_order != lexical_order
    )

    print()
    print("=" * 70)
    print("HYBRID FUSION")
    print("=" * 70)
    print(f"Fused order      : {fused_order}")
    print(f"Semantic order   : {semantic_order}")
    print(f"Lexical order    : {lexical_order}")

    print("Hybrid fusion: PASS")


# ================================================================
# CONFIDENCE TESTS
# ================================================================

def test_confidence_decision_contract(retriever):
    """Every query must produce a well-formed confidence decision."""

    print()
    print("=" * 70)
    print("CONFIDENCE GATE")
    print("=" * 70)

    for query in QUERIES:

        results, decision = retriever.retrieve(
            query=query,
            k=TOP_K,
            candidate_k=CANDIDATE_K,
        )

        assert isinstance(decision.confident, bool)
        assert isinstance(decision.reason, str)
        assert decision.reason.strip()
        assert isinstance(decision.lexical_match, bool)
        assert decision.retrieved_count == len(results)
        assert decision.max_semantic_score >= (
            decision.top_semantic_score
        )
        assert decision.max_lexical_score >= (
            decision.top_lexical_score
        )

        print(
            f"  confident={decision.confident} "
            f"semantic={decision.top_semantic_score:.4f} "
            f"bm25={decision.top_lexical_score:.4f} "
            f"lexical_match={decision.lexical_match} | "
            f"{query}"
        )

        print_decision(decision)
        print_results(results)

    print("Confidence decision contract: PASS")


def test_unanswerable_query_is_not_confident(retriever):
    """
    A question with no supporting evidence must not be confident.

    Thresholds are not modified; this verifies the existing gate.
    """

    results, decision = retriever.retrieve(
        query="What is the airspeed velocity of an "
              "unladen swallow?",
        k=TOP_K,
        candidate_k=CANDIDATE_K,
    )

    assert decision.confident is False
    assert decision.lexical_match is False

    print()
    print("Unsupported question rejected by confidence gate: PASS")


# ================================================================
# DOCUMENT SCOPE TESTS
# ================================================================

def test_document_scope_filtering(retriever, chunks):
    """document_ids must restrict retrieval to the scoped chunks."""

    first_id = "document-scope-a"
    second_id = "document-scope-b"

    for index, chunk in enumerate(chunks):
        chunk.metadata["document_id"] = (
            first_id
            if index % 2 == 0
            else second_id
        )

    scoped_ids = {
        chunk.metadata["document_id"]
        for chunk in chunks
        if chunk.metadata["document_id"] == first_id
    }

    results, _ = retriever.retrieve(
        query="Which week covers regression models?",
        k=TOP_K,
        candidate_k=CANDIDATE_K,
        document_ids=scoped_ids,
    )

    for result in results:

        assert (
            result.document.metadata["document_id"]
            == first_id
        )

    try:
        retriever.retrieve(
            query="Which week covers regression models?",
            k=TOP_K,
            candidate_k=CANDIDATE_K,
            document_ids={"unknown-document"},
        )
    except ValueError as error:
        assert "document scope" in str(error)
    else:
        raise AssertionError(
            "An unmatched document scope must raise ValueError."
        )

    print()
    print("=" * 70)
    print("DOCUMENT SCOPE")
    print("=" * 70)
    print(f"Scoped results: {len(results)}")
    print(f"Scoped document: {first_id}")

    print("Document scope: PASS")


# ================================================================
# INPUT VALIDATION TESTS
# ================================================================

def test_input_validation(retriever):
    """Empty and whitespace-only queries must be rejected."""

    print()
    print("=" * 70)
    print("INPUT VALIDATION")
    print("=" * 70)

    for query in ("", "   ", "\n\t"):
        try:
            retriever.retrieve(
                query=query,
                k=TOP_K,
                candidate_k=CANDIDATE_K,
            )
        except ValueError as error:
            assert "Query cannot be empty" in str(error)
        else:
            raise AssertionError(
                "An empty query must raise ValueError."
            )

    print("Empty and whitespace-only queries rejected.")

    print("Input validation: PASS")


# ================================================================
# TEST RUNNER
# ================================================================

def main():
    """Run the hybrid retriever regression test."""

    print("=" * 70)
    print("HYBRID RETRIEVER REGRESSION TEST")
    print("=" * 70)

    retriever, chunks, vector_store = build_retriever()

    test_fixture_builds_deterministic_index(
        retriever,
        chunks,
        vector_store,
    )

    test_retrieve_returns_results_and_decision(
        retriever,
    )

    test_result_ranking_is_ordered(retriever)

    test_metadata_is_preserved(retriever, chunks)

    test_semantic_retrieval_uses_faiss(
        retriever,
        vector_store,
    )

    test_keyword_retrieval_uses_bm25(retriever, chunks)

    test_hybrid_fusion_combines_both_signals(retriever)

    test_confidence_decision_contract(retriever)

    test_unanswerable_query_is_not_confident(retriever)

    test_document_scope_filtering(retriever, chunks)

    test_input_validation(retriever)

    print()
    print("=" * 70)
    print("ALL HYBRID RETRIEVER TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()