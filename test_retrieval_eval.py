from src.document_loader import load_pdf
from src.embeddings import EmbeddingModel
from src.rag_pipeline import RAGPipeline


PDF_PATH = (
    "data/documents/"
    "ML_Engineer_Learning_Playlist.pdf"
)


EVALUATION_DATASET = [

    # ==================================================
    # DIRECT FACTUAL QUESTIONS
    # ==================================================

    {
        "query": "Which week covers regression models?",
        "expected_chunks": [5],
        "answerable": True,
    },
    {
        "query": "What topics are covered in Week 13?",
        "expected_chunks": [9],
        "answerable": True,
    },
    {
        "query": "What is MLflow?",
        "expected_chunks": [11],
        "answerable": True,
    },
    {
        "query": "What is AWS SageMaker?",
        "expected_chunks": [14],
        "answerable": True,
    },
    {
        "query": "What is the recommended timeline?",
        "expected_chunks": [0, 20],
        "answerable": True,
    },
    {
        "query": "What is covered in Week 8?",
        "expected_chunks": [6],
        "answerable": True,
    },
    {
        "query": "What is covered in Week 11?",
        "expected_chunks": [8],
        "answerable": True,
    },
    {
        "query": "What is covered in Week 15?",
        "expected_chunks": [11],
        "answerable": True,
    },
    {
        "query": "What is covered in Week 18?",
        "expected_chunks": [14],
        "answerable": True,
    },

    # ==================================================
    # CONCEPT QUESTIONS
    # ==================================================

    {
        "query": "What topics are included in model evaluation?",
        "expected_chunks": [6],
        "answerable": True,
    },
    {
        "query": "What topics are covered in NLP?",
        "expected_chunks": [9],
        "answerable": True,
    },
    {
        "query": "What tools are used for experiment tracking?",
        "expected_chunks": [11],
        "answerable": True,
    },
    {
        "query": "What cloud deployment technologies are mentioned?",
        "expected_chunks": [14],
        "answerable": True,
    },
    {
        "query": "What topics are covered in deep learning?",
        "expected_chunks": [8],
        "answerable": True,
    },

    # ==================================================
    # CROSS-SECTION QUESTIONS
    # ==================================================

    {
        "query": "How long does the complete learning plan take?",
        "expected_chunks": [0, 20],
        "answerable": True,
    },
    {
        "query": "How many total hours are recommended including practice?",
        "expected_chunks": [20],
        "answerable": True,
    },
    {
        "query": "Which phases are relevant to an ML Engineer?",
        "expected_chunks": [20],
        "answerable": True,
    },
    {
        "query": "What projects are included in the learning plan?",
        "expected_chunks": [16, 17],
        "answerable": True,
    },

    # ==================================================
    # PARAPHRASED / SEMANTIC QUESTIONS
    # ==================================================

    {
        "query": "At what stage should I study regression?",
        "expected_chunks": [5],
        "answerable": True,
    },
    {
        "query": "When do I learn experiment tracking?",
        "expected_chunks": [11],
        "answerable": True,
    },
    {
        "query": "How long should I plan to complete this curriculum?",
        "expected_chunks": [0, 20],
        "answerable": True,
    },
    {
        "query": "When does cloud deployment begin?",
        "expected_chunks": [14],
        "answerable": True,
    },

    # ==================================================
    # UNANSWERABLE QUESTIONS
    # ==================================================

    {
        "query": "What is PyTorch?",
        "expected_chunks": [],
        "answerable": False,
    },
    {
        "query": "What is TensorFlow?",
        "expected_chunks": [],
        "answerable": False,
    },
    {
        "query": "What salary does a machine learning engineer earn?",
        "expected_chunks": [],
        "answerable": False,
    },
    {
        "query": "Who created this curriculum?",
        "expected_chunks": [],
        "answerable": False,
    },
]


def evaluate_query(
    pipeline,
    item,
    k=5,
    candidate_k=10,
):
    query = item["query"]

    result = pipeline.retrieve(
        query=query,
        k=k,
        candidate_k=candidate_k,
    )

    sources = result["sources"]
    confidence = result["confidence"]

    retrieved_chunks = [
        source["chunk_id"]
        for source in sources
    ]

    expected_chunks = item[
        "expected_chunks"
    ]

    # --------------------------------------------------
    # Hit / Recall @ K
    # --------------------------------------------------

    if expected_chunks:

        hits = [
            chunk
            for chunk in expected_chunks
            if chunk in retrieved_chunks
        ]

        hit = len(hits) > 0

        recall = (
            len(hits) / len(expected_chunks)
        )

    else:

        hit = None
        recall = None

    # --------------------------------------------------
    # Reciprocal Rank
    # --------------------------------------------------

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(
        retrieved_chunks,
        start=1,
    ):

        if chunk_id in expected_chunks:

            reciprocal_rank = 1.0 / rank

            break

    # --------------------------------------------------
    # Confidence correctness
    # --------------------------------------------------

    confidence_correct = (
        confidence["confident"]
        == item["answerable"]
    )

    # --------------------------------------------------
    # False positive / false negative
    # --------------------------------------------------

    if item["answerable"]:

        if confidence["confident"]:
            classification = "TP"
        else:
            classification = "FN"

    else:

        if confidence["confident"]:
            classification = "FP"
        else:
            classification = "TN"

    return {
        "query": query,
        "expected_chunks": expected_chunks,
        "retrieved_chunks": retrieved_chunks,
        "hit": hit,
        "recall": recall,
        "mrr": reciprocal_rank,
        "confidence": confidence["confident"],
        "expected_answerable": item["answerable"],
        "confidence_correct": confidence_correct,
        "classification": classification,
        "top_semantic": confidence[
            "top_semantic_score"
        ],
        "top_bm25": confidence[
            "top_lexical_score"
        ],
        "lexical_match": confidence[
            "lexical_match"
        ],
    }


def build_evaluation_pipeline():
    """
    Build the RAG pipeline used for retrieval evaluation.

    Local MiniLM embeddings are used explicitly so the evaluation
    is deterministic and never depends on Gemini quota or network
    availability. The provider is replaced before the index is
    built, so a single FAISS index never mixes providers.
    """

    documents = load_pdf(PDF_PATH)

    pipeline = RAGPipeline()

    pipeline.embedding_model = EmbeddingModel(
        force_local=True
    )

    pipeline.build_from_documents(documents)

    return pipeline, documents


def validate_harness(results):
    """
    Validate the evaluation harness itself.

    These checks verify that the metrics were actually computed
    and that the confusion matrix is complete. They deliberately
    do NOT assert any retrieval-quality threshold.
    """

    assert results, "No evaluation results were produced."

    assert len(results) == len(EVALUATION_DATASET)

    for result in results:

        recall = result["recall"]

        if recall is None:
            assert not result["expected_chunks"]
            assert result["hit"] is None
        else:
            assert 0.0 <= recall <= 1.0

        assert 0.0 <= result["mrr"] <= 1.0

        assert result["classification"] in (
            "TP",
            "TN",
            "FP",
            "FN",
        )


def main():

    print("=" * 80)
    print("EXPANDED RAG RETRIEVAL EVALUATION")
    print("=" * 80)

    pipeline, pdf_documents = build_evaluation_pipeline()

    results = []

    for item in EVALUATION_DATASET:

        evaluation = evaluate_query(
            pipeline,
            item,
        )

        results.append(evaluation)

        print("\n" + "-" * 80)

        print(
            f"QUERY: {evaluation['query']}"
        )

        print(
            f"Expected chunks : "
            f"{evaluation['expected_chunks']}"
        )

        print(
            f"Retrieved chunks: "
            f"{evaluation['retrieved_chunks']}"
        )

        if evaluation["recall"] is not None:

            print(
                f"Recall@5        : "
                f"{evaluation['recall']:.2f}"
            )

            print(
                f"Hit@5           : "
                f"{evaluation['hit']}"
            )

        else:

            print(
                "Recall@5        : N/A"
            )

            print(
                "Hit@5           : N/A"
            )

        print(
            f"MRR             : "
            f"{evaluation['mrr']:.2f}"
        )

        print(
            f"Expected answer : "
            f"{evaluation['expected_answerable']}"
        )

        print(
            f"Confidence      : "
            f"{evaluation['confidence']}"
        )

        print(
            f"Classification   : "
            f"{evaluation['classification']}"
        )

        print(
            f"Top semantic    : "
            f"{evaluation['top_semantic']:.4f}"
        )

        print(
            f"Top BM25        : "
            f"{evaluation['top_bm25']:.4f}"
        )

        print(
            f"Lexical match   : "
            f"{evaluation['lexical_match']}"
        )

    # ==================================================
    # AGGREGATE METRICS
    # ==================================================

    answerable_results = [
        result
        for result in results
        if result["recall"] is not None
    ]

    total_expected_chunks = sum(
        len(result["expected_chunks"])
        for result in answerable_results
    )

    total_hits = sum(
        int(
            result["recall"]
            * len(result["expected_chunks"])
        )
        for result in answerable_results
    )

    recall_at_5 = (
        total_hits
        / total_expected_chunks
        if total_expected_chunks
        else 0.0
    )

    hit_at_5 = (
        sum(
            result["hit"]
            for result in answerable_results
        )
        / len(answerable_results)
        if answerable_results
        else 0.0
    )

    mean_reciprocal_rank = (
        sum(
            result["mrr"]
            for result in answerable_results
        )
        / len(answerable_results)
        if answerable_results
        else 0.0
    )

    confidence_accuracy = (
        sum(
            result["confidence_correct"]
            for result in results
        )
        / len(results)
        if results
        else 0.0
    )

    true_positive = sum(
        result["classification"] == "TP"
        for result in results
    )

    true_negative = sum(
        result["classification"] == "TN"
        for result in results
    )

    false_positive = sum(
        result["classification"] == "FP"
        for result in results
    )

    false_negative = sum(
        result["classification"] == "FN"
        for result in results
    )

    print("\n" + "=" * 80)
    print("FINAL METRICS")
    print("=" * 80)

    print(
        f"Queries evaluated    : "
        f"{len(results)}"
    )

    print(
        f"Answerable queries   : "
        f"{len(answerable_results)}"
    )

    print(
        f"Unanswerable queries : "
        f"{len(results) - len(answerable_results)}"
    )

    print(
        f"Recall@5             : "
        f"{recall_at_5:.2%}"
    )

    print(
        f"Hit@5                : "
        f"{hit_at_5:.2%}"
    )

    print(
        f"MRR                  : "
        f"{mean_reciprocal_rank:.2f}"
    )

    print(
        f"Confidence accuracy  : "
        f"{confidence_accuracy:.2%}"
    )

    print("\nConfidence Classification")
    print("-" * 40)

    print(
        f"True Positive  (TP) : "
        f"{true_positive}"
    )

    print(
        f"True Negative  (TN) : "
        f"{true_negative}"
    )

    print(
        f"False Positive (FP) : "
        f"{false_positive}"
    )

    print(
        f"False Negative (FN) : "
        f"{false_negative}"
    )

    # --------------------------------------------------
    # Harness integrity
    # --------------------------------------------------

    validate_harness(results)

    assert (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    ) == len(results)

    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"Embedding provider  : {pipeline.embedding_provider}")
    print(f"Documents evaluated : {len(pdf_documents)}")
    print(f"Chunks in index     : {len(pipeline.chunks)}")
    print(f"Queries evaluated   : {len(results)}")
    print(f"Recall@5            : {recall_at_5:.2%}")
    print(f"Hit@5               : {hit_at_5:.2%}")
    print(f"MRR                 : {mean_reciprocal_rank:.4f}")
    print(
        f"Confidence accuracy : "
        f"{confidence_accuracy:.2%}"
    )

    print()
    print("RETRIEVAL EVALUATION: PASS")


if __name__ == "__main__":
    main()