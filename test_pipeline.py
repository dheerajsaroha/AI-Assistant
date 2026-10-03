from src.document_loader import load_pdf
from src.rag_pipeline import RAGPipeline


PDF_PATH = "data/documents/ML_Engineer_Learning_Playlist.pdf"

QUERIES = [
    "Which week covers regression models?",
    "What is PyTorch?",
    "What is MLflow?",
    "What is AWS SageMaker?",
]


def main():
    print("=" * 80)
    print("RAG PIPELINE — RETRIEVAL TEST")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load document
    # ---------------------------------------------------------
    print("\nLoading document...")

    documents = load_pdf(PDF_PATH)

    print(f"Loaded pages: {len(documents)}")

    # ---------------------------------------------------------
    # 2. Build RAG pipeline
    # ---------------------------------------------------------
    print("\nBuilding RAG pipeline...")

    pipeline = RAGPipeline()

    pipeline.build_from_documents(documents)

    print(f"Created chunks: {len(pipeline.chunks)}")
    print("HybridRetriever ready.")

    # ---------------------------------------------------------
    # 3. Test retrieval
    # ---------------------------------------------------------
    for query in QUERIES:
        print("\n" + "-" * 80)
        print(f"QUERY: {query}")
        print("-" * 80)

        result = pipeline.retrieve(
            query=query,
            k=5,
            candidate_k=10,
        )

        confidence = result["confidence"]
        sources = result["sources"]

        print(
            f"Confidence: {confidence['confident']}"
        )

        print(
            f"Reason: {confidence['reason']}"
        )

        print(
            f"Top semantic: "
            f"{confidence['top_semantic_score']:.4f}"
        )

        print(
            f"Top BM25: "
            f"{confidence['top_lexical_score']:.4f}"
        )

        print(
            f"Lexical match: "
            f"{confidence['lexical_match']}"
        )

        print("\nSources:")

        for source in sources:
            print(
                f"  Rank {source['rank']} | "
                f"Document: {source['document_name']} | "
                f"Page: {source['page']} | "
                f"Chunk: {source['chunk_id']} | "
                f"Semantic: {source['semantic_score']:.4f} | "
                f"BM25: {source['bm25_score']:.4f}"
            )


if __name__ == "__main__":
    main()