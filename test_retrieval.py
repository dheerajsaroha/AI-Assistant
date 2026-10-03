from src.document_loader import load_pdf
from src.text_splitter import clean_documents, split_documents
from src.embeddings import EmbeddingModel
from src.vector_stores import FAISSVectorStore


def main():
    pdf_path = "data/documents/ML_Engineer_Learning_Playlist.pdf"

    # -------------------------
    # 1. Load documents
    # -------------------------
    documents = load_pdf(pdf_path)

    # -------------------------
    # 2. Clean documents
    # -------------------------
    cleaned_documents = clean_documents(documents)

    # -------------------------
    # 3. Split documents
    # -------------------------
    chunks = split_documents(cleaned_documents)

    print(f"Pages: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    # -------------------------
    # 4. Load embedding model
    # -------------------------
    print("\nLoading embedding model...")

    embedding_model = EmbeddingModel()

    # -------------------------
    # 5. Generate chunk embeddings
    # -------------------------
    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedding_model.embed_documents(texts)

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    # -------------------------
    # 6. Create FAISS index
    # -------------------------
    dimension = len(embeddings[0])

    vector_store = FAISSVectorStore(
        dimension=dimension
    )

    # -------------------------
    # 7. Add documents
    # -------------------------
    vector_store.add_documents(
        chunks,
        embeddings,
    )

    print(
        f"FAISS index contains "
        f"{vector_store.index.ntotal} vectors."
    )

    # -------------------------
    # 8. Search
    # -------------------------
    query = "What topics are covered in machine learning?"

    print(f"\nQuery: {query}")

    query_embedding = embedding_model.embed_query(
        query
    )

    results = vector_store.similarity_search(
        query_embedding,
        k=3,
    )

    # -------------------------
    # 9. Display results
    # -------------------------
    print("\nTop retrieved chunks:")
    print("=" * 70)

    for rank, (document, score) in enumerate(
        results,
        start=1,
    ):
        print(f"\nResult {rank}")
        print("-" * 70)

        print(
            f"Similarity score: {score:.4f}"
        )

        print(
            f"Page: "
            f"{document.metadata.get('page')}"
        )

        print(
            f"Chunk ID: "
            f"{document.metadata.get('chunk_id')}"
        )

        print("\nContent:")
        print(document.page_content)


if __name__ == "__main__":
    main()

