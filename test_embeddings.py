from src.document_loader import load_pdf
from src.text_splitter import clean_documents, split_documents
from src.embeddings import EmbeddingModel


def main():
    pdf_path = "/home/dheeraj/NIELIT/Project/AI-Assistant/data/documents/ML_Engineer_Learning_Playlist.pdf"

    # Load PDF
    documents = load_pdf(pdf_path)

    # Clean text
    cleaned_documents = clean_documents(documents)

    # Split into chunks
    chunks = split_documents(cleaned_documents)

    print(f"Pages: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    # Load embedding model
    print("\nLoading embedding model...")

    embedding_model = EmbeddingModel()

    # Generate embeddings
    texts = [chunk.page_content for chunk in chunks]

    print("Generating embeddings...")

    embeddings = embedding_model.embed_documents(texts)

    print("\nEmbedding generation successful.")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimensions: {len(embeddings[0])}")

    # Test query embedding
    query = "What is covered in machine learning?"

    query_embedding = embedding_model.embed_query(query)

    print(f"\nQuery: {query}")
    print(f"Query embedding dimensions: {len(query_embedding)}")


if __name__ == "__main__":
    main()
