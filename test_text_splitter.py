from src.document_loader import load_pdf
from src.text_splitter import split_documents


def main():
    pdf_path = "/home/dheeraj/NIELIT/Project/AI-Assistant/data/documents/ML_Engineer_Learning_Playlist.pdf"

    documents = load_pdf(pdf_path)

    chunks = split_documents(documents)

    print(f"Original documents/pages: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    print("\n" + "=" * 70)

    for index, chunk in enumerate(chunks[:5]):
        print(f"\nChunk {index + 1}")
        print("-" * 70)

        print("Metadata:")
        print(chunk.metadata)

        print("\nContent:")
        print(chunk.page_content)

        print(f"\nCharacter count: {len(chunk.page_content)}")


if __name__ == "__main__":
    main()
