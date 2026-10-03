from src.document_loader import load_pdf


def main():
    pdf_path = "/home/dheeraj/NIELIT/Project/AI-Assistant/data/documents/ML_Engineer_Learning_Playlist.pdf"

    documents = load_pdf(pdf_path)

    print(f"\nLoaded {len(documents)} pages.\n")

    for document in documents[:3]:
        print("=" * 60)
        print("Metadata:")
        print(document.metadata)
        print("\nContent:")
        print(document.page_content[:1000])
        print()


if __name__ == "__main__":
    main()

