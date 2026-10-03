from pathlib import Path

from src.document_loader import load_pdf


PDF_PATH = Path(
    "data/documents/ML_Engineer_Learning_Playlist.pdf"
)


class TestUploadedFile:
    """Minimal Streamlit UploadedFile-like object for testing."""

    def __init__(self, path: Path):
        self.name = path.name
        self._bytes = path.read_bytes()

    def getvalue(self):
        return self._bytes


def main():
    uploaded_file = TestUploadedFile(PDF_PATH)

    documents = load_pdf(uploaded_file)

    print(f"Uploaded file: {uploaded_file.name}")
    print(f"Pages loaded: {len(documents)}")

    assert len(documents) == 7

    for document in documents:
        assert document.metadata["document_name"] == uploaded_file.name
        assert document.metadata["source"] == uploaded_file.name

    print("Uploaded-file ingestion test: PASS")


if __name__ == "__main__":
    main()
