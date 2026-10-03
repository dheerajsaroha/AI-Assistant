from pathlib import Path
from typing import BinaryIO

from src.document_loader import load_pdf
from src.text_splitter import clean_documents, split_documents


SUPPORTED_EXTENSIONS = {".pdf"}


class DocumentManager:
    """Handles user-provided documents before indexing."""

    def validate_file(self, file: BinaryIO) -> Path:
        filename = getattr(file, "name", "")

        if not filename:
            raise ValueError("Uploaded file must have a filename.")

        extension = Path(filename).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {supported}"
            )

        return Path(filename)

    def process_file(
        self,
        file: BinaryIO,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        filename = self.validate_file(file)

        # Streamlit UploadedFile provides getvalue().
        file_bytes = file.getvalue()

        if not file_bytes:
            raise ValueError(f"{filename.name} is empty.")

        # Temporary in-memory file is not yet used because the
        # existing loader currently expects a filesystem path.
        raise NotImplementedError(
            "Filesystem-backed ingestion will be added in the next step."
        )
