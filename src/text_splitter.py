import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """
    Perform conservative cleanup of common PDF extraction artifacts.
    """

    # Fix words where PDF extraction inserted spaces after the first letter.
    text = re.sub(r"\bW\s+eek\b", "Week", text)
    text = re.sub(r"\bT\s+ext\b", "Text", text)
    text = re.sub(r"\bA\s+WS\b", "AWS", text)
    text = re.sub(r"\bF\s+ull-Stack\b", "Full-Stack", text)
    text = re.sub(r"\bTOT\s+AL\b", "TOTAL", text)
    text = re.sub(r"\bOV\s+ER\s+VIEW\b", "OVERVIEW", text)

    # Normalize excessive whitespace while preserving line structure.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Clean extracted document text while preserving metadata.
    """

    cleaned_documents = []

    for document in documents:
        cleaned_document = Document(
            page_content=clean_text(document.page_content),
            metadata=document.metadata.copy(),
        )

        cleaned_documents.append(cleaned_document)

    return cleaned_documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Split documents into smaller chunks for semantic retrieval.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks