from langchain_core.documents import Document

from src.rag_pipeline import RAGPipeline


def make_document(
    name: str,
    document_id: str,
    content: str,
):
    return Document(
        page_content=content,
        metadata={
            "document_name": name,
            "document_id": document_id,
            "page": 0,
        },
    )


def main():
    pipeline = RAGPipeline()

    pipeline.chunks = [
        make_document(
            "Candidate DMC.pdf",
            "dmc123",
            "Result cum Semester Grade Point Card. "
            "CGPA 7.08. Percentage 70.80.",
        ),
        make_document(
            "Dheeraj_Resume.pdf",
            "resume456",
            "Python, SQL, Machine Learning, "
            "FastAPI and LangChain projects.",
        ),
        make_document(
            "ML_Engineer_Learning_Playlist.pdf",
            "playlist789",
            "Week 7 covers Regression Models.",
        ),
    ]

    scope = pipeline._resolve_document_scope(
        "Give me a summary of Candidate DMC document"
    )

    print("Candidate DMC scope:", scope)

    assert scope == {"dmc123"}

    scope = pipeline._resolve_document_scope(
        "Summarize Dheeraj Resume"
    )

    print("Resume scope:", scope)

    assert scope == {"resume456"}

    scope = pipeline._resolve_document_scope(
        "What is the CGPA?"
    )

    print("Global scope:", scope)

    assert scope is None

    print("\nDOCUMENT SCOPE TEST: PASS")


if __name__ == "__main__":
    main()
