import time

from src.embeddings import EmbeddingModel
from src.gemini_client import generate_response
from src.vector_stores import FAISSVectorStore
from src.local_llm import generate_local_response

def generate_with_fallback(prompt: str) -> tuple[str, str]:
    try:
        answer = generate_response(prompt)
        return answer, "Gemini 2.5 Flash"

    except RuntimeError as error:
        error_message = str(error).upper()

        fallback_conditions = (
            "429" in error_message
            or "RESOURCE_EXHAUSTED" in error_message
            or "QUOTA" in error_message
            or "503" in error_message
            or "UNAVAILABLE" in error_message
            or "SERVICE UNAVAILABLE" in error_message
            # Missing or invalid credentials should not hard-fail
            # generation. They are an availability problem, not a
            # content problem, so the local model answers instead.
            or "API KEY" in error_message
            or "NOT CONFIGURED" in error_message
            or "API_KEY" in error_message
        )

        if not fallback_conditions:
            raise

        print(
            "Gemini generation unavailable "
            f"({error_message[:120]}). "
            "Switching to SmolLM2-360M-Instruct."
        )

        answer = generate_local_response(
            prompt=prompt,
            max_new_tokens=256,
        )

        return answer, "SmolLM2-360M-Instruct (Fallback)"
def retrieve_context(
    query: str,
    embedding_model: EmbeddingModel,
    vector_store: FAISSVectorStore,
    k: int = 5,
) -> list[tuple]:
    """
    Retrieve relevant document chunks using
    hybrid semantic + keyword retrieval.
    """

    query_embedding = (
        embedding_model.embed_query(query)
    )

    return vector_store.hybrid_search(
        query=query,
        query_embedding=query_embedding,
        k=k,
        candidate_k=10,
    )



def build_context(
    results: list[tuple],
) -> str:
    """
    Convert retrieved documents into a context string.

    `results` follows the `FAISSVectorStore.hybrid_search()`
    contract:
        (document, final_score, semantic_score, keyword_score)

    The ranking scores are deliberately kept out of the prompt
    because they are retrieval metadata rather than document
    facts. Including them would change generation behaviour.
    """

    context_parts = []

    for rank, (
        document,
        final_score,
        semantic_score,
        keyword_score,
    ) in enumerate(
        results,
        start=1,
    ):
        page = document.metadata.get("page")
        chunk_id = document.metadata.get("chunk_id")

        context_parts.append(
            f"[Source {rank} | Page {page} | Chunk {chunk_id}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)


def generate_rag_answer(
    query: str,
    context: str,
) -> tuple[str, str]:
    """
    Generate a grounded answer using retrieved document context.
    """

    prompt = f"""
You are a document-based AI assistant.

Your job is to answer the user's question using the provided
document context.

Follow these rules strictly:

1. Use the document context as the primary source of information.
2. Do not invent facts that are not supported by the context.
3. Do not use your general knowledge to answer factual questions
   about the document.
4. If the document directly answers the question, answer it clearly.
5. If the document does not directly answer the question but contains
   related information, explain that the exact answer is not stated
   and summarize the relevant information that IS present.
6. If the document contains no useful information related to the
   question, say that the information could not be found.
7. Do not pretend that related information is a direct answer.
8. Keep the answer concise.
9. When possible, mention the relevant week, phase, or topic.

DOCUMENT CONTEXT
================
{context}
================

USER QUESTION
================
{query}
================

ANSWER:
"""
    return generate_with_fallback(prompt)


def run_rag(
    query: str,
    embedding_model: EmbeddingModel,
    vector_store: FAISSVectorStore,
    k: int = 3,
) -> dict:
    """
    Run retrieval and generation while measuring
    the latency of each stage.
    """

    # -------------------------
    # Query embedding + retrieval
    # -------------------------

    retrieval_start = time.perf_counter()

    results = retrieve_context(
        query=query,
        embedding_model=embedding_model,
        vector_store=vector_store,
        k=k,
    )

    retrieval_time = (
        time.perf_counter() - retrieval_start
    )

    # -------------------------
    # Context construction
    # -------------------------

    context_start = time.perf_counter()

    context = build_context(results)

    context_time = (
        time.perf_counter() - context_start
    )

    # -------------------------
    # Gemini generation
    # -------------------------

    generation_start = time.perf_counter()

    answer = generate_rag_answer(
        query=query,
        context=context,
    )

    generation_time = (
        time.perf_counter() - generation_start
    )

    return {
        "answer": answer,
        "sources": [
            {
                "page": document.metadata.get("page"),
                "chunk_id": document.metadata.get(
                    "chunk_id"
                ),
                "score": final_score,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "content": document.page_content,
            }
            for (
                document,
                final_score,
                semantic_score,
                keyword_score,
            ) in results
        ],
        "timing": {
            "retrieval": retrieval_time,
            "context": context_time,
            "generation": generation_time,
            "total": (
                retrieval_time
                + context_time
                + generation_time
            ),
        },
    }