import os
from threading import Lock

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY


GENERATION_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)

EMBEDDING_MODEL = "gemini-embedding-2"

_client = None
_client_lock = Lock()


def get_client():
    """
    Return the Gemini client, creating it on first use.

    The client is created lazily so that importing this module
    never requires Gemini credentials. This allows local-only
    providers and fallbacks to work without a GEMINI_API_KEY.

    No fallback decision is made here. Missing credentials are
    reported to the caller, which owns fallback behaviour.
    """

    global _client

    if _client is not None:
        return _client

    with _client_lock:

        if _client is not None:
            return _client

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "Gemini API key is not configured. "
                "Set GEMINI_API_KEY to use Gemini generation "
                "or embedding."
            )

        _client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    return _client


def generate_response(prompt: str) -> str:
    """
    Generate a response using Gemini.
    """

    if not prompt.strip():
        raise ValueError(
            "Prompt cannot be empty."
        )

    try:
        response = get_client().models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=512,
                temperature=0.2,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0
                ),
            ),
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return response.text

    except Exception as e:
        error_message = str(e)

        if (
            "429" in error_message
            or "RESOURCE_EXHAUSTED" in error_message
        ):
            raise RuntimeError(
                "Gemini generation quota has been exhausted. "
                "Please wait for the quota to reset."
            ) from e

        raise RuntimeError(
            f"Gemini generation failed: {error_message}"
        ) from e


def embed_text(text: str) -> list[float]:
    """
    Generate an embedding for a single query.
    """

    if not text.strip():
        raise ValueError("Text cannot be empty.")

    try:
        response = get_client().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )

        return response.embeddings[0].values

    except Exception as e:
        raise RuntimeError(
            f"Gemini query embedding failed: {e}"
        ) from e


def embed_texts(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate document embeddings.

    Generates one embedding per document chunk.
    """

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError(
            "Texts cannot contain empty strings."
        )

    embeddings = []

    try:
        for text in texts:
            response = get_client().models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768,
                ),
            )

            embeddings.append(
                response.embeddings[0].values
            )

        return embeddings

    except Exception as e:
        raise RuntimeError(
            f"Gemini document embedding failed: {e}"
        ) from e