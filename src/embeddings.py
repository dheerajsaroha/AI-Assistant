from sentence_transformers import SentenceTransformer

from src.gemini_client import embed_text, embed_texts
from src.config import LLM_PROVIDER


GEMINI_DIMENSION = 768
LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_DIMENSION = 384


class EmbeddingModel:
    """
    Embedding provider with automatic Gemini -> local MiniLM fallback.

    Primary:
        Gemini Embedding 2

    Fallback:
        sentence-transformers/all-MiniLM-L6-v2

    Important:
        A single EmbeddingModel instance uses only one provider for its
        entire vector index. This prevents mixing embeddings from
        different vector spaces.
    """

    def __init__(self, force_local: bool = False):
        # Respect LLM_PROVIDER when set explicitly. This is the only
        # place the configuration actually takes effect for embeddings;
        # without it, "LLM_PROVIDER=local" would silently do nothing and
        # the model would still try Gemini first.
        if force_local or LLM_PROVIDER == "local":
            self.provider = "local"
        else:
            self.provider = "gemini"

        self.local_model = None

        if self.provider == "local":
            self._load_local_model()

    def _load_local_model(self):
        """Load the local MiniLM model only when fallback is required."""

        if self.local_model is None:
            print(
                "Loading local embedding model: "
                f"{LOCAL_MODEL_NAME}"
            )

            self.local_model = SentenceTransformer(
                LOCAL_MODEL_NAME
            )

            print(
                "Local embedding model loaded successfully."
            )

    def _switch_to_local(self):
        """Switch permanently to the local embedding provider."""

        if self.provider != "local":
            print(
                "Gemini embedding unavailable. "
                "Switching to all-MiniLM-L6-v2."
            )

        self.provider = "local"
        self._load_local_model()

    @staticmethod
    def _is_gemini_quota_error(error: Exception) -> bool:
        """
        Detect Gemini quota/rate-limit exhaustion.

        We specifically treat 429 / RESOURCE_EXHAUSTED as a reason
        to switch to the local embedding model.
        """

        message = str(error).upper()

        return (
            "429" in message
            or "RESOURCE_EXHAUSTED" in message
            or "QUOTA" in message
        )

    def _embed_documents_local(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate document embeddings locally."""

        self._load_local_model()

        embeddings = self.local_model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def _embed_query_local(
        self,
        text: str,
    ) -> list[float]:
        """Generate a query embedding locally."""

        self._load_local_model()

        embedding = self.local_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for document chunks.

        Gemini is attempted first.

        If Gemini embedding quota/API exhaustion occurs,
        the complete document batch is embedded using MiniLM.
        """

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError(
                "Texts cannot contain empty strings."
            )

        # Once fallback has happened, keep using the local model.
        if self.provider == "local":
            return self._embed_documents_local(texts)

        try:
            embeddings = embed_texts(texts)

            if len(embeddings) != len(texts):
                raise RuntimeError(
                    "Gemini returned an incorrect number of embeddings."
                )

            return embeddings

        except Exception as error:

            # Build-time fallback is broad on purpose.
            #
            # At this point no index exists yet, so falling back to
            # the local model is always safe: there is no pre-built
            # Gemini index that could become inconsistent with
            # MiniLM vectors.
            #
            # This is what makes the pipeline robust against a
            # missing or invalid API key, a network failure, or any
            # other non-quota Gemini error during indexing.
            #
            # The narrow quota-only check is deliberately kept for
            # embed_query(), where an existing Gemini index cannot
            # be queried with a different provider's vectors.
            if not self._is_gemini_quota_error(error):
                print(
                    "Gemini document embedding failed "
                    f"({str(error)[:160]}). "
                    "Falling back to all-MiniLM-L6-v2."
                )

            # Do NOT use partially generated Gemini embeddings.
            # Rebuild the entire batch using MiniLM.
            self._switch_to_local()

            return self._embed_documents_local(texts)

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate a query embedding using the same provider
        used to build the vector index.
        """

        if not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        if self.provider == "local":
            return self._embed_query_local(text)

        try:
            return embed_text(text)

        except Exception as error:

            if not self._is_gemini_quota_error(error):
                raise

            """
            IMPORTANT:

            We cannot silently switch an already-built Gemini
            FAISS index to MiniLM because the vector dimensions
            and vector spaces are different.

            Therefore, query fallback requires the pipeline to
            rebuild its index using MiniLM.
            """

            raise RuntimeError(
                "Gemini embedding quota was exhausted while "
                "embedding the query. The current FAISS index "
                "was built with Gemini embeddings, so it cannot "
                "be queried with MiniLM embeddings. Rebuild the "
                "pipeline so it can automatically fall back to "
                "all-MiniLM-L6-v2."
            ) from error