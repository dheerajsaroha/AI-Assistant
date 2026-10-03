import hashlib
from pathlib import Path
import re

from langchain_core.documents import Document

from src.text_splitter import clean_documents, split_documents
from src.embeddings import EmbeddingModel
from src.vector_stores import FAISSVectorStore
from src.hybrid_retriever import HybridRetriever
from src.rag_chain import generate_rag_answer


class RAGPipeline:
    """
    End-to-end RAG processing pipeline.

    Document loading is handled outside this class.

    The pipeline supports:
        Gemini embeddings
            ↓
        automatic MiniLM fallback

    A single FAISS index always uses one embedding provider.
    """

    def __init__(self):
        self.embedding_model = EmbeddingModel()

        self.vector_store = None
        self.chunks = []
        self.retriever = None

        # Keeps the original documents so the index can be rebuilt
        # if the embedding provider changes.
        self.documents = []

    @property
    def embedding_provider(self) -> str:
        """Return the currently active embedding provider."""

        if self.embedding_model.provider == "gemini":
            return "Gemini Embedding 2"

        if self.embedding_model.provider == "local":
            return "all-MiniLM-L6-v2 (Fallback)"

        return self.embedding_model.provider

    def build_from_documents(self, documents):
        """
        Build the complete retrieval pipeline from documents.

        Steps:
        1. Assign document IDs
        2. Clean documents
        3. Split into chunks
        4. Generate embeddings
        5. Build FAISS vector store
        6. Build hybrid retriever
        """

        if not documents:
            raise ValueError("No documents were provided.")

        documents = self._assign_document_ids(documents)

        # Keep the processed source documents so the vector index
        # can be rebuilt if the embedding provider changes.
        self.documents = documents

        # ---------------------------------------------------------
        # 1. Clean documents
        # ---------------------------------------------------------
        cleaned_documents = clean_documents(documents)

        if not cleaned_documents:
            raise ValueError(
                "No usable documents after cleaning."
            )

        # ---------------------------------------------------------
        # 2. Split documents
        # ---------------------------------------------------------
        self.chunks = split_documents(
            cleaned_documents
        )

        if not self.chunks:
            raise ValueError(
                "No chunks were created from the documents."
            )

        # ---------------------------------------------------------
        # 3. Generate embeddings
        # ---------------------------------------------------------
        self._build_vector_index()

        return self

    def _build_vector_index(self):
        """
        Generate embeddings and build the FAISS index.

        The embedding provider is controlled by EmbeddingModel.
        """

        if not self.chunks:
            raise RuntimeError(
                "No chunks are available to build the vector index."
            )

        texts = [
            chunk.page_content
            for chunk in self.chunks
        ]

        embeddings = self.embedding_model.embed_documents(
            texts
        )

        if len(embeddings) != len(self.chunks):
            raise ValueError(
                "Number of embeddings does not match "
                "number of chunks."
            )

        if not embeddings:
            raise ValueError(
                "No embeddings were generated."
            )

        # ---------------------------------------------------------
        # 4. Build FAISS vector store
        # ---------------------------------------------------------
        dimension = len(embeddings[0])

        self.vector_store = FAISSVectorStore(
            dimension=dimension
        )

        self.vector_store.add_documents(
            self.chunks,
            embeddings,
        )

        # ---------------------------------------------------------
        # 5. Build hybrid retriever
        # ---------------------------------------------------------
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            documents=self.chunks,
        )

    def _rebuild_with_local_embeddings(self):
        """
        Rebuild the complete FAISS index using MiniLM.

        This is required if Gemini embedding becomes unavailable
        after a Gemini-based index has already been created.
        """

        print(
            "Rebuilding RAG index using "
            "all-MiniLM-L6-v2..."
        )

        # Force the same EmbeddingModel instance to use the
        # local provider.
        self.embedding_model._switch_to_local()

        self._build_vector_index()

        print(
            "RAG index rebuilt successfully using "
            "all-MiniLM-L6-v2."
        )

    def _resolve_document_scope(
        self,
        query: str,
    ) -> set[str] | None:
        """
        Detect whether the user explicitly refers to one or more
        uploaded documents.

        Returns:
            set of document IDs when a document is explicitly named.
            None when the query should search across all documents.
        """

        query_normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            query.lower(),
        ).strip()

        if not query_normalized:
            return None

        document_map = {}

        for chunk in self.chunks:
            document_id = chunk.metadata.get(
                "document_id"
            )

            document_name = chunk.metadata.get(
                "document_name",
                chunk.metadata.get(
                    "source",
                    "",
                ),
            )

            if not document_id or not document_name:
                continue

            document_map[document_id] = document_name

        matched_document_ids = set()

        for document_id, document_name in document_map.items():

            document_stem = Path(
                document_name
            ).stem

            normalized_name = re.sub(
                r"[^a-z0-9]+",
                " ",
                document_stem.lower(),
            ).strip()

            if not normalized_name:
                continue

            # Exact normalized filename/stem match.
            if normalized_name in query_normalized:
                matched_document_ids.add(
                    document_id
                )
                continue

            # Token-based match for natural references such as:
            # "Candidate DMC document"
            document_tokens = [
                token
                for token in normalized_name.split()
                if len(token) > 2
            ]

            if not document_tokens:
                continue

            matched_tokens = sum(
                token in query_normalized.split()
                for token in document_tokens
            )

            match_ratio = (
                matched_tokens / len(document_tokens)
            )

            if (
                matched_tokens >= 2
                and match_ratio >= 0.5
            ):
                matched_document_ids.add(
                    document_id
                )

        return (
            matched_document_ids
            if matched_document_ids
            else None
        )

    def retrieve(
        self,
        query: str,
        k: int = 5,
        candidate_k: int = 10,
        document_ids: set[str] | None = None,
    ):
        """
        Retrieve relevant chunks without calling the LLM.

        If Gemini query embedding is exhausted after the index was
        built with Gemini, the complete index is rebuilt with MiniLM
        before retrying the query.
        """

        if self.retriever is None:
            raise RuntimeError(
                "RAG pipeline has not been built. "
                "Call build_from_documents() first."
            )

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )
        document_scope = self._resolve_document_scope(
                query
            )
        try:
            results, decision = self.retriever.retrieve(
                query=query,
                k=k,
                candidate_k=candidate_k,
                document_ids = document_scope
            )

        except RuntimeError as error:
            error_message = str(error).upper()

            is_embedding_quota_error = (
                "GEMINI" in error_message
                and (
                    "429" in error_message
                    or "RESOURCE_EXHAUSTED" in error_message
                    or "QUOTA" in error_message
                )
            )

            if not is_embedding_quota_error:
                raise

            if self.embedding_model.provider != "gemini":
                raise

            self._rebuild_with_local_embeddings()

            # Retry retrieval using the newly rebuilt MiniLM index.
            results, decision = self.retriever.retrieve(
                query=query,
                k=k,
                candidate_k=candidate_k,
                document_ids=document_ids,
                
            )

        confidence = {
            "confident": decision.confident,
            "reason": decision.reason,
            "top_semantic_score": decision.top_semantic_score,
            "top_lexical_score": decision.top_lexical_score,
            "lexical_match": decision.lexical_match,
            "retrieved_count": decision.retrieved_count,
        }

        sources = []

        for result in results:
            sources.append(
                {
                    "rank": result.rank,
                    "document_id": result.document.metadata.get(
                        "document_id"
                    ),
                    "page": result.document.metadata.get(
                        "page"
                    ),
                    "chunk_id": result.document.metadata.get(
                        "chunk_id"
                    ),
                    "document_name": result.document.metadata.get(
                        "document_name",
                        result.document.metadata.get(
                            "source",
                            "unknown",
                        ),
                    ),
                    "semantic_score": result.semantic_score,
                    "bm25_score": result.lexical_score,
                    "content": result.document.page_content,
                }
            )

        return {
            "results": results,
            "sources": sources,
            "confidence": confidence,
            "embedding_provider": self.embedding_provider,
        }

    def ask(
        self,
        query: str,
        k: int = 5,
        candidate_k: int = 10,
        document_ids: set[str] | None = None,
    ):
        """
        Retrieve relevant context and generate a grounded answer.
        """

        retrieval = self.retrieve(
            query=query,
            k=k,
            candidate_k=candidate_k,
            document_ids=document_ids
        )

        confidence = retrieval["confidence"]
        sources = retrieval["sources"]

        if not confidence["confident"]:
            return {
                "answer": (
                    "I couldn't find enough relevant information "
                    "in the uploaded document(s) to answer this "
                    "question."
                ),
                "sources": [],
                "confidence": confidence,
                "embedding_provider": retrieval[
                    "embedding_provider"
                ],
            }

        context_parts = []

        for result in retrieval["results"]:
            document = result.document

            document_name = document.metadata.get(
                "document_name",
                document.metadata.get(
                    "source",
                    "unknown",
                ),
            )

            page = document.metadata.get(
                "page"
            )

            context_parts.append(
                f"[Document: {document_name} | Page: {page}]\n"
                f"{document.page_content}"
            )

        context = "\n\n".join(
            context_parts
        )

        answer, generation_provider = generate_rag_answer(
            query=query,
            context=context,
        )

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "embedding_provider": retrieval[
                "embedding_provider"
            ],
            "generation_provider": generation_provider,

        }

    def _assign_document_ids(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """
        Assign a stable document identifier based on
        the document source.
        """

        document_ids = {}

        for document in documents:
            source = document.metadata.get(
                "document_name",
                document.metadata.get(
                    "source",
                    "unknown",
                ),
            )

            if source not in document_ids:
                document_ids[source] = hashlib.sha256(
                    source.encode("utf-8")
                ).hexdigest()[:16]

            document.metadata[
                "document_id"
            ] = document_ids[source]

        return documents