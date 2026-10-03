import faiss
import numpy as np

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

import re


class FAISSVectorStore:
    """
    Hybrid document store using:

    1. FAISS for semantic retrieval
    2. BM25 for lexical retrieval
    3. Reciprocal Rank Fusion for combining results
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

        # Normalized embeddings + Inner Product
        # approximates cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)

        self.documents: list[Document] = []

        self.bm25 = None

    @staticmethod
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Normalize text for lexical retrieval.

        Removes punctuation differences such as:

            MLflow,
            MLflow.
            MLflow

        and converts all text to lowercase.
        """

        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "be",
            "do",
            "does",
            "for",
            "how",
            "in",
            "is",
            "of",
            "on",
            "the",
            "to",
            "what",
            "which",
            "where",
            "when",
            "who",
            "why",
        }

        tokens = re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )

        return [
            token
            for token in tokens
            if token not in stopwords
        ]

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add documents to both FAISS and BM25.
        """

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match "
                "number of embeddings."
            )

        vectors = np.asarray(
            embeddings,
            dtype="float32",
        )

        self.index.add(vectors)

        self.documents.extend(documents)

        # Build BM25 index
        tokenized_documents = [
            self._tokenize(document.page_content)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Semantic search using FAISS.
        """

        if not self.documents:
            raise ValueError(
                "Vector store is empty."
            )

        k = min(k, len(self.documents))

        query_vector = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        scores, indices = self.index.search(
            query_vector,
            k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            results.append(
                (
                    self.documents[index],
                    float(score),
                )
            )

        return results

    def keyword_search(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Keyword retrieval using BM25.
        """

        if not self.documents:
            raise ValueError(
                "Vector store is empty."
            )

        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built."
            )

        k = min(k, len(self.documents))

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = np.argsort(
            scores
        )[::-1][:k]

        results = []

        for index in ranked_indices:
            results.append(
                (
                    self.documents[index],
                    float(scores[index]),
                )
            )

        return results

    def hybrid_search(
    self,
    query: str,
    query_embedding: list[float],
    k: int = 5,
    candidate_k: int = 10,
):
        """
        Perform hybrid retrieval using:
        - FAISS semantic similarity
        - BM25 keyword matching
        - Weighted score fusion

        Returns:
            List of tuples:
            (
                document,
                final_score,
                semantic_score,
                keyword_score
            )
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not self.documents:
            return []

        # ---------------------------------------------------------
        # 1. Semantic retrieval using FAISS
        # ---------------------------------------------------------
        semantic_results = self.similarity_search(
            query_embedding=query_embedding,
            k=min(candidate_k, len(self.documents)),
        )

        # Map document index -> semantic score
        semantic_scores = {}

        for document, score in semantic_results:
            chunk_id = document.metadata.get("chunk_id")

            if chunk_id is not None:
                semantic_scores[chunk_id] = float(score)

        # ---------------------------------------------------------
        # 2. Keyword retrieval using BM25
        # ---------------------------------------------------------
        keyword_results = self.keyword_search(
            query=query,
            k=min(candidate_k, len(self.documents)),
        )

        # Map document index -> BM25 score
        keyword_scores = {}

        for document, score in keyword_results:
            chunk_id = document.metadata.get("chunk_id")

            if chunk_id is not None:
                keyword_scores[chunk_id] = float(score)

        # ---------------------------------------------------------
        # 3. Collect candidate chunks
        # ---------------------------------------------------------
        candidate_chunk_ids = set(semantic_scores) | set(keyword_scores)

        if not candidate_chunk_ids:
            return []

        # ---------------------------------------------------------
        # 4. Normalize semantic scores
        # ---------------------------------------------------------
        semantic_values = list(semantic_scores.values())

        semantic_min = min(semantic_values)
        semantic_max = max(semantic_values)

        normalized_semantic = {}

        for chunk_id, score in semantic_scores.items():

            if semantic_max == semantic_min:
                normalized_score = 1.0
            else:
                normalized_score = (
                    (score - semantic_min)
                    / (semantic_max - semantic_min)
                )

            normalized_semantic[chunk_id] = normalized_score

        # ---------------------------------------------------------
        # 5. Normalize BM25 scores
        # ---------------------------------------------------------
        keyword_values = list(keyword_scores.values())

        keyword_min = min(keyword_values)
        keyword_max = max(keyword_values)

        normalized_keyword = {}

        for chunk_id, score in keyword_scores.items():

            if keyword_max == keyword_min:
                normalized_score = 1.0
            else:
                normalized_score = (
                    (score - keyword_min)
                    / (keyword_max - keyword_min)
                )

            normalized_keyword[chunk_id] = normalized_score

        # ---------------------------------------------------------
        # 6. Weighted score fusion
        #
        # Semantic similarity = 60%
        # Keyword relevance   = 40%
        # ---------------------------------------------------------
        SEMANTIC_WEIGHT = 0.60
        KEYWORD_WEIGHT = 0.40

        fused_results = []

        for chunk_id in candidate_chunk_ids:

            semantic_score = semantic_scores.get(
                chunk_id,
                0.0,
            )

            keyword_score = keyword_scores.get(
                chunk_id,
                0.0,
            )

            normalized_semantic_score = normalized_semantic.get(
                chunk_id,
                0.0,
            )

            normalized_keyword_score = normalized_keyword.get(
                chunk_id,
                0.0,
            )

            final_score = (
                SEMANTIC_WEIGHT * normalized_semantic_score
                + KEYWORD_WEIGHT * normalized_keyword_score
            )

            document = self.documents[chunk_id]

            fused_results.append(
                (
                    document,
                    float(final_score),
                    float(semantic_score),
                    float(keyword_score),
                )
            )

        # ---------------------------------------------------------
        # 7. Sort by final hybrid score
        # ---------------------------------------------------------
        fused_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        # ---------------------------------------------------------
        # 8. Return top-k results
        # ---------------------------------------------------------
        return fused_results[:k]