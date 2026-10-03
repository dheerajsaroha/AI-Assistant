import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from langchain_core.documents import Document

from src.embeddings import EmbeddingModel
from src.vector_stores import FAISSVectorStore


# Query-side terms that indicate the question is about pay.
SALARY_QUERY_TERMS = (
    "salary",
    "compensation",
    "income",
    "pay",
    "earning",
    "earn",
)

# Document-side terms that count as salary evidence.
SALARY_EVIDENCE_TERMS = (
    "salary",
    "compensation",
    "income",
    "pay",
    "earnings",
    "earning",
)


def contains_term(text: str, terms) -> bool:
    """
    Return True when any term occurs in the text as a whole word.

    Substring containment is deliberately avoided here. "learn"
    contains "earn" and "learning" contains "earning", so plain
    substring matching would classify unrelated questions as
    pay-related and suppress valid answers.
    """

    return any(
        re.search(
            rf"\b{re.escape(term)}\b",
            text,
        )
        for term in terms
    )


@dataclass
class RetrievalResult:
    """
    Represents one retrieved document chunk.
    """

    document: Document
    semantic_score: float
    lexical_score: float
    rank: int


@dataclass
class RetrievalDecision:
    """
    Represents the evidence-based confidence decision.

    Retrieval ranking and confidence evaluation are
    intentionally kept separate.
    """

    confident: bool
    reason: str

    top_semantic_score: float
    max_semantic_score: float

    top_lexical_score: float
    max_lexical_score: float

    lexical_match: bool
    retrieved_count: int


class HybridRetriever:
    """
    Hybrid semantic + lexical retriever.

    Semantic retrieval:
        Sentence Transformer + FAISS

    Lexical retrieval:
        BM25

    Confidence gate:
        Determines whether the retrieved evidence
        is strong enough to pass to the generator.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FAISSVectorStore,
        documents: list[Document],
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.documents = documents

        self.bm25 = self._build_bm25(documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be",
            "by", "for", "from", "how", "in", "is", "it",
            "of", "on", "or", "that", "the", "this", "to",
            "was", "what", "when", "where", "which", "who",
            "why", "with"
        }

        tokens = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )

        return [
            token
            for token in tokens
            if token not in stopwords
        ]

    def _build_bm25(
        self,
        documents: list[Document],
    ) -> BM25Okapi:
        """
        Build BM25 index from document chunks.
        """

        tokenized_documents = [
            self._tokenize(
                document.page_content
            )
            for document in documents
        ]

        return BM25Okapi(
            tokenized_documents
        )

    def _lexical_scores(
        self,
        query: str,
    ) -> list[float]:
        """
        Calculate BM25 scores for all documents.
        """

        query_tokens = self._tokenize(query)

        return self.bm25.get_scores(
            query_tokens
        ).tolist()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        candidate_k: int = 10,
        document_ids: set[str] | None = None,
    ) -> tuple[
        list[RetrievalResult],
        RetrievalDecision,
    ]:
        """
        Retrieve relevant documents and evaluate
        retrieval confidence.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if document_ids:
            allowed_indices = {
                index
                for index, document in enumerate(self.documents)
                if document.metadata.get("document_id") in document_ids
            }

            if not allowed_indices:
                raise ValueError(
                    "No chunks were found for the requested document scope."
                )

            scoped_candidate_k = min(
                candidate_k,
                len(allowed_indices),
            )
        else:
            allowed_indices = set(range(len(self.documents)))

            scoped_candidate_k = min(
                candidate_k,
                len(self.documents),
            )

        # --------------------------------
        # Semantic retrieval
        # --------------------------------

        query_embedding = (
            self.embedding_model.embed_query(
                query
            )
        )

        semantic_search_k = (
    len(self.documents)
    if document_ids
    else scoped_candidate_k
)

        semantic_results = (
            self.vector_store.similarity_search(
                query_embedding,
                k=semantic_search_k,
            )
        )

        if document_ids:
            semantic_results = [
                (
                    document,
                    score,
                )
                for document, score in semantic_results
                if document.metadata.get("document_id")
                in document_ids
            ][:scoped_candidate_k]

        # --------------------------------
        # Lexical retrieval
        # --------------------------------

        lexical_scores = self._lexical_scores(
            query
        )

        lexical_ranked_indices = sorted(
            allowed_indices,
            key=lambda index: lexical_scores[index],
            reverse=True,
        )[:scoped_candidate_k]

        lexical_rank = {
            index: rank + 1
            for rank, index in enumerate(
                lexical_ranked_indices
            )
        }

        # --------------------------------
        # Combine candidates
        # --------------------------------

        candidate_indices = set()

        for document, _ in semantic_results:
            candidate_indices.add(
                document.metadata["chunk_id"]
            )

        candidate_indices.update(
            lexical_ranked_indices
        )

        results = []

        semantic_scores_by_chunk = {
            document.metadata["chunk_id"]: score
            for document, score in semantic_results
        }

        for chunk_id in candidate_indices:

            document = self.documents[chunk_id]

            semantic_score = (
                semantic_scores_by_chunk.get(
                    chunk_id,
                    0.0,
                )
            )

            lexical_score = lexical_scores[
                chunk_id
            ]

            results.append(
                RetrievalResult(
                    document=document,
                    semantic_score=semantic_score,
                    lexical_score=lexical_score,
                    rank=0,
                )
            )

        # --------------------------------
        # RRF ranking
        # --------------------------------
        #
        # We are NOT changing the semantic
        # similarity formula.
        #
        # RRF only combines the independent
        # semantic and lexical rankings.

        semantic_rank = {}

        for rank, (document, _) in enumerate(
            semantic_results,
            start=1,
        ):
            semantic_rank[
                document.metadata["chunk_id"]
            ] = rank

        def rrf_score(result: RetrievalResult):
            chunk_id = result.document.metadata[
                "chunk_id"
            ]

            semantic_component = (
                1 / (
                    60
                    + semantic_rank.get(
                        chunk_id,
                        scoped_candidate_k + 1,
                    )
                )
            )

            lexical_component = (
                1 / (
                    60
                    + lexical_rank.get(
                        chunk_id,
                        candidate_k + 1,
                    )
                )
            )

            return (
                semantic_component
                + lexical_component
            )

        results.sort(
            key=rrf_score,
            reverse=True,
        )

        results = results[:k]

        for rank, result in enumerate(
            results,
            start=1,
        ):
            result.rank = rank

        # --------------------------------
        # Confidence gate
        # --------------------------------

        decision = self._confidence_gate(
            query=query,
            results=results,
        )

        return results, decision
    def _normalize_query_tokens(
        self,
        query: str,
    ) -> set[str]:

        tokens = set(
            self._tokenize(query)
        )

        # Common domain abbreviations
        expansions = {
            "nlp": {
                "nlp",
                "natural",
                "language",
                "processing",
            },
            "ml": {
                "ml",
                "machine",
                "learning",
            },
            "dl": {
                "dl",
                "deep",
                "learning",
            },
            "ai": {
                "ai",
                "artificial",
                "intelligence",
            },
            "cnn": {
                "cnn",
                "convolutional",
                "neural",
                "networks",
            },
            "ann": {
                "ann",
                "artificial",
                "neural",
                "networks",
            },
        }

        expanded_tokens = set(tokens)

        for token in tokens:

            if token in expansions:
                expanded_tokens.update(
                    expansions[token]
                )

        return expanded_tokens

    def _get_query_focus_tokens(
        self,
        query: str,
    ) -> set[str]:

        tokens = list(
            self._tokenize(query)
        )

        stopwords = {
            "what",
            "which",
            "when",
            "where",
            "who",
            "whom",
            "whose",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
            "do",
            "does",
            "did",
            "can",
            "could",
            "would",
            "should",
            "will",
            "the",
            "a",
            "an",
            "this",
            "that",
            "these",
            "those",
            "in",
            "on",
            "at",
            "for",
            "of",
            "to",
            "from",
            "with",
            "about",
            "and",
            "or",
            "i",
            "my",
            "me",
            "it",
        }

        focus_tokens = {
            token
            for token in tokens
            if token not in stopwords
        }

        return focus_tokens

    def _has_answer_specific_evidence(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> bool:
        query_lower = query.lower().strip()

        evidence = " ".join(
            result.document.page_content.lower()
            for result in results
        )

        # Salary questions require salary-related evidence.
        if contains_term(
            query_lower,
            SALARY_QUERY_TERMS,
        ):
            return contains_term(
                evidence,
                SALARY_EVIDENCE_TERMS,
            )

        # "Who" questions require explicit authorship/creator evidence.
        if query_lower.startswith("who "):
            creator_phrases = [
                "created by",
                "creator",
                "author",
                "authored by",
                "written by",
            ]

            return any(
                phrase in evidence
                for phrase in creator_phrases
            )

        return True

    

    def _confidence_gate(
        self,
        query: str,
        results: list[RetrievalResult],
        min_semantic_score: float = 0.35,
        strong_semantic_score: float = 0.75,
        semantic_only_score: float = 0.60,
    ) -> RetrievalDecision:

        if not results:
            return RetrievalDecision(
                confident=False,
                reason="No documents were retrieved.",
                top_semantic_score=0.0,
                max_semantic_score=0.0,
                top_lexical_score=0.0,
                max_lexical_score=0.0,
                lexical_match=False,
                retrieved_count=0,
            )

        top_result = results[0]

        top_semantic_score = (
            top_result.semantic_score
        )

        top_lexical_score = (
            top_result.lexical_score
        )

        max_semantic_score = max(
            result.semantic_score
            for result in results
        )

        max_lexical_score = max(
            result.lexical_score
            for result in results
        )

        # --------------------------------------------------
        # Query analysis
        # --------------------------------------------------

        query_tokens = (
            self._normalize_query_tokens(query)
        )

        focus_tokens = (
            self._get_query_focus_tokens(query)
        )

        # --------------------------------------------------
        # Retrieved document tokens
        # --------------------------------------------------

        retrieved_tokens = set()

        for result in results:

            retrieved_tokens.update(
                self._tokenize(
                    result.document.page_content
                )
            )

        # --------------------------------------------------
        # Query token coverage
        # --------------------------------------------------

        matched_focus_tokens = (
            focus_tokens.intersection(
                retrieved_tokens
            )
        )

        focus_match = bool(
            matched_focus_tokens
        )
        

        # Expanded semantic/lexical concept match
        expanded_query_match = bool(
            query_tokens.intersection(
                retrieved_tokens
            )
        )
        print("\n[CONFIDENCE DEBUG]")
        print("Query:", query)
        print("Focus tokens:", sorted(focus_tokens))
        print("Matched focus tokens:", sorted(matched_focus_tokens))
        print("Expanded query match:", expanded_query_match)
        print("Top semantic:", top_semantic_score)
        print("Max semantic:", max_semantic_score)
        print("Max lexical:", max_lexical_score)
        print("[END DEBUG]\n")

        lexical_match = (
            expanded_query_match
        )
                # --------------------------------------------------
        # Answer-specific evidence protection
        # --------------------------------------------------

        query_lower = query.lower()

        # Salary-related questions must have salary-specific
        # evidence in the retrieved document.
        if contains_term(
            query_lower,
            SALARY_QUERY_TERMS,
        ):
            salary_evidence = any(
                term in retrieved_tokens
                for term in SALARY_EVIDENCE_TERMS
            )

            if not salary_evidence:
                return RetrievalDecision(
                    confident=False,
                    reason=(
                        "The query asks about salary or "
                        "earnings, but the retrieved evidence "
                        "does not contain salary-specific information."
                    ),
                    top_semantic_score=top_semantic_score,
                    max_semantic_score=max_semantic_score,
                    top_lexical_score=top_lexical_score,
                    max_lexical_score=max_lexical_score,
                    lexical_match=lexical_match,
                    retrieved_count=len(results),
                )

        # --------------------------------------------------
        # Question type
        # --------------------------------------------------

        raw_tokens = self._tokenize(
            query
        )

        is_who_question = (
            "who" in raw_tokens
        )

        is_when_question = (
            "when" in raw_tokens
        )

        is_where_question = (
            "where" in raw_tokens
        )
#         answer_specific_evidence = self._has_answer_specific_evidence(
#     query,
#     results,
# )

#         if not answer_specific_evidence:
#             return RetrievalDecision(
#                 confident=False,
#                 reason=(
#                     "Retrieved content is semantically related "
#                     "but does not contain answer-specific evidence."
#                 ),
#                 top_semantic_score=top_semantic_score,
#                 max_semantic_score=max_semantic_score,
#                 top_lexical_score=top_lexical_score,
#                 max_lexical_score=max_lexical_score,
#                 lexical_match=lexical_match,
#                 retrieved_count=len(results),
#             )
        

        # --------------------------------------------------
        # 1. Very strong semantic evidence
        #
        # Still require meaningful query evidence.
        # --------------------------------------------------

        if (
            top_semantic_score
            >= strong_semantic_score
            and focus_match
        ):
            return RetrievalDecision(
                confident=True,
                reason=(
                    "Retrieved evidence has very "
                    "strong semantic similarity and "
                    "contains query-relevant terms."
                ),
                top_semantic_score=(
                    top_semantic_score
                ),
                max_semantic_score=(
                    max_semantic_score
                ),
                top_lexical_score=(
                    top_lexical_score
                ),
                max_lexical_score=(
                    max_lexical_score
                ),
                lexical_match=lexical_match,
                retrieved_count=len(results),
            )

        # --------------------------------------------------
        # 2. Strong lexical + semantic support
        # --------------------------------------------------

        if (
            focus_match
            and max_semantic_score
            >= min_semantic_score
        ):
            return RetrievalDecision(
                confident=True,
                reason=(
                    "Retrieved evidence contains "
                    "meaningful query terms together "
                    "with semantic support."
                ),
                top_semantic_score=(
                    top_semantic_score
                ),
                max_semantic_score=(
                    max_semantic_score
                ),
                top_lexical_score=(
                    top_lexical_score
                ),
                max_lexical_score=(
                    max_lexical_score
                ),
                lexical_match=lexical_match,
                retrieved_count=len(results),
            )

        # --------------------------------------------------
        # 3. Semantic-only evidence
        #
        # Only permit this when the normalized query
        # concept is represented in the retrieved text.
        # This handles NLP -> Natural Language Processing.
        # --------------------------------------------------

        if (
            top_semantic_score
            >= semantic_only_score
            and expanded_query_match
        ):
            return RetrievalDecision(
                confident=True,
                reason=(
                    "Top retrieved evidence has strong "
                    "semantic relevance and matches the "
                    "normalized query concept."
                ),
                top_semantic_score=(
                    top_semantic_score
                ),
                max_semantic_score=(
                    max_semantic_score
                ),
                top_lexical_score=(
                    top_lexical_score
                ),
                max_lexical_score=(
                    max_lexical_score
                ),
                lexical_match=lexical_match,
                retrieved_count=len(results),
            )
        

        # --------------------------------------------------
        # 4. Question-type protection
        #
        # Prevent generic semantic matches from answering
        # "who", "when", or "where" questions without
        # corresponding evidence.
        # --------------------------------------------------

        # if is_who_question:

        #     return RetrievalDecision(
        #         confident=False,
        #         reason=(
        #             "The query asks for a person or "
        #             "creator, but the retrieved evidence "
        #             "does not provide sufficient "
        #             "answer-specific support."
        #         ),
        #         top_semantic_score=(
        #             top_semantic_score
        #         ),
        #         max_semantic_score=(
        #             max_semantic_score
        #         ),
        #         top_lexical_score=(
        #             top_lexical_score
        #         ),
        #         max_lexical_score=(
        #             max_lexical_score
        #         ),
        #         lexical_match=lexical_match,
        #         retrieved_count=len(results),
        #     )

        # --------------------------------------------------
# Question-type protection
# --------------------------------------------------

        # A "who" question should only be accepted when
        # there is explicit person/creator evidence.
        if is_who_question:
            return RetrievalDecision(
                confident=False,
                reason=(
                    "The query asks for a person or creator, "
                    "but the retrieved evidence does not provide "
                    "sufficient answer-specific support."
                ),
                top_semantic_score=top_semantic_score,
                max_semantic_score=max_semantic_score,
                top_lexical_score=top_lexical_score,
                max_lexical_score=max_lexical_score,
                lexical_match=lexical_match,
                retrieved_count=len(results),
            )

# Do not reject "when" questions here.
# A valid timing query can be answered when its
# semantic and lexical evidence already passed the
# confidence branches below.

    # A "where" question should only be accepted when
    # location evidence is explicitly available.

        if is_where_question:
            return RetrievalDecision(
                confident=False,
                reason=(
                    "The query asks for a location, but the "
                    "retrieved evidence does not provide "
                    "sufficient answer-specific support."
                ),
                top_semantic_score=top_semantic_score,
                max_semantic_score=max_semantic_score,
                top_lexical_score=top_lexical_score,
                max_lexical_score=max_lexical_score,
                lexical_match=lexical_match,
                retrieved_count=len(results),
            )

        if is_when_question:

            return RetrievalDecision(
                confident=False,
                reason=(
                    "The query asks for timing, but the "
                    "retrieved evidence does not provide "
                    "sufficient answer-specific support."
                ),
                top_semantic_score=(
                    top_semantic_score
                ),
                max_semantic_score=(
                    max_semantic_score
                ),
                top_lexical_score=(
                    top_lexical_score
                ),
                max_lexical_score=(
                    max_lexical_score
                ),
                lexical_match=lexical_match,
                retrieved_count=len(results),
            )

        if is_where_question:

            return RetrievalDecision(
                confident=False,
                reason=(
                    "The query asks for a location, but "
                    "the retrieved evidence does not "
                    "provide sufficient answer-specific "
                    "support."
                ),
                top_semantic_score=(
                    top_semantic_score
                ),
                max_semantic_score=(
                    max_semantic_score
                ),
                top_lexical_score=(
                    top_lexical_score
                ),
                max_lexical_score=(
                    max_lexical_score
                ),
                lexical_match=lexical_match,
                retrieved_count=len(results),
            )

        # --------------------------------------------------
        # 5. Reject
        # --------------------------------------------------

        return RetrievalDecision(
            confident=False,
            reason=(
                "Retrieved evidence does not contain "
                "sufficient answer-specific support."
            ),
            top_semantic_score=(
                top_semantic_score
            ),
            max_semantic_score=(
                max_semantic_score
            ),
            top_lexical_score=(
                top_lexical_score
            ),
            max_lexical_score=(
                max_lexical_score
            ),
            lexical_match=lexical_match,
            retrieved_count=len(results),
        )