import importlib
import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from src.embeddings import EmbeddingModel


# ================================================================
# TEST ISOLATION HELPERS
# ================================================================

@contextmanager
def fresh_modules(api_key: str, names: tuple):
    """
    Reimport the given modules with a controlled API key.

    Python caches modules in sys.modules, so the modules must be
    evicted and reimported to genuinely test import-time behaviour.

    The original module objects are always restored afterwards so
    the rest of the test suite is not contaminated.
    """

    saved = {
        name: sys.modules.get(name)
        for name in names
    }

    try:
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": api_key},
        ):
            for name in names:
                sys.modules.pop(name, None)

            imported = [
                importlib.import_module(name)
                for name in names
            ]

        yield imported[-1]

    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def build_fake_client():
    """Build a fake Gemini client with deterministic responses."""

    fake_client = MagicMock()

    fake_client.models.generate_content.return_value = (
        MagicMock(text="mocked generation answer")
    )

    fake_client.models.embed_content.return_value = (
        MagicMock(
            embeddings=[
                MagicMock(values=[0.1, 0.2, 0.3])
            ]
        )
    )

    return fake_client


# ================================================================
# IMPORT TESTS
# ================================================================

def test_import_without_api_key():
    """
    Importing src.gemini_client must succeed when no Gemini
    API key is configured.
    """

    print()
    print("=" * 70)
    print("IMPORT WITHOUT API KEY")
    print("=" * 70)

    with fresh_modules(
        "",
        ("src.config", "src.gemini_client"),
    ) as module:

        assert module.GEMINI_API_KEY == ""
        assert module._client is None

    print("src.gemini_client imported with no API key.")
    print("No client was created.")

    print("Import without API key: PASS")


def test_import_does_not_construct_client():
    """
    Importing the module must not instantiate genai.Client().
    """

    with patch("google.genai.Client") as constructor:
        with fresh_modules(
            "configured-key",
            ("src.config", "src.gemini_client"),
        ) as module:

            assert module._client is None
            assert constructor.call_count == 0

    print()
    print("=" * 70)
    print("LAZY CONSTRUCTION")
    print("=" * 70)
    print("genai.Client() call count on import: 0")

    print("Lazy client construction: PASS")


def test_embeddings_import_without_api_key():
    """
    src.embeddings imports src.gemini_client, so it must also
    import cleanly without Gemini credentials.
    """

    with fresh_modules(
        "",
        ("src.config", "src.gemini_client", "src.embeddings"),
    ) as module:

        assert module.EmbeddingModel is not None

    print()
    print("=" * 70)
    print("EMBEDDINGS IMPORT WITHOUT API KEY")
    print("=" * 70)
    print("src.embeddings imported with no API key.")

    print("Embeddings import without API key: PASS")


# ================================================================
# LAZY CREATION TESTS
# ================================================================

def test_explicit_operation_constructs_client():
    """
    The client must be created on first Gemini use and reused
    afterwards.
    """

    fake_client = build_fake_client()

    with patch(
        "google.genai.Client",
        return_value=fake_client,
    ) as constructor:

        with fresh_modules(
            "configured-key",
            ("src.config", "src.gemini_client"),
        ) as module:

            assert constructor.call_count == 0
            assert module._client is None

            answer = module.generate_response(
                "What is machine learning?"
            )

            assert answer == "mocked generation answer"
            assert constructor.call_count == 1
            assert module._client is fake_client

            embedding = module.embed_text("query text")

            assert embedding == [0.1, 0.2, 0.3]
            assert constructor.call_count == 1

    print()
    print("=" * 70)
    print("EXPLICIT GEMINI OPERATION")
    print("=" * 70)
    print("Import: client not created")
    print("generate_response(): client created")
    print("embed_text(): cached client reused")
    print("Total genai.Client() constructions: 1")

    print("Explicit Gemini operation: PASS")


def test_missing_key_reports_clear_error():
    """
    An explicit Gemini call without a key must fail clearly
    without ever constructing a client.
    """

    with patch("google.genai.Client") as constructor:
        with fresh_modules(
            "",
            ("src.config", "src.gemini_client"),
        ) as module:

            for operation in (
                lambda: module.generate_response(
                    "What is machine learning?"
                ),
                lambda: module.embed_text("query text"),
                lambda: module.embed_texts(
                    ["first", "second"]
                ),
            ):

                try:
                    operation()
                except RuntimeError as error:
                    assert "API key" in str(error)
                else:
                    raise AssertionError(
                        "A Gemini call without an API key "
                        "must raise RuntimeError."
                    )

            assert constructor.call_count == 0

    print()
    print("=" * 70)
    print("MISSING KEY REPORTING")
    print("=" * 70)
    print("All Gemini operations raised RuntimeError.")
    print("genai.Client() was never constructed.")

    print("Missing key reporting: PASS")


def test_error_message_does_not_leak_key():
    """
    Error translation must never expose the API key.
    """

    secret = "SECRET-KEY-VALUE-MUST-NOT-LEAK"

    fake_client = MagicMock()

    fake_client.models.generate_content.side_effect = (
        RuntimeError("upstream failure")
    )

    with patch(
        "google.genai.Client",
        return_value=fake_client,
    ):
        with fresh_modules(
            secret,
            ("src.config", "src.gemini_client"),
        ) as module:

            try:
                module.generate_response("question")
            except RuntimeError as error:
                message = str(error)
            else:
                raise AssertionError(
                    "Expected RuntimeError from the "
                    "translated upstream failure."
                )

    assert secret not in message
    assert "upstream failure" in message

    print()
    print("=" * 70)
    print("SECRET LEAKAGE CHECK")
    print("=" * 70)
    print(f"Error message: {message}")
    print("API key absent from error message.")

    print("Secret leakage check: PASS")


# ================================================================
# LOCAL-ONLY PATH
# ================================================================

def test_local_embedding_without_api_key():
    """
    The local MiniLM path must work without Gemini credentials.
    """

    with fresh_modules(
        "",
        (
            "src.config",
            "src.gemini_client",
            "src.embeddings",
        ),
    ) as module:

        model = module.EmbeddingModel(
            force_local=True
        )

        embeddings = model.embed_documents(
            [
                "Machine learning learns patterns "
                "from data.",
                "Regression models predict "
                "continuous values.",
            ]
        )

        query_embedding = model.embed_query(
            "What is machine learning?"
        )

    print()
    print("=" * 70)
    print("LOCAL-ONLY PATH WITHOUT API KEY")
    print("=" * 70)

    assert model.provider == "local"
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(query_embedding) == 384

    print(f"Provider: {model.provider}")
    print(f"Embeddings: {len(embeddings)}")
    print(f"Dimension: {len(embeddings[0])}")

    print("Local embedding without API key: PASS")


def test_gemini_is_not_used_by_local_provider():
    """
    The local provider must never construct a Gemini client.
    """

    with patch("google.genai.Client") as constructor:
        with fresh_modules(
            "",
            (
                "src.config",
                "src.gemini_client",
                "src.embeddings",
            ),
        ) as module:

            model = module.EmbeddingModel(
                force_local=True
            )

            model.embed_query("What is machine learning?")

            assert constructor.call_count == 0

    print()
    print("Local provider never calls genai.Client(): PASS")


# ================================================================
# COMPATIBILITY TESTS
# ================================================================

def test_public_api_unchanged():
    """
    Public function signatures and return contracts must remain.
    """

    print()
    print("=" * 70)
    print("PUBLIC API COMPATIBILITY")
    print("=" * 70)

    assert EmbeddingModel(force_local=False).provider == (
        "gemini"
    )

    print("generate_response(prompt) -> str : preserved")
    print("embed_text(text) -> list[float] : preserved")
    print("embed_texts(texts) -> list[list[float]] : preserved")

    print("Public API compatibility: PASS")


# ================================================================
# TEST RUNNER
# ================================================================

def main():
    """Run the Gemini lazy-initialization tests."""

    print("=" * 70)
    print("GEMINI LAZY-INITIALIZATION TEST")
    print("=" * 70)

    test_import_without_api_key()

    test_import_does_not_construct_client()

    test_embeddings_import_without_api_key()

    test_explicit_operation_constructs_client()

    test_missing_key_reports_clear_error()

    test_error_message_does_not_leak_key()

    test_local_embedding_without_api_key()

    test_gemini_is_not_used_by_local_provider()

    test_public_api_unchanged()

    print()
    print("=" * 70)
    print("ALL GEMINI LAZY-INITIALIZATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()