from unittest.mock import patch

from src.rag_chain import generate_with_fallback


PROMPT = (
    "Answer in one sentence using only this context:\n"
    "Machine learning uses data to learn patterns.\n\n"
    "Question: What does machine learning use?"
)


def test_429_generation_fallback():
    """Verify fallback on Gemini quota exhaustion."""

    with patch(
        "src.rag_chain.generate_response",
        side_effect=RuntimeError(
            "Gemini generation quota has been exhausted. "
            "Please wait for the quota to reset."
        ),
    ):
        answer, provider = generate_with_fallback(
            PROMPT
        )

    print()
    print("429 FALLBACK")
    print(f"Provider: {provider}")
    print(f"Answer: {answer}")

    assert provider == (
        "SmolLM2-360M-Instruct (Fallback)"
    )

    assert answer.strip()

    print("429 generation fallback: PASS")


def test_503_generation_fallback():
    with patch(
        "src.rag_chain.generate_response",
        side_effect=RuntimeError(
            "Gemini generation failed: 503 UNAVAILABLE. "
            "This model is currently experiencing high demand."
        ),
    ):
        answer, provider = generate_with_fallback(
            "Answer in one sentence: What is machine learning?"
        )

    print("\n503 FALLBACK")
    print(f"Provider: {provider}")
    print(f"Answer: {answer}")

    assert provider == "SmolLM2-360M-Instruct (Fallback)"
    assert answer.strip()

    print("503 generation fallback: PASS")


def test_missing_key_generation_fallback():
    """
    Verify fallback when Gemini has no API key configured.

    This is the case that fails on a host without GEMINI_API_KEY
    set. It must not hard-fail generation; the local model answers
    instead.
    """

    with patch(
        "src.rag_chain.generate_response",
        side_effect=RuntimeError(
            "Gemini API key is not configured. "
            "Set GEMINI_API_KEY to use Gemini generation "
            "or embedding."
        ),
    ):
        answer, provider = generate_with_fallback(
            "Answer in one sentence: What is machine learning?"
        )

    print("\nMISSING-KEY FALLBACK")
    print(f"Provider: {provider}")
    print(f"Answer: {answer}")

    assert provider == "SmolLM2-360M-Instruct (Fallback)"
    assert answer.strip()

    print("Missing-key generation fallback: PASS")
    
if __name__ == "__main__":

    print("=" * 80)
    print("GENERATION FALLBACK TEST")
    print("=" * 80)

    test_429_generation_fallback()
    test_503_generation_fallback()
    test_missing_key_generation_fallback()

    print()
    print("=" * 80)
    print("ALL GENERATION FALLBACK TESTS PASSED")
    print("=" * 80)