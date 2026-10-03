import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "gemini" if GEMINI_API_KEY else "local",
).lower()

if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Set LLM_PROVIDER=local to use the free local provider, "
        "or provide a GEMINI_API_KEY."
    )

