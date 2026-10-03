from threading import Lock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


class LocalLLM:
    """
    Lightweight local text-generation fallback.

    The model is loaded lazily so normal Gemini usage does not
    unnecessarily consume local memory.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = Lock()

    def _load_model(self):
        """Load the local model only when fallback is required."""

        if self.model is not None:
            return

        print(
            f"Loading local generation model: {MODEL_NAME}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME
        ).to(self.device)

        self.model.eval()

        print(
            f"Local generation model loaded on {self.device}."
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate text locally.
        """

        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        with self._lock:
            self._load_model()

            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            input_text = (
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_tokens = outputs[
                0
            ][
                inputs["input_ids"].shape[-1]:
            ]

            answer = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            if not answer:
                raise RuntimeError(
                    "Local LLM returned an empty response."
                )

            return answer


_local_llm = LocalLLM()


def generate_local_response(
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """
    Generate a response using the local fallback LLM.
    """

    return _local_llm.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
