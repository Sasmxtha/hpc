"""Generic LLM fallback chain: Groq API -> local TinyLlama -> (caller's own rule-based
backstop). Shared plumbing for anything in the AI brain that needs "ask an LLM for a
structured JSON answer, degrade gracefully if it's unavailable" -- the 3-way classifier
today, the LLM forensics engine next.

Detection itself (observer, transformer, CBF) never depends on any LLM -- this chain exists
for the *interpretive* layer on top (classification rationale, forensics narrative), where
an LLM materially helps but its absence must never stop the system from producing an answer.

Design: each backend is prompt-in / JSON-dict-out and raises BackendUnavailable on any
failure (no API key, no network, no model file, unparseable response) rather than letting
exceptions escape -- that uniform failure signal is what lets FallbackChain try the next
level without each backend needing to know about the others. The final backstop (hardcoded
rules) is deliberately NOT implemented as an LLMBackend here: it needs the caller's actual
structured evidence, not a text prompt, so callers should catch a fully-exhausted
FallbackChain and fall through to their own rule-based function -- see
classifier_3way.classify() for the pattern.
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


class BackendUnavailable(Exception):
    """Raised by a backend when it cannot produce an answer -- missing credentials, no
    network, missing local model file, or a response that fails to parse as the expected
    JSON. Signals FallbackChain to move on to the next backend.
    """


class LLMBackend(ABC):
    name: str

    @abstractmethod
    def complete_json(self, prompt: str, system: str = "") -> dict:
        """Send prompt (+ optional system prompt), return the parsed JSON response body.
        Must raise BackendUnavailable rather than propagate a raw exception on failure.
        """


def _extract_json(text: str) -> str:
    """Best-effort extraction of the first {...} object from raw LLM text output.

    Small local models routinely wrap JSON in commentary, markdown code fences, or a
    trailing explanation despite being told not to -- this pulls out the object rather than
    requiring the whole response to be exactly parseable.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model output")
    return text[start : end + 1]


class GroqBackend(LLMBackend):
    """Primary backend: Groq's free-tier API, OpenAI-compatible chat completions endpoint.

    Requires GROQ_API_KEY in the environment and network access. Verified against the live
    API: the model named in the spec, "llama3-70b-8192", has since been decommissioned by
    Groq (confirmed via a live 400 model_decommissioned response, then cross-checked against
    GET /openai/v1/models for this key's currently available models). No Llama-3 model
    remains on Groq at all -- openai/gpt-oss-120b is the closest replacement in spirit for
    "primary, large, capable" and is what's used by default here. Groq's lineup shifts over
    time; re-run GET https://api.groq.com/openai/v1/models if this default 400s again.
    """

    name = "groq"

    def __init__(
        self, model: str = "openai/gpt-oss-120b", api_key_env: str = "GROQ_API_KEY", timeout: float = 10.0
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout

    def complete_json(self, prompt: str, system: str = "") -> dict:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise BackendUnavailable(f"{self.api_key_env} not set")
        try:
            import requests
        except ImportError as e:
            raise BackendUnavailable(f"requests not installed: {e}") from e

        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except BackendUnavailable:
            raise
        except Exception as e:
            raise BackendUnavailable(f"Groq request failed: {e}") from e


class TinyLlamaBackend(LLMBackend):
    """Secondary backend: TinyLlama-1.1B running locally via llama-cpp-python, no network.

    Requires llama-cpp-python installed and a local GGUF model file (e.g. a Q4_K_M quantized
    TinyLlama-1.1B-Chat, a few hundred MB). Neither is installed/downloaded in this dev
    environment -- downloading a multi-hundred-MB model file needs your explicit go-ahead
    first. This class's failure path (raising BackendUnavailable when the file/library is
    missing) is what's actually been exercised here.
    """

    name = "tinyllama"

    def __init__(self, model_path: str, n_ctx: int = 2048, max_tokens: int = 256):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self._llm = None

    def _load(self):
        if self._llm is not None:
            return self._llm
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise BackendUnavailable(f"llama-cpp-python not installed: {e}") from e
        if not os.path.exists(self.model_path):
            raise BackendUnavailable(f"model file not found: {self.model_path}")
        self._llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx, verbose=False)
        return self._llm

    def complete_json(self, prompt: str, system: str = "") -> dict:
        llm = self._load()
        full_prompt = f"{system}\n\n{prompt}\n\nRespond with only a JSON object, nothing else."
        try:
            out = llm(full_prompt, max_tokens=self.max_tokens, temperature=0.0)
            text = out["choices"][0]["text"]
            return json.loads(_extract_json(text))
        except BackendUnavailable:
            raise
        except Exception as e:
            raise BackendUnavailable(f"TinyLlama generation/parse failed: {e}") from e


@dataclass
class BackendResult:
    data: dict
    backend_used: str


class FallbackChain:
    """Tries each backend in order, falling through to the next on BackendUnavailable.

    Raises RuntimeError only if every backend in the chain fails -- callers (e.g.
    classifier_3way.classify) should catch that and fall through to their own hardcoded
    rule-based function, which is the true final backstop and has zero dependencies.
    """

    def __init__(self, backends: List[LLMBackend]):
        self.backends = backends

    def run(self, prompt: str, system: str = "") -> BackendResult:
        errors = []
        for backend in self.backends:
            try:
                data = backend.complete_json(prompt, system=system)
                return BackendResult(data=data, backend_used=backend.name)
            except BackendUnavailable as e:
                errors.append(f"{backend.name}: {e}")
                continue
        raise RuntimeError("all backends exhausted: " + "; ".join(errors))
