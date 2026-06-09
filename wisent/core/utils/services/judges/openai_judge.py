"""OpenAI-compatible chat-completions judge (AxBench reference: gpt-4o-mini).

Talks to ``$OPENAI_BASE_URL`` (default https://api.openai.com/v1) with
``$OPENAI_API_KEY``. Missing key or any non-200 response raises — the judge
never substitutes another provider.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import requests

from wisent.core.utils.services.judges.base import BaseJudge

__all__ = ["OpenAIJudge"]

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIJudge(BaseJudge):
    """Judge over an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        *,
        model: str,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"OPENAI_API_KEY is not set but the 'openai:{model}' judge "
                "was requested. Export OPENAI_API_KEY, or pass an explicit "
                "HuggingFace model id as --judge-model for a local judge."
            )
        self.model = model
        self.api_key = api_key
        self.base_url = os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _complete_one(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            },
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI judge request failed: HTTP {response.status_code} "
                f"from {self.base_url}/chat/completions: {response.text}"
            )
        payload = response.json()
        choices = payload.get("choices")
        if not choices:
            raise RuntimeError(f"OpenAI judge returned no choices: {payload}")
        return choices[0]["message"]["content"]

    def complete_batch(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=self.batch_size) as pool:
            return list(pool.map(self._complete_one, prompts))
