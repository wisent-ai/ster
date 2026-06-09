"""Local HuggingFace judge: runs an explicitly named open model via WisentModel.

Each AxBench rubric prompt is sent as a single user chat message; the raw
completion text is returned for rating parsing. Greedy decoding is used when
temperature is 0 so ratings are deterministic.
"""

from __future__ import annotations

from typing import List

from wisent.core.utils.services.judges.base import BaseJudge

__all__ = ["HFJudge"]


class HFJudge(BaseJudge):
    """Judge backed by a locally loaded HF causal LM."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str | None,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        from wisent.core.primitives.models.wisent_model import WisentModel

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = WisentModel(model_name, device=device)

    def complete_batch(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        completions: List[str] = []
        gen_kwargs: dict = {"max_new_tokens": self.max_new_tokens}
        if self.temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start:start + self.batch_size]
            messages = [[{"role": "user", "content": p}] for p in chunk]
            texts = self.model.generate(inputs=messages, **gen_kwargs)
            completions.extend(texts)
        if len(completions) != len(prompts):
            raise RuntimeError(
                f"HF judge returned {len(completions)} completions for "
                f"{len(prompts)} prompts."
            )
        return completions
