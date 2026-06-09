"""Judge selection: OpenAI-compatible or local HF judge behind one interface.

The AxBench reference judge is OpenAI gpt-4o-mini. Selection rule (exactly
one judge is constructed; the other is never substituted):
  - judge_model starting with "openai:" -> OpenAIJudge over the chat
    completions API; raises immediately when OPENAI_API_KEY is unset.
  - any other non-empty judge_model    -> HFJudge running that HF model
    locally via WisentModel.
  - empty/None judge_model             -> error naming both options.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

__all__ = ["BaseJudge", "make_judge", "OPENAI_JUDGE_PREFIX"]

OPENAI_JUDGE_PREFIX = "openai:"


class BaseJudge(ABC):
    """A judge maps rubric prompts to raw completion texts."""

    @abstractmethod
    def complete_batch(self, prompts: List[str]) -> List[str]:
        """Return one completion per prompt, in order."""


def make_judge(
    judge_model: str | None,
    *,
    device: str | None = None,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> BaseJudge:
    """Construct the judge selected by ``judge_model``.

    arguments:
        judge_model: "openai:<model>" for the OpenAI-compatible API judge
            (e.g. "openai:gpt-4o-mini", the AxBench reference judge), or a
            HuggingFace model id for a local judge.
        device: device for the local HF judge (unused by the OpenAI judge).
        batch_size: concurrent requests (OpenAI) / generation batch (HF).
        max_new_tokens: completion budget per rubric prompt.
        temperature: sampling temperature for judge completions.
    """
    if not judge_model:
        raise ValueError(
            "A judge model is required. Pass --judge-model "
            f"'{OPENAI_JUDGE_PREFIX}gpt-4o-mini' (AxBench reference judge, "
            "requires OPENAI_API_KEY) or an explicit HuggingFace model id "
            "for a local judge."
        )
    if judge_model.startswith(OPENAI_JUDGE_PREFIX):
        from wisent.core.utils.services.judges.openai_judge import OpenAIJudge

        model = judge_model[len(OPENAI_JUDGE_PREFIX):]
        if not model:
            raise ValueError(
                f"'{OPENAI_JUDGE_PREFIX}' judge requires a model name, "
                f"e.g. '{OPENAI_JUDGE_PREFIX}gpt-4o-mini'."
            )
        return OpenAIJudge(
            model=model,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    from wisent.core.utils.services.judges.hf_judge import HFJudge

    return HFJudge(
        model_name=judge_model,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
