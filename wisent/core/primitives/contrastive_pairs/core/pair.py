from __future__ import annotations

from dataclasses import dataclass, replace 

from wisent.core.primitives.contrastive_pairs.core.atoms import AtomContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.core.utils.infra_tools.errors import PromptMustBeStringError

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, RawActivationMap

__all__ = [
    "ContrastivePair",
]

@dataclass(frozen=True, slots=True)
class ContrastivePair(AtomContrastivePair):
    """A single contrastive pair: (prompt, positive_response, negative_response).

    attributes:
        prompt: The input prompt string. For example, a question or instruction.
        positive_response: The response considered "harmless" or "correct".
        negative_response: The response considered "harmful" or "incorrect".
        label: Optional label for the pair, e.g., "toxic", "biased", etc.
        trait_description: Optional description of the trait being tested. For example, "hallucinatory", "toxic", "biased", etc.
        metadata: Optional dictionary for task-specific data (e.g., test_code, entry_point for coding tasks).
    """

    prompt: str
    positive_response: PositiveResponse
    negative_response: NegativeResponse
    label: str | None = None
    trait_description: str | None = None
    metadata: dict | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise PromptMustBeStringError()
        if not isinstance(self.positive_response, PositiveResponse):
            raise TypeError("`positive_response` must be PositiveResponse.")
        if not isinstance(self.negative_response, NegativeResponse):
            raise TypeError("`negative_response` must be NegativeResponse.")
        
    def __repr__(self) -> str:
        return (
            f"ContrastivePair(\n"
            f"  prompt={self.prompt!r},\n"
            f"  positive_response={self.positive_response!r},\n"
            f"  negative_response={self.negative_response!r},\n"
            f"  label={self.label!r},\n"
            f"  trait_description={self.trait_description!r}\n"
            f")"
        )

    def with_activations(
        self,
        positive: LayerActivations | RawActivationMap | None,
        negative: LayerActivations | RawActivationMap | None,
    ) -> ContrastivePair:
        """Return a copy of the ContrastivePair with updated activations.

        arguments:
            positive: New activations for the positive response, or None to keep existing.
            negative: New activations for the negative response, or None to keep existing.

        returns:
            A new ContrastivePair with updated activations.

        example:
        >>> pair = ContrastivePair(
        ...     prompt="Is the sky blue?",
        ...     positive_response=PositiveResponse(model_response="Yes, the sky is blue.", layers_activations=None),
        ...     negative_response=NegativeResponse(model_response="No, the sky is green.", layers_activations=None),
        ... )
        >>> new_positive_activations = {"blocks.0.mlp": torch.randn(2, 4)}
        >>> new_negative_activations = {"blocks.0.mlp": torch.randn(2, 4)}
        >>> updated_pair = pair.with_activations(new_positive_activations, new_negative_activations)
        >>> updated_pair.positive_response.layers_activations
        LayerActivations({'blocks.0.mlp': tensor([[ 0.1234, -0.5678, ...]])})
        >>> updated_pair.negative_response.layers_activations
        LayerActivations({'blocks.0.mlp': tensor([[ 0.8765, -0.4321, ...]])})
        """
        new_pos = self.positive_response if positive is None else self.positive_response.with_activations(positive)
        new_neg = self.negative_response if negative is None else self.negative_response.with_activations(negative)
        return replace(self, positive_response=new_pos, negative_response=new_neg)

    def to_dict(self) -> dict[str, str | dict[str, RawActivationMap | str | None] | None]:
        """Return a plain dict representation of this ContrastivePair.
        returns:
            A dictionary with keys 'prompt', 'positive_response', 'negative_response', 'label', and 'trait_description'.

        example:
         >>> pair = ContrastivePair(
         ...     prompt="Is the sky blue?",
         ...     positive_response=PositiveResponse(
         ...         model_response="Yes, the sky is blue.",
         ...         layers_activations={"blocks.0.mlp": torch.randn(2, 4)},
         ...         label="harmless"
         ...     ),
         ...     negative_response=NegativeResponse(
         ...         model_response="No, the sky is green.",
         ...         layers_activations={"blocks.0.mlp": torch.randn(2, 4)},
         ...         label="toxic"
         ...     ),
         ...     label="color_question",
         ...     trait_description="hallucinatory"
         ... )
         >>> pair_dict = pair.to_dict()
         >>> print(pair_dict)
         {
             "prompt": "Is the sky blue?",
             "positive_response": {
                 "model_response": "Yes, the sky is blue.",
                 "layers_activations": {"blocks.0.mlp": tensor([[ 0.1234, -0.5678, ...]])},
                 "label": "harmless"
             },
             "negative_response": {
                 "model_response": "No, the sky is green.",
                 "layers_activations": {"blocks.0.mlp": tensor([[ 0.8765, -0.4321, ...]])},
                 "label": "toxic"
             },
             "label": "color_question",
             "trait_description": "hallucinatory"
         }       
        """

        data: dict[str, str | dict[str, RawActivationMap | str | None] | None] = {
            "prompt": self.prompt,
            "positive_response": self.positive_response.to_dict(),
            "negative_response": self.negative_response.to_dict(),
            "label": self.label,
            "trait_description": self.trait_description,
            "metadata": self.metadata,
            "stable_id": self.stable_id(),
        }
        return data

    def stable_id(self) -> str:
        """Stable global ID derived from pair content. SHA-256 of
        (prompt + positive_response.text + negative_response.text), truncated.

        Used to track which pairs have had activations extracted across runs
        so re-runs don't redo work and partial-coverage jobs (e.g. --limit 500
        on a 1997-pair task) can incrementally extend coverage to the
        remaining 1497 pairs without overwriting existing shards. Earlier the
        only "pair_id" was the positional index inside a single pairs.json,
        which carries no meaning across runs.
        """
        import hashlib
        pos_text = getattr(self.positive_response, "text", None) \
                   or getattr(self.positive_response, "model_response", "") or ""
        neg_text = getattr(self.negative_response, "text", None) \
                   or getattr(self.negative_response, "model_response", "") or ""
        h = hashlib.sha256(
            (self.prompt + "\x1f" + pos_text + "\x1f" + neg_text).encode("utf-8")
        )
        return h.hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: dict[str, str | RawActivationMap | None]) -> ContrastivePair:
        ''' Create a ContrastivePair from a plain dict.

        arguments:
            data: A dictionary with keys 'prompt', 'positive_response', 'negative_response', 'label', and 'trait_description'.
                    'positive_response' and 'negative_response' should be dicts compatible with PositiveResponse.from_dict and NegativeResponse.from_dict respectively. 

        example:
         >>> data = {
         ...     "prompt": "Is the sky blue?",
         ...     "positive_response": {
         ...         "model_response": "Yes, the sky is blue.",
         ...         "layers_activations": {"blocks.0.mlp": torch.randn(2, 4)},
         ...         "label": "harmless"
         ...     },
         ...     "negative_response": {
         ...         "model_response": "No, the sky is green.",
         ...         "layers_activations": {"blocks.0.mlp": torch.randn(2, 4)},
         ...         "label": "toxic"
         ...     },
         ...     "label": "color_question",
         ...     "trait_description": "hallucinatory"
         ... }
         >>> pair = ContrastivePair.from_dict(data)
         >>> print(pair)
         ContrastivePair(
             prompt='Is the sky blue?',
             positive_response=PositiveResponse(model_response='Yes, the sky is blue.', layers_activations=LayerActivations(...), label='harmless'),
             negative_response=NegativeResponse(model_response='No, the sky is green.', layers_activations=LayerActivations(...), label='toxic'),
             label='color_question',
             trait_description='hallucinatory'
         )          
        '''

        from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse

        return cls(
            prompt=str(data["prompt"]),
            positive_response=PositiveResponse.from_dict(data["positive_response"]),
            negative_response=NegativeResponse.from_dict(data["negative_response"]),
            label=data.get("label"),
            trait_description=data.get("trait_description"),
            metadata=data.get("metadata"),
        )