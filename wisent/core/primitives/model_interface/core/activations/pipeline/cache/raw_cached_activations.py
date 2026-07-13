"""
RawCachedActivations for storing full hidden state sequences.

Stores complete sequences so any extraction strategy in the same text family
can be applied without re-running the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import torch

from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy, extract_activation
from .cached_activations import CachedActivations, get_strategy_text_family
from ..pair_identity import PairId, validate_pair_id


@dataclass
class RawPairData:
    """Full hidden states plus the exact forwarded token sequence for one pair."""

    pair_id: PairId
    stable_id: str
    pos_hidden_states: Dict[str, torch.Tensor]
    neg_hidden_states: Dict[str, torch.Tensor]
    pos_answer_text: str
    neg_answer_text: str
    pos_input_ids: torch.Tensor
    neg_input_ids: torch.Tensor
    pos_attention_mask: torch.Tensor
    neg_attention_mask: torch.Tensor
    pos_effective_length: int
    neg_effective_length: int
    pos_answer_onset: int
    neg_answer_onset: int
    pos_answer_end_exclusive: int
    neg_answer_end_exclusive: int

    def __post_init__(self) -> None:
        validate_pair_id(self.pair_id)
        _validate_identity("stable_id", self.stable_id)
        _validate_answer_text("positive", self.pos_answer_text)
        _validate_answer_text("negative", self.neg_answer_text)
        _validate_sequence(
            "positive",
            self.pos_input_ids,
            self.pos_attention_mask,
            self.pos_effective_length,
            self.pos_answer_onset,
            self.pos_answer_end_exclusive,
            self.pos_hidden_states,
        )
        _validate_sequence(
            "negative",
            self.neg_input_ids,
            self.neg_attention_mask,
            self.neg_effective_length,
            self.neg_answer_onset,
            self.neg_answer_end_exclusive,
            self.neg_hidden_states,
        )


def _validate_identity(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_revision(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} revision must be a lowercase 40-hex commit")


def _validate_answer_text(polarity: str, answer_text: str) -> None:
    if not isinstance(answer_text, str) or not answer_text:
        raise ValueError(f"{polarity} answer_text must be a non-empty string")


def _validate_sequence(
    polarity: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    effective_length: int,
    answer_onset: int,
    answer_end_exclusive: int,
    hidden_states: Dict[str, torch.Tensor],
) -> None:
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
        raise ValueError(f"{polarity} input_ids must be a 1-D tensor")
    if input_ids.numel() == 0:
        raise ValueError(f"{polarity} input_ids must not be empty")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 1:
        raise ValueError(f"{polarity} attention_mask must be a 1-D tensor")
    if attention_mask.shape != input_ids.shape:
        raise ValueError(f"{polarity} token IDs and attention mask must have equal length")
    if not bool(torch.all((attention_mask == 0) | (attention_mask == 1))):
        raise ValueError(f"{polarity} attention_mask must be binary")
    if isinstance(effective_length, bool) or not isinstance(effective_length, int):
        raise ValueError(f"{polarity} effective_length must be an integer")
    if effective_length != int(attention_mask.sum().item()) or effective_length <= 0:
        raise ValueError(f"{polarity} effective_length must equal the attended token count")
    if isinstance(answer_onset, bool) or not isinstance(answer_onset, int):
        raise ValueError(f"{polarity} answer_onset must be an integer")
    if not 0 <= answer_onset < input_ids.numel():
        raise ValueError(f"{polarity} answer_onset is outside the token sequence")
    if int(attention_mask[answer_onset].item()) != 1:
        raise ValueError(f"{polarity} answer_onset must identify an attended token")
    if isinstance(answer_end_exclusive, bool) or not isinstance(answer_end_exclusive, int):
        raise ValueError(f"{polarity} answer_end_exclusive must be an integer")
    if not answer_onset < answer_end_exclusive <= input_ids.numel():
        raise ValueError(f"{polarity} answer_end_exclusive is outside the answer span")
    if int(attention_mask[answer_end_exclusive - 1].item()) != 1:
        raise ValueError(f"{polarity} answer_end_exclusive must follow an attended token")
    if not isinstance(hidden_states, dict):
        raise ValueError(f"{polarity} hidden_states must be a layer mapping")
    for layer_name, hidden in hidden_states.items():
        if not isinstance(layer_name, str) or not isinstance(hidden, torch.Tensor):
            raise ValueError(f"{polarity} hidden_states must map layer names to tensors")
        if hidden.ndim != 2 or hidden.shape[0] != input_ids.numel():
            raise ValueError(
                f"{polarity} hidden-state sequence length must match token IDs"
            )


@dataclass
class RawCachedActivations:
    """
    Cache full hidden states per (benchmark, text_family).

    Stores complete sequences so any extraction strategy in the same
    text family can be applied without re-running the model.
    """
    benchmark: str
    text_family: str  # "chat", "role_play", "mc", "completion"
    model_name: str
    num_layers: int
    model_revision: str
    tokenizer_revision: str

    pairs: List[RawPairData] = field(default_factory=list)
    num_pairs: int = 0
    hidden_size: int = 0

    def __post_init__(self) -> None:
        _validate_revision("model", self.model_revision)
        _validate_revision("tokenizer", self.tokenizer_revision)

    def add_pair(
        self,
        *,
        pair_id: PairId,
        stable_id: str,
        pos_hidden_states: Dict[str, torch.Tensor],
        neg_hidden_states: Dict[str, torch.Tensor],
        pos_answer_text: str,
        neg_answer_text: str,
        pos_input_ids: torch.Tensor,
        neg_input_ids: torch.Tensor,
        pos_attention_mask: torch.Tensor,
        neg_attention_mask: torch.Tensor,
        pos_effective_length: int,
        neg_effective_length: int,
        pos_answer_onset: int,
        neg_answer_onset: int,
        pos_answer_end_exclusive: int,
        neg_answer_end_exclusive: int,
    ) -> None:
        """Add a validated raw pair without approximating prompt length."""
        pair_data = RawPairData(
            pair_id=pair_id,
            stable_id=stable_id,
            pos_hidden_states={k: v.clone() for k, v in pos_hidden_states.items()},
            neg_hidden_states={k: v.clone() for k, v in neg_hidden_states.items()},
            pos_answer_text=pos_answer_text,
            neg_answer_text=neg_answer_text,
            pos_input_ids=pos_input_ids.clone(),
            neg_input_ids=neg_input_ids.clone(),
            pos_attention_mask=pos_attention_mask.clone(),
            neg_attention_mask=neg_attention_mask.clone(),
            pos_effective_length=pos_effective_length,
            neg_effective_length=neg_effective_length,
            pos_answer_onset=pos_answer_onset,
            neg_answer_onset=neg_answer_onset,
            pos_answer_end_exclusive=pos_answer_end_exclusive,
            neg_answer_end_exclusive=neg_answer_end_exclusive,
        )
        self.pairs.append(pair_data)
        self.num_pairs = len(self.pairs)

        if self.hidden_size == 0 and pos_hidden_states:
            first_tensor = next(iter(pos_hidden_states.values()))
            self.hidden_size = first_tensor.shape[-1]

    def extract_with_strategy(
        self,
        strategy: ExtractionStrategy,
        tokenizer,
        layer: int | str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract activations using a specific strategy.

        Args:
            strategy: Extraction strategy (must be in same text family)
            tokenizer: Supplies the padding side only; it is never called
            layer: Layer to extract from

        Returns:
            Tuple of (pos_activations, neg_activations) each [num_pairs, hidden_size]
        """
        if get_strategy_text_family(strategy) != self.text_family:
            raise ValueError(
                f"Strategy {strategy.value} (family: {get_strategy_text_family(strategy)}) "
                f"incompatible with cached family: {self.text_family}"
            )

        layer_name = str(layer)
        padding_side = getattr(tokenizer, "padding_side", None)
        pos_acts = []
        neg_acts = []

        for pair_data in self.pairs:
            pos_hs = pair_data.pos_hidden_states.get(layer_name)
            if pos_hs is not None:
                pos_vec = extract_activation(
                    strategy,
                    pos_hs,
                    pair_data.pos_answer_text,
                    tokenizer,
                    pair_data.pos_answer_onset,
                    attention_mask=pair_data.pos_attention_mask,
                    effective_length=pair_data.pos_effective_length,
                    answer_onset=pair_data.pos_answer_onset,
                    answer_end_exclusive=pair_data.pos_answer_end_exclusive,
                    padding_side=padding_side,
                )
                pos_acts.append(pos_vec)

            neg_hs = pair_data.neg_hidden_states.get(layer_name)
            if neg_hs is not None:
                neg_vec = extract_activation(
                    strategy,
                    neg_hs,
                    pair_data.neg_answer_text,
                    tokenizer,
                    pair_data.neg_answer_onset,
                    attention_mask=pair_data.neg_attention_mask,
                    effective_length=pair_data.neg_effective_length,
                    answer_onset=pair_data.neg_answer_onset,
                    answer_end_exclusive=pair_data.neg_answer_end_exclusive,
                    padding_side=padding_side,
                )
                neg_acts.append(neg_vec)

        return torch.stack(pos_acts), torch.stack(neg_acts)

    def to_cached_activations(
        self,
        strategy: ExtractionStrategy,
        tokenizer,
    ) -> CachedActivations:
        """Convert to CachedActivations for a specific strategy."""
        cached = CachedActivations(
            benchmark=self.benchmark,
            strategy=strategy,
            model_name=self.model_name,
            num_layers=self.num_layers,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            hidden_size=self.hidden_size,
        )
        padding_side = getattr(tokenizer, "padding_side", None)

        for pair_data in self.pairs:
            pos_dict = {}
            neg_dict = {}

            for layer_name in pair_data.pos_hidden_states.keys():
                pos_hs = pair_data.pos_hidden_states[layer_name]
                neg_hs = pair_data.neg_hidden_states[layer_name]

                pos_dict[layer_name] = extract_activation(
                    strategy,
                    pos_hs,
                    pair_data.pos_answer_text,
                    tokenizer,
                    pair_data.pos_answer_onset,
                    attention_mask=pair_data.pos_attention_mask,
                    effective_length=pair_data.pos_effective_length,
                    answer_onset=pair_data.pos_answer_onset,
                    answer_end_exclusive=pair_data.pos_answer_end_exclusive,
                    padding_side=padding_side,
                )
                neg_dict[layer_name] = extract_activation(
                    strategy,
                    neg_hs,
                    pair_data.neg_answer_text,
                    tokenizer,
                    pair_data.neg_answer_onset,
                    attention_mask=pair_data.neg_attention_mask,
                    effective_length=pair_data.neg_effective_length,
                    answer_onset=pair_data.neg_answer_onset,
                    answer_end_exclusive=pair_data.neg_answer_end_exclusive,
                    padding_side=padding_side,
                )

            cached.pair_activations.append((pos_dict, neg_dict))

        cached.num_pairs = len(cached.pair_activations)
        return cached

    def get_available_layers(self) -> List[str]:
        """Get list of available layer names."""
        if not self.pairs:
            return []
        return list(self.pairs[0].pos_hidden_states.keys())
