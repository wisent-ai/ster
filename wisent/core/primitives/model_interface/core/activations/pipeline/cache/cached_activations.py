"""
CachedActivations dataclass and strategy text family mapping.

Stores extracted activation vectors per (benchmark, strategy) combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import torch

from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations
from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy


def get_strategy_text_family(strategy: ExtractionStrategy) -> str:
    """
    Get the text family for a strategy.

    Strategies in the same family use the same input text (prompt construction),
    they only differ in which token(s) they extract from the output.
    """
    if strategy in (
        ExtractionStrategy.CHAT_MEAN,
        ExtractionStrategy.CHAT_FIRST,
        ExtractionStrategy.CHAT_LAST,
        ExtractionStrategy.CHAT_MAX_NORM,
        ExtractionStrategy.CHAT_WEIGHTED,
    ):
        return "chat"
    elif strategy == ExtractionStrategy.ROLE_PLAY:
        return "role_play"
    elif strategy in (
        ExtractionStrategy.MC_BALANCED, ExtractionStrategy.MC_COMPLETION
    ):
        return "mc"
    elif strategy in (
        ExtractionStrategy.COMPLETION_LAST, ExtractionStrategy.COMPLETION_MEAN
    ):
        return "completion"
    else:
        return strategy.value


def validate_revision_identity(
    model_revision: str | None,
    tokenizer_revision: str | None,
) -> None:
    """Require either an unpinned identity or two exact commit SHAs."""
    if model_revision is None and tokenizer_revision is None:
        return
    for name, revision in (
        ("model_revision", model_revision),
        ("tokenizer_revision", tokenizer_revision),
    ):
        if (
            revision is None
            or not isinstance(revision, str)
            or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
        ):
            raise ValueError(
                "activation cache revision identity requires a complete pair of "
                f"lowercase 40-character commit SHAs; invalid {name}"
            )


@dataclass
class CachedActivations:
    """
    Cached activations for a single (benchmark, strategy) pair.

    Contains activations for ALL layers for all pairs.
    Layer combinations can be extracted without re-running the model.
    """
    benchmark: str
    strategy: ExtractionStrategy
    model_name: str
    num_layers: int
    model_revision: str | None = None
    tokenizer_revision: str | None = None

    pair_activations: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = field(default_factory=list)
    num_pairs: int = 0
    hidden_size: int = 0

    def __post_init__(self) -> None:
        validate_revision_identity(self.model_revision, self.tokenizer_revision)

    def add_pair(self, positive: LayerActivations, negative: LayerActivations) -> None:
        """Add activations for a contrastive pair."""
        pos_dict = {k: v.clone() for k, v in positive.items() if v is not None}
        neg_dict = {k: v.clone() for k, v in negative.items() if v is not None}
        self.pair_activations.append((pos_dict, neg_dict))
        self.num_pairs = len(self.pair_activations)

        if self.hidden_size == 0 and pos_dict:
            first_tensor = next(iter(pos_dict.values()))
            self.hidden_size = first_tensor.shape[-1]

    def get_layer_subset(self, layers: List[int]) -> "CachedActivations":
        """Get a new CachedActivations with only the specified layers."""
        layer_names = [str(l) for l in layers]

        new_pairs = []
        for pos_dict, neg_dict in self.pair_activations:
            new_pos = {k: v for k, v in pos_dict.items() if k in layer_names}
            new_neg = {k: v for k, v in neg_dict.items() if k in layer_names}
            new_pairs.append((new_pos, new_neg))

        result = CachedActivations(
            benchmark=self.benchmark,
            strategy=self.strategy,
            model_name=self.model_name,
            num_layers=len(layers),
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            hidden_size=self.hidden_size,
        )
        result.pair_activations = new_pairs
        result.num_pairs = len(new_pairs)
        return result

    def get_available_layers(self) -> List[str]:
        """Get list of available layer names."""
        if not self.pair_activations:
            return []
        return list(self.pair_activations[0][0].keys())

    def get_positive_activations(self, layer: int | str) -> torch.Tensor:
        """Get stacked positive activations for a single layer. [num_pairs, hidden_size]"""
        layer_name = str(layer)
        tensors = [pos[layer_name] for pos, _ in self.pair_activations if layer_name in pos]
        if not tensors:
            raise KeyError(f"Layer {layer_name} not found. Available: {self.get_available_layers()}")
        return torch.stack(tensors, dim=0)

    def get_negative_activations(self, layer: int | str) -> torch.Tensor:
        """Get stacked negative activations for a single layer. [num_pairs, hidden_size]"""
        layer_name = str(layer)
        tensors = [neg[layer_name] for _, neg in self.pair_activations if layer_name in neg]
        if not tensors:
            raise KeyError(f"Layer {layer_name} not found. Available: {self.get_available_layers()}")
        return torch.stack(tensors, dim=0)

    def get_diff_activations(self, layer: int | str) -> torch.Tensor:
        """Get positive - negative activation differences. [num_pairs, hidden_size]"""
        return self.get_positive_activations(layer) - self.get_negative_activations(layer)

    def get_all_layers_diff(self) -> Dict[str, torch.Tensor]:
        """Get activation differences for all layers."""
        result = {}
        if not self.pair_activations:
            return result

        layer_names = list(self.pair_activations[0][0].keys())
        for layer_name in layer_names:
            pos_tensors = []
            neg_tensors = []
            for pos, neg in self.pair_activations:
                if layer_name in pos and layer_name in neg:
                    pos_tensors.append(pos[layer_name])
                    neg_tensors.append(neg[layer_name])
            if pos_tensors:
                result[layer_name] = torch.stack(pos_tensors) - torch.stack(neg_tensors)
        return result

    def to_device(self, device: str) -> "CachedActivations":
        """Move all tensors to a device."""
        new_pairs = []
        for pos, neg in self.pair_activations:
            new_pos = {k: v.to(device) for k, v in pos.items()}
            new_neg = {k: v.to(device) for k, v in neg.items()}
            new_pairs.append((new_pos, new_neg))

        result = CachedActivations(
            benchmark=self.benchmark,
            strategy=self.strategy,
            model_name=self.model_name,
            num_layers=self.num_layers,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            hidden_size=self.hidden_size,
        )
        result.pair_activations = new_pairs
        result.num_pairs = self.num_pairs
        return result
