"""Activation wrapper for classifier feature extraction."""

from typing import Any
import torch
from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
from wisent.core.utils.infra_tools.errors import InvalidValueError


class Activations:
    """Wrapper for activation tensors with extraction strategy.

    This class wraps activation tensors and provides methods to extract
    features for classifier input based on the specified extraction strategy.
    """

    def __init__(self, tensor: torch.Tensor, layer: Any, extraction_strategy: ExtractionStrategy = ExtractionStrategy.default(), weighted_decay: float = 0.1):
        """Initialize Activations wrapper.

        Args:
            tensor: Activation tensor (typically shape [batch, seq_len, hidden_dim])
            layer: Layer object containing layer metadata
            extraction_strategy: The extraction strategy to use
            weighted_decay: Decay factor for CHAT_WEIGHTED strategy
        """
        self.tensor = tensor
        self.layer = layer
        self.extraction_strategy = extraction_strategy
        self.weighted_decay = weighted_decay

    def extract_features_for_classifier(self) -> torch.Tensor:
        """Extract features from activations for classifier input.

        Aggregates the activation tensor based on the specified strategy
        to produce a single feature vector suitable for classifier input.

        Returns:
            torch.Tensor: Aggregated features (typically shape [hidden_dim])
        """
        if self.tensor is None:
            raise InvalidValueError(param="tensor", reason="Cannot extract features from None tensor")

        # Ensure tensor is 3D: [batch, seq_len, hidden_dim]
        if len(self.tensor.shape) == 2:
            tensor = self.tensor.unsqueeze(0)
        else:
            tensor = self.tensor

        # Apply extraction strategy
        strategy = self.extraction_strategy
        
        if strategy in (ExtractionStrategy.CHAT_MEAN,):
            features = tensor.mean(dim=1).squeeze(0)
        elif strategy in (ExtractionStrategy.CHAT_LAST, ExtractionStrategy.ROLE_PLAY, ExtractionStrategy.MC_BALANCED):
            features = tensor[:, -1, :].squeeze(0)
        elif strategy == ExtractionStrategy.CHAT_FIRST:
            features = tensor[:, 0, :].squeeze(0)
        elif strategy == ExtractionStrategy.CHAT_MAX_NORM:
            norms = torch.norm(tensor, dim=2)
            max_idx = torch.argmax(norms, dim=1)
            features = tensor[0, max_idx[0], :]
        elif strategy == ExtractionStrategy.CHAT_WEIGHTED:
            seq_len = tensor.shape[1]
            weights = torch.exp(-torch.arange(seq_len, dtype=tensor.dtype, device=tensor.device) * self.weighted_decay)
            weights = weights / weights.sum()
            features = (tensor * weights.unsqueeze(0).unsqueeze(2)).sum(dim=1).squeeze(0)
        else:
            raise InvalidValueError(param="extraction_strategy", reason=f"Unknown extraction strategy: {strategy}")

        return features

    def cpu(self):
        """Move tensor to CPU."""
        if self.tensor is not None:
            self.tensor = self.tensor.cpu()
        return self

    def detach(self):
        """Detach tensor from computation graph."""
        if self.tensor is not None:
            self.tensor = self.tensor.detach()
        return self
