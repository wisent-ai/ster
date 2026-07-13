"""GROM neural network components: gating, intensity, direction routing."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from wisent.core.utils.config_tools.constants import (
    COMBO_OFFSET,
    RECURSION_INITIAL_DEPTH,
)


def _require_finite_gating_tensor(value: torch.Tensor, name: str) -> None:
    """Reject corrupted gating inputs without changing the autograd graph."""
    finite_mask = torch.isfinite(value.detach())
    if not bool(finite_mask.all()):
        non_finite_count = value.numel() - int(finite_mask.sum().item())
        raise RuntimeError(
            "GROM gating numerical invariant failed: "
            f"{name} contains {non_finite_count}/{value.numel()} non-finite values"
        )


class GatingNetwork(nn.Module):
    """Learned gating network that predicts whether steering should activate."""

    def __init__(self, input_dim: int, hidden_dim: int, *, shrink_factor: int):
        super().__init__()
        shrunk_dim = hidden_dim // shrink_factor
        self.shrink_factor = shrink_factor
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, shrunk_dim),
            nn.GELU(),
            nn.Linear(shrunk_dim, COMBO_OFFSET),
        )

    def forward(self, h: torch.Tensor, temperature: float) -> torch.Tensor:
        """Predict a finite gate in ``[0, 1]`` or fail before it is consumed."""
        try:
            temperature_value = float(temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "GROM gate_temperature must be a finite number greater than zero; "
                f"got {temperature!r}"
            ) from exc
        if not math.isfinite(temperature_value) or temperature_value <= 0.0:
            raise ValueError(
                "GROM gate_temperature must be finite and greater than zero; "
                f"got {temperature!r}"
            )
        _require_finite_gating_tensor(h, "gate input activations")
        if h.dim() == COMBO_OFFSET:
            h = h.unsqueeze(RECURSION_INITIAL_DEPTH)
        logit = self.net(h).squeeze(-COMBO_OFFSET)
        _require_finite_gating_tensor(logit, "gate logits")
        gate = torch.sigmoid(logit / temperature_value)
        _require_finite_gating_tensor(gate, "gate output")
        detached_gate = gate.detach()
        if bool(((detached_gate < 0.0) | (detached_gate > 1.0)).any()):
            raise RuntimeError(
                "GROM gating numerical invariant failed: sigmoid output is outside [0, 1]; "
                f"minimum={detached_gate.min().item():.6g}, "
                f"maximum={detached_gate.max().item():.6g}"
            )
        return gate


class IntensityNetwork(nn.Module):
    """Learned intensity network that predicts per-layer steering strength."""

    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int, max_alpha: float):
        super().__init__()
        self.max_alpha = max_alpha
        self.num_layers = num_layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict per-layer intensity."""
        if h.dim() == COMBO_OFFSET:
            h = h.unsqueeze(RECURSION_INITIAL_DEPTH)
        raw = self.net(h)
        return torch.sigmoid(raw) * self.max_alpha


class DirectionWeightNetwork(nn.Module):
    """Learned network that predicts weights for combining directions in manifold."""

    def __init__(self, input_dim: int, num_directions: int, hidden_dim: int):
        super().__init__()
        self.num_directions = num_directions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_directions),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict direction weights."""
        if h.dim() == COMBO_OFFSET:
            h = h.unsqueeze(RECURSION_INITIAL_DEPTH)
        logits = self.net(h)
        return F.softmax(logits, dim=-COMBO_OFFSET)
