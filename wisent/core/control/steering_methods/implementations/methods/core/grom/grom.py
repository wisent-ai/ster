"""
GROM - Total Integrated Targeted Activation Navigation.

The most powerful steering method that jointly optimizes:
- WHAT to steer (multi-directional manifold discovery)
- WHEN to steer (learned gating network)
- WHERE to steer (learned layer importance)
- HOW MUCH to steer (input-dependent intensity prediction)

Combines insights from:
- TECZA: Gradient-optimized multi-directional discovery
- TETNO: Conditional gating and layer-adaptive steering
- DAC/CAST: Dynamic intensity modulation

Key innovations:
1. Learned gating head (MLP) instead of fixed cosine similarity
2. Per-input intensity prediction instead of global strength
3. Direction weighting within the manifold
4. Joint end-to-end optimization of all components
5. Sparse layer activation for minimal side effects
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

from wisent.core.control.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.utils.infra_tools.errors import InsufficientDataError
from wisent.core.control.steering_methods.configs.optimal import get_optimal
from wisent.core.control.steering_methods.methods.grom._config import (
    GROMConfig, GatingNetwork, IntensityNetwork,
    DirectionWeightNetwork, GeometryAdaptation,
)

__all__ = [
    "GROMMethod",
    "GROMConfig",
    "GROMResult",
    "GeometryAdaptation",
    "GatingNetwork",
    "IntensityNetwork",
]


def _require(name: str, kwargs: dict):
    """Raise ValueError if a required hyperparameter is missing."""
    if name not in kwargs:
        raise ValueError(
            f"Parameter '{name}' is required. "
            f"Run 'wisent optimize-steering auto' first, or pass it explicitly."
        )
    return kwargs[name]


_GROM_REQUIRED_KWARGS = [
    "num_directions", "optimization_steps", "learning_rate",
    "warmup_steps", "behavior_weight", "retain_weight",
    "sparse_weight", "smooth_weight", "independence_weight",
    "max_alpha", "gate_temperature", "min_cosine_similarity",
    "max_cosine_similarity", "weight_decay", "max_grad_norm",
    "eta_min_factor", "linear_threshold", "adapt_cone_threshold",
    "adapt_manifold_threshold", "adapt_linear_directions",
    "adapt_complex_directions", "adapt_max_directions",
    "significant_directions_default", "min_adapted_directions",
    "caa_similarity_skip", "contrastive_margin", "contrastive_weight",
    "utility_weight", "concentration_weight", "gate_warmup_weight",
    "caa_alignment_weight", "gate_dim_min", "gate_dim_max",
    "gate_dim_divisor", "intensity_dim_min", "intensity_dim_max",
    "intensity_dim_divisor", "gate_shrink_factor",
    "create_noise_scale", "create_gate_threshold",
]


@dataclass
class GROMResult:
    """Result containing all GROM components."""
    
    # Manifold
    directions: Dict[LayerName, torch.Tensor]
    """Per-layer directions [num_directions, hidden_dim]."""
    
    # Networks
    gate_network: Optional[GatingNetwork]
    """Learned gating network (None if disabled due to linear structure)."""
    
    intensity_network: IntensityNetwork
    """Learned intensity prediction network."""
    
    direction_weights: Dict[LayerName, torch.Tensor]
    """Learned static direction weights per layer [num_directions]."""
    
    # Layer mapping
    layer_order: List[LayerName]
    """Ordered list of layer names (for intensity network output indexing)."""

    sensor_layer: LayerName
    """Exact layer supplying the gate and intensity network input."""
    
    # Trained hyperparameters (stored from training config)
    gate_temperature: float
    """Temperature for gate sigmoid, stored from training."""

    # Metadata
    metadata: Dict[str, Any]
    """Training metadata and diagnostics."""

    # Geometry analysis
    geometry_adaptation: Optional[GeometryAdaptation] = None
    """Results from geometry analysis (if adapt_to_geometry was True)."""
    
    def get_effective_direction(self, layer: LayerName, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the effective steering direction for a layer.
        
        If h is provided, uses direction weight network (if available).
        Otherwise uses static learned weights.
        
        Args:
            layer: Layer name
            h: Optional hidden state for dynamic weighting
            
        Returns:
            Weighted direction vector [hidden_dim]
        """
        dirs = self.directions[layer]  # [K, H]
        weights = self.direction_weights[layer]  # [K]
        
        # Weighted sum of directions
        effective = (weights.unsqueeze(-1) * dirs).sum(dim=0)  # [H]
        return F.normalize(effective, p=2, dim=-1)
    
    def predict_gate(self, h: torch.Tensor, temperature: float = None) -> torch.Tensor:
        """Predict gate value from hidden state.

        Uses stored gate_temperature from training if not overridden.
        """
        if temperature is None:
            temperature = self.gate_temperature
        if self.gate_network is None:
            # No gating - always return 1.0 (always steer)
            if h.dim() == 1:
                return torch.ones(1, device=h.device)
            return torch.ones(h.shape[0], device=h.device)
        return self.gate_network(h, temperature)
    
    def predict_intensity(self, h: torch.Tensor) -> Dict[LayerName, torch.Tensor]:
        """
        Predict per-layer intensity from hidden state.
        
        Returns dict mapping layer name to intensity value.
        """
        intensities = self.intensity_network(h)  # [batch, num_layers]
        if intensities.dim() == 1:
            intensities = intensities.unsqueeze(0)
        
        result = {}
        for i, layer in enumerate(self.layer_order):
            result[layer] = intensities[:, i]
        return result
    
    def apply_steering(
        self,
        activations: Dict[LayerName, torch.Tensor],
        sensor_hidden: torch.Tensor,
    ) -> Dict[LayerName, torch.Tensor]:
        """
        Apply full GROM steering to activations.
        
        Args:
            activations: Dict of layer activations
            sensor_hidden: Hidden state at sensor layer for gating/intensity
            
        Returns:
            Steered activations
        """
        # Get gate and intensities
        gate = self.predict_gate(sensor_hidden)
        intensities = self.predict_intensity(sensor_hidden)
        
        steered = {}
        for layer, h in activations.items():
            if layer in self.directions:
                direction = self.get_effective_direction(layer)
                intensity = intensities.get(layer, torch.ones(1))
                
                # Apply steering: h' = h + gate * intensity * direction
                if h.dim() == 1:
                    steered[layer] = h + gate.squeeze() * intensity.squeeze() * direction
                else:
                    steered[layer] = h + gate.unsqueeze(-1) * intensity.unsqueeze(-1) * direction
            else:
                steered[layer] = h
        
        return steered


class GROMMethod(BaseSteeringMethod):
    """
    GROM - Total Integrated Targeted Activation Navigation.
    
    The most powerful steering method combining:
    - Multi-directional manifold discovery (TECZA-style)
    - Learned gating network (improved TETNO)
    - Per-input intensity prediction
    - Direction weighting within manifold
    - Joint end-to-end optimization
    
    Usage:
        method = GROMMethod(num_directions=5, optimization_steps=200)
        result = method.train_grom(pair_set)
        
        # At inference:
        steered = result.apply_steering(activations, sensor_hidden)
    """
    
    name = "grom"
    description = "Total Integrated Targeted Activation Navigation - joint optimization of manifold, gating, and intensity"
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if "config" in kwargs and isinstance(kwargs["config"], GROMConfig):
            self.config = kwargs["config"]
        else:
            cfg = {k: _require(k, kwargs) for k in _GROM_REQUIRED_KWARGS}
            cfg["steering_layers"] = kwargs.get("steering_layers")
            cfg["sensor_layer"] = kwargs.get("sensor_layer")
            cfg["num_layers"] = kwargs.get("num_layers")
            cfg["gate_hidden_dim"] = kwargs.get("gate_hidden_dim")
            cfg["intensity_hidden_dim"] = kwargs.get("intensity_hidden_dim")
            cfg["use_caa_init"] = kwargs.get("use_caa_init", get_optimal("use_caa_init"))
            cfg["normalize"] = kwargs.get("normalize", get_optimal("normalize"))
            self.config = GROMConfig(**cfg)
        self.log_interval: int = _require("log_interval", kwargs)
        self._training_logs: List[Dict[str, float]] = []
    
    def train(self, pair_set: ContrastivePairSet) -> LayerActivations:
        """
        Train GROM (simplified interface for compatibility).
        
        Returns primary directions as LayerActivations.
        """
        result = self.train_grom(pair_set)
        
        # Return effective directions
        primary_map: RawActivationMap = {}
        for layer in result.directions:
            primary_map[layer] = result.get_effective_direction(layer)
        
        dtype = self.kwargs.get("dtype", None)
        return LayerActivations(primary_map, dtype=dtype)
    
    def train_grom(self, pair_set: ContrastivePairSet) -> "GROMResult":
        """Full GROM training with all components."""
        from wisent.core.control.steering_methods.methods.grom.impl._optimization import train_grom_impl
        return train_grom_impl(self, pair_set)

    def _analyze_and_adapt_geometry(self, *a, **kw):
        from wisent.core.control.steering_methods.methods.grom.impl._training import _analyze_and_adapt_geometry_impl
        return _analyze_and_adapt_geometry_impl(self, *a, **kw)

    def _initialize_directions(self, *a, **kw):
        from wisent.core.control.steering_methods.methods.grom.impl._training import _initialize_directions_impl
        return _initialize_directions_impl(self, *a, **kw)

    def _prepare_data_tensors(self, *a, **kw):
        from wisent.core.control.steering_methods.methods.grom.impl._training import _prepare_data_tensors_impl
        return _prepare_data_tensors_impl(self, *a, **kw)

    def _find_sensor_layer(self, *a, **kw):
        from wisent.core.control.steering_methods.methods.grom.impl._training import _find_sensor_layer_impl
        return _find_sensor_layer_impl(self, *a, **kw)

    def _joint_optimization(self, *a, **kw):
        from wisent.core.control.steering_methods.methods.grom.impl._optimization import _joint_optimization_impl
        return _joint_optimization_impl(self, *a, **kw)

    def _compute_grom_loss(self, *a, **kw):
        from wisent.core.control.steering_methods.methods.grom.impl._loss import _compute_grom_loss_impl
        return _compute_grom_loss_impl(self, *a, **kw)

    def _apply_direction_constraints(self, directions):
        from wisent.core.control.steering_methods.methods.grom.impl._loss import _apply_direction_constraints_impl
        return _apply_direction_constraints_impl(self, directions)

    def _collect_from_set(self, *a, **kw):
        from wisent.core.control.steering_methods.methods.grom.impl._loss import _collect_from_set_impl
        return _collect_from_set_impl(self, *a, **kw)

    def get_training_logs(self) -> List[Dict[str, Any]]:
        from wisent.core.control.steering_methods.methods.grom.impl._loss import get_training_logs_impl
        return get_training_logs_impl(self)
