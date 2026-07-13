"""
Unified Steering Objects for all steering methods.

Each steering method produces a SteeringObject that contains:
- The steering vectors/directions
- Method-specific components (gates, networks, thresholds)
- Metadata about training
- Methods to apply steering at inference time

This preserves the full structure of complex methods like TETNO and GROM
rather than flattening everything to simple vectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from wisent.core.utils.config_tools.constants import JSON_INDENT, BASE_CLASS_NAME
from wisent.core.control.steering_methods.configs.optimal import get_optimal


LayerName = str


@dataclass
class SteeringObjectMetadata:
    """Common metadata for all steering objects."""
    method: str
    model_name: str
    benchmark: str
    category: str
    extraction_strategy: str
    num_pairs: int
    layers: List[int]
    hidden_dim: int
    created_at: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    # Calibration data: average hidden state norm per layer
    calibration_norms: Dict[int, float] = field(default_factory=dict)
    extraction_component: Optional[str] = None
    # Sensor input is persisted independently of vector-bearing steering layers.
    sensor_layer: Optional[int] = None
    
    def get_calibrated_strength(self, layer: int, target_percentage: float) -> float:
        """
        Compute calibrated strength for a layer based on hidden state norms.
        
        Args:
            layer: Layer index
            target_percentage: Target steering magnitude as percentage of hidden state norm (default 100%)
            
        Returns:
            Calibrated strength value
        """
        if layer not in self.calibration_norms or self.calibration_norms[layer] == 0:
            # No calibration data, return a reasonable default
            return target_percentage * 100  # Assume norm ~100 for typical models
        
        return target_percentage * self.calibration_norms[layer]


class BaseSteeringObject(ABC):
    """
    Base class for all steering objects.
    
    A steering object encapsulates everything needed to apply steering at inference:
    - Vectors/directions
    - Gating logic (when to steer)
    - Intensity computation (how much to steer)
    """
    
    method_name: str = BASE_CLASS_NAME
    
    def __init__(self, metadata: SteeringObjectMetadata):
        self.metadata = metadata
    
    @abstractmethod
    def get_steering_vector(self, layer: int, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get the steering vector for a layer, optionally conditioned on hidden state."""
        pass
    
    @abstractmethod
    def compute_gate(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Compute gate value (0-1) determining whether to apply steering."""
        pass
    
    @abstractmethod
    def compute_intensity(self, hidden_state: torch.Tensor, layer: int) -> torch.Tensor:
        """Compute steering intensity for a layer."""
        pass
    
    def get_calibrated_strength(self, target_percentage: float) -> float:
        """
        Get auto-calibrated strength based on hidden state norms.
        
        Args:
            target_percentage: Target steering magnitude as percentage of hidden state norm (default 100%)
            
        Returns:
            Calibrated strength to use with apply_steering
        """
        if not self.metadata.calibration_norms:
            # No calibration data - return reasonable default for normalized vectors
            return target_percentage * 100
        
        # Use average calibration norm across layers
        avg_norm = sum(self.metadata.calibration_norms.values()) / len(self.metadata.calibration_norms)
        return target_percentage * avg_norm
    
    def apply_steering(
        self,
        hidden_state: torch.Tensor,
        layer: int,
        base_strength: float,
    ) -> torch.Tensor:
        """
        Apply steering to hidden state.

        Args:
            hidden_state: [batch, seq, hidden] or [seq, hidden] or [hidden]
            layer: Layer index
            base_strength: Base strength multiplier (use get_calibrated_strength() for auto-calibration)
            
        Returns:
            Steered hidden state
        """
        original_shape = hidden_state.shape
        
        # Flatten to [batch, hidden] for gate computation
        if hidden_state.dim() == 1:
            h_flat = hidden_state.unsqueeze(0)
        elif hidden_state.dim() == 2:
            h_flat = hidden_state.mean(dim=0, keepdim=True)  # avg over seq
        else:
            h_flat = hidden_state.mean(dim=1)  # [batch, hidden]
        
        original_dtype = hidden_state.dtype
        gate = self.compute_gate(h_flat)
        intensity = self.compute_intensity(h_flat, layer)
        direction = self.get_steering_vector(layer, h_flat)

        # Ensure direction matches hidden state device/dtype
        direction = direction.to(hidden_state.device, original_dtype)

        # Compute steering delta - cast scale to match hidden state dtype
        scale = (gate * intensity * base_strength).to(original_dtype)

        # Broadcast direction to match hidden_state shape
        if hidden_state.dim() == 1:
            delta = scale.squeeze() * direction
        elif hidden_state.dim() == 2:
            delta = scale.unsqueeze(-1) * direction.unsqueeze(0)
        else:
            delta = scale.unsqueeze(-1).unsqueeze(-1) * direction.unsqueeze(0).unsqueeze(0)

        return (hidden_state + delta).to(original_dtype)
    
    def to_steering_plan(self, scale: float, normalize: bool = get_optimal("normalize")) -> "SteeringPlan":
        """
        Convert this steering object to a SteeringPlan for use with WisentModel.

        This allows steering objects to be used with the existing model infrastructure.
        Complex methods (TETNO, GROM) are simplified to their primary vectors.

        Args:
            scale: Base scale for all vectors
            normalize: Whether to normalize vectors

        Returns:
            SteeringPlan compatible with WisentModel.apply_steering()
        """
        from wisent.core.primitives.models.core.atoms import SteeringPlan, SteeringVector
        
        layers_dict = {}
        for layer in self.metadata.layers:
            try:
                vec = self.get_steering_vector(layer)
                layers_dict[str(layer)] = SteeringVector(
                    vector=vec,
                    scale=scale,
                    normalize=normalize,
                    layer_description=f"{self.method_name}_{self.metadata.benchmark}",
                )
            except (KeyError, IndexError):
                continue
        
        return SteeringPlan(
            layers=layers_dict,
            layers_description=[f"{self.method_name}_{self.metadata.benchmark}"],
        )
    
    def to_raw_activation_map(self) -> Dict[str, torch.Tensor]:
        """
        Convert to RawActivationMap format (layer_name -> tensor).
        
        This is useful for integrating with existing code that expects this format.
        """
        raw = {}
        for layer in self.metadata.layers:
            try:
                raw[str(layer)] = self.get_steering_vector(layer)
            except (KeyError, IndexError):
                continue
        return raw
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for saving."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseSteeringObject":
        """Deserialize from dictionary."""
        pass
    
    def save(self, path: str):
        """Save steering object to file."""
        data = self.to_dict()
        
        if path.endswith('.pt'):
            torch.save(data, path)
        else:
            # Convert tensors to lists for JSON
            def convert(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj
            
            with open(path, 'w') as f:
                json.dump(convert(data), f, indent=JSON_INDENT)
    
    @classmethod
    def load(cls, path: str) -> "BaseSteeringObject":
        """Load steering object from file."""
        if path.endswith('.pt'):
            data = torch.load(path, map_location='cpu', weights_only=False)
        else:
            with open(path, 'r') as f:
                data = json.load(f)
        
        method = data.get('method', 'caa')
        
        # Dispatch to appropriate class (lazy imports to avoid circular deps)
        if method == 'caa':
            from wisent.core.control.steering_methods._steering_object_simple import CAASteeringObject
            return CAASteeringObject.from_dict(data)
        elif method == 'ostrze':
            from wisent.core.control.steering_methods._steering_object_simple import OstrzeSteeringObject
            return OstrzeSteeringObject.from_dict(data)
        elif method == 'mlp':
            from wisent.core.control.steering_methods._steering_object_simple import MLPSteeringObject
            return MLPSteeringObject.from_dict(data)
        elif method == 'tecza':
            from wisent.core.control.steering_methods._steering_object_advanced import TECZASteeringObject
            return TECZASteeringObject.from_dict(data)
        elif method == 'tetno':
            from wisent.core.control.steering_methods._steering_object_advanced import TETNOSteeringObject
            return TETNOSteeringObject.from_dict(data)
        elif method == 'grom':
            from wisent.core.control.steering_methods._steering_object_grom import GROMSteeringObject
            return GROMSteeringObject.from_dict(data)
        elif method == 'nurt':
            from wisent.core.control.steering_methods.methods.nurt import NurtSteeringObject
            return NurtSteeringObject.from_dict(data)
        elif method == 'szlak':
            from wisent.core.control.steering_methods.methods.szlak import SzlakSteeringObject
            return SzlakSteeringObject.from_dict(data)
        elif method == 'wicher':
            from wisent.core.control.steering_methods.methods.wicher import WicherSteeringObject
            return WicherSteeringObject.from_dict(data)
        elif method == 'przelom':
            from wisent.core.control.steering_methods.methods.przelom import PrzelomSteeringObject
            return PrzelomSteeringObject.from_dict(data)
        else:
            raise ValueError(f"Unknown steering method: {method}")

