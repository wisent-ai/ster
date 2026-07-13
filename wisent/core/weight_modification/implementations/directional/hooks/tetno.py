"""TETNO runtime hooks and GROM convenience function."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Callable

from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch.nn import Module

_LOG = setup_logger(__name__)


class TETNORuntimeHooks:
    """Runtime hooks for TETNO conditional steering."""

    def __init__(
        self,
        model: Module,
        tetno_result,
        base_strength: float,
        gate_temperature: float,
        strength_provider: Callable[[torch.Tensor], float] | None = None,
    ) -> None:
        self.model = model
        self.tetno_result = tetno_result
        self.base_strength = base_strength
        self.gate_temperature = gate_temperature
        self.strength_provider = strength_provider
        self._hooks = []
        self._current_gate = None
        self._sensor_activation = None
        self._current_strategy_strength = 1.0
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self._layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self._layers = model.transformer.h
        elif hasattr(model, "layers"):
            self._layers = model.layers
        else:
            raise ValueError("TETNO model has no supported decoder layers")

        self._layer_name_to_idx = {
            layer_name: self._model_index(layer_name, "steering layer")
            for layer_name in tetno_result.behavior_vectors
        }
        sensor_value = getattr(tetno_result, "sensor_layer", None)
        metadata = getattr(tetno_result, "metadata", None)
        metadata_sensor = (
            metadata.get("sensor_layer")
            if isinstance(metadata, dict)
            else getattr(metadata, "sensor_layer", None)
        )
        metadata_component = (
            metadata.get("extraction_component")
            if isinstance(metadata, dict)
            else getattr(metadata, "extraction_component", None)
        )
        if metadata_component != "residual_stream":
            raise ValueError(
                "TETNO requires extraction_component='residual_stream'"
            )
        if sensor_value is None:
            sensor_value = metadata_sensor
        elif metadata_sensor is not None and self._layer_number(sensor_value) != self._layer_number(metadata_sensor):
            raise ValueError("TETNO sensor_layer conflicts with metadata.sensor_layer")
        if sensor_value is None:
            raise ValueError("TETNO sensor_layer is required")
        self._sensor_layer_idx = self._model_index(sensor_value, "sensor_layer")
        if (
            self._layer_name_to_idx
            and self._sensor_layer_idx >= min(self._layer_name_to_idx.values())
        ):
            raise ValueError(
                "TETNO sensor_layer must be strictly earlier than every steering layer"
            )

    @staticmethod
    def _layer_number(layer) -> int:
        try:
            number = int(str(layer).split("_")[-1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"Invalid TETNO layer identifier: {layer!r}") from exc
        if number <= 0:
            raise ValueError(f"TETNO sensor_layer/steering layer must be >= 1, got {number}")
        return number

    def _model_index(self, layer, label: str) -> int:
        number = self._layer_number(layer)
        if number > len(self._layers):
            raise ValueError(
                f"TETNO {label} {number} exceeds model layer count {len(self._layers)}"
            )
        return number - 1

    def install(self) -> None:
        """Install sensor first, then steering hooks in causal layer order."""
        self.remove()
        self._hooks.append(
            self._layers[self._sensor_layer_idx].register_forward_hook(self._sensor_hook)
        )
        for layer_name in self.tetno_result.behavior_vectors:
            layer_idx = self._layer_name_to_idx[layer_name]
            steering_hook = self._layers[layer_idx].register_forward_hook(
                lambda module, input, output, ln=layer_name: self._steering_hook(
                    module, input, output, ln
                )
            )
            self._hooks.append(steering_hook)

    def begin_generation(self) -> None:
        """Reset per-request gate and absolute generated-token schedule state."""
        self._current_gate = None
        self._sensor_activation = None
        self._current_strategy_strength = 1.0
        reset = getattr(self.strength_provider, "reset", None)
        if callable(reset):
            reset()

    def remove(self) -> None:
        """Remove all installed hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self.begin_generation()

    def _sensor_hook(self, module, input, output):
        """Capture the exact configured sensor activation and compute the gate."""
        hidden_states = output[0] if isinstance(output, tuple) else output
        sensor_hidden = hidden_states[:, -1, :] if hidden_states.dim() == 3 else hidden_states
        self._sensor_activation = sensor_hidden
        self._current_strategy_strength = (
            self.strength_provider(hidden_states)
            if self.strength_provider is not None
            else 1.0
        )
        if hasattr(self.tetno_result, "threshold"):
            self._current_gate = self.tetno_result.compute_gate(sensor_hidden)
        elif hasattr(self.tetno_result, "optimal_threshold"):
            self._current_gate = self.tetno_result.compute_gate(
                sensor_hidden, self.gate_temperature
            )
        else:
            raise ValueError("TETNO result has no persisted gate threshold")
        return output

    def _steering_hook(self, module, input, output, layer_name):
        """Apply conditional steering using the previously computed sensor gate."""
        if self._current_gate is None:
            return output
        hidden_states = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        behavior_vector = self.tetno_result.behavior_vectors[layer_name].to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        if layer_name not in self.tetno_result.layer_scales:
            raise KeyError(f"No layer_scale for '{layer_name}' in TETNO result")
        layer_scale = self.tetno_result.layer_scales[layer_name]
        gate = self._current_gate.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        strategy_strength = self._current_strategy_strength
        if hidden_states.dim() == 3:
            gate = gate.reshape(-1, 1, 1)
            behavior_vector = behavior_vector.view(1, 1, -1)
        elif hidden_states.dim() == 2:
            gate = gate.reshape(-1, 1)
            behavior_vector = behavior_vector.view(1, -1)
        steering_delta = (
            gate
            * self.base_strength
            * strategy_strength
            * layer_scale
            * behavior_vector
        )
        hidden_states = hidden_states + steering_delta
        return (hidden_states,) + rest if rest is not None else hidden_states

    def get_current_gate(self) -> float | None:
        """Get the current gate value."""
        return self._current_gate.mean().item() if self._current_gate is not None else None


def apply_grom_steering(
    model: Module, grom_result, base_strength: float, mode: str,
    components: list[str] | None = None, verbose: bool = True
) -> dict:
    """Apply GROM steering to a model with the specified mode."""
    from .grom import GROMRuntimeHooks, project_weights_grom
    result = {}
    if mode in ("static", "hybrid"):
        blw = grom_result.metadata.get("base_layer_weight")
        stats = project_weights_grom(model=model, grom_result=grom_result, base_layer_weight=blw, components=components,
                                       base_strength=base_strength if mode == "static" else 1.0,
                                       use_learned_intensities=True, verbose=verbose)
        result["stats"] = stats
    if mode in ("dynamic", "hybrid"):
        hooks = GROMRuntimeHooks(model=model, grom_result=grom_result, base_strength=base_strength, use_soft_gating=True)
        hooks.install()
        result["hooks"] = hooks
        if verbose:
            print(f"\nGROM Runtime Hooks installed\n  Sensor layer: {hooks._sensor_layer_idx}\n  Steering layers: {len(grom_result.layer_order)}\n  Mode: {mode}")
    return result
