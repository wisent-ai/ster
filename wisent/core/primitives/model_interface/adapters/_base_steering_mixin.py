"""Steering hook mixin and utility methods for BaseAdapter.

Extracted from base.py to keep file under 300 lines.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict

import torch
import torch.nn as nn

from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations
from wisent.core.utils.config_tools.constants import NORM_EPS


class SteeringHookMixin:
    """Mixin providing steering hook management and utility methods for BaseAdapter."""

    @contextmanager
    def _steering_hooks(
        self,
        steering_vectors: LayerActivations,
        config: Any = None,
    ):
        """
        Context manager that registers steering hooks and cleans up after.

        Args:
            steering_vectors: Steering vectors to apply
            config: Steering configuration

        Yields:
            None (hooks are active within context)
        """
        from wisent.core.primitives.model_interface.adapters.base import SteeringConfig
        config = config or SteeringConfig()
        handles = []

        try:
            # Get intervention points
            intervention_points = {ip.name: ip for ip in self.get_intervention_points()}

            # Register hooks for each layer with a steering vector
            for layer_name, vector in steering_vectors.items():
                if vector is None:
                    continue
                if config.layers is not None and layer_name not in config.layers:
                    continue

                ip = intervention_points.get(layer_name)
                if ip is None:
                    continue

                # Get the module
                module = self._get_module_by_path(ip.module_path)
                if module is None:
                    continue

                # Create and register hook (pass layer_name for per-layer config)
                hook = self._create_steering_hook(vector, config, layer_name)
                handle = module.register_forward_hook(hook)
                handles.append(handle)

            yield

        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()

    def _get_module_by_path(self, path: str) -> nn.Module | None:
        """Get a module by its dot-separated path."""
        module = self.model
        try:
            for attr in path.split("."):
                if attr.isdigit():
                    module = module[int(attr)]
                else:
                    module = getattr(module, attr)
            return module
        except (AttributeError, IndexError, KeyError):
            return None

    def _create_steering_hook(self, vector: torch.Tensor, config: Any, layer_name: str = None):
        """Create a forward hook that adds the steering vector."""
        # Get per-layer method if config.method is a dict
        method = config.method.get(layer_name) if isinstance(config.method, dict) else config.method
        if isinstance(config.scale, dict):
            if layer_name not in config.scale:
                raise KeyError(f"No scale for layer '{layer_name}' in steering config")
            scale = config.scale[layer_name]
        else:
            scale = config.scale

        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
            if method is not None:
                # Non-linear method.transform path. Honors temporal_mode the
                # same way the linear path below does, so non-text modalities
                # (image patches, audio frames, video tokens) can opt into
                # per_step (broadcast across all positions) instead of being
                # silently confined to the last sequence position.
                if output.dim() < 3:
                    return method.transform(output)
                if config.temporal_mode == "last":
                    steered = output.clone()
                    steered[:, -1] = method.transform(output[:, -1])
                    return steered
                if config.temporal_mode == "first":
                    steered = output.clone()
                    steered[:, 0] = method.transform(output[:, 0])
                    return steered
                if config.temporal_mode == "per_step":
                    B, S = output.shape[0], output.shape[1]
                    tail = output.shape[2:]
                    flat = output.reshape(B * S, *tail)
                    transformed = method.transform(flat)
                    return transformed.reshape(output.shape)
                # Backward-compatible default for callers (predominantly text)
                # that did not set temporal_mode and relied on last-position
                # semantics. Modality adapters whose intervention points have
                # no last-token meaning should set temporal_mode explicitly
                # (ImageAdapter.forward_with_steering does this).
                steered = output.clone()
                steered[:, -1] = method.transform(output[:, -1])
                return steered
            # Linear steering (no method): unchanged.
            v = vector.to(output.device, output.dtype)
            if config.normalize:
                v = v / (v.norm(dim=-1, keepdim=True) + NORM_EPS)
            v = v * scale
            while v.dim() < output.dim():
                v = v.unsqueeze(0)
            if output.dim() >= 3 and config.temporal_mode != "per_step":
                steered = output.clone()
                if config.temporal_mode == "last":
                    steered[:, -1] = steered[:, -1] + v.squeeze(1)
                elif config.temporal_mode == "first":
                    steered[:, 0] = steered[:, 0] + v.squeeze(1)
                return steered
            return output + v
        return hook

    @property
    def hidden_size(self) -> int:
        """Get the model's hidden dimension."""
        raise NotImplementedError("Subclass must implement hidden_size property")

    @property
    def num_layers(self) -> int:
        """Get the number of steerable layers."""
        return len(self.get_intervention_points())

    def build_pair_set(
        self,
        positive_contents: list,
        negative_contents: list,
        layers: list[str] | None = None,
        name: str = "contrastive_pairs",
        task_type: str | None = None,
    ):
        """
        Build a ContrastivePairSet from parallel positive/negative content
        lists by running self.extract_activations on each entry. The pair's
        prompt is taken from a TextContent.text field when present, otherwise
        synthesized as f"<{modality}-pair-{i}>" so the upstream non-empty
        string check on ContrastivePair.prompt passes for non-text modalities.

        Output is fed directly to any wisent steering method (CAA, GROM,
        NURT, OSTRZE, MLP, TECZA, TETNO, SZLAK, WICHER, PRZELOM):
            method = CAAMethod(); vectors = method.train(pair_set)
        """
        from wisent.core.primitives.contrastive_pairs.core.io.response import (
            PositiveResponse, NegativeResponse,
        )
        from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
        from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet

        if len(positive_contents) != len(negative_contents):
            raise ValueError(
                f"positive_contents (len={len(positive_contents)}) and "
                f"negative_contents (len={len(negative_contents)}) must be parallel"
            )

        pairs = []
        for i, (pos, neg) in enumerate(zip(positive_contents, negative_contents)):
            pos_acts = self.extract_activations(pos, layers)
            neg_acts = self.extract_activations(neg, layers)
            pos_text = getattr(pos, "text", None)
            modality_name = getattr(getattr(pos, "modality", None), "name", "unknown").lower()
            prompt = pos_text if isinstance(pos_text, str) and pos_text.strip() else f"<{modality_name}-pair-{i}>"
            pair = ContrastivePair(
                prompt=prompt,
                positive_response=PositiveResponse(
                    model_response=pos,
                    layers_activations=pos_acts,
                ),
                negative_response=NegativeResponse(
                    model_response=neg,
                    layers_activations=neg_acts,
                ),
            )
            pairs.append(pair)

        return ContrastivePairSet(name=name, pairs=pairs, task_type=task_type)

    @classmethod
    def list_registered(cls) -> Dict[str, type]:
        """List all registered adapters."""
        return dict(cls._REGISTRY)

    @classmethod
    def get(cls, name: str) -> type:
        """
        Get a registered adapter class by name.

        Args:
            name: Adapter name

        Returns:
            Adapter class

        Raises:
            AdapterError: If adapter not found
        """
        from wisent.core.primitives.model_interface.adapters.base import AdapterError
        try:
            return cls._REGISTRY[name]
        except KeyError as exc:
            raise AdapterError(f"Unknown adapter: {name!r}") from exc

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r}, device={self.device!r})"
