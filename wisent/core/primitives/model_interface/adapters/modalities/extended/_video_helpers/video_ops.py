"""
Video adapter operations: steering, activations, temporal analysis.

Extracted from video.py to keep files under 300 lines.
"""
from __future__ import annotations

from typing import Any, Dict, List, Union

import torch
import torch.nn as nn

from wisent.core.primitives.model_interface.adapters.base import (
    InterventionPoint,
    SteeringConfig,
)
from wisent.core.primitives.models.modalities import VideoContent
from wisent.core.utils.infra_tools.errors import UnknownTypeError
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations
from wisent.core.primitives.model_interface.adapters.modalities.extended._video_helpers.video_core import VideoSteeringConfig
from wisent.core.utils.config_tools.constants import NORM_EPS, TEMPORAL_RAMP_MAX


class VideoOpsMixin:
    """Mixin with steering, activation, and temporal methods for video."""

    def get_intervention_points(self) -> List[InterventionPoint]:
        """Get available intervention points in the video model."""
        points = []
        encoder_layers = self._resolve_encoder_layers()
        for i, _ in enumerate(encoder_layers):
            recommended = (len(encoder_layers) // 3) <= i <= (2 * len(encoder_layers) // 3)
            points.append(
                InterventionPoint(
                    name=f"encoder.{i}",
                    module_path=f"{self._encoder_path}.{i}",
                    description=f"Encoder layer {i}",
                    recommended=recommended,
                )
            )
        return points

    def extract_activations(
        self,
        content: VideoContent,
        layers: List[str] | None = None,
    ) -> LayerActivations:
        """Extract activations from video model layers."""
        all_points = {ip.name: ip for ip in self.get_intervention_points()}
        target_layers = layers if layers else list(all_points.keys())
        activations: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                activations[layer_name] = output.detach().cpu()
            return hook

        try:
            for layer_name in target_layers:
                if layer_name not in all_points:
                    continue
                ip = all_points[layer_name]
                module = self._get_module_by_path(ip.module_path)
                if module is not None:
                    handle = module.register_forward_hook(make_hook(layer_name))
                    hooks.append(handle)
            _ = self.encode(content)
        finally:
            for handle in hooks:
                handle.remove()
        return LayerActivations(activations)

    def _create_temporal_steering_hook(
        self,
        vector: torch.Tensor,
        config: VideoSteeringConfig,
        num_frames: int,
        *,
        temporal_ramp_min: float,
    ):
        """Create a hook with temporal-aware steering."""
        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
            v = vector.to(output.device, output.dtype)
            if config.normalize:
                v = v / (v.norm(dim=-1, keepdim=True) + NORM_EPS)
            if config.frame_mode == "keyframes":
                mask = torch.zeros(num_frames, device=output.device)
                mask[::config.keyframe_interval] = 1.0
                v = v * mask.view(-1, 1, 1) if v.dim() >= 3 else v
            elif config.frame_mode == "temporal_decay":
                decay = torch.tensor(
                    [config.temporal_decay_rate ** i for i in range(num_frames)],
                    device=output.device,
                )
                v = v * decay.view(-1, 1, 1) if v.dim() >= 3 else v * decay.mean()
            elif config.frame_mode == "temporal_ramp":
                ramp = torch.linspace(temporal_ramp_min, TEMPORAL_RAMP_MAX, num_frames, device=output.device)
                v = v * ramp.view(-1, 1, 1) if v.dim() >= 3 else v * ramp.mean()
            v = v * config.scale
            while v.dim() < output.dim():
                v = v.unsqueeze(0)
            return output + v
        return hook

    def forward_with_steering(
        self,
        content: VideoContent,
        steering_vectors: LayerActivations,
        config: SteeringConfig | VideoSteeringConfig,
    ) -> torch.Tensor:
        """Process video with steering applied."""
        with self._steering_hooks(steering_vectors, config):
            return self.encode(content)

    def _generate_unsteered(self, content: VideoContent, **kwargs: Any) -> torch.Tensor:
        """Generate output without steering."""
        return self.encode(content)

    def extract_frame_activations(
        self,
        content: VideoContent,
        layers: List[str] | None = None,
    ) -> Dict[int, LayerActivations]:
        """Extract activations per frame (for detailed temporal analysis)."""
        frames = self._sample_frames(content)
        frame_activations = {}
        for i in range(frames.shape[0]):
            single_frame = VideoContent(frames=frames[i:i+1], fps=content.fps)
            frame_activations[i] = self.extract_activations(single_frame, layers)
        return frame_activations

    def compute_temporal_steering_vector(
        self,
        positive_video: VideoContent,
        negative_video: VideoContent,
        layer: str,
        aggregation: str,
    ) -> torch.Tensor:
        """Compute a steering vector from positive/negative video pair."""
        pos_acts = self.extract_activations(positive_video, [layer])
        neg_acts = self.extract_activations(negative_video, [layer])
        pos_tensor = pos_acts[layer]
        neg_tensor = neg_acts[layer]
        if aggregation == "mean":
            pos_agg = pos_tensor.mean(dim=1)
            neg_agg = neg_tensor.mean(dim=1)
        elif aggregation == "last":
            pos_agg = pos_tensor[:, -1]
            neg_agg = neg_tensor[:, -1]
        elif aggregation == "first":
            pos_agg = pos_tensor[:, 0]
            neg_agg = neg_tensor[:, 0]
        else:
            raise UnknownTypeError(
                entity_type="aggregation",
                value=aggregation,
                valid_values=["mean", "max", "last", "first"],
            )
        return pos_agg.mean(dim=0) - neg_agg.mean(dim=0)
