"""
Image-generation adapter operations: intervention points, activation
extraction, steering hooks, contrastive vector computation, decode.

Extracted from image.py to keep files under 300 lines.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch

from wisent.core.primitives.model_interface.adapters.base import (
    AdapterError,
    InterventionPoint,
    SteeringConfig,
)
from wisent.core.primitives.models.modalities import TextContent, ImageContent
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations


class ImageOpsMixin:
    """
    Mixin with intervention-point enumeration, activation extraction during
    text-encoding or denoising, steering-hook generation, and contrastive
    cross-modal vector computation for ImageAdapter.
    """

    def get_intervention_points(self) -> List[InterventionPoint]:
        """
        Enumerate text_encoder.{i}, dit_block.{i}, and cross_attn.{i}
        intervention points across the loaded diffusion pipeline. Mirrors
        MultimodalAdapter's named per-modality intervention point convention.
        """
        points: List[InterventionPoint] = []

        te_layers = self._resolve_text_encoder_layers()
        for i, _ in enumerate(te_layers):
            recommended = i >= len(te_layers) // 2
            points.append(InterventionPoint(
                name=f"text_encoder.{i}",
                module_path=f"{self._text_encoder_path}.{i}",
                description=f"Text encoder layer {i}",
                recommended=recommended,
            ))

        blocks = self._resolve_dit_blocks()
        block_paths = self._dit_block_paths or []
        for i, (_blk, path) in enumerate(zip(blocks, block_paths)):
            recommended = (len(blocks) // 3) <= i <= (2 * len(blocks) // 3)
            points.append(InterventionPoint(
                name=f"dit_block.{i}",
                module_path=path,
                description=f"DiT/UNet block {i}",
                recommended=recommended,
            ))

        cross_attns = self._resolve_cross_attn_modules()
        cross_paths = self._cross_attn_paths or []
        for i, (attn, path) in enumerate(zip(cross_attns, cross_paths)):
            if attn is None or not path:
                continue
            recommended = (len(cross_attns) // 3) <= i <= (2 * len(cross_attns) // 3)
            points.append(InterventionPoint(
                name=f"cross_attn.{i}",
                module_path=path,
                description=f"Cross-attention at block {i}",
                recommended=recommended,
            ))

        return points

    def extract_activations(
        self,
        content: TextContent,
        layers: List[str] | None = None,
    ) -> LayerActivations:
        """
        Extract activations for the requested layers. text_encoder.* layers
        are captured from a pure text-encoder forward (timestep-independent,
        cheap). dit_block.* and cross_attn.* layers are captured from a full
        pipeline run; the hook fires every denoising step and the value at
        capture_timestep_index is retained.
        """
        from wisent.core.primitives.model_interface.adapters.modalities.extended._image_helpers.image_core import (
            ImageSteeringConfig,
        )
        pipe = self.model
        text = content.text if hasattr(content, "text") else str(content)
        all_points = {ip.name: ip for ip in self.get_intervention_points()}
        target_layers = layers if layers else list(all_points.keys())

        activations: Dict[str, torch.Tensor] = {}

        te_layers = [l for l in target_layers if l.startswith("text_encoder.") and l in all_points]
        diff_layers = [
            l for l in target_layers
            if l in all_points and not l.startswith("text_encoder.")
        ]

        if te_layers:
            hooks = []
            try:
                for name in te_layers:
                    ip = all_points[name]
                    module = self._get_module_by_path(ip.module_path)
                    if module is None:
                        continue
                    handle = module.register_forward_hook(_make_capture_hook(name, activations))
                    hooks.append(handle)
                self.encode(content)
            finally:
                for h in hooks:
                    h.remove()

        if diff_layers:
            capture_idx = 0
            backbone = pipe.transformer if self._detect_model_type() == self.MODEL_TYPE_DIT else getattr(pipe, "unet", None)
            if backbone is None:
                raise AdapterError("Pipeline has no transformer/unet for DiT activation capture")
            step_counter = [0]
            hooks = []

            def step_advance(_module, _input, _output):
                step_counter[0] += 1

            try:
                hooks.append(backbone.register_forward_hook(step_advance))
                for name in diff_layers:
                    ip = all_points[name]
                    module = self._get_module_by_path(ip.module_path)
                    if module is None:
                        continue
                    handle = module.register_forward_hook(
                        _make_step_indexed_hook(name, activations, step_counter, capture_idx)
                    )
                    hooks.append(handle)
                with torch.no_grad():
                    pipe(text, output_type="pt", num_inference_steps=max(capture_idx + 1, 4))
            finally:
                for h in hooks:
                    h.remove()

        return LayerActivations(activations)

    def forward_with_steering(
        self,
        content: TextContent,
        steering_vectors: LayerActivations,
        config: SteeringConfig | None = None,
    ) -> ImageContent:
        """
        Generate an image with steering active across every denoising step.
        Filters steering_vectors by config.steer_surfaces so callers can
        steer only the text-encoder side (cheap, concept-only) or only the
        DiT side (image-feature steering) or both.
        """
        from wisent.core.primitives.model_interface.adapters.modalities.extended._image_helpers.image_core import (
            ImageSteeringConfig,
        )
        config = config or ImageSteeringConfig()
        # Image patches / text-encoder padding tokens have no "last position"
        # semantics, so default the non-linear hook path to broadcast across
        # every position. Caller can still pin to "last" / "first" explicitly.
        if config.temporal_mode is None:
            config.temporal_mode = "per_step"
        pipe = self.model
        text = content.text if hasattr(content, "text") else str(content)

        if isinstance(config, ImageSteeringConfig) and config.steer_surfaces != "all":
            surfaces = (
                [config.steer_surfaces] if isinstance(config.steer_surfaces, str)
                else config.steer_surfaces
            )
            filtered: Dict[str, torch.Tensor] = {}
            for name, vec in steering_vectors.items():
                surface = name.split(".")[0]
                if surface in surfaces:
                    filtered[name] = vec
            steering_vectors = LayerActivations(filtered)

        pipe_kwargs: Dict[str, Any] = {"output_type": "pt"}
        if isinstance(config, ImageSteeringConfig):
            if config.num_inference_steps is not None:
                pipe_kwargs["num_inference_steps"] = config.num_inference_steps
            if config.guidance_scale is not None:
                pipe_kwargs["guidance_scale"] = config.guidance_scale

        with self._steering_hooks(steering_vectors, config):
            with torch.no_grad():
                out = pipe(text, **pipe_kwargs)

        return ImageContent(pixels=out.images[0])

    def _generate_unsteered(self, content: TextContent, **kwargs: Any) -> ImageContent:
        """Plain pipe(prompt) call wrapped to return ImageContent."""
        pipe = self.model
        text = content.text if hasattr(content, "text") else str(content)
        pipe_kwargs: Dict[str, Any] = {"output_type": "pt"}
        pipe_kwargs.update(kwargs)
        with torch.no_grad():
            out = pipe(text, **pipe_kwargs)
        return ImageContent(pixels=out.images[0])

    def decode(self, latent: torch.Tensor) -> ImageContent:
        """
        Decode a VAE latent to image pixels via the pipeline's VAE. The
        diffusers convention divides the latent by vae.config.scaling_factor
        before the decoder pass; we apply the same convention here.
        """
        pipe = self.model
        vae = getattr(pipe, "vae", None)
        if vae is None:
            raise AdapterError("Pipeline has no VAE for decoding")
        with torch.no_grad():
            if latent.dim() == 3:
                latent = latent.unsqueeze(0)
            latent = latent.to(vae.device)
            scaling = float(getattr(vae.config, "scaling_factor", 1.0))
            decoded = vae.decode(latent / scaling).sample
        return ImageContent(pixels=decoded[0])

    def compute_cross_modal_steering_vector(
        self,
        positive_content: TextContent,
        negative_content: TextContent,
        layer: str,
    ) -> torch.Tensor:
        """
        Compute a steering vector from a positive/negative text-prompt pair
        at a single intervention point. Pools over batch and sequence dims.
        """
        pos_acts = self.extract_activations(positive_content, [layer])
        neg_acts = self.extract_activations(negative_content, [layer])
        pos_tensor = pos_acts[layer]
        neg_tensor = neg_acts[layer]
        pos_pooled = _pool_to_vector(pos_tensor)
        neg_pooled = _pool_to_vector(neg_tensor)
        return pos_pooled - neg_pooled


def _make_capture_hook(name: str, sink: Dict[str, torch.Tensor]):
    """Forward hook that stores the layer output (single occurrence)."""
    def hook(_module, _input, output):
        val = output[0] if isinstance(output, tuple) else output
        sink[name] = val.detach().cpu()
    return hook


def _make_step_indexed_hook(
    name: str,
    sink: Dict[str, torch.Tensor],
    step_counter: List[int],
    target_idx: int,
):
    """Forward hook that stores the output only on the target denoising step."""
    def hook(_module, _input, output):
        if step_counter[0] == target_idx:
            val = output[0] if isinstance(output, tuple) else output
            sink[name] = val.detach().cpu()
    return hook


def _pool_to_vector(tensor: torch.Tensor) -> torch.Tensor:
    """Mean-pool all leading dims so a single hidden vector remains."""
    if tensor.dim() <= 1:
        return tensor
    pooled = tensor.mean(dim=tuple(range(tensor.dim() - 1)))
    return pooled.squeeze()
