"""
Image adapter for steering text-to-image diffusion generation models.

Verified target: Z-Image-Turbo (Tongyi). Generic across diffusers
DiffusionPipeline-compatible DiT/UNet pipelines (SDXL, Flux, SD3, ...).
Enables contrastive steering for:
- Concept selection at conditioning time (text encoder hooks)
- Image-feature shaping during denoising (DiT/UNet block hooks)
- Concept-to-image binding control (cross-attention hooks)

Implementation split into _image_helpers/image_core.py and
_image_helpers/image_ops.py to keep files under 300 lines.
"""
from __future__ import annotations

from wisent.core.primitives.model_interface.adapters.modalities.extended._image_helpers.image_core import (
    ImageAdapterCore,
    ImageSteeringConfig,
)
from wisent.core.primitives.model_interface.adapters.modalities.extended._image_helpers.image_ops import (
    ImageOpsMixin,
)

__all__ = ["ImageAdapter", "ImageSteeringConfig"]


class ImageAdapter(ImageOpsMixin, ImageAdapterCore):
    """
    Adapter for diffusion text-to-image model steering.

    Supports any diffusers DiffusionPipeline. The intervention surface is
    selected at steering-config time:
    - text_encoder.{i}: cheap concept-selection steering at conditioning
    - dit_block.{i} / unet_block.{i}: image-feature steering during denoising
    - cross_attn.{i}: text-conditioning binding control

    Example:
        >>> from wisent import ImageAdapter, TextContent, ImageSteeringConfig
        >>> adapter = ImageAdapter(model_name="Tongyi-MAI/Z-Image-Turbo")
        >>> pos = TextContent(text="a photorealistic landscape")
        >>> neg = TextContent(text="a cartoon landscape")
        >>> vec = adapter.compute_cross_modal_steering_vector(pos, neg, "text_encoder.20")
        >>> from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations
        >>> img = adapter.forward_with_steering(
        ...     TextContent(text="a mountain at dawn"),
        ...     LayerActivations({"text_encoder.20": vec}),
        ...     ImageSteeringConfig(scale=1.5, steer_surfaces="text_encoder"),
        ... )
    """
    pass
