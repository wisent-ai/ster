"""
Core image-generation adapter: lazy diffusers pipeline loading,
DiT vs UNet structure detection, and intervention-point layer resolution.

Extracted from image.py to keep files under 300 lines.
"""
from __future__ import annotations

from typing import Any, List, Optional
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn

from wisent.core.primitives.model_interface.adapters.base import (
    BaseAdapter,
    AdapterError,
    SteeringConfig,
)
from wisent.core.primitives.models.modalities import (
    Modality,
    TextContent,
    ImageContent,
)
from wisent.core.utils import preferred_dtype

logger = logging.getLogger(__name__)


@dataclass
class ImageSteeringConfig(SteeringConfig):
    """
    Steering config for text-to-image diffusion. Extends SteeringConfig:
    steer_surfaces selects which intervention surface(s) ("text_encoder",
    "dit_block", "cross_attn", a list, or "all"); num_inference_steps and
    guidance_scale override the pipeline defaults; capture_timestep_index
    picks which denoising step's activations to capture in extract_activations.
    """
    steer_surfaces: str | List[str] = "all"
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    capture_timestep_index: int = 0


class ImageAdapterCore(BaseAdapter[TextContent, ImageContent]):
    """
    Core adapter for diffusion text-to-image model steering. Loads any
    diffusers DiffusionPipeline (Z-Image-Turbo as the verified target,
    generic across DiT-based pipelines like Flux / SD3 and UNet-based
    pipelines like SDXL / SD2). Detects model class, resolves text encoder
    transformer layers, DiT/UNet block lists, and per-block cross-attention
    modules. Companion ops mixin (image_ops.py) implements intervention
    points + steering hooks on top of these resolvers.
    """

    name = "image"
    modality = Modality.IMAGE

    MODEL_TYPE_DIT = "dit"
    MODEL_TYPE_UNET = "unet"
    MODEL_TYPE_GENERIC = "generic"

    def __init__(
        self,
        model_name: str | None = None,
        model: Any | None = None,
        device: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=None, model_name=model_name, device=device, **kwargs)
        self._pipeline: Any | None = model
        self._model_type = model_type
        self._text_encoder_layers: List[nn.Module] | None = None
        self._dit_blocks: List[nn.Module] | None = None
        self._dit_block_paths: List[str] | None = None
        self._cross_attn_modules: List[nn.Module] | None = None
        self._cross_attn_paths: List[str] | None = None
        self._text_encoder_path: str | None = None
        self._hidden_size: int | None = None

    @property
    def model(self) -> Any:
        """Diffusers DiffusionPipeline (loaded on first access)."""
        if self._pipeline is None:
            self._pipeline = self._load_model()
        return self._pipeline

    def _load_model(self) -> Any:
        """Lazily import diffusers and load the pipeline."""
        if self.model_name is None:
            raise AdapterError(
                "model_name is required when an existing pipeline is not provided"
            )
        try:
            from diffusers import DiffusionPipeline
        except ImportError as e:
            raise AdapterError(
                "diffusers is required for ImageAdapter. "
                "Install with: pip install diffusers"
            ) from e
        try:
            pipe = DiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=preferred_dtype(self.device),
                trust_remote_code=True,
                **self._kwargs,
            )
            if self.device:
                pipe = pipe.to(self.device)
            return pipe
        except Exception as e:
            raise AdapterError(
                f"Failed to load diffusion pipeline {self.model_name}: {e}"
            ) from e

    def _detect_model_type(self) -> str:
        """Identify whether the loaded pipeline is DiT-based or UNet-based."""
        if self._model_type:
            return self._model_type
        pipe = self.model
        if getattr(pipe, "transformer", None) is not None:
            self._model_type = self.MODEL_TYPE_DIT
        elif getattr(pipe, "unet", None) is not None:
            self._model_type = self.MODEL_TYPE_UNET
        else:
            self._model_type = self.MODEL_TYPE_GENERIC
        return self._model_type

    def _resolve_text_encoder_layers(self) -> List[nn.Module]:
        """Find the transformer layers inside pipe.text_encoder."""
        if self._text_encoder_layers is not None:
            return self._text_encoder_layers
        pipe = self.model
        text_encoder = getattr(pipe, "text_encoder", None)
        if text_encoder is None:
            self._text_encoder_layers = []
            return self._text_encoder_layers
        candidates = [
            ("text_encoder.encoder.block", "encoder.block"),
            ("text_encoder.text_model.encoder.layers", "text_model.encoder.layers"),
            ("text_encoder.encoder.layers", "encoder.layers"),
            ("text_encoder.transformer.h", "transformer.h"),
        ]
        for full_path, rel_path in candidates:
            obj = text_encoder
            try:
                for attr in rel_path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, (nn.ModuleList, nn.Sequential)):
                    self._text_encoder_layers = list(obj)
                    self._text_encoder_path = full_path
                    return self._text_encoder_layers
            except AttributeError:
                continue
        self._text_encoder_layers = []
        return self._text_encoder_layers

    def _resolve_dit_blocks(self) -> List[nn.Module]:
        """
        Find DiT transformer blocks (or UNet block list). For DiT pipelines:
        pipe.transformer.{transformer_blocks|blocks|layers}. For UNet pipelines:
        flatten down_blocks / mid_block / up_blocks.
        """
        if self._dit_blocks is not None:
            return self._dit_blocks
        pipe = self.model
        model_type = self._detect_model_type()
        if model_type == self.MODEL_TYPE_DIT:
            transformer = pipe.transformer
            candidates = [
                ("transformer.transformer_blocks", "transformer_blocks"),
                ("transformer.blocks", "blocks"),
                ("transformer.layers", "layers"),
            ]
            for full_path, rel_path in candidates:
                obj = transformer
                try:
                    for attr in rel_path.split("."):
                        obj = getattr(obj, attr)
                    if isinstance(obj, (nn.ModuleList, nn.Sequential)):
                        block_list = list(obj)
                        self._dit_blocks = block_list
                        self._dit_block_paths = [
                            f"{full_path}.{i}" for i in range(len(block_list))
                        ]
                        return self._dit_blocks
                except AttributeError:
                    continue
        elif model_type == self.MODEL_TYPE_UNET:
            unet = pipe.unet
            blocks: List[nn.Module] = []
            paths: List[str] = []
            for section in ("down_blocks", "mid_block", "up_blocks"):
                obj = getattr(unet, section, None)
                if obj is None:
                    continue
                if isinstance(obj, nn.ModuleList):
                    for i, blk in enumerate(obj):
                        blocks.append(blk)
                        paths.append(f"unet.{section}.{i}")
                else:
                    blocks.append(obj)
                    paths.append(f"unet.{section}")
            self._dit_blocks = blocks
            self._dit_block_paths = paths
            return self._dit_blocks
        self._dit_blocks = []
        self._dit_block_paths = []
        return self._dit_blocks

    def _resolve_cross_attn_modules(self) -> List[nn.Module]:
        """
        Cross-attention modules (attn2 / cross_attn convention) per DiT/UNet
        block. Returned list is parallel to _resolve_dit_blocks; entries are
        None for blocks that have no cross-attention.
        """
        if self._cross_attn_modules is not None:
            return self._cross_attn_modules
        modules: List[nn.Module] = []
        paths: List[str] = []
        block_paths = self._dit_block_paths or [
            f"block.{i}" for i in range(len(self._resolve_dit_blocks()))
        ]
        for blk, blk_path in zip(self._resolve_dit_blocks(), block_paths):
            attn = getattr(blk, "attn2", None)
            attr_name = "attn2"
            if attn is None:
                attn = getattr(blk, "cross_attn", None)
                attr_name = "cross_attn"
            modules.append(attn)
            paths.append(f"{blk_path}.{attr_name}" if attn is not None else "")
        self._cross_attn_modules = modules
        self._cross_attn_paths = paths
        return modules

    @property
    def hidden_size(self) -> int:
        """Hidden dim from the transformer/unet config."""
        if self._hidden_size is not None:
            return self._hidden_size
        pipe = self.model
        model_type = self._detect_model_type()
        backbone = (
            pipe.transformer if model_type == self.MODEL_TYPE_DIT
            else getattr(pipe, "unet", None)
        )
        if backbone is None:
            raise AdapterError(
                "Could not determine hidden size: no transformer/unet on pipeline"
            )
        config = getattr(backbone, "config", None)
        if config is None:
            raise AdapterError("Backbone has no .config object")
        for attr in (
            "hidden_size", "inner_dim", "joint_attention_dim",
            "cross_attention_dim", "block_out_channels",
        ):
            val = getattr(config, attr, None)
            if val is None:
                continue
            if isinstance(val, (list, tuple)):
                val = val[0]
            if isinstance(val, int):
                self._hidden_size = val
                return self._hidden_size
        raise AdapterError("Could not determine hidden size from backbone config")

    def encode(self, content: TextContent) -> torch.Tensor:
        """
        Encode a text prompt via the pipeline's text encoder. Returns the
        last hidden state — used by extract_activations and the contrastive
        vector computation for text-encoder-based steering.
        """
        pipe = self.model
        text = content.text if hasattr(content, "text") else str(content)
        tokenizer = getattr(pipe, "tokenizer", None)
        text_encoder = getattr(pipe, "text_encoder", None)
        if tokenizer is None or text_encoder is None:
            raise AdapterError("Pipeline has no tokenizer / text_encoder")
        inputs = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(text_encoder.device)
        with torch.no_grad():
            outputs = text_encoder(input_ids, output_hidden_states=True)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs.hidden_states[-1]
