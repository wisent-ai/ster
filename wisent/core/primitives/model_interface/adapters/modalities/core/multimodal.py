"""Multimodal adapter for vision-language and other combined modality models.
Supports LLaVA, Qwen-VL, IDEFICS, ImageBind, and generic multimodal models."""
from __future__ import annotations
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import logging
import torch
import torch.nn as nn
from wisent.core.primitives.model_interface.adapters.base import BaseAdapter, AdapterError, InterventionPoint, SteeringConfig
from wisent.core.primitives.models.modalities import (
    Modality, ModalityContent, MultimodalContent, TextContent, ImageContent, AudioContent, VideoContent,
)
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations
from wisent.core.utils import preferred_dtype
from wisent.core.primitives.models.config import get_generate_kwargs

__all__ = ["MultimodalAdapter", "MultimodalSteeringConfig"]
logger = logging.getLogger(__name__)

@dataclass
class MultimodalSteeringConfig(SteeringConfig):
    """Extended steering config for multimodal models."""
    steer_modalities: str | List[str] = "all"
    cross_modal_steering: bool = True
    fusion_steering: bool = True

class MultimodalAdapter(BaseAdapter[MultimodalContent, Union[str, torch.Tensor]]):
    """Adapter for multimodal model steering (VLMs, audio-text, etc.)."""
    name = "multimodal"
    modality = Modality.MULTIMODAL
    MODEL_TYPE_LLAVA = "llava"
    MODEL_TYPE_QWEN_VL = "qwen_vl"
    MODEL_TYPE_IDEFICS = "idefics"
    MODEL_TYPE_IMAGEBIND = "imagebind"
    MODEL_TYPE_GENERIC = "generic"

    def __init__(self, model_name: str | None = None, model: nn.Module | None = None,
                 processor: Any | None = None, device: str | None = None,
                 model_type: str | None = None, **kwargs: Any):
        super().__init__(model=model, model_name=model_name, device=device, **kwargs)
        self._processor = processor
        self._model_type = model_type
        self._vision_encoder_layers: List[nn.Module] | None = None
        self._language_layers: List[nn.Module] | None = None
        self._cross_attention_layers: List[nn.Module] | None = None
        self._hidden_size: int | None = None

    def _detect_model_type(self) -> str:
        if self._model_type:
            return self._model_type
        name_lower = (self.model_name or "").lower()
        if "llava" in name_lower:
            return self.MODEL_TYPE_LLAVA
        elif "qwen" in name_lower and ("vl" in name_lower or "visual" in name_lower):
            return self.MODEL_TYPE_QWEN_VL
        elif "idefics" in name_lower:
            return self.MODEL_TYPE_IDEFICS
        elif "imagebind" in name_lower:
            return self.MODEL_TYPE_IMAGEBIND
        if self._model is not None:
            model_class = self._model.__class__.__name__.lower()
            if "llava" in model_class:
                return self.MODEL_TYPE_LLAVA
            elif "qwen" in model_class:
                return self.MODEL_TYPE_QWEN_VL
            elif "idefics" in model_class:
                return self.MODEL_TYPE_IDEFICS
        return self.MODEL_TYPE_GENERIC

    def _load_model(self) -> nn.Module:
        if self.model_name is None:
            raise AdapterError("model_name is required when model is not provided")
        model_type = self._detect_model_type() if self._model_type is None else self._model_type
        try:
            dtype = preferred_dtype(self.device)
            device_map = "auto" if self.device in ("cuda", "auto") else None
            if model_type == self.MODEL_TYPE_LLAVA:
                from transformers import LlavaForConditionalGeneration
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name, torch_dtype=dtype, device_map=device_map, **self._kwargs)
            elif model_type == self.MODEL_TYPE_QWEN_VL:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, trust_remote_code=True, torch_dtype=dtype,
                    device_map=device_map, **self._kwargs)
            elif model_type == self.MODEL_TYPE_IDEFICS:
                from transformers import IdeficsForVisionText2Text
                model = IdeficsForVisionText2Text.from_pretrained(
                    self.model_name, torch_dtype=dtype, device_map=device_map, **self._kwargs)
            else:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    self.model_name, trust_remote_code=True, **self._kwargs)
                if self.device and self.device not in ("auto",):
                    model = model.to(self.device)
            return model
        except ImportError as e:
            raise AdapterError(f"Required library not installed: {e}")
        except Exception as e:
            raise AdapterError(f"Failed to load model {self.model_name}: {e}")

    @property
    def processor(self) -> Any:
        if self._processor is None:
            if self.model_name is None:
                raise AdapterError("model_name required for processor loading")
            try:
                from transformers import AutoProcessor
                self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            except Exception as e:
                raise AdapterError(f"Failed to load processor: {e}")
        return self._processor

    def _resolve_vision_encoder_layers(self) -> List[nn.Module]:
        if self._vision_encoder_layers is not None:
            return self._vision_encoder_layers
        m = self.model
        candidates = [
            "vision_tower.vision_model.encoder.layers", "vision_model.encoder.layers",
            "visual.transformer.resblocks", "vision_encoder.layers", "image_encoder.layers",
        ]
        for path in candidates:
            obj = m
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, (nn.ModuleList, list)):
                    self._vision_encoder_layers = list(obj)
                    self._vision_path = path
                    return self._vision_encoder_layers
            except AttributeError:
                continue
        self._vision_encoder_layers = []
        return self._vision_encoder_layers

    def _resolve_language_layers(self) -> List[nn.Module]:
        if self._language_layers is not None:
            return self._language_layers
        m = self.model
        candidates = [
            "language_model.model.layers", "model.layers", "transformer.h",
            "llm.model.layers", "text_model.encoder.layers",
        ]
        for path in candidates:
            obj = m
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, (nn.ModuleList, list)):
                    self._language_layers = list(obj)
                    self._language_path = path
                    return self._language_layers
            except AttributeError:
                continue
        self._language_layers = []
        return self._language_layers

    @property
    def hidden_size(self) -> int:
        if self._hidden_size is not None:
            return self._hidden_size
        config = self.model.config
        for attr in ["hidden_size", "d_model", "text_config.hidden_size"]:
            parts = attr.split(".")
            obj = config
            try:
                for part in parts:
                    obj = getattr(obj, part)
                if isinstance(obj, int):
                    self._hidden_size = obj
                    return self._hidden_size
            except AttributeError:
                continue
        for p in self.model.parameters():
            if p.ndim >= 2:
                self._hidden_size = int(p.shape[-1])
                return self._hidden_size
        raise AdapterError("Could not determine hidden size")

    def _prepare_inputs(self, content: MultimodalContent) -> Dict[str, Any]:
        text = content.get_text()
        image = content.get_image()
        video = content.get_video()
        text_input = text.text if text else ""
        if image is not None:
            if image.pixels is None and image.file_path:
                from PIL import Image
                pil_image = Image.open(image.file_path).convert("RGB")
            else:
                pixels = image.to_tensor()
                if pixels.max() <= 1.0:
                    pixels = (pixels * 255).byte()
                pil_image = pixels.permute(1, 2, 0).numpy()
                from PIL import Image
                pil_image = Image.fromarray(pil_image)
            inputs = self.processor(text=text_input, images=pil_image, return_tensors="pt")
        elif video is not None:
            frames = video.to_tensor()
            from PIL import Image
            pil_frames = []
            for i in range(frames.shape[0]):
                frame = frames[i]
                if frame.max() <= 1.0:
                    frame = (frame * 255).byte()
                frame_np = frame.permute(1, 2, 0).numpy()
                pil_frames.append(Image.fromarray(frame_np))
            inputs = self.processor(text=text_input, images=pil_frames, return_tensors="pt")
        else:
            inputs = self.processor(text=text_input, return_tensors="pt")
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        return inputs

    def encode(self, content: MultimodalContent) -> torch.Tensor:
        inputs = self._prepare_inputs(content)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states"):
                return outputs.hidden_states[-1]
            else:
                raise AdapterError("Model output format not recognized")

    def decode(self, latent: torch.Tensor, *, display_token_limit: int) -> str:
        from wisent.core.primitives.model_interface.adapters.modalities.core.multimodal_decode import (
            decode_llava, decode_qwen_vl, decode_idefics, decode_generic,
        )
        model_type = self._detect_model_type()
        with torch.no_grad():
            if latent.dim() == 2:
                latent = latent.unsqueeze(0)
            latent = latent.to(self.model.device)
            if model_type == self.MODEL_TYPE_LLAVA:
                return decode_llava(self.model, latent, self.processor, display_token_limit=display_token_limit)
            elif model_type == self.MODEL_TYPE_QWEN_VL:
                return decode_qwen_vl(self.model, latent, self.processor, display_token_limit=display_token_limit)
            elif model_type == self.MODEL_TYPE_IDEFICS:
                return decode_idefics(self.model, latent, self.processor, display_token_limit=display_token_limit)
            else:
                return decode_generic(self.model, latent, self.processor, display_token_limit=display_token_limit)

    def get_intervention_points(self) -> List[InterventionPoint]:
        from wisent.core.primitives.model_interface.adapters.modalities.core.multimodal_steering import get_intervention_points_multimodal
        return get_intervention_points_multimodal(self)

    def extract_activations(self, content: MultimodalContent, layers: List[str] | None = None) -> LayerActivations:
        from wisent.core.primitives.model_interface.adapters.modalities.core.multimodal_steering import extract_activations_multimodal
        return extract_activations_multimodal(self, content, layers)

    def forward_with_steering(self, content: MultimodalContent, steering_vectors: LayerActivations,
                              config: SteeringConfig | MultimodalSteeringConfig | None = None) -> str:
        from wisent.core.primitives.model_interface.adapters.modalities.core.multimodal_steering import forward_with_steering_multimodal
        return forward_with_steering_multimodal(self, content, steering_vectors, config)

    def _generate_unsteered(self, content: MultimodalContent, **kwargs: Any) -> str:
        from wisent.core.primitives.model_interface.adapters.modalities.core.multimodal_steering import generate_unsteered_multimodal
        return generate_unsteered_multimodal(self, content, **kwargs)

    def generate(self, content: MultimodalContent, steering_vectors: LayerActivations | None = None,
                 config: SteeringConfig | MultimodalSteeringConfig | None = None, **generation_kwargs: Any) -> str:
        if steering_vectors is not None:
            return self.forward_with_steering(content, steering_vectors, config)
        return self._generate_unsteered(content, **generation_kwargs)

    def compute_cross_modal_steering_vector(self, positive_content: MultimodalContent,
                                            negative_content: MultimodalContent, layer: str) -> torch.Tensor:
        from wisent.core.primitives.model_interface.adapters.modalities.core.multimodal_steering import compute_cross_modal_steering_vector
        return compute_cross_modal_steering_vector(self, positive_content, negative_content, layer)
