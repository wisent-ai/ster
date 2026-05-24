"""Multimodal adapter: steering, generation, and cross-modal methods."""
from __future__ import annotations
from typing import Any, Dict, List, Union
import torch
import torch.nn as nn
from wisent.core.primitives.model_interface.adapters.base import SteeringConfig, InterventionPoint
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations
from wisent.core.primitives.models.config import get_generate_kwargs

__all__ = [
    "get_intervention_points_multimodal",
    "extract_activations_multimodal",
    "forward_with_steering_multimodal",
    "generate_unsteered_multimodal",
    "compute_cross_modal_steering_vector",
]


def get_intervention_points_multimodal(adapter) -> List[InterventionPoint]:
    """Get available intervention points across all modality encoders."""
    points = []
    vision_layers = adapter._resolve_vision_encoder_layers()
    for i, _ in enumerate(vision_layers):
        recommended = i >= len(vision_layers) // 2
        points.append(InterventionPoint(
            name=f"vision.{i}", module_path=f"{adapter._vision_path}.{i}",
            description=f"Vision encoder layer {i}", recommended=recommended,
        ))
    language_layers = adapter._resolve_language_layers()
    for i, _ in enumerate(language_layers):
        recommended = (len(language_layers) // 3) <= i <= (2 * len(language_layers) // 3)
        points.append(InterventionPoint(
            name=f"language.{i}", module_path=f"{adapter._language_path}.{i}",
            description=f"Language model layer {i}", recommended=recommended,
        ))
    m = adapter.model
    projection_candidates = [
        ("multi_modal_projector", "Multimodal projector"),
        ("mm_projector", "MM projector"),
        ("vision_projection", "Vision projection"),
    ]
    for path, desc in projection_candidates:
        try:
            module = getattr(m, path, None)
            if module is not None:
                points.append(InterventionPoint(
                    name="projection", module_path=path,
                    description=desc, recommended=True,
                ))
                break
        except AttributeError:
            continue
    return points


def extract_activations_multimodal(
    adapter, content, layers: List[str] | None = None,
) -> LayerActivations:
    """Extract activations from multimodal model layers."""
    all_points = {ip.name: ip for ip in adapter.get_intervention_points()}
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
            module = adapter._get_module_by_path(ip.module_path)
            if module is not None:
                handle = module.register_forward_hook(make_hook(layer_name))
                hooks.append(handle)
        inputs = adapter._prepare_inputs(content)
        with torch.no_grad():
            adapter.model(**inputs)
    finally:
        for handle in hooks:
            handle.remove()
    return LayerActivations(activations)


def forward_with_steering_multimodal(adapter, content, steering_vectors, config=None) -> str:
    """Generate output with steering applied."""
    from wisent.core.primitives.model_interface.adapters.modalities.core.multimodal import MultimodalSteeringConfig
    config = config or MultimodalSteeringConfig()
    if isinstance(config, MultimodalSteeringConfig):
        if config.steer_modalities != "all":
            modalities = (
                [config.steer_modalities] if isinstance(config.steer_modalities, str)
                else config.steer_modalities
            )
            filtered = {}
            for name, vec in steering_vectors.items():
                modality = name.split(".")[0]
                if modality in modalities or name == "projection":
                    filtered[name] = vec
            steering_vectors = LayerActivations(filtered)
    inputs = adapter._prepare_inputs(content)
    gen_kwargs = get_generate_kwargs()
    with adapter._steering_hooks(steering_vectors, config):
        with torch.no_grad():
            outputs = adapter.model.generate(
                **inputs, **gen_kwargs,
            )
    generated = adapter.processor.decode(outputs[0], skip_special_tokens=True)
    return generated


def generate_unsteered_multimodal(
    adapter, content, **kwargs,
) -> str:
    """Generate output without steering."""
    inputs = adapter._prepare_inputs(content)
    gen_kwargs = get_generate_kwargs(**kwargs)
    with torch.no_grad():
        outputs = adapter.model.generate(
            **inputs, **gen_kwargs,
        )
    return adapter.processor.decode(outputs[0], skip_special_tokens=True)


def compute_cross_modal_steering_vector(
    adapter, positive_content, negative_content, layer: str,
) -> torch.Tensor:
    """Compute steering vector from multimodal content pairs."""
    pos_acts = adapter.extract_activations(positive_content, [layer])
    neg_acts = adapter.extract_activations(negative_content, [layer])
    pos_tensor = pos_acts[layer]
    neg_tensor = neg_acts[layer]
    pos_pooled = pos_tensor.mean(dim=1) if pos_tensor.dim() > 2 else pos_tensor.mean(dim=0)
    neg_pooled = neg_tensor.mean(dim=1) if neg_tensor.dim() > 2 else neg_tensor.mean(dim=0)
    return pos_pooled.squeeze() - neg_pooled.squeeze()
