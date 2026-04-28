"""
Transformer-specific analysis for different components.

Analyze activations from different transformer components
(attention, MLP, residual) to find optimal intervention points.
"""

import torch
import numpy as np
from enum import Enum
from typing import Dict, Any, List, Optional
"""Formerly imported CV_FOLDS is now a required parameter."""


class TransformerComponent(Enum):
    """Transformer component types for activation extraction."""
    RESIDUAL = "residual"
    ATTENTION = "attention"
    MLP = "mlp"
    ATTENTION_OUTPUT = "attention_output"
    MLP_OUTPUT = "mlp_output"
    PER_HEAD = "per_head"
    MLP_INTERMEDIATE = "mlp_intermediate"
    POST_ATTN_RESIDUAL = "post_attn_residual"
    PRE_ATTN_LAYERNORM = "pre_attn_layernorm"
    Q_PROJ = "q_proj"
    K_PROJ = "k_proj"
    V_PROJ = "v_proj"
    MLP_GATE = "mlp_gate"
    ATTENTION_SCORES = "attention_scores"
    EMBEDDING = "embedding"
    FINAL_LAYERNORM = "final_layernorm"
    LOGITS = "logits"


def get_component_hook_points(
    model_type: str,
    layer: int,
    component: TransformerComponent,
) -> List[str]:
    """
    Get hook point names for a given component.
    
    Different model architectures use different naming conventions.
    """
    # gpt_oss (and similar modern decoder-only families) use Llama-style
    # `model.layers.{i}.self_attn.{q,k,v,o}_proj` paths despite their
    # model_type containing 'gpt'. Verified via the model.safetensors.index
    # weight_map for openai/gpt-oss-20b which references model.layers.0.*.
    # Match these BEFORE the generic 'gpt' branch (which is GPT-2 style).
    mt = model_type.lower()
    if "llama" in mt or "mistral" in mt or "gpt_oss" in mt or "gptoss" in mt:
        base = f"model.layers.{layer}"
        mapping = {
            TransformerComponent.RESIDUAL: [base],
            TransformerComponent.ATTENTION: [f"{base}.self_attn"],
            TransformerComponent.MLP: [f"{base}.mlp"],
            TransformerComponent.ATTENTION_OUTPUT: [f"{base}.self_attn.o_proj"],
            TransformerComponent.MLP_OUTPUT: [f"{base}.mlp.down_proj"],
            TransformerComponent.PER_HEAD: [f"{base}.self_attn.o_proj"],
            TransformerComponent.MLP_INTERMEDIATE: [f"{base}.mlp.down_proj"],
            TransformerComponent.POST_ATTN_RESIDUAL: [f"{base}.post_attention_layernorm"],
            TransformerComponent.PRE_ATTN_LAYERNORM: [f"{base}.input_layernorm"],
            TransformerComponent.Q_PROJ: [f"{base}.self_attn.q_proj"],
            TransformerComponent.K_PROJ: [f"{base}.self_attn.k_proj"],
            TransformerComponent.V_PROJ: [f"{base}.self_attn.v_proj"],
            TransformerComponent.MLP_GATE: [f"{base}.mlp.gate_proj"],
            TransformerComponent.ATTENTION_SCORES: [f"{base}.self_attn.q_proj"],
            TransformerComponent.EMBEDDING: ["model.embed_tokens"],
            TransformerComponent.FINAL_LAYERNORM: ["model.norm"],
            TransformerComponent.LOGITS: ["lm_head"],
        }
        return mapping.get(component, [base])

    elif "qwen" in model_type.lower():
        base = f"model.layers.{layer}"
        mapping = {
            TransformerComponent.RESIDUAL: [base],
            TransformerComponent.ATTENTION: [f"{base}.self_attn"],
            TransformerComponent.MLP: [f"{base}.mlp"],
            TransformerComponent.ATTENTION_OUTPUT: [f"{base}.self_attn.o_proj"],
            TransformerComponent.MLP_OUTPUT: [f"{base}.mlp.down_proj"],
            TransformerComponent.PER_HEAD: [f"{base}.self_attn.o_proj"],
            TransformerComponent.MLP_INTERMEDIATE: [f"{base}.mlp.down_proj"],
            TransformerComponent.POST_ATTN_RESIDUAL: [f"{base}.post_attention_layernorm"],
            TransformerComponent.PRE_ATTN_LAYERNORM: [f"{base}.input_layernorm"],
            TransformerComponent.Q_PROJ: [f"{base}.self_attn.q_proj"],
            TransformerComponent.K_PROJ: [f"{base}.self_attn.k_proj"],
            TransformerComponent.V_PROJ: [f"{base}.self_attn.v_proj"],
            TransformerComponent.MLP_GATE: [f"{base}.mlp.gate_proj"],
            TransformerComponent.ATTENTION_SCORES: [f"{base}.self_attn.q_proj"],
            TransformerComponent.EMBEDDING: ["model.embed_tokens"],
            TransformerComponent.FINAL_LAYERNORM: ["model.norm"],
            TransformerComponent.LOGITS: ["lm_head"],
        }
        return mapping.get(component, [base])

    elif "gpt" in model_type.lower():
        base = f"transformer.h.{layer}"
        mapping = {
            TransformerComponent.RESIDUAL: [base],
            TransformerComponent.ATTENTION: [f"{base}.attn"],
            TransformerComponent.MLP: [f"{base}.mlp"],
            TransformerComponent.ATTENTION_OUTPUT: [f"{base}.attn.c_proj"],
            TransformerComponent.MLP_OUTPUT: [f"{base}.mlp.c_proj"],
            TransformerComponent.PER_HEAD: [f"{base}.attn.c_proj"],
            TransformerComponent.MLP_INTERMEDIATE: [f"{base}.mlp.c_proj"],
            TransformerComponent.POST_ATTN_RESIDUAL: [f"{base}.ln_2"],
            TransformerComponent.PRE_ATTN_LAYERNORM: [f"{base}.ln_1"],
            TransformerComponent.Q_PROJ: [f"{base}.attn.c_attn"],
            TransformerComponent.K_PROJ: [f"{base}.attn.c_attn"],
            TransformerComponent.V_PROJ: [f"{base}.attn.c_attn"],
            TransformerComponent.MLP_GATE: [f"{base}.mlp.c_fc"],
            TransformerComponent.ATTENTION_SCORES: [f"{base}.attn.c_attn"],
            TransformerComponent.EMBEDDING: ["transformer.wte"],
            TransformerComponent.FINAL_LAYERNORM: ["transformer.ln_f"],
            TransformerComponent.LOGITS: ["lm_head"],
        }
        return mapping.get(component, [base])

    # Default pattern (Llama-like)
    base = f"layers.{layer}"
    return [base]


def analyze_transformer_components(
    model,
    tokenizer,
    pos_texts: List[str],
    neg_texts: List[str],
    layer: int,
    device: str,
) -> Dict[str, Any]:
    """
    Analyze different transformer components for steering potential.
    
    Compares residual stream, attention output, and MLP output.
    """
    from wisent.core.reading.modules.utilities.metrics.probe.probe_metrics import compute_linear_probe_accuracy, compute_signal_strength

    results = {}
    model_type = model.config.model_type if hasattr(model.config, 'model_type') else "unknown"
    
    for component in [TransformerComponent.RESIDUAL, TransformerComponent.MLP]:
        try:
            hook_points = get_component_hook_points(model_type, layer, component)
            
            # This is a simplified version - full implementation would use hooks
            # For now, we just return placeholder
            results[component.value] = {
                "hook_points": hook_points,
                "status": "not_implemented",
            }
        except Exception as e:
            results[component.value] = {"error": str(e)}
    
    return results


def compare_components_for_benchmark(
    model,
    tokenizer,
    pos_activations_by_component: Dict[str, torch.Tensor],
    neg_activations_by_component: Dict[str, torch.Tensor],
    min_clusters: int,
    *,
    cv_folds: int,
    probe_min_per_class: int,
    blend_default: float,
) -> Dict[str, Any]:
    """Compare which component has best steering signal for a benchmark."""
    from wisent.core.reading.modules.utilities.metrics.probe.probe_metrics import compute_linear_probe_accuracy
    from wisent.core.reading.modules.modules.steering.analysis.steerability import compute_steerability_metrics

    results = {}

    for component in pos_activations_by_component:
        if component not in neg_activations_by_component:
            continue

        pos = pos_activations_by_component[component]
        neg = neg_activations_by_component[component]

        try:
            linear_acc = compute_linear_probe_accuracy(pos, neg, cv_folds, probe_min_per_class=probe_min_per_class, blend_default=blend_default)
            steerability = compute_steerability_metrics(pos, neg, min_clusters=min_clusters)

            results[component] = {
                "linear_accuracy": linear_acc,
                "caa_probe_alignment": steerability.get("caa_probe_alignment"),
                "diff_mean_alignment": steerability.get("diff_mean_alignment"),
            }
        except Exception as e:
            results[component] = {"error": str(e)}
    
    # Find best component
    if results:
        best_component = max(
            results.keys(),
            key=lambda c: results[c].get("linear_accuracy", 0) if "error" not in results[c] else 0
        )
        results["best_component"] = best_component
    
    return results


def compare_concept_granularity(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    layers: List[int],
    model=None,
    tokenizer=None,
) -> Dict[str, Any]:
    """
    Compare concept representation across different layers.
    
    Earlier layers often have more fine-grained concepts,
    later layers have more abstract concepts.
    """
    from wisent.core.reading.modules.utilities.metrics.probe.probe_metrics import compute_linear_probe_accuracy
    from wisent.core.reading.modules.modules.geo_utils.icd import compute_icd

    # This would need per-layer activations
    # Placeholder implementation
    return {
        "layers": layers,
        "status": "requires_per_layer_activations",
    }
