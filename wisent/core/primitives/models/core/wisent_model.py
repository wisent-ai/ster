from __future__ import annotations

from wisent.core.primitives.models.layer import extract_token_ids

import logging
import random
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer
)


# HF Hub load helpers extracted to ._hf 2026-05-10. The 8-attempt 429
# retry was removed at the same time — wait_for_hf_token (installed
# at top-level wisent import) covers every HfApi method that touches
# the network in 0.11.35, so 429s should never reach this layer.
from wisent.core.primitives.models.core._hf import (
    _is_hf_rate_limit_exc,
    _retry_on_hf_rate_limit,
    _load_cache_first,
)



from wisent.core.primitives.models.core.atoms import SteeringPlan, SteeringVector, HookHandleGroup, GenerationStats, TopLogits
from wisent.core.primitives.model_interface.core.activations.core.atoms import RawActivationMap

from wisent.core.control.generation.prompts.core.atom import ChatMessage
from wisent.core.utils import resolve_default_device, resolve_torch_device, preferred_dtype
from wisent.core.utils.config_tools.constants import STEERING_SCALE_IDENTITY
from wisent.core.primitives.contrastive_pairs.diagnostics import run_control_steering_diagnostics
from wisent.core.utils.infra_tools.errors import (
    ChatTemplateNotAvailableError,
    DecoderLayersNotFoundError,
    HiddenSizeNotFoundError,
    TokenizerMissingMethodError,
    ControlVectorDiagnosticsError,
    LayerNotFoundError,
    InsufficientDataError,
)

import threading

from wisent.core.primitives.models._model_parts.steering_application import (
    _apply_steering_object,
    _encode_one,
    _batch_encode,
    _extract_assistant_response,
)
from wisent.core.primitives.models._model_parts.generation import (
    _generate,
    _set_steering_from_raw,
    _clear_steering,
)
from wisent.core.primitives.models._model_parts.generation_analytics import _generate_with_stats
from wisent.core.primitives.models._model_parts.generation_streaming import _generate_stream

__all__ = ["WisentModel"]


logger = logging.getLogger(__name__)

class WisentModel:
    """
    Wrapper around a causal LM (HF transformers) with steering capabilities.

    atributes:
        model_name:
            HF repo id or local path.
        device:
            'cuda', 'cuda:0', 'cpu', etc. If None, leave to HF defaults/accelerate.
        hf_model:
            the loaded PreTrainedModel instance.
        tokenizer:
            the loaded PreTrainedTokenizerBase instance.
        hidden_size:
            model hidden size (last dim of residual stream).
        num_layers:
            number of decoder blocks we can hook.
        _steering_plan:
            current SteeringPlan (can be empty).
        _hook_group:
            manages active hooks for clean detach.
    """
    def __init__(
            self,
            model_name: str,
            steering_layers: list[RawActivationMap] | RawActivationMap | None = None,
            steering_weights: list[float] | None = None,
            layers_description: list[str] | None = None,
            device: str | None = None,
            hf_model: AutoModelForCausalLM | None = None
        ):
        """
        Initialize the wrapper (model + tokenizer + default steering plan).

        arguments:
            model_name:
                HF repo id or local path.
            steering_layers:
                list of RawActivationMap or single RawActivationMap of steering vectors.
            steering_weights:
                list of weights for each steering vector, optional.
            device:
                'cuda', 'cuda:0', 'cpu', etc. If None, leave to HF defaults/accelerate.
            hf_model:
                optional preloaded model (skips from_pretrained if provided).
        """
        self.model_name = model_name
        self.device = resolve_default_device() if device is None or device == "auto" else device

        # Determine appropriate dtype and settings for the device
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
        }

        load_kwargs["torch_dtype"] = preferred_dtype(self.device)
        if self.device == "mps":
            load_kwargs["device_map"] = "mps"
        elif self.device == "cuda" or self.device == "auto":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = None

        self.hf_model: PreTrainedModel = hf_model or _load_cache_first(
            AutoModelForCausalLM.from_pretrained,
            model_name,
            **load_kwargs,
        )

        device_map_used = load_kwargs.get("device_map")

        # Only move to device if device_map wasn't used
        if device_map_used is None:
            self.hf_model.to(self.device)

        self.tokenizer: PreTrainedTokenizerBase = _load_cache_first(
            AutoTokenizer.from_pretrained,
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )

        if not self._is_chat_tokenizer():
            raise TokenizerMissingMethodError("apply_chat_template")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.hf_model.generation_config, "pad_token_id", None) is None:
            self.hf_model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self._steering_plan: SteeringPlan = SteeringPlan.from_raw(
            raw=steering_layers,
            scale=STEERING_SCALE_IDENTITY,
            weights=steering_weights,
            layers_description=layers_description,
            )
        self._hook_group = HookHandleGroup()

        self._layers, self._hidden_size = self._resolve_decoder_layers_and_hidden()


    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def _resolve_decoder_layers_and_hidden(self) -> tuple[list[nn.Module], int]:
        m = self.hf_model
        hidden_size = getattr(m.config, "hidden_size", None) or getattr(m.config, "n_embd", None)
        layers: list[nn.Module] = []

        candidates = [
            "layers",
            "model.layers",
            "model.decoder.layers",
            "transformer.h",
            "base_model.model.layers",
            "blocks", "model.blocks",
            "gpt_neox.layers",
        ]
        for path in candidates:
            obj = m
            try:
                for attr in path.split("."):
                    if attr:
                        obj = getattr(obj, attr)
                if (isinstance(obj, nn.ModuleList) or
                    (isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], nn.Module))):
                    layers = list(obj)
                    break
            except AttributeError:
                continue

        if not layers:
            raise DecoderLayersNotFoundError()

        if hidden_size is None:
            for p in m.parameters():
                if p.ndim >= 2:
                    hidden_size = int(p.shape[-1]); break
        if hidden_size is None:
            raise HiddenSizeNotFoundError()

        return layers, int(hidden_size)

    def _is_chat_tokenizer(self) -> bool:
        return hasattr(self.tokenizer, "apply_chat_template") and callable(
            getattr(self.tokenizer, "apply_chat_template"))

    def apply_steering(self, plan: SteeringPlan | None = None) -> None:
        """
        Register forward hooks to add steering vectors *after* the selected decoder blocks.
        If plan is None, use the internal plan set at init or via set_steering_from_raw().
        """
        p = plan or self._steering_plan
        if p.is_empty():
            return

        p.validate_hidden_size(hidden_size=self._hidden_size)
        self.detach()

        name_to_index = {str(i + 1): i for i in range(len(self._layers))}

        for lname, vec in p.layers.items():
            if lname not in name_to_index:
                continue
            idx = name_to_index[lname]
            layer = self._layers[idx]

            def _hook_factory(v: SteeringVector):
                def _hook(_mod: nn.Module, _inp: tuple, out: torch.Tensor | tuple) -> torch.Tensor | tuple:
                    if isinstance(out, tuple):
                        hs = out[0]
                        delta = torch.zeros_like(hs)
                        delta = delta + v.materialize(hs)
                        return (hs + delta,) + out[1:]
                    else:
                        hs = out
                        delta = torch.zeros_like(hs)
                        delta = delta + v.materialize(hs)
                        return hs + delta
                return _hook

            handle = layer.register_forward_hook(_hook_factory(vec))
            self._hook_group.add(handle)

    def detach(self) -> None:
        """Remove all registered steering hooks; model returns to unsteered behavior."""
        self._hook_group.remove_all()

    @contextmanager
    def detached(self):
        """Context manager: guarantees a vanilla (unsteered) model inside the block."""
        self.detach()
        try:
            yield
        finally:
            self.detach()

    # -- Methods imported from part files for 300-line compliance --
    apply_steering_object = _apply_steering_object
    _encode_one = _encode_one
    _batch_encode = _batch_encode
    _extract_assistant_response = _extract_assistant_response
    generate = _generate
    generate_with_stats = _generate_with_stats
    generate_stream = _generate_stream
    set_steering_from_raw = _set_steering_from_raw
    clear_steering = _clear_steering
