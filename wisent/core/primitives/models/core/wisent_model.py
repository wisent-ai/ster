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


# HF Hub returns 429 on the public 1000-req-per-5-min ceiling once enough
# concurrent VMs are loading the same model. AutoTokenizer.from_pretrained
# silently calls model_info() (via is_base_mistral) and AutoModelForCausalLM
# .from_pretrained does its own metadata calls; both raise HfHubHTTPError 429
# directly. We chain-walk the cause/context tree because the same exception
# can be re-wrapped by transformers' tokenization_utils_base around the raw
# huggingface_hub error.
_HF_RETRY_MAX_ATTEMPTS = 8
_HF_RETRY_BASE_WAIT_S = 15.0
_HF_RETRY_CAP_S = 600.0


def _is_hf_rate_limit_exc(exc: BaseException) -> bool:
    """Match any of:
    - HfHubHTTPError 429 directly (msg contains '429' or 'Too Many Requests')
    - HF Hub's English rate-limit response ('We had to rate limit you...')
    - transformers' aggregated cached_files OSError 'does not appear to have
      files named (...)' — that error swallows per-shard HfHubHTTPError 429s
      and re-raises a non-429 OSError, so we treat the *shape* of the message
      as a strong signal of a transient HF 429 burst whenever we know the
      repo exists. False positives only fire for genuinely-missing files;
      worst case is wait_total ~= sum(backoff)*8 ~= 35min before failing.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = str(cur)
        msg_l = msg.lower()
        if (
            "429" in msg
            or "too many requests" in msg_l
            or "rate limit" in msg_l
            or "rate-limit" in msg_l
            or "we had to rate limit" in msg_l
            or "does not appear to have files named" in msg
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _retry_on_hf_rate_limit(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Call fn, retry only on HF 429 with exponential backoff + jitter.

    Non-429 exceptions propagate immediately (we are not a generic catch-all).
    Backoff capped at _HF_RETRY_CAP_S so a long outage isn't stuck waiting
    hours.
    """
    last_err: BaseException | None = None
    for attempt in range(_HF_RETRY_MAX_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not _is_hf_rate_limit_exc(e):
                raise
            last_err = e
            wait = min(
                _HF_RETRY_CAP_S,
                _HF_RETRY_BASE_WAIT_S * (2 ** attempt) + random.uniform(0, 5),
            )
            print(
                f"[hf-retry] attempt {attempt + 1}/{_HF_RETRY_MAX_ATTEMPTS} "
                f"hit HF 429 in {fn.__name__}; sleeping {wait:.0f}s",
                flush=True,
            )
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def _load_cache_first(
    fn: Callable[..., Any], model_name: str, **kwargs: Any
) -> Any:
    """Try loading from local HF cache without any network calls; on miss
    fall through to network with HF-429 retry.

    This is the single biggest lever we have for cutting HF Hub traffic on a
    fleet of agent VMs: once a VM has fetched a given model once, every
    subsequent job on that VM avoids ALL HF Hub calls (no model_info, no
    file metadata, no etag refresh, no shard listing). Without this, every
    job's WisentModel.__init__ hammers HF Hub even when the files are
    already on local disk — and at 50+ concurrent VMs that exceeds the
    1000-req/5min anonymous+free-tier ceiling.

    Phase 1 sets local_files_only=True so transformers/huggingface_hub stay
    fully offline. Any failure (cache miss, partial cache, malformed cache)
    propagates as some flavor of OSError / LocalEntryNotFoundError /
    HFValidationError; we swallow all of those and try Phase 2 on the
    network. Phase 2 is the same call without local_files_only, wrapped in
    our existing 429 backoff. If the model genuinely doesn't exist on the
    Hub, Phase 2 will surface the real error.
    """
    if kwargs.get("local_files_only"):
        return _retry_on_hf_rate_limit(fn, model_name, **kwargs)
    try:
        return fn(model_name, local_files_only=True, **kwargs)
    except Exception as e:
        # Surface 429 immediately so we don't double-hammer the Hub: the
        # local-only path can't hit 429 itself, but defensively if anything
        # in the cache layer ever emits a 429-shaped message we want our
        # retry path to handle it instead of falling through unconditionally.
        if _is_hf_rate_limit_exc(e):
            return _retry_on_hf_rate_limit(fn, model_name, **kwargs)
    return _retry_on_hf_rate_limit(fn, model_name, **kwargs)



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
