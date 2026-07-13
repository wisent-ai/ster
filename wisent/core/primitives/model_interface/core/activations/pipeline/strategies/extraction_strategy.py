"""
Unified extraction strategies for activation collection.

These strategies combine prompt construction and token extraction into a single
unified approach, based on empirical testing of what actually works.
"""

from enum import Enum
from typing import Tuple, Optional
import argparse
import torch
from wisent.core.utils.config_tools.constants import ROLE_PLAY_TOKENS

class ExtractionStrategy(str, Enum):
    """
    Unified extraction strategies combining prompt format and token selection.

    These replace the old separate PromptConstructionStrategy and ActivationAggregationStrategy.
    """

    CHAT_MEAN = "chat_mean"
    """Chat template prompt with Q+A, extract mean of answer tokens."""

    CHAT_FIRST = "chat_first"
    """Chat template prompt with Q+A, extract first answer token."""

    CHAT_LAST = "chat_last"
    """Chat template prompt with Q+A, extract EOT token (has seen full answer)."""

    CHAT_MAX_NORM = "chat_max_norm"
    """Chat template prompt with Q+A, extract token with max norm in answer region."""

    CHAT_WEIGHTED = "chat_weighted"
    """Chat template prompt with Q+A, position-weighted mean (earlier tokens weighted more)."""

    ROLE_PLAY = "role_play"
    """'Behave like person who answers Q with A' format, extract EOT token."""

    MC_BALANCED = "mc_balanced"
    """Multiple choice format with balanced A/B assignment, extract the A/B choice token."""

    # Base model strategies (no chat template required)
    COMPLETION_LAST = "completion_last"
    """Direct Q+A completion without chat template, extract last token. For base models."""

    COMPLETION_MEAN = "completion_mean"
    """Direct Q+A completion without chat template, extract mean of answer tokens. For base models."""

    MC_COMPLETION = "mc_completion"
    """Multiple choice without chat template, extract A/B token. For base models."""

    # Optimal extraction (requires raw activations + steering direction)
    CHAT_OPTIMAL = "chat_optimal"
    """Extract at token with max signal. Requires raw activations and steering direction."""

    @property
    def description(self) -> str:
        descriptions = {
            ExtractionStrategy.CHAT_MEAN: "Chat template with mean of answer tokens",
            ExtractionStrategy.CHAT_FIRST: "Chat template with first answer token",
            ExtractionStrategy.CHAT_LAST: "Chat template with EOT token",
            ExtractionStrategy.CHAT_MAX_NORM: "Chat template with max-norm answer token",
            ExtractionStrategy.CHAT_WEIGHTED: "Chat template with position-weighted mean",
            ExtractionStrategy.ROLE_PLAY: "Role-playing format with EOT token",
            ExtractionStrategy.MC_BALANCED: "Balanced multiple choice with A/B token",
            ExtractionStrategy.COMPLETION_LAST: "Direct completion with last token (base models)",
            ExtractionStrategy.COMPLETION_MEAN: "Direct completion with mean of answer tokens (base models)",
            ExtractionStrategy.MC_COMPLETION: "Multiple choice completion with A/B token (base models)",
            ExtractionStrategy.CHAT_OPTIMAL: "Chat template with optimal position extraction (requires steering direction)",
        }
        return descriptions.get(self, "Unknown strategy")

    @classmethod
    def default(cls) -> "ExtractionStrategy":
        """Return the empirically optimal strategy.

        Reads from parameters_to_validate.json via get_optimal_extraction_strategy().
        """
        from wisent.core.control.steering_methods.configs.optimal import (
            get_optimal_extraction_strategy,
        )
        return cls(get_optimal_extraction_strategy())

    @classmethod
    def list_all(cls) -> list[str]:
        """List all strategy names."""
        return [s.value for s in cls]

    @classmethod
    def for_tokenizer(cls, tokenizer, prefer_mc: bool = False) -> "ExtractionStrategy":
        """
        Select the appropriate strategy based on whether tokenizer supports chat template.
        """
        has_chat = (hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template"))
                    and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None)

        if has_chat:
            return cls.MC_BALANCED if prefer_mc else cls.CHAT_LAST
        else:
            return cls.MC_COMPLETION if prefer_mc else cls.COMPLETION_LAST

    @classmethod
    def is_base_model_strategy(cls, strategy: "ExtractionStrategy") -> bool:
        """Check if a strategy is designed for base models (no chat template)."""
        return strategy in (cls.COMPLETION_LAST, cls.COMPLETION_MEAN, cls.MC_COMPLETION)

    @classmethod
    def get_equivalent_for_model_type(cls, strategy: "ExtractionStrategy", tokenizer) -> "ExtractionStrategy":
        """
        Get the equivalent strategy for the given tokenizer type.

        If strategy requires chat template but tokenizer doesn't have it,
        returns the base model equivalent. And vice versa.
        """
        has_chat = (hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template"))
                    and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None)
        is_base_strategy = cls.is_base_model_strategy(strategy)

        if has_chat and is_base_strategy:
            mapping = {
                cls.COMPLETION_LAST: cls.CHAT_LAST,
                cls.COMPLETION_MEAN: cls.CHAT_MEAN,
                cls.MC_COMPLETION: cls.MC_BALANCED,
            }
            return mapping.get(strategy, strategy)

        elif not has_chat and not is_base_strategy:
            mapping = {
                cls.CHAT_LAST: cls.COMPLETION_LAST,
                cls.CHAT_FIRST: cls.COMPLETION_LAST,
                cls.CHAT_MEAN: cls.COMPLETION_MEAN,
                cls.CHAT_MAX_NORM: cls.COMPLETION_LAST,
                cls.CHAT_WEIGHTED: cls.COMPLETION_MEAN,
                cls.ROLE_PLAY: cls.COMPLETION_LAST,
                cls.MC_BALANCED: cls.MC_COMPLETION,
            }
            return mapping.get(strategy, cls.COMPLETION_LAST)

        return strategy

class ExtractionComponent(str, Enum):
    """
    Where in the transformer to extract activations from.

    Orthogonal to ExtractionStrategy (which controls prompt format + token position).
    Component controls which part of the transformer block to read.
    """
    RESIDUAL_STREAM = "residual_stream"
    """Full layer output (default). Uses output_hidden_states, no hooks needed."""
    ATTN_OUTPUT = "attn_output"
    """Self-attention block output before residual add. Forward hook on self_attn."""
    MLP_OUTPUT = "mlp_output"
    """MLP block output before residual add. Forward hook on mlp."""
    PER_HEAD = "per_head"
    """Individual attention head outputs (pre-o_proj). Pre-hook on o_proj."""
    MLP_INTERMEDIATE = "mlp_intermediate"
    """High-dim state inside MLP (input to down_proj). Pre-hook on down_proj."""
    POST_ATTN_RESIDUAL = "post_attn_residual"
    """After attention add, before MLP. Pre-hook on post_attention_layernorm."""
    PRE_ATTN_LAYERNORM = "pre_attn_layernorm"
    """Input to self_attn after input_layernorm. Forward hook on input_layernorm."""
    EMBEDDING_OUTPUT = "embedding_output"
    """Token embeddings before layer 0. Forward hook on embed_tokens. Global (not per-layer)."""
    FINAL_LAYERNORM = "final_layernorm"
    """After last layer layernorm, before lm_head. Forward hook on model.norm. Global."""
    Q_PROJ = "q_proj"
    """Query projection output per layer. Forward hook on self_attn.q_proj."""
    K_PROJ = "k_proj"
    """Key projection output per layer. Forward hook on self_attn.k_proj."""
    V_PROJ = "v_proj"
    """Value projection output per layer. Forward hook on self_attn.v_proj."""
    MLP_GATE_ACTIVATION = "mlp_gate_activation"
    """Gate projection output in SwiGLU MLP (pre-SiLU). Forward hook on mlp.gate_proj."""
    ATTENTION_SCORES = "attention_scores"
    """Per-head attention score scaling via Q-proj. Dim = num_heads."""
    LOGITS = "logits"
    """Output of lm_head (vocab space). Forward hook on lm_head. Global."""
    KV_CACHE = "kv_cache"
    """KV cache (past_key_values). No hooks — reads cache directly after forward pass."""
    @classmethod
    def default(cls) -> "ExtractionComponent":
        return cls.RESIDUAL_STREAM
    @classmethod
    def list_all(cls) -> list[str]:
        return [c.value for c in cls]
    @property
    def needs_hooks(self) -> bool:
        """Whether this component requires forward hooks (vs hidden_states)."""
        return self not in (ExtractionComponent.RESIDUAL_STREAM, ExtractionComponent.KV_CACHE)
    @property
    def needs_cache(self) -> bool:
        """Whether this component reads from past_key_values instead of hidden states."""
        return self == ExtractionComponent.KV_CACHE

def tokenizer_has_chat_template(tokenizer) -> bool:
    """Check if tokenizer supports chat template."""
    return (hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template"))
            and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None)


def _retained_answer_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    effective_length: int,
    answer_onset: int,
    answer_end_exclusive: int,
    padding_side: str,
) -> torch.Tensor:
    """Return the exact stored, attended answer span."""
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("hidden_states must be a 2-D tensor")
    if hidden_states.shape[0] == 0:
        raise ValueError("hidden_states must contain at least one token")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 1:
        raise ValueError("attention_mask must be a 1-D tensor")
    if attention_mask.shape[0] != hidden_states.shape[0]:
        raise ValueError("attention_mask must match the hidden-state sequence length")
    if not bool(torch.all((attention_mask == 0) | (attention_mask == 1))):
        raise ValueError("attention_mask must be binary")
    if isinstance(effective_length, bool) or not isinstance(effective_length, int):
        raise ValueError("effective_length must be an integer")
    if effective_length <= 0 or effective_length != int(attention_mask.sum().item()):
        raise ValueError("effective_length must equal the attended token count")
    if padding_side not in ("left", "right"):
        raise ValueError("padding_side must be 'left' or 'right'")

    expected_mask = torch.zeros_like(attention_mask)
    if padding_side == "left":
        expected_mask[-effective_length:] = 1
        attended_end = hidden_states.shape[0]
    else:
        expected_mask[:effective_length] = 1
        attended_end = effective_length
    if not torch.equal(attention_mask, expected_mask):
        raise ValueError(f"attention_mask is not contiguous {padding_side}-padded data")

    if isinstance(answer_onset, bool) or not isinstance(answer_onset, int):
        raise ValueError("answer_onset must be an integer")
    if not 0 <= answer_onset < hidden_states.shape[0]:
        raise ValueError("answer_onset is outside the hidden-state sequence")
    if int(attention_mask[answer_onset].item()) != 1:
        raise ValueError("answer_onset must identify an attended token")
    if isinstance(answer_end_exclusive, bool) or not isinstance(answer_end_exclusive, int):
        raise ValueError("answer_end_exclusive must be an integer")
    if not answer_onset < answer_end_exclusive <= attended_end:
        raise ValueError("answer_end_exclusive is outside the attended answer span")
    if int(attention_mask[answer_end_exclusive - 1].item()) != 1:
        raise ValueError("answer_end_exclusive must follow an attended token")

    answer_hidden = hidden_states[answer_onset:answer_end_exclusive]
    if answer_hidden.shape[0] == 0:
        raise ValueError("no retained answer tokens are available for extraction")
    return answer_hidden


def extract_activation(
    strategy: ExtractionStrategy,
    hidden_states: torch.Tensor,
    answer_text: str,
    tokenizer,
    prompt_len: int,
    weighted_decay: float = 0.1,
    *,
    attention_mask: torch.Tensor | None = None,
    effective_length: int | None = None,
    answer_onset: int | None = None,
    answer_end_exclusive: int | None = None,
    padding_side: str | None = None,
) -> torch.Tensor:
    """Extract one activation vector for ``strategy``.

    Raw-cache callers provide all four keyword-only token metadata fields. In
    that mode this function uses only the stored retained span and never invokes
    the tokenizer. Existing live-collection callers retain the legacy tokenizer
    path when none of those fields is supplied.
    """
    metadata = (
        attention_mask,
        effective_length,
        answer_onset,
        answer_end_exclusive,
        padding_side,
    )
    uses_stored_span = any(value is not None for value in metadata)
    if uses_stored_span and not all(value is not None for value in metadata):
        raise ValueError(
            "attention_mask, effective_length, answer_onset, "
            "answer_end_exclusive, and padding_side must be supplied together"
        )

    if uses_stored_span:
        answer_hidden = _retained_answer_hidden_states(
            hidden_states,
            attention_mask,
            effective_length,
            answer_onset,
            answer_end_exclusive,
            padding_side,
        )
        if strategy == ExtractionStrategy.CHAT_FIRST:
            return answer_hidden[0]
        if strategy in (
            ExtractionStrategy.CHAT_LAST,
            ExtractionStrategy.ROLE_PLAY,
            ExtractionStrategy.COMPLETION_LAST,
            ExtractionStrategy.MC_COMPLETION,
        ):
            return answer_hidden[-1]
        if strategy in (
            ExtractionStrategy.CHAT_MEAN,
            ExtractionStrategy.COMPLETION_MEAN,
        ):
            return answer_hidden.mean(dim=0)
        if strategy == ExtractionStrategy.CHAT_MAX_NORM:
            return answer_hidden[torch.argmax(torch.norm(answer_hidden, dim=1))]
        if strategy == ExtractionStrategy.CHAT_WEIGHTED:
            positions = torch.arange(
                answer_hidden.shape[0],
                dtype=answer_hidden.dtype,
                device=answer_hidden.device,
            )
            weights = torch.exp(-positions * weighted_decay)
            weights = weights / weights.sum()
            return (answer_hidden * weights.unsqueeze(1)).sum(dim=0)
        if strategy == ExtractionStrategy.MC_BALANCED:
            return answer_hidden[0]
        raise ValueError(f"Unknown extraction strategy: {strategy}")

    seq_len = hidden_states.shape[0]
    answer_tokens = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    if strategy == ExtractionStrategy.CHAT_LAST:
        return hidden_states[-1]

    elif strategy == ExtractionStrategy.CHAT_FIRST:
        first_answer_idx = max(0, seq_len - num_answer_tokens - 1)
        return hidden_states[first_answer_idx]

    elif strategy == ExtractionStrategy.CHAT_MEAN:
        if num_answer_tokens > 0 and seq_len > num_answer_tokens:
            answer_hidden = hidden_states[-num_answer_tokens-1:-1]
            return answer_hidden.mean(dim=0)
        return hidden_states[-1]

    elif strategy == ExtractionStrategy.CHAT_MAX_NORM:
        if num_answer_tokens > 0 and seq_len > num_answer_tokens:
            answer_hidden = hidden_states[-num_answer_tokens-1:-1]
            norms = torch.norm(answer_hidden, dim=1)
            max_idx = torch.argmax(norms)
            return answer_hidden[max_idx]
        return hidden_states[-1]

    elif strategy == ExtractionStrategy.CHAT_WEIGHTED:
        if num_answer_tokens > 0 and seq_len > num_answer_tokens:
            answer_hidden = hidden_states[-num_answer_tokens-1:-1]
            weights = torch.exp(-torch.arange(answer_hidden.shape[0], dtype=answer_hidden.dtype, device=answer_hidden.device) * weighted_decay)
            weights = weights / weights.sum()
            return (answer_hidden * weights.unsqueeze(1)).sum(dim=0)
        return hidden_states[-1]

    elif strategy == ExtractionStrategy.ROLE_PLAY:
        return hidden_states[-1]

    elif strategy == ExtractionStrategy.MC_BALANCED:
        return hidden_states[-2]

    elif strategy == ExtractionStrategy.COMPLETION_LAST:
        return hidden_states[-1]

    elif strategy == ExtractionStrategy.COMPLETION_MEAN:
        if num_answer_tokens > 0 and seq_len > num_answer_tokens:
            answer_hidden = hidden_states[-num_answer_tokens:]
            return answer_hidden.mean(dim=0)
        return hidden_states[-1]

    elif strategy == ExtractionStrategy.MC_COMPLETION:
        return hidden_states[-1]

    else:
        raise ValueError(f"Unknown extraction strategy: {strategy}")

def add_extraction_strategy_args(parser: argparse.ArgumentParser) -> None:
    """Add --extraction-strategy argument to an argument parser."""
    parser.add_argument(
        "--extraction-strategy",
        type=str,
        default=ExtractionStrategy.default().value,
        choices=ExtractionStrategy.list_all(),
        help=f"Extraction strategy for activations. Default: {ExtractionStrategy.default().value}",
    )

def add_extraction_component_args(parser: argparse.ArgumentParser) -> None:
    """Add --extraction-component argument to an argument parser."""
    parser.add_argument(
        "--extraction-component",
        type=str,
        default=ExtractionComponent.default().value,
        choices=ExtractionComponent.list_all(),
        help=f"Transformer component to extract from. Default: {ExtractionComponent.default().value}",
    )

