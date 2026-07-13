"""Validation helpers shared by raw activation collection paths."""

from __future__ import annotations

from wisent.core.primitives.model_interface.core.activations import ExtractionComponent


def require_supported_raw_component(
    component: ExtractionComponent,
) -> ExtractionComponent:
    """Fail closed before raw collection can mislabel another component."""
    if not isinstance(component, ExtractionComponent):
        raise TypeError("raw extraction component must be an ExtractionComponent")
    if component.needs_cache:
        raise ValueError(
            "raw collection does not support cache-backed components; "
            "use the standard activation collector for KV_CACHE"
        )
    if component is not ExtractionComponent.RESIDUAL_STREAM:
        raise ValueError("raw collection supports only the residual_stream component")
    return component


def require_answer_text(name: str, value: object) -> str:
    """Return an exact answer string, rejecting coercion and empty content."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def response_text(response: object) -> str:
    """Read supported response text without stringifying invalid payloads."""
    if isinstance(response, str):
        return require_answer_text("response", response)
    for attribute in ("model_response", "text"):
        if hasattr(response, attribute):
            return require_answer_text(
                f"response.{attribute}", getattr(response, attribute)
            )
    raise ValueError("response object must provide non-empty string answer text")


def require_identity(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def resolved_revision(model, kind: str) -> str:
    if kind == "model":
        revision = getattr(model, "resolved_model_revision", None)
        if revision is None:
            revision = getattr(
                getattr(model.hf_model, "config", None), "_commit_hash", None
            )
    else:
        revision = getattr(model, "resolved_tokenizer_revision", None)
        tokenizer = model.tokenizer
        if revision is None:
            revision = getattr(tokenizer, "_commit_hash", None)
        if revision is None:
            revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    revision_width = len("0123456789abcdef0123456789abcdef01234567")
    if (
        not isinstance(revision, str)
        or len(revision) != revision_width
        or any(char not in "0123456789abcdef" for char in revision)
    ):
        raise ValueError(f"resolved {kind} revision must be a lowercase 40-hex commit")
    return revision


def longest_common_prefix_length(left: str, right: str) -> int:
    for index, (left_char, right_char) in enumerate(zip(left, right)):
        if left_char != right_char:
            return index
    return min(len(left), len(right))
