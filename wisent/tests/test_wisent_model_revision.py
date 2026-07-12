from types import SimpleNamespace

import pytest
import torch.nn as nn

from wisent.core.primitives.models.core import wisent_model


REVISION = "0123456789abcdef0123456789abcdef01234567"
MODEL_NAME = "example/model"
_MISSING = object()


def _install_loaders(
    monkeypatch,
    *,
    model_revision=REVISION,
    tokenizer_revision=REVISION,
):
    calls = {"model": [], "tokenizer": []}

    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=4),
        generation_config=SimpleNamespace(pad_token_id=0),
        layers=nn.ModuleList([nn.Identity()]),
        to=lambda _device: None,
    )
    if model_revision is not _MISSING:
        model.config._commit_hash = model_revision

    tokenizer = SimpleNamespace(
        init_kwargs={},
        pad_token_id=0,
        eos_token="<eos>",
        apply_chat_template=lambda *args, **kwargs: "prompt",
    )
    if tokenizer_revision is not _MISSING:
        tokenizer._commit_hash = tokenizer_revision

    def load_model(model_name, **kwargs):
        calls["model"].append((model_name, kwargs.get("revision")))
        return model

    def load_tokenizer(model_name, **kwargs):
        calls["tokenizer"].append((model_name, kwargs.get("revision")))
        return tokenizer

    monkeypatch.setattr(
        wisent_model,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=load_model),
    )
    monkeypatch.setattr(
        wisent_model,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=load_tokenizer),
    )
    return calls


def test_pinned_revision_is_loaded_and_recorded_for_model_and_tokenizer(monkeypatch):
    calls = _install_loaders(monkeypatch)

    model = wisent_model.WisentModel(MODEL_NAME, device="cpu", revision=REVISION)

    assert calls == {
        "model": [(MODEL_NAME, REVISION)],
        "tokenizer": [(MODEL_NAME, REVISION)],
    }
    assert model.requested_revision == REVISION
    assert model.resolved_model_revision == REVISION
    assert model.resolved_tokenizer_revision == REVISION


@pytest.mark.parametrize("model_revision", ["f" * 40, _MISSING], ids=["mismatch", "missing"])
def test_pinned_revision_rejects_unverified_model_commit(monkeypatch, model_revision):
    _install_loaders(monkeypatch, model_revision=model_revision)

    with pytest.raises(
        ValueError,
        match="loaded model revision does not match the requested immutable revision",
    ):
        wisent_model.WisentModel(MODEL_NAME, device="cpu", revision=REVISION)


@pytest.mark.parametrize("tokenizer_revision", ["f" * 40, _MISSING], ids=["mismatch", "missing"])
def test_pinned_revision_rejects_unverified_tokenizer_commit(monkeypatch, tokenizer_revision):
    _install_loaders(monkeypatch, tokenizer_revision=tokenizer_revision)

    with pytest.raises(
        ValueError,
        match="loaded tokenizer revision does not match the requested immutable revision",
    ):
        wisent_model.WisentModel(MODEL_NAME, device="cpu", revision=REVISION)


def test_unpinned_load_accepts_artifacts_without_commit_metadata(monkeypatch):
    _install_loaders(
        monkeypatch,
        model_revision=_MISSING,
        tokenizer_revision=_MISSING,
    )

    model = wisent_model.WisentModel(MODEL_NAME, device="cpu", revision=None)

    assert model.hidden_size == 4
    assert model.num_layers == 1
