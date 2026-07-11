import json

import pytest
import torch

from wisent.core.reading.modules.utilities.data import enriched_builder
from wisent.core.reading.modules.utilities.data.sources.hf import hf_loaders


MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TASK = "winogrande"
FORMAT = "chat_last"
LAYER = 6


def _error_code(exc_info):
    return json.loads(str(exc_info.value))["code"]


def _write_activation_artifacts(tmp_path, *, pair_ids=(9, 2, 7), manifest_ids=None, rows=None):
    safetensors = pytest.importorskip("safetensors.torch")
    pair_ids = list(pair_ids)
    manifest_ids = pair_ids if manifest_ids is None else list(manifest_ids)
    rows = len(pair_ids) if rows is None else rows
    activation_path = tmp_path / "activations.safetensors"
    values = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
    safetensors.save_file(
        {"pos_activations": values, "neg_activations": values + 100},
        str(activation_path),
        metadata={"pair_ids": json.dumps(pair_ids)},
    )
    manifest_path = tmp_path / "_complete.json"
    manifest_path.write_text(json.dumps({
        "complete": True,
        "model": MODEL,
        "benchmark": TASK,
        "extraction_strategy": FORMAT,
        "layers": [LAYER],
        "pair_ids": manifest_ids,
    }))
    return activation_path, manifest_path


def test_strict_activation_loader_selects_subset_in_requested_pair_id_order(tmp_path, monkeypatch):
    activation_path, manifest_path = _write_activation_artifacts(tmp_path)
    monkeypatch.setattr(hf_loaders, "_hf_hub_download", lambda _path: str(activation_path))

    positive, negative, pair_ids = hf_loaders.load_activations_from_hf_strict(
        MODEL, TASK, LAYER, FORMAT, [7, 9], str(manifest_path)
    )

    assert pair_ids == [7, 9]
    assert positive.tolist() == [[4.0, 5.0], [0.0, 1.0]]
    assert negative.tolist() == [[104.0, 105.0], [100.0, 101.0]]


@pytest.mark.parametrize(
    ("pair_ids", "manifest_ids", "rows", "expected_code"),
    [
        ((9, 9), (9, 9), 2, "duplicate_expected_pair_id"),
        ((9, 2, 7), (9, 7, 2), 3, "manifest_support_mismatch"),
        ((9, 2, 7), (9, 2, 7), 2, "activation_pair_id_count_mismatch"),
    ],
)
def test_strict_activation_loader_rejects_ambiguous_or_mismatched_support(
    tmp_path, monkeypatch, pair_ids, manifest_ids, rows, expected_code
):
    activation_path, manifest_path = _write_activation_artifacts(
        tmp_path, pair_ids=pair_ids, manifest_ids=manifest_ids, rows=rows
    )
    monkeypatch.setattr(hf_loaders, "_hf_hub_download", lambda _path: str(activation_path))

    with pytest.raises(ValueError) as exc_info:
        hf_loaders.load_activations_from_hf_strict(
            MODEL, TASK, LAYER, FORMAT, [9], str(manifest_path)
        )

    assert _error_code(exc_info) == expected_code


def _write_pair_texts(tmp_path, pairs):
    path = tmp_path / "pair_texts.json"
    path.write_text(json.dumps({"num_pairs": len(pairs), "pairs": pairs}))
    return path


def test_pair_text_loader_parses_real_nested_schema_and_preserves_requested_order(tmp_path, monkeypatch):
    path = _write_pair_texts(tmp_path, [
        {"pair_id": 7, "prompt": "p7", "positive_response": {"model_response": "yes7"},
         "negative_response": {"model_response": "no7"}},
        {"pair_id": 2, "prompt": "p2", "positive_response": {"model_response": "yes2"},
         "negative_response": {"model_response": "no2"}},
        {"pair_id": 9, "prompt": "p9", "positive_response": {"model_response": "yes9"},
         "negative_response": {"model_response": "no9"}},
    ])
    monkeypatch.setattr(hf_loaders, "_hf_hub_download", lambda _path: str(path))

    pairs = hf_loaders.load_pair_texts_from_hf_strict(TASK, [9, 7])

    assert list(pairs) == [9, 7]
    assert pairs == {
        9: {"prompt": "p9", "positive": "yes9", "negative": "no9"},
        7: {"prompt": "p7", "positive": "yes7", "negative": "no7"},
    }


@pytest.mark.parametrize(
    ("pairs", "requested", "expected_code"),
    [
        ([
            {"pair_id": 3, "prompt": "a", "positive": "a+", "negative": "a-"},
            {"pair_id": 3, "prompt": "b", "positive": "b+", "negative": "b-"},
        ], [3], "duplicate_pair_id"),
        ([{"pair_id": 3, "prompt": "a", "positive": "a+", "negative": "a-"}],
         [4], "pair_text_support_mismatch"),
    ],
)
def test_pair_text_loader_rejects_duplicate_and_missing_ids(
    tmp_path, monkeypatch, pairs, requested, expected_code
):
    path = _write_pair_texts(tmp_path, pairs)
    monkeypatch.setattr(hf_loaders, "_hf_hub_download", lambda _path: str(path))

    with pytest.raises(ValueError) as exc_info:
        hf_loaders.load_pair_texts_from_hf_strict(TASK, requested)

    assert _error_code(exc_info) == expected_code


def test_strict_enriched_builder_joins_text_and_activations_only_by_pair_id(tmp_path, monkeypatch):
    activation_path, manifest_path = _write_activation_artifacts(tmp_path)
    pair_text_path = _write_pair_texts(tmp_path, [
        {"pair_id": 7, "prompt": "p7", "positive_response": {"model_response": "yes7"},
         "negative_response": {"model_response": "no7"}},
        {"pair_id": 2, "prompt": "p2", "positive_response": {"model_response": "yes2"},
         "negative_response": {"model_response": "no2"}},
        {"pair_id": 9, "prompt": "p9", "positive_response": {"model_response": "yes9"},
         "negative_response": {"model_response": "no9"}},
    ])
    monkeypatch.setattr(
        hf_loaders,
        "_hf_hub_download",
        lambda path: str(pair_text_path if path.startswith("pair_texts/") else activation_path),
    )

    output_path = enriched_builder.build_enriched_from_hf_strict(
        MODEL, TASK, LAYER, FORMAT, str(tmp_path), [7, 9], str(manifest_path)
    )
    output = json.loads(open(output_path).read())

    assert output["pair_ids"] == [7, 9]
    assert output["extraction_component"] == "residual_stream"
    assert [pair["prompt"] for pair in output["pairs"]] == ["p7", "p9"]
    assert [pair["positive_response"]["layers_activations"][str(LAYER)] for pair in output["pairs"]] == [
        [4.0, 5.0], [0.0, 1.0]
    ]
