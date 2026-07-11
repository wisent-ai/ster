import pytest

from wisent.core.utils.cli.commands.optimize_steering.pipeline.pipeline import _build_config
from wisent.core.utils.cli.commands.optimize_steering.pipeline.search_space import get_method_space


FORMATS = [
    "chat_first",
    "chat_last",
    "chat_mean",
    "chat_max_norm",
    "chat_weighted",
    "mc_balanced",
    "role_play",
]
STANDARD_METHODS = ("caa", "ostrze", "mlp", "tecza", "tetno", "grom", "nurt", "wicher")


def _structural_params(method):
    params = {
        "extraction_strategy": "role_play",
        "extraction_component": "q_proj",
        "steering_strategy": "constant",
        "strength": 1.75,
    }
    if method in ("tetno", "grom"):
        params.update(sensor_layer=2, steering_start=5, steering_end=3)
    else:
        params["layer"] = 4
    return params


@pytest.mark.parametrize("method", STANDARD_METHODS)
def test_standard_method_searches_exact_formats_without_component(method):
    space = get_method_space(method, num_layers=16)

    assert space["extraction_strategy"].choices == FORMATS
    assert "extraction_component" not in space


@pytest.mark.parametrize("method", STANDARD_METHODS)
def test_standard_config_uses_selected_format_and_forces_residual_stream(method):
    config, strength = _build_config(method, _structural_params(method))

    assert config.extraction_strategy == "role_play"
    assert config._extra_args["extraction_component"] == "residual_stream"
    assert strength == 1.75
    if method in ("tetno", "grom"):
        assert config.steering_layers == [3, 4, 5]


@pytest.mark.parametrize("method", ("szlak", "przelom"))
def test_deferred_methods_retain_component_search_instead_of_format_search(method):
    space = get_method_space(method, num_layers=16)

    assert "extraction_strategy" not in space
    assert set(space["extraction_component"].choices) >= {"residual_stream", "q_proj", "k_proj"}

    config, _ = _build_config(
        method,
        {
            "layer": 4,
            "extraction_component": "q_proj",
            "steering_strategy": "constant",
            "strength": 1.0,
        },
    )
    assert config._extra_args["extraction_component"] == "q_proj"
