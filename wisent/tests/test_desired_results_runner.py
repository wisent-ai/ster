import copy
import math

import pytest
import sys
from collections import Counter
from types import ModuleType, SimpleNamespace
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts" / "steering" / "desired_results_runner.py"
SPEC = importlib.util.spec_from_file_location("desired_results_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _contract_artifacts(tmp_path):
    manifest = {
        "schema_version": 1,
        "purpose": "calibration",
        "execution_mode": "calibration",
        "calibration_protocol": {
            "id": "desired-results-bounded-rerun-v1",
            "revision": 1,
            "run_class": "bounded_calibration_rerun",
            "prior_owner": "scripts/steering/desired_results_runner.py",
            "methods": ["caa", "grom", "mlp", "nurt", "ostrze", "tecza", "tetno", "wicher"],
            "extraction_component": "residual_stream",
            "extraction_strategies": list(runner.STRATEGIES),
            "trials_per_format": 2,
            "format_count": 7,
            "trials_per_method": 14,
            "selection_split": "validation",
            "fit_splits": ["train"],
            "final_fit_splits": ["train"],
            "test_evaluations": 0,
            "exploratory_run_disposition": "invalid_unbounded_priors_excluded",
        },
        "job_unit": {"model": runner.MODEL, "benchmark": runner.BENCHMARK, "method": "caa"},
        "split": {
            "counts": {"train": 2, "validation": 2},
            "pair_ids": {"train": [1, 4], "validation": [2, 5]},
            "hpo_reads": ["train"],
            "selection_split": "validation",
            "final_fit": ["train"],
            "test_evaluations": 0,
        },
        "activation_search_scope": {
            "extraction_component": "residual_stream",
            "extraction_strategies": list(runner.STRATEGIES),
            "layers": list(range(1, 17)),
        },
        "saved_activation_policy": {
            "automatic_regeneration": "forbidden",
            "fallback": "forbidden",
            "positional_join": "forbidden",
        },
        "mode_contracts": {
            "hpo": {
                "strict_loader_pair_ids": "train_plus_validation_only",
                "objective_reports": "validation_only",
                "writes_under": "desired-results/bounded-rerun-v1/hpo/",
                "required_output": "frozen_config.json",
            },
        },
        "revisions": {"model": "a" * 40, "activation": "b" * 40},
        "output_prefix": "desired-results/",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    artifacts = []
    for strategy in runner.STRATEGIES:
        for layer in range(1, 17):
            proof_path = tmp_path / f"{strategy}-{layer}.json"
            proof_path.write_text(json.dumps({
                "complete": True,
                "model": runner.MODEL,
                "benchmark": runner.BENCHMARK,
                "extraction_strategy": strategy,
                "layers": [layer],
                "pair_ids": [1, 2, 4, 5],
            }))
            artifacts.append({
                "extraction_strategy": strategy,
                "layer": layer,
                "completion_manifest": proof_path.name,
            })
    index_path = tmp_path / "completion-index.json"
    index_path.write_text(json.dumps({"artifacts": artifacts}))
    return manifest_path, index_path, artifacts


def test_contract_routes_every_format_layer_and_materializes_canonical_plan(capsys, tmp_path):
    manifest_path, index_path, _ = _contract_artifacts(tmp_path)

    result = runner.main([
        "contract", "--manifest", str(manifest_path),
        "--completion-index", str(index_path),
    ])
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["status"] == "ok"
    assert output["mode"] == "contract"
    assert output["model_loaded"] is False
    assert output["method"] == "caa"
    assert output["route_count"] == len(runner.STRATEGIES) * 16
    assert {(route["extraction_strategy"], route["layer"]) for route in output["routes"]} == {
        (strategy, layer) for strategy in runner.STRATEGIES for layer in range(1, 17)
    }
    assert output["fit_splits"] == ["train"]
    assert output["selection_split"] == "validation"
    assert output["final_fit_splits"] == ["train"]
    assert output["test_evaluations"] == 0

    budget = output["stratified_budget"]
    assert budget == {
        "method_count": 8,
        "format_count": 7,
        "trials_per_format": 2,
        "trials_per_method": 14,
        "route_count": 112,
    }
    plan = output["calibration_plan"]
    assert plan["route_count"] == len(plan["routes"]) == 112
    assert Counter(
        (route["method"], route["extraction_strategy"])
        for route in plan["routes"]
    ) == Counter({
        (method, strategy): 2
        for method in runner.METHODS
        for strategy in runner.STRATEGIES
    })
    assert all(route["extraction_strategy"] in runner.STRATEGIES for route in plan["routes"])
    assert all(route["test_enabled"] is False for route in plan["routes"])
    assert all("test_pair_ids" not in route for route in plan["routes"])
    assert len({route["run_key"] for route in plan["routes"]}) == 112
    assert len({route["staging_prefix"] for route in plan["routes"]}) == 112


def test_contract_rejects_an_unrouted_format_layer(capsys, tmp_path):
    manifest_path, index_path, artifacts = _contract_artifacts(tmp_path)
    index_path.write_text(json.dumps({"artifacts": artifacts[:-1]}))

    result = runner.main([
        "contract", "--manifest", str(manifest_path),
        "--completion-index", str(index_path),
    ])

    assert result == 2
    assert "completion index is missing routes" in capsys.readouterr().err


def test_contract_accepts_full_preflight_proof_support(capsys, tmp_path):
    manifest_path, index_path, artifacts = _contract_artifacts(tmp_path)
    for artifact in artifacts:
        proof_path = tmp_path / artifact["completion_manifest"]
        proof = json.loads(proof_path.read_text())
        proof["pair_ids"].extend([0, 3])
        proof_path.write_text(json.dumps(proof))

    result = runner.main([
        "contract", "--manifest", str(manifest_path),
        "--completion-index", str(index_path),
    ])

    assert result == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_contract_rejects_route_proof_missing_calibration_support(capsys, tmp_path):
    manifest_path, index_path, artifacts = _contract_artifacts(tmp_path)
    proof_path = tmp_path / artifacts[0]["completion_manifest"]
    proof = json.loads(proof_path.read_text())
    proof["pair_ids"].remove(5)
    proof_path.write_text(json.dumps(proof))

    result = runner.main([
        "contract", "--manifest", str(manifest_path),
        "--completion-index", str(index_path),
    ])

    assert result == 2
    assert "is missing calibration support: [5]" in capsys.readouterr().err


def test_noncanonical_trial_budget_fails_before_model_or_manifest_load(monkeypatch, capsys, tmp_path):
    manifest_path, index_path, _ = _contract_artifacts(tmp_path)
    model_constructions = []
    model_module = ModuleType("wisent.core.primitives.models.wisent_model")
    model_module.WisentModel = lambda *args, **kwargs: model_constructions.append((args, kwargs))
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    monkeypatch.setattr(
        runner,
        "_manifest",
        lambda path: (_ for _ in ()).throw(AssertionError("manifest must not be read")),
    )

    result = runner.main([
        "contract", "--manifest", str(manifest_path),
        "--completion-index", str(index_path),
        "--trials-per-format", "3",
    ])

    assert result == 2
    assert "requires exactly 2 trials per format" in capsys.readouterr().err
    assert model_constructions == []


def test_hpo_stratifies_optimizer_calls_and_never_materializes_validation(monkeypatch, capsys, tmp_path):
    manifest_path, index_path, _ = _contract_artifacts(tmp_path)
    plan_path = tmp_path / "calibration-plan.json"
    contract_result = runner.main([
        "contract", "--manifest", str(manifest_path),
        "--completion-index", str(index_path),
        "--calibration-plan", str(plan_path),
    ])
    assert contract_result == 0
    capsys.readouterr()

    materialized_ids = []
    text_ids = []
    optimizer_calls = []
    publish_observations = []
    durable_artifacts = {
        "best_config.json",
        "validation_summary.json",
        "trials.json",
        "frozen_config.json",
        "validation_pairs.json",
        "provenance.json",
    }
    real_publish = runner._publish


    def fake_materialize(root, pair_ids, routes):
        materialized_ids.append(list(pair_ids))
        root.mkdir(parents=True)
        (root / "cached-activations.pt").write_bytes(b"transient")
        return {route: str(root / f"{route[0]}-{route[1]}.json") for route in routes}


    def fake_pair_file(path, pair_ids):
        text_ids.append(list(pair_ids))
        path.write_text(json.dumps({"pair_ids": list(pair_ids)}))


    class FakeHPOConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeOptimizer:
        def optimize_fn(self, objective, space, n_trials, *, cfg, extra_trials):
            optimizer_calls.append((space, n_trials, cfg, extra_trials))
            sample = {}
            for name, param in space.items():
                spec = runner._serialize_param(param)
                sample[name] = spec["choices"][0] if spec["kind"] == "categorical" else spec["low"]
            score = objective(sample)
            return SimpleNamespace(
                backend="optuna",
                n_trials=2,
                all_trials=[
                    {"params": dict(sample), "score": score - 0.5},
                    {"params": dict(sample), "score": score},
                ],
                best_params=dict(sample),
                best_score=score,
            )

    pipeline_module = ModuleType("wisent.core.utils.cli.commands.optimize_steering.pipeline.pipeline")

    def fake_create_objective(**kwargs):
        (Path(kwargs["work_dir"]) / "optimizer-cache.db").write_bytes(b"transient")
        return lambda params: float(runner.STRATEGIES.index(params["extraction_strategy"]))

    pipeline_module.create_objective = fake_create_objective


    atoms_module = ModuleType("wisent.core.utils.services.optimization.core.atoms")
    atoms_module.BaseOptimizer = FakeOptimizer
    atoms_module.HPOConfig = FakeHPOConfig
    model_module = ModuleType("wisent.core.primitives.models.wisent_model")

    model_module.WisentModel = lambda *args, **kwargs: object()
    for module in (pipeline_module, atoms_module, model_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(runner, "_materialize", fake_materialize)
    monkeypatch.setattr(runner, "_pair_file", fake_pair_file)

    def fake_publish(staging, destination):
        publish_observations.append({
            "strict_train_exists": (staging / "strict_train").exists(),
            "trial_work_exists": (staging / "trial_work").exists(),
            "artifacts": {path.name for path in staging.iterdir()},
        })
        real_publish(staging, destination)

    monkeypatch.setattr(runner, "_publish", fake_publish)


    output_root = tmp_path / "outputs"
    legacy = output_root / "desired-results" / "hpo"
    legacy.mkdir(parents=True)
    (legacy / "exploratory.json").write_text("untouched")
    result = runner.main([
        "hpo", "--manifest", str(manifest_path),
        "--completion-index", str(index_path),
        "--calibration-plan", str(plan_path),
        "--output-root", str(output_root),
    ])

    assert result == 0
    destination = Path(capsys.readouterr().out.strip())
    assert destination == output_root / "desired-results" / "bounded-rerun-v1" / "hpo"
    assert (legacy / "exploratory.json").read_text() == "untouched"
    assert publish_observations == [{
        "strict_train_exists": False,
        "trial_work_exists": False,
        "artifacts": durable_artifacts,
    }]
    assert {path.name for path in destination.iterdir()} == durable_artifacts
    assert materialized_ids == [[1, 4]]
    assert text_ids == [[2, 5]]
    assert len(optimizer_calls) == len(runner.STRATEGIES)
    assert [call[1] for call in optimizer_calls] == [2] * len(runner.STRATEGIES)
    assert [call[2].n_trials for call in optimizer_calls] == [2] * len(runner.STRATEGIES)
    assert [call[3] for call in optimizer_calls] == [0] * len(runner.STRATEGIES)
    assert [call[2].sampler for call in optimizer_calls] == ["random"] * len(runner.STRATEGIES)
    assert len({call[2].seed for call in optimizer_calls}) == len(runner.STRATEGIES)
    assert [call[0]["extraction_strategy"].choices for call in optimizer_calls] == [
        [strategy] for strategy in runner.STRATEGIES
    ]

    frozen = json.loads((destination / "frozen_config.json").read_text())
    assert frozen["fit_pair_ids"] == [1, 4]
    assert frozen["selection_pair_ids"] == [2, 5]
    assert frozen["test_pair_ids_read"] == []
    assert frozen["trial_count"] == 14
    assert [item["trial_count"] for item in frozen["per_format"]] == [2] * 7
    assert frozen["best_params"]["extraction_strategy"] == runner.STRATEGIES[-1]
    assert set(frozen["best_params"]) == set(runner._calibration_plan()["prior_definitions"]["caa"])
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["protocol_identity"]["id"] == runner.BOUNDED_PROTOCOL_ID
    assert provenance["eligible_for_selection"] is True
    assert provenance["old_exploratory_run"] == {
        "protocol_id": "steering_effectiveness_initial",
        "excluded": True,
        "consumed_as_prior_or_resume": False,
    }
    assert provenance["optimizer"]["integer_bounds"] == "inclusive"


def _definition_endpoints(spec):
    if spec["kind"] == "categorical":
        return list(spec["choices"])
    return [spec["low"], spec["high"]]


def test_calibration_plan_v2_freezes_identity_hash_and_exact_cross_product():
    plan = runner._calibration_plan()

    assert plan["schema_version"] == 2
    assert plan["protocol_identity"] == {
        "id": "desired-results-bounded-rerun-v1",
        "revision": 1,
        "run_class": "bounded_calibration_rerun",
        "model": runner.MODEL,
        "benchmark": runner.BENCHMARK,
        "extraction_component": "residual_stream",
    }
    assert plan["optimizer_contract"] == {
        "backend": "optuna",
        "sampler": "random",
        "pruner": "nop",
        "base_seed": 0,
        "seed_algorithm": "sha256-first-u32-be-v1",
        "integer_bounds": "inclusive",
        "load_if_exists": False,
        "extra_trials": 0,
        "trials_per_format": 2,
    }
    assert plan["prior_definitions_sha256"] == runner._canonical_json_sha256(
        plan["prior_definitions"]
    )
    assert plan["prior_definitions_sha256"] == (
        "d9c8c9cefd107c86835cf486bf673ea62ecbe2f4b648ed82992d66fcc3bb5858"
    )
    assert len(plan["routes"]) == plan["route_count"] == 112
    assert Counter(
        (route["method"], route["extraction_strategy"], route["repeat"])
        for route in plan["routes"]
    ) == Counter(
        (method, strategy, repeat)
        for method in runner.METHODS
        for strategy in runner.STRATEGIES
        for repeat in range(2)
    )
    assert all(route["protocol_id"] == runner.BOUNDED_PROTOCOL_ID for route in plan["routes"])
    assert all(
        route["study_seed"] == runner._study_seed(0, route["method"], route["extraction_strategy"])
        for route in plan["routes"]
    )


@pytest.mark.parametrize("method", sorted(runner.METHODS))
def test_all_final_spaces_are_finite_exact_and_dependency_safe(method):
    plan = runner._calibration_plan()
    definitions = plan["prior_definitions"][method]
    space = runner._search_space(method, list(range(1, 17)), runner.STRATEGIES[0], plan)

    expected = copy.deepcopy(definitions)
    expected["extraction_strategy"] = {"kind": "categorical", "choices": [runner.STRATEGIES[0]]}
    assert {name: runner._serialize_param(param) for name, param in space.items()} == expected
    for name, spec in definitions.items():
        assert spec["kind"] in {"float", "int", "categorical"}, name
        assert spec.get("distribution") not in {"normal", "lognormal", "qnormal", "qlognormal"}
        if spec["kind"] == "categorical":
            assert spec["choices"]
            assert all(not isinstance(value, float) or math.isfinite(value) for value in spec["choices"])
        else:
            assert math.isfinite(spec["low"]) and math.isfinite(spec["high"])
            assert spec["low"] <= spec["high"]
    runner._validate_dependency_contract(method, definitions, list(range(1, 17)))

    endpoints = {name: _definition_endpoints(spec) for name, spec in definitions.items()}
    if method == "tecza":
        assert max(endpoints["num_directions"]) <= min(endpoints["max_directions"])
        assert max(endpoints["min_cosine_similarity"]) <= min(endpoints["max_cosine_similarity"])
    elif method == "tetno":
        assert max(endpoints["entropy_floor"]) < min(endpoints["entropy_ceiling"])
        assert endpoints["sensor_layer"] == endpoints["steering_start"] == endpoints["steering_end"]
    elif method == "grom":
        assert max(endpoints["warmup_steps"]) < min(endpoints["optimization_steps"])
        assert endpoints["sensor_layer"] == endpoints["steering_start"] == endpoints["steering_end"]
        assert max(endpoints["min_cosine_sim"]) <= min(endpoints["max_cosine_sim"])
        assert max(endpoints["gate_dim_min"]) <= min(endpoints["gate_hidden_dim"])
        assert max(endpoints["gate_hidden_dim"]) <= min(endpoints["gate_dim_max"])
        assert max(endpoints["intensity_dim_min"]) <= min(endpoints["intensity_hidden_dim"])
        assert max(endpoints["intensity_hidden_dim"]) <= min(endpoints["intensity_dim_max"])
        ordered = ["min_adapted_directions", "adapt_linear_directions", "significant_directions_default", "adapt_complex_directions", "adapt_max_directions"]
        assert all(max(endpoints[left]) <= min(endpoints[right]) for left, right in zip(ordered, ordered[1:]))
    elif method == "nurt":
        assert max(endpoints["num_dims"]) <= min(endpoints["max_concept_dim"])
        assert max(endpoints["lr_min"]) <= min(endpoints["lr"])


@pytest.mark.parametrize("method", ["tetno", "grom"])
@pytest.mark.parametrize(
    ("sample", "layer_count", "expected_layers"),
    [
        ({"sensor_layer": 1, "steering_start": 3, "steering_end": 5}, 8, (1, 3, 5)),
        ({"sensor_layer": 10, "steering_start": 5, "steering_end": 2}, 10, (1, 2, 5)),
        ({"sensor_layer": 1, "steering_start": 1, "steering_end": 1}, 8, (1, 2, 2)),
        ({"sensor_layer": 8, "steering_start": 8, "steering_end": 8}, 8, (7, 8, 8)),
    ],
)
def test_tetno_and_grom_normalize_sensor_before_steering(
    method, sample, layer_count, expected_layers,
):
    normalized = runner._normalize_sample(method, sample, hidden_size=32, layer_count=layer_count)
    layers = (
        normalized["sensor_layer"],
        normalized["steering_start"],
        normalized["steering_end"],
    )

    assert layers == expected_layers
    assert 1 <= layers[0] < layers[1] <= layers[2] <= layer_count


@pytest.mark.parametrize("method", ["tetno", "grom"])
def test_tetno_and_grom_reject_single_layer_normalization(method):
    with pytest.raises(runner.ContractError, match="requires at least two activation layers"):
        runner._normalize_sample(
            method,
            {"sensor_layer": 1, "steering_start": 1, "steering_end": 1},
            hidden_size=32,
            layer_count=1,
        )


@pytest.mark.parametrize("method", ["tetno", "grom"])
@pytest.mark.parametrize(
    ("sensor", "start", "end"),
    [(2, 2, 4), (1, 4, 3)],
)
def test_tetno_and_grom_selected_config_rejects_invalid_layer_order(
    method, sensor, start, end,
):
    with pytest.raises(runner.ContractError, match="valid sensor/steering layer identity"):
        runner._selected_config(method, {
            "strategy": runner.STRATEGIES[0],
            "best_params": {
                "extraction_strategy": runner.STRATEGIES[0],
                "sensor_layer": sensor,
                "steering_start": start,
                "steering_end": end,
            },
        })


def test_plan_mutations_are_rejected_fail_closed():
    canonical = runner._calibration_plan()
    mutations = []
    for mutate in (
        lambda plan: plan["protocol_identity"].__setitem__("id", "other"),
        lambda plan: plan["protocol_identity"].__setitem__("run_class", "exploratory"),
        lambda plan: plan["optimizer_contract"].__setitem__("sampler", "tpe"),
        lambda plan: plan["optimizer_contract"].__setitem__("base_seed", 1),
        lambda plan: plan["optimizer_contract"].__setitem__("integer_bounds", "half_open"),
        lambda plan: plan["data_contract"].__setitem__("selection_split", "test"),
        lambda plan: plan["invalid_exploration_disposition"].__setitem__("eligible_for_selection", True),
        lambda plan: plan["routes"][0].__setitem__("repeat", 2),
        lambda plan: plan.__setitem__("prior_definitions_sha256", "0" * 64),
    ):
        changed = copy.deepcopy(canonical)
        mutate(changed)
        mutations.append(changed)
    removed = copy.deepcopy(canonical)
    removed["prior_definitions"]["caa"].pop("strength")
    mutations.append(removed)
    added = copy.deepcopy(canonical)
    added["prior_definitions"]["caa"]["unknown"] = {"kind": "categorical", "choices": [1]}
    mutations.append(added)
    changed_bound = copy.deepcopy(canonical)
    changed_bound["prior_definitions"]["caa"]["strength"]["high"] = 5.0
    mutations.append(changed_bound)

    for plan in mutations:
        with pytest.raises(runner.ContractError):
            runner._validate_calibration_plan(plan, 2)


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "float", "distribution": "normal", "low": 0.1, "high": 1.0, "log_scale": False},
        {"kind": "float", "distribution": "uniform", "low": float("nan"), "high": 1.0, "log_scale": False},
        {"kind": "float", "distribution": "uniform", "low": 0.0, "high": 1.0, "log_scale": True},
        {"kind": "int", "distribution": "randint", "low": 1.5, "high": 2},
        {"kind": "categorical", "choices": []},
        {"kind": "categorical", "choices": [1, 1]},
    ],
)
def test_malformed_parameter_definitions_are_rejected(spec):
    with pytest.raises(runner.ContractError):
        runner._validate_param_definition("bad", spec)


def test_study_seeds_and_two_random_samples_are_deterministic():
    optuna = pytest.importorskip("optuna")
    plan = runner._calibration_plan()
    specs = plan["prior_definitions"]["caa"]

    def sample(seed):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
        def objective(trial):
            for name, spec in sorted(specs.items()):
                if spec["kind"] == "categorical":
                    trial.suggest_categorical(name, spec["choices"])
                elif spec["kind"] == "int":
                    trial.suggest_int(name, spec["low"], spec["high"])
                else:
                    trial.suggest_float(name, spec["low"], spec["high"], log=spec["log_scale"])
            return 0.0
        study.optimize(objective, n_trials=2)
        return [trial.params for trial in study.trials]

    seed = runner._study_seed(0, "caa", "chat_first")
    assert sample(seed) == sample(seed)
    assert seed != runner._study_seed(0, "caa", "chat_last")
    assert seed != runner._study_seed(0, "mlp", "chat_first")


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed"])
def test_global_space_drift_is_rejected_without_mutation(monkeypatch, mutation):
    search_module = __import__(
        "wisent.core.utils.cli.commands.optimize_steering.pipeline.search_space",
        fromlist=["get_method_space"],
    )
    parameters_module = __import__(
        "wisent.core.utils.services.optimization.core.parameters",
        fromlist=["CategoricalParam"],
    )
    original = search_module.get_method_space
    before = {name: repr(param) for name, param in original("caa", 17).items()}

    def drifted(method, num_layers):
        space = original(method, num_layers)
        if mutation == "missing":
            space.pop("steering_strategy")
        elif mutation == "extra":
            space["unexpected"] = parameters_module.CategoricalParam(choices=[1])
        else:
            space["steering_strategy"] = parameters_module.CategoricalParam(choices=["constant"])
        return space

    monkeypatch.setattr(search_module, "get_method_space", drifted)
    with pytest.raises(runner.ContractError, match="drift|space"):
        runner._search_space("caa", list(range(1, 17)), "chat_first", runner._calibration_plan())
    monkeypatch.setattr(search_module, "get_method_space", original)
    after = {name: repr(param) for name, param in original("caa", 17).items()}
    assert after == before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.pop("calibration_protocol"),
        lambda manifest: manifest["calibration_protocol"].__setitem__("id", "other"),
        lambda manifest: manifest["calibration_protocol"].__setitem__("selection_split", "test"),
        lambda manifest: manifest["split"].__setitem__("hpo_reads", ["validation"]),
        lambda manifest: manifest["activation_search_scope"].__setitem__("layers", [1, 2]),
    ],
)
def test_malformed_calibration_manifest_is_rejected_before_model_load(tmp_path, mutate):
    manifest_path, _, _ = _contract_artifacts(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(runner.ContractError):
        runner._manifest(manifest_path)


@pytest.mark.parametrize("failure_mode", ["invalid_sample", "nan_score", "missing_trial", "unobserved_best"])
def test_invalid_hpo_results_abort_without_retry_or_publish(monkeypatch, tmp_path, failure_mode):
    manifest_path, index_path, _ = _contract_artifacts(tmp_path)
    manifest = runner._manifest(manifest_path)
    routes = runner._completion_routes(index_path, manifest)
    plan = runner._calibration_plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    objective_calls = []
    publish_calls = []

    def fake_materialize(root, pair_ids, available_routes):
        root.mkdir(parents=True)
        return {route: str(root / "unused.json") for route in available_routes}

    def fake_pair_file(path, pair_ids):
        path.write_text(json.dumps({"pair_ids": pair_ids}))

    class FakeHPOConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FailingOptimizer:
        def optimize_fn(self, objective, space, n_trials, *, cfg, extra_trials):
            sample = {}
            for name, param in space.items():
                spec = runner._serialize_param(param)
                sample[name] = spec["choices"][0] if spec["kind"] == "categorical" else spec["low"]
            if failure_mode == "invalid_sample":
                sample["strength"] = float("inf")
                objective(sample)
                raise AssertionError("invalid sample unexpectedly reached optimizer return")
            if failure_mode == "nan_score":
                objective(sample)
                raise AssertionError("NaN objective unexpectedly returned")
            if failure_mode == "missing_trial":
                return SimpleNamespace(backend="optuna", n_trials=2, all_trials=[{"params": sample, "score": 0.0}], best_params=sample, best_score=0.0)
            alternate = dict(sample)
            alternate["strength"] = 4
            return SimpleNamespace(
                backend="optuna",
                n_trials=2,
                all_trials=[{"params": sample, "score": 0.0}, {"params": sample, "score": 1.0}],
                best_params=alternate,
                best_score=0.5,
            )

    pipeline_module = ModuleType("wisent.core.utils.cli.commands.optimize_steering.pipeline.pipeline")
    def fake_create_objective(**kwargs):
        def real_objective(params):
            objective_calls.append(dict(params))
            return float("nan") if failure_mode == "nan_score" else 0.0
        return real_objective
    pipeline_module.create_objective = fake_create_objective
    atoms_module = ModuleType("wisent.core.utils.services.optimization.core.atoms")
    atoms_module.BaseOptimizer = FailingOptimizer
    atoms_module.HPOConfig = FakeHPOConfig
    model_module = ModuleType("wisent.core.primitives.models.wisent_model")
    model_module.WisentModel = lambda *args, **kwargs: object()
    for module in (pipeline_module, atoms_module, model_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(runner, "_materialize", fake_materialize)
    monkeypatch.setattr(runner, "_pair_file", fake_pair_file)
    monkeypatch.setattr(runner, "_publish", lambda *args: publish_calls.append(args))

    args = SimpleNamespace(
        output_root=tmp_path / "outputs",
        device="cpu",
        completion_index=index_path,
        calibration_plan=plan_path,
    )
    destination = args.output_root / "desired-results" / "bounded-rerun-v1" / "hpo"
    with pytest.raises(runner.ContractError):
        runner._run_hpo(args, manifest_path, manifest, routes, plan)

    assert publish_calls == []
    assert not destination.exists()
    assert not list(destination.parent.glob(".hpo.*"))
    if failure_mode == "invalid_sample":
        assert objective_calls == []
    elif failure_mode == "nan_score":
        assert len(objective_calls) == 1
