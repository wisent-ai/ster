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
        "job_unit": {"model": runner.MODEL, "benchmark": runner.BENCHMARK, "method": "caa"},
        "split": {
            "pair_ids": {"train": [1, 4], "validation": [2, 5]},
            "selection_split": "validation",
            "final_fit": ["train"],
            "test_evaluations": 0,
        },
        "activation_search_scope": {
            "extraction_component": "residual_stream",
            "extraction_strategies": list(runner.STRATEGIES),
            "layers": [3, 8],
        },
        "saved_activation_policy": {
            "automatic_regeneration": "forbidden",
            "fallback": "forbidden",
            "positional_join": "forbidden",
        },
        "output_prefix": "desired-results",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    artifacts = []
    for strategy in runner.STRATEGIES:
        for layer in (3, 8):
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
    assert output["route_count"] == len(runner.STRATEGIES) * 2
    assert {(route["extraction_strategy"], route["layer"]) for route in output["routes"]} == {
        (strategy, layer) for strategy in runner.STRATEGIES for layer in (3, 8)
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

    class FakeCategoricalParam:
        def __init__(self, choices):
            self.choices = choices

    class FakeHPOConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeOptimizer:
        def optimize_fn(self, objective, space, n_trials, *, cfg, extra_trials):
            optimizer_calls.append((space, n_trials, cfg, extra_trials))
            strategy = space["extraction_strategy"].choices[0]
            score = float(runner.STRATEGIES.index(strategy))
            return SimpleNamespace(
                all_trials=[{"score": score - 0.5}, {"score": score}],
                best_params={"layer": 8},
                best_score=score,
            )

    search_module = ModuleType(
        "wisent.core.utils.cli.commands.optimize_steering.pipeline.search_space"
    )
    search_module.get_method_space = lambda method, num_layers: {"strength": object()}
    parameters_module = ModuleType("wisent.core.utils.services.optimization.core.parameters")
    parameters_module.CategoricalParam = FakeCategoricalParam
    pipeline_module = ModuleType("wisent.core.utils.cli.commands.optimize_steering.pipeline.pipeline")

    def fake_create_objective(**kwargs):
        (Path(kwargs["work_dir"]) / "optimizer-cache.db").write_bytes(b"transient")
        return object()

    pipeline_module.create_objective = fake_create_objective


    atoms_module = ModuleType("wisent.core.utils.services.optimization.core.atoms")
    atoms_module.BaseOptimizer = FakeOptimizer
    atoms_module.HPOConfig = FakeHPOConfig
    model_module = ModuleType("wisent.core.primitives.models.wisent_model")

    model_module.WisentModel = lambda *args, **kwargs: object()
    for module in (search_module, parameters_module, pipeline_module, atoms_module, model_module):
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
    result = runner.main([
        "hpo", "--manifest", str(manifest_path),
        "--completion-index", str(index_path),
        "--calibration-plan", str(plan_path),
        "--output-root", str(output_root),
    ])

    assert result == 0
    destination = Path(capsys.readouterr().out.strip())
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
    assert [call[0]["extraction_strategy"].choices for call in optimizer_calls] == [
        [strategy] for strategy in runner.STRATEGIES
    ]

    frozen = json.loads((destination / "frozen_config.json").read_text())
    assert frozen["fit_pair_ids"] == [1, 4]
    assert frozen["selection_pair_ids"] == [2, 5]
    assert frozen["test_pair_ids_read"] == []
    assert frozen["trial_count"] == 14
    assert [item["trial_count"] for item in frozen["per_format"]] == [2] * 7
    assert frozen["best_params"] == {
        "layer": 8,
        "extraction_strategy": runner.STRATEGIES[-1],
    }
