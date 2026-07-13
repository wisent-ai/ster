"""Backend conversion functions for Hyperopt and Optuna param spaces."""
from __future__ import annotations

import math
from typing import Any, Callable

from wisent.core.utils.config_tools.constants import (
    EXTRA_TRIALS_DEFAULT,
    HYPEROPT_BACKEND_NAME,
    INDEX_FIRST,
    OPTUNA_BACKEND_NAME,
    OPTUNA_SIGMA_SPREAD_FACTOR,
    QUANTIZATION_STEP_DEFAULT,
)
from wisent.core.utils.services.optimization.core.parameters import (
    CategoricalParam,
    FloatParam,
    IntParam,
    OptimizationRun,
    Param,
)


def _validate_optuna_resume_distributions(study, space: dict[str, Param]) -> None:
    """Reject persisted distributions that do not exactly match current schemas."""
    from optuna.distributions import CategoricalDistribution, IntDistribution

    qlognormal_params = {
        name: param
        for name, param in space.items()
        if isinstance(param, IntParam) and param.distribution == "qlognormal"
    }
    if not qlognormal_params:
        return

    categorical_supports = {
        name: _qlognormal_categorical_support(name, param)
        for name, param in qlognormal_params.items()
        if param.q != QUANTIZATION_STEP_DEFAULT
    }
    for trial in study.trials:
        for name in qlognormal_params.keys() & trial.distributions.keys():
            param = qlognormal_params[name]
            distribution = trial.distributions[name]
            if (
                param.q != QUANTIZATION_STEP_DEFAULT
                and isinstance(distribution, IntDistribution)
            ):
                incompatibility = (
                    "the pre-categorical IntDistribution schema, but qlognormal "
                    "with q > 1 now requires CategoricalDistribution"
                )
            elif param.q != QUANTIZATION_STEP_DEFAULT:
                expected = categorical_supports[name]
                persisted = (
                    list(distribution.choices)
                    if isinstance(distribution, CategoricalDistribution)
                    else None
                )
                exact_match = persisted is not None and len(persisted) == len(expected)
                if exact_match:
                    exact_match = all(
                        type(old) is type(new) and old == new
                        for old, new in zip(persisted, expected)
                    )
                if exact_match:
                    continue
                incompatibility = (
                    f"categorical choices {persisted!r}, but the newly computed "
                    f"qlognormal support is {expected!r}"
                )
            elif (
                isinstance(distribution, IntDistribution)
                and not distribution.log
            ):
                incompatibility = (
                    "the legacy IntDistribution(log=False) schema, but qlognormal "
                    "with q=1 now requires IntDistribution(log=True)"
                )
            else:
                continue
            raise ValueError(
                f"Optuna study {study.study_name!r} stores parameter {name!r} with "
                f"{incompatibility}. Resume is unsafe because the search support "
                "changed. Choose a new study_name, or explicitly remove/migrate "
                "the persisted study before retrying."
            )


def _qlognormal_categorical_support(name: str, p: IntParam) -> list[int]:
    """Return the exact positive qlognormal support used by Optuna."""
    spread = OPTUNA_SIGMA_SPREAD_FACTOR
    raw_lo = math.exp(p.mu - spread * p.sigma)
    raw_hi = math.exp(p.mu + spread * p.sigma)
    lo_units = raw_lo / p.q
    hi_units = raw_hi / p.q
    lo_nearest = round(lo_units)
    hi_nearest = round(hi_units)
    if abs(lo_units - lo_nearest) <= 4 * math.ulp(lo_units):
        lo_units = float(lo_nearest)
    if abs(hi_units - hi_nearest) <= 4 * math.ulp(hi_units):
        hi_units = float(hi_nearest)
    quantized_lo = math.ceil(lo_units) * p.q
    quantized_hi = math.floor(hi_units) * p.q
    if quantized_lo > quantized_hi:
        raise ValueError(
            f"qlognormal parameter '{name}' has no positive multiple of q={p.q} "
            f"inside [{raw_lo}, {raw_hi}]"
        )
    return list(range(int(quantized_lo), int(quantized_hi) + p.q, p.q))


def run_hyperopt(
    objective_fn: Callable[[dict[str, Any]], float],
    space: dict[str, Param],
    n_trials: int,
    direction: str,
    seed: int | None,
    model: str | None = None,
    benchmark: str | None = None,
    method: str | None = None,
    extra_trials: int = EXTRA_TRIALS_DEFAULT,
) -> OptimizationRun:
    """Run optimization using the Hyperopt backend."""
    from hyperopt import STATUS_OK, Trials, fmin, tpe

    hp_space = {
        name: _param_to_hyperopt(name, param)
        for name, param in space.items()
    }
    maximize = direction == "maximize"
    trials = Trials()
    if model and benchmark and method:
        from wisent.core.utils.services.optimization.core.study_persistence import (
            download_hyperopt_trials, upload_hyperopt_trials,
        )
        cached = download_hyperopt_trials(model, benchmark, method)
        if cached is not None:
            trials = cached
            print(f"  [study] Resuming from {len(trials.trials)} prior trials")

    _should_upload = bool(model and benchmark and method)
    prior_count = len(trials.trials)

    target = n_trials + extra_trials
    if prior_count >= target:
        print(f"  [study] Already has {prior_count} trials (target {target}), skipping")
    else:
        remaining = target - prior_count
        if prior_count:
            print(f"  [study] Resuming from {prior_count}, running {remaining} more")

        def wrapped(params):
            score = objective_fn(params)
            if _should_upload:
                upload_hyperopt_trials(model, benchmark, method, trials)
            return {"loss": -score if maximize else score, "status": STATUS_OK}

        rstate_kwargs = {}
        if seed is not None:
            import numpy as _np
            rstate_kwargs["rstate"] = _np.random.default_rng(seed)
        fmin(
            fn=wrapped, space=hp_space, algo=tpe.suggest,
            max_evals=prior_count + remaining, trials=trials,
            show_progressbar=True, **rstate_kwargs,
        )
        if _should_upload:
            upload_hyperopt_trials(model, benchmark, method, trials)

    all_trials = []
    best_score = None
    best_params = None
    for t in trials.trials:
        vals = t["misc"]["vals"]
        params = {k: v[INDEX_FIRST] for k, v in vals.items()}
        params = resolve_categorical_indices(params, space)
        loss = t["result"]["loss"]
        score = -loss if maximize else loss
        all_trials.append({"params": params, "score": score})
        if best_score is None or (
            (maximize and score > best_score)
            or (not maximize and score < best_score)
        ):
            best_score = score
            best_params = params
    if best_params is None:
        best_raw = {
            k: v[INDEX_FIRST]
            for k, v in trials.best_trial["misc"]["vals"].items()
        }
        best_params = resolve_categorical_indices(best_raw, space)
        best_loss = trials.best_trial["result"]["loss"]
        best_score = -best_loss if maximize else best_loss
    return OptimizationRun(
        best_params=best_params, best_score=best_score,
        all_trials=all_trials, n_trials=len(trials.trials),
        backend=HYPEROPT_BACKEND_NAME,
    )


def run_optuna_functional(
    objective_fn: Callable[[dict[str, Any]], float],
    space: dict[str, Param],
    n_trials: int,
    direction: str,
    sampler,
    pruner,
    storage: str | None,
    study_name: str | None,
    load_if_exists: bool,
    model: str | None = None,
    benchmark: str | None = None,
    method: str | None = None,
    extra_trials: int = EXTRA_TRIALS_DEFAULT,
) -> OptimizationRun:
    """Run optimization using Optuna with Param-based space."""
    import optuna
    import tempfile, os

    # Try to download existing study from HF
    hf_db_path = None
    if not storage and model and benchmark and method:
        from wisent.core.utils.services.optimization.core.study_persistence import (
            download_optuna_db, upload_optuna_db,
        )
        work = tempfile.mkdtemp(prefix="optuna_study_")
        hf_db_path = download_optuna_db(model, benchmark, method, work)
        db_path = hf_db_path or os.path.join(work, "study.db")
        storage = f"sqlite:///{db_path}"
        study_name = study_name or f"{benchmark}_{method.lower()}"
        load_if_exists = True

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: param_to_optuna(trial, name, param)
            for name, param in space.items()
        }
        return objective_fn(params)

    study = optuna.create_study(
        direction=direction, sampler=sampler, pruner=pruner,
        storage=storage, study_name=study_name,
        load_if_exists=bool(storage and study_name and load_if_exists),
    )
    _validate_optuna_resume_distributions(study, space)
    prior_count = len(study.trials)
    target = n_trials + extra_trials
    if prior_count >= target:
        print(f"  [study] Already has {prior_count} trials (target {target}), skipping")
    else:
        remaining = target - prior_count
        if prior_count:
            print(f"  [study] Resuming from {prior_count}, running {remaining} more")
        callbacks = []
        if model and benchmark and method and storage and storage.startswith("sqlite:///"):
            db_file = storage.replace("sqlite:///", "")

            def _upload_cb(study, trial):
                upload_optuna_db(model, benchmark, method, db_file)

            callbacks.append(_upload_cb)
        study.optimize(
            objective, n_trials=remaining, show_progress_bar=True,
            callbacks=callbacks,
        )

    all_trials = [
        {"params": t.params, "score": t.value}
        for t in study.trials if t.value is not None
    ]
    return OptimizationRun(
        best_params=study.best_params, best_score=study.best_value,
        all_trials=all_trials, n_trials=len(study.trials),
        backend=OPTUNA_BACKEND_NAME,
    )


def _param_to_hyperopt(name: str, param: Param):
    """Map a Param dataclass to a hyperopt search expression."""
    from hyperopt import hp

    if isinstance(param, FloatParam):
        return _float_to_hyperopt(name, param)
    if isinstance(param, IntParam):
        return _int_to_hyperopt(name, param)
    if isinstance(param, CategoricalParam):
        return hp.choice(name, param.choices)
    raise TypeError(f"Unknown param type: {type(param)}")


def _float_to_hyperopt(name: str, p: FloatParam):
    from hyperopt import hp

    d = p.distribution
    if d == "normal":
        return hp.normal(name, p.mu, p.sigma)
    if d == "lognormal":
        return hp.lognormal(name, p.mu, p.sigma)
    if d == "uniform":
        return hp.uniform(name, p.low, p.high)
    raise ValueError(f"Unknown float distribution: {d}")


def _int_to_hyperopt(name: str, p: IntParam):
    from hyperopt import hp
    from hyperopt.pyll import scope

    d = p.distribution
    if d == "randint":
        return scope.int(
            hp.quniform(name, p.low, p.high, QUANTIZATION_STEP_DEFAULT),
        )
    if d == "qnormal":
        return scope.int(hp.qnormal(name, p.mu, p.sigma, p.q))
    if d == "qlognormal":
        return scope.int(hp.qlognormal(name, p.mu, p.sigma, p.q))
    raise ValueError(f"Unknown int distribution: {d}")


def param_to_optuna(trial, name: str, param: Param):
    """Map a Param dataclass to an Optuna trial suggestion."""
    if isinstance(param, FloatParam):
        return _float_to_optuna(trial, name, param)
    if isinstance(param, IntParam):
        return _int_to_optuna(trial, name, param)
    if isinstance(param, CategoricalParam):
        return trial.suggest_categorical(name, param.choices)
    raise TypeError(f"Unknown param type: {type(param)}")


def _float_to_optuna(trial, name: str, p: FloatParam):
    d = p.distribution
    if d == "uniform":
        return trial.suggest_float(name, p.low, p.high, log=p.log_scale)
    spread = OPTUNA_SIGMA_SPREAD_FACTOR
    if d == "normal":
        lo = p.mu - spread * p.sigma
        hi = p.mu + spread * p.sigma
        return trial.suggest_float(name, lo, hi)
    if d == "lognormal":
        lo = math.exp(p.mu - spread * p.sigma)
        hi = math.exp(p.mu + spread * p.sigma)
        return trial.suggest_float(name, lo, hi, log=True)
    raise ValueError(f"Unknown float distribution: {d}")


def _int_to_optuna(trial, name: str, p: IntParam):
    d = p.distribution
    if d == "randint":
        return trial.suggest_int(name, p.low, p.high)
    spread = OPTUNA_SIGMA_SPREAD_FACTOR
    if d == "qnormal":
        lo = max(INDEX_FIRST, int(p.mu - spread * p.sigma))
        hi = int(p.mu + spread * p.sigma)
        return trial.suggest_int(name, lo, hi, step=p.q)
    if d == "qlognormal":
        if p.q != QUANTIZATION_STEP_DEFAULT:
            return trial.suggest_categorical(
                name, _qlognormal_categorical_support(name, p),
            )
        raw_lo = math.exp(p.mu - spread * p.sigma)
        raw_hi = math.exp(p.mu + spread * p.sigma)
        lo_units = raw_lo / p.q
        hi_units = raw_hi / p.q
        lo_nearest = round(lo_units)
        hi_nearest = round(hi_units)
        if abs(lo_units - lo_nearest) <= 4 * math.ulp(lo_units):
            lo_units = float(lo_nearest)
        if abs(hi_units - hi_nearest) <= 4 * math.ulp(hi_units):
            hi_units = float(hi_nearest)
        quantized_lo = math.ceil(lo_units) * p.q
        quantized_hi = math.floor(hi_units) * p.q
        if quantized_lo > quantized_hi:
            raise ValueError(
                f"qlognormal parameter '{name}' has no positive multiple of q={p.q} "
                f"inside [{raw_lo}, {raw_hi}]"
            )
        return trial.suggest_int(
            name, int(quantized_lo), int(quantized_hi), log=True,
        )
    raise ValueError(f"Unknown int distribution: {d}")


def resolve_categorical_indices(
    params: dict, space: dict[str, Param],
) -> dict:
    """Hyperopt returns indices for hp.choice; resolve to values."""
    resolved = {}
    for k, v in params.items():
        if k in space and isinstance(space[k], CategoricalParam):
            idx = int(v)
            choices = space[k].choices
            resolved[k] = choices[idx] if idx < len(choices) else v
        else:
            resolved[k] = v
    return resolved
