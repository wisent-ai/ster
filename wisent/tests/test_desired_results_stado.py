import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
STADO_PATH = ROOT / "scripts" / "steering" / "desired_results_stado.py"
SPEC = importlib.util.spec_from_file_location("desired_results_stado", STADO_PATH)
stado = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stado)


def _artifact_ref(uri):
    return {"uri": uri, "generation": "7", "size": "1", "sha256": "a" * 64}


def _build_job(phase, command, **bindings):
    revision = "b" * 40
    return stado.build_job(
        phase=phase,
        command=command,
        policy_sha256="c" * 64,
        code_commit=revision,
        resources={"accelerator": "none", "memory_bytes": 1, "runtime_seconds": 60},
        image={"name": stado.STADO_IMAGE_NAME, "project": stado.STADO_IMAGE_PROJECT},
        dependency_lock_ref=_artifact_ref("gs://wisent-runtime/requirements.lock"),
        secrets=[],
        runtime={"package": stado.RUNTIME_PACKAGE, "device": "cpu", "revision": revision},
        output_prefix="gs://wisent-results/attempt",
        **bindings,
    )


@pytest.mark.parametrize(
    ("phase", "module", "arguments", "bindings"),
    [
        pytest.param(
            "calibration",
            "scripts.steering.desired_results_worker",
            [
                "--calibration-manifest", "gs://wisent-inputs/calibration.json",
                "--calibration-manifest-generation", "7",
                "--attempt-number", "1",
            ],
            {"manifest_ref": _artifact_ref("gs://wisent-inputs/calibration.json"), "attempt": 1},
            id="calibration",
        ),
        pytest.param(
            "arm",
            "scripts.steering.desired_results_final_test_worker",
            [
                "--seal-ref", "gs://wisent-inputs/seal.json",
                "--seal-ref-generation", "7",
                "--arm-manifest", "gs://wisent-inputs/arm.json",
                "--arm-manifest-generation", "7",
                "--attempt-number", "1",
                "--device", "cpu",
            ],
            {
                "seal_ref": _artifact_ref("gs://wisent-inputs/seal.json"),
                "manifest_ref": _artifact_ref("gs://wisent-inputs/arm.json"),
                "attempt": 1,
            },
            id="final-test",
        ),
    ],
)
def test_stado_requires_checkout_resolved_worker_commands(phase, module, arguments, bindings):
    module_command = ["python", "-m", module, *arguments]
    job = _build_job(phase, module_command, **bindings)

    request = stado.build_submission_request(job)

    assert request["command"] == (
        "timeout --signal=TERM --kill-after=30s -- 60s " + " ".join(module_command)
    )

    repo_relative_command = ["python", f"scripts/steering/{module.rsplit('.', 1)[-1]}.py", *arguments]
    with pytest.raises(stado.StadoSubmissionError, match=f"exact {phase} worker argv schema"):
        _build_job(phase, repo_relative_command, **bindings)
