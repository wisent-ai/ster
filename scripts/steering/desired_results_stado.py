#!/usr/bin/env python3
"""Source-pinned Stado submission adapter for desired-results execution.

Planning is pure and deterministic.  Live dispatch is explicit and deliberately
uses only arguments supported by the installed :mod:`stado` client.  Runtime
identity is measured by the worker and is never represented as a scheduler
option.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Callable

try:
    from scripts.steering import desired_results_execution_contract as execution
except (ImportError, ModuleNotFoundError):
    import importlib.util

    _EXECUTION_PATH = Path(__file__).resolve().with_name("desired_results_execution_contract.py")
    _EXECUTION_SPEC = importlib.util.spec_from_file_location(
        "desired_results_execution_contract_stado", _EXECUTION_PATH
    )
    if _EXECUTION_SPEC is None or _EXECUTION_SPEC.loader is None:
        raise ImportError(f"cannot load execution contract from {_EXECUTION_PATH}")
    execution = importlib.util.module_from_spec(_EXECUTION_SPEC)
    _EXECUTION_SPEC.loader.exec_module(execution)

CANONICAL_REPO = "https://github.com/wisent-ai/wisent.git"
DEFAULT_REPO_WORKDIR = "wisent"
DEFAULT_REPO_EXTRAS = "harness"
PROVIDER = "gcp"
STADO_IMAGE_NAME = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260408"
STADO_IMAGE_PROJECT = "deeplearning-platform-release"
STADO_QUEUE_BUCKET = "wisent-compute"
IMAGE_KEYS = frozenset({"name", "project"})
RUNTIME_KEYS = frozenset({"package", "device", "revision"})
RUNTIME_PACKAGE = "wisent-runtime"
DEPENDENCY_LOCK_FORMAT = "pip-requirements-with-hashes-v1"
DEPENDENCY_LOCK_SUFFIX = ".lock"
DEPENDENCY_LOCK_PATH = '"$WORK/dependency-lock.requirements"'
RUNTIME_TIMEOUT_KILL_AFTER_SECONDS = 30

_DEPENDENCY_LOCK_URI = re.compile(r"^gs://[a-z0-9][a-z0-9._-]*/[^\s]+\.lock$")

SUBMISSION_KEYS = frozenset({"provider", "repo", "repo_workdir", "repo_extras", "code_commit"})
SUBMISSION_SOURCE_KEYS = frozenset(
    {"repo", "code_commit", "repo_workdir", "repo_extras", "pre_command_sha256"}
)
RESOURCE_KEYS = frozenset({"accelerator", "memory_bytes", "runtime_seconds"})
SUPPORTED_SUBMIT_KWARGS = frozenset(
    {
        "provider", "batch_id", "bucket", "pin_to_provider", "repo",
        "repo_workdir", "repo_extras", "gpu_type", "vram_gb", "pre_command", "run_id",
    }
)
IMMUTABLE_JOB_KEYS = frozenset(
    {"command", "dependency_lock_ref", "image", "resources", "secrets", "runtime", "submission", "submission_source"}
)
RECEIPT_BINDING_KEYS = frozenset(
    {"job_id", "scheduler_job_id", "plan_sha256", "policy_sha256", "submission_source_sha256"}
)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_SHELL_OR_CONTROL = re.compile(r"[\x00-\x20\x7f;&|`$<>\\\"'(){}\[\]*?!~#]")
_SAFE_DEVICE = re.compile(r"^[A-Za-z0-9._:+-]+$")


class StadoSubmissionError(RuntimeError):
    """A planned submission or scheduler receipt is unsafe or inconsistent."""


def _queue_bucket() -> str:
    return os.environ.get("WC_BUCKET", STADO_QUEUE_BUCKET).strip() or STADO_QUEUE_BUCKET


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StadoSubmissionError("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StadoSubmissionError(f"{label} must be an object with string keys")
    return dict(value)


def _string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise StadoSubmissionError(f"{label} must be a{' non-empty' if nonempty else ''} string")
    return value


def _digest(value: Any, label: str) -> str:
    text = _string(value, label)
    if not _HEX64.fullmatch(text):
        raise StadoSubmissionError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _exact(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    root = _mapping(value, label)
    if set(root) != set(keys):
        missing = sorted(set(keys) - set(root))
        extra = sorted(set(root) - set(keys))
        raise StadoSubmissionError(f"{label} has wrong keys (missing={missing}, extra={extra})")
    return root


def _validate_dependency_lock_ref(value: Any, label: str = "job.dependency_lock_ref") -> dict[str, str]:
    """Validate the sole lock format: a generation-pinned pip requirements lock with hashes."""
    try:
        ref = execution.validate_artifact_ref(value, label)
    except execution.ContractError as exc:
        raise StadoSubmissionError(str(exc)) from exc
    uri = ref["uri"]
    if _DEPENDENCY_LOCK_URI.fullmatch(uri) is None:
        raise StadoSubmissionError(
            f"{label}.uri must be a gs:// object ending in {DEPENDENCY_LOCK_SUFFIX!r} "
            f"for {DEPENDENCY_LOCK_FORMAT}"
        )
    generation = ref["generation"]
    if not generation.isdigit() or generation.startswith("0"):
        raise StadoSubmissionError(f"{label}.generation must be a canonical positive decimal string")
    return ref




def validate_submission_source(value: Any, label: str = "submission") -> dict[str, str]:
    """Validate the exact repository identity accepted by production dispatch."""
    root = _exact(value, SUBMISSION_KEYS, label)
    clean = {name: _string(root[name], f"{label}.{name}")
             for name in ("provider", "repo", "repo_workdir", "code_commit")}
    extras = _string(root["repo_extras"], f"{label}.repo_extras", nonempty=False)
    clean["repo_extras"] = extras
    if clean["provider"] != PROVIDER:
        raise StadoSubmissionError(f"{label}.provider must equal 'gcp'")
    if clean["repo"] != CANONICAL_REPO:
        raise StadoSubmissionError(f"{label}.repo must equal the canonical Wisent HTTPS URL")
    workdir = clean["repo_workdir"]
    if workdir in {".", ".."} or not _SAFE_NAME.fullmatch(workdir):
        raise StadoSubmissionError(f"{label}.repo_workdir must be one safe relative directory name")
    if extras and any(not _SAFE_NAME.fullmatch(part) for part in extras.split(",")):
        raise StadoSubmissionError(f"{label}.repo_extras must be safe comma-separated extra names")
    if not _HEX40.fullmatch(clean["code_commit"]):
        raise StadoSubmissionError(f"{label}.code_commit must be a 40-character lowercase Git commit")
    return clean


def build_pre_command(submission: Mapping[str, Any], dependency_lock_ref: Mapping[str, Any]) -> str:
    """Fetch and verify the pinned pip lock before installing it and the detached checkout."""
    source = validate_submission_source(submission)
    lock = _validate_dependency_lock_ref(dependency_lock_ref)
    repo_dir = f'"$WORK/{source["repo_workdir"]}"'
    commit = shlex.quote(source["code_commit"])
    target = "." if not source["repo_extras"] else f'.[{source["repo_extras"]}]'
    return "\n".join(
        (
            "set -eu",
            f"rm -f -- {DEPENDENCY_LOCK_PATH}",
            "gsutil -h "
            f"x-goog-if-generation-match:{shlex.quote(lock['generation'])} cp "
            f"{shlex.quote(lock['uri'])} {DEPENDENCY_LOCK_PATH}",
            f'test "$(/usr/bin/wc -c < {DEPENDENCY_LOCK_PATH} | tr -d \'[:space:]\')" = '
            f"{shlex.quote(lock['size'])}",
            f'test "$(sha256sum {DEPENDENCY_LOCK_PATH} | cut -d \' \' -f 1)" = '
            f"{shlex.quote(lock['sha256'])}",
            f"python -m pip install --require-hashes --no-deps -r {DEPENDENCY_LOCK_PATH}",
            f"git -C {repo_dir} fetch --depth 1 origin {commit}",
            f"git -C {repo_dir} checkout --detach {commit}",
            f'test "$(git -C {repo_dir} rev-parse HEAD)" = {commit}',
            f"cd {repo_dir}",
            f"python -m pip install --no-deps -e {shlex.quote(target)}",
        )
    )


def submission_source(value: Mapping[str, Any], dependency_lock_ref: Mapping[str, Any]) -> dict[str, str]:
    source = validate_submission_source(value)
    pre_command = build_pre_command(source, dependency_lock_ref)
    return {
        "repo": source["repo"],
        "code_commit": source["code_commit"],
        "repo_workdir": source["repo_workdir"],
        "repo_extras": source["repo_extras"],
        "pre_command_sha256": hashlib.sha256(pre_command.encode("utf-8")).hexdigest(),
    }


def submission_source_sha256(job: Mapping[str, Any]) -> str:
    root = _mapping(job, "job")
    planned = _exact(root.get("submission_source"), SUBMISSION_SOURCE_KEYS, "job.submission_source")
    expected = submission_source(
        _mapping(root.get("submission"), "job.submission"),
        _mapping(root.get("dependency_lock_ref"), "job.dependency_lock_ref"),
    )
    if planned != expected:
        raise StadoSubmissionError("job.submission_source differs from its exact checkout pre-command")
    return canonical_sha256(planned)


def _validate_resources(value: Any) -> dict[str, Any]:
    root = _exact(value, RESOURCE_KEYS, "job.resources")
    accelerator = _string(root["accelerator"], "job.resources.accelerator")
    for key in ("memory_bytes", "runtime_seconds"):
        if isinstance(root[key], bool) or not isinstance(root[key], int) or root[key] <= 0:
            raise StadoSubmissionError(f"job.resources.{key} must be a positive integer")
    return {"accelerator": accelerator, "memory_bytes": root["memory_bytes"],
            "runtime_seconds": root["runtime_seconds"]}


def _validate_command(value: Any, job: Mapping[str, Any]) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise StadoSubmissionError("job.command must be a non-empty argv array")
    result = [_string(item, f"job.command[{index}]") for index, item in enumerate(value)]
    if any(_SHELL_OR_CONTROL.search(item) is not None for item in result):
        raise StadoSubmissionError("job.command must not contain shell or control characters")

    phase = _string(job.get("phase"), "job.phase")
    if phase == "calibration":
        ref = _mapping(job.get("manifest_ref"), "job.manifest_ref")
        uri = _string(ref.get("uri"), "job.manifest_ref.uri")
        generation = _string(ref.get("generation"), "job.manifest_ref.generation")
        expected = [
            "python", "-m", "scripts.steering.desired_results_worker",
            "--calibration-manifest", uri,
            "--calibration-manifest-generation", generation,
            "--attempt-number", str(job.get("attempt")),
        ]
    elif phase == "arm":
        seal_ref = _mapping(job.get("seal_ref"), "job.seal_ref")
        seal_uri = _string(seal_ref.get("uri"), "job.seal_ref.uri")
        seal_generation = _string(seal_ref.get("generation"), "job.seal_ref.generation")
        ref = _mapping(job.get("manifest_ref"), "job.manifest_ref")
        uri = _string(ref.get("uri"), "job.manifest_ref.uri")
        generation = _string(ref.get("generation"), "job.manifest_ref.generation")
        device = _string(job.get("runtime", {}).get("device"), "job.runtime.device")
        if _SAFE_DEVICE.fullmatch(device) is None:
            raise StadoSubmissionError("job.runtime.device contains unsafe characters")
        expected = [
            "python", "-m", "scripts.steering.desired_results_final_test_worker",
            "--seal-ref", seal_uri, "--seal-ref-generation", seal_generation,
            "--arm-manifest", uri, "--arm-manifest-generation", generation,
            "--attempt-number", str(job.get("attempt")), "--device", device,
        ]
        if not seal_generation.isdigit() or seal_generation.startswith("0"):
            raise StadoSubmissionError("final seal generation must be a canonical positive decimal")
    else:
        raise StadoSubmissionError("job.phase must be exactly calibration or arm")
    if not generation.isdigit() or generation.startswith("0"):
        raise StadoSubmissionError("worker input generation must be a canonical positive decimal")
    if type(job.get("attempt")) is not int or job["attempt"] <= 0:
        raise StadoSubmissionError("job.attempt must be a positive integer")
    if result != expected:
        raise StadoSubmissionError(f"job.command does not match the exact {phase} worker argv schema")
    return result


def _validate_production_refs(value: Any, path: str = "job") -> None:
    if isinstance(value, Mapping):
        if "uri" in value:
            uri = value.get("uri")
            generation = value.get("generation")
            if not isinstance(uri, str) or not uri.startswith("gs://"):
                raise StadoSubmissionError(f"{path} contains a non-production artifact URI")
            if isinstance(generation, bool) or not (
                isinstance(generation, int) and generation > 0
                or isinstance(generation, str) and generation.isdigit() and int(generation) > 0
            ):
                raise StadoSubmissionError(f"{path} contains an unpinned artifact generation")
        for key, child in value.items():
            _validate_production_refs(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _validate_production_refs(child, f"{path}[{index}]")
    elif isinstance(value, str) and value.startswith(("file://", "local://", "bundle://")):
        raise StadoSubmissionError(f"{path} contains a local reference")


def _logical_payload(job: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key not in {"job_id", "plan_sha256"}}


def validate_submission_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Return a defensive canonical copy of one immutable planned job."""
    root = _mapping(job, "job")
    missing = sorted(IMMUTABLE_JOB_KEYS - set(root))
    if missing:
        raise StadoSubmissionError(f"job is missing immutable fields: {missing}")
    clean = json.loads(canonical_bytes(root))
    clean["submission"] = validate_submission_source(clean["submission"], "job.submission")
    clean["dependency_lock_ref"] = _validate_dependency_lock_ref(clean["dependency_lock_ref"])
    planned_source = _exact(clean["submission_source"], SUBMISSION_SOURCE_KEYS, "job.submission_source")
    if planned_source != submission_source(clean["submission"], clean["dependency_lock_ref"]):
        raise StadoSubmissionError("job.submission_source differs from its exact checkout pre-command")
    clean["resources"] = _validate_resources(clean["resources"])
    if not isinstance(clean["secrets"], list) or any(not isinstance(x, str) or not x for x in clean["secrets"]):
        raise StadoSubmissionError("job.secrets must be an array of secret names")
    if len(set(clean["secrets"])) != len(clean["secrets"]):
        raise StadoSubmissionError("job.secrets must not contain duplicates")
    runtime = _exact(clean["runtime"], RUNTIME_KEYS, "job.runtime")
    if _string(runtime["package"], "job.runtime.package") != RUNTIME_PACKAGE:
        raise StadoSubmissionError(f"job.runtime.package must equal {RUNTIME_PACKAGE!r}")
    _string(runtime["device"], "job.runtime.device")
    revision = _string(runtime["revision"], "job.runtime.revision")
    if not _HEX40.fullmatch(revision):
        raise StadoSubmissionError("job.runtime.revision must be a pinned 40-character lowercase revision")
    if revision != clean["submission"]["code_commit"]:
        raise StadoSubmissionError("job.runtime.revision must equal the detached code commit")
    clean["command"] = _validate_command(clean["command"], clean)
    image = _exact(clean["image"], IMAGE_KEYS, "job.image")
    if image != {"name": STADO_IMAGE_NAME, "project": STADO_IMAGE_PROJECT}:
        raise StadoSubmissionError("job.image must equal the installed Stado fixed image identity")
    _validate_production_refs(clean)
    if "policy_sha256" in clean:
        _digest(clean["policy_sha256"], "job.policy_sha256")
    logical_digest = canonical_sha256(_logical_payload(clean))
    if "plan_sha256" in clean and clean["plan_sha256"] != logical_digest:
        raise StadoSubmissionError("job.plan_sha256 does not match the immutable planned job")
    if "job_id" in clean:
        _string(clean["job_id"], "job.job_id")
    return clean


validate_job = validate_submission_job


def build_job(*, phase: str, command: Sequence[str], policy_sha256: str,
              code_commit: str, resources: Mapping[str, Any], image: Mapping[str, Any],
              dependency_lock_ref: Mapping[str, Any], secrets: Sequence[str], runtime: Mapping[str, Any],
              output_prefix: str, input_refs: Sequence[Mapping[str, Any]] = (),
              dependencies: Sequence[str] = (), provider: str = PROVIDER,
              repo: str = CANONICAL_REPO, repo_workdir: str = DEFAULT_REPO_WORKDIR,
              repo_extras: str = DEFAULT_REPO_EXTRAS, **bindings: Any) -> dict[str, Any]:
    """Construct and content-address one stable planned job for RunV3."""
    source = {"provider": provider, "repo": repo, "repo_workdir": repo_workdir,
              "repo_extras": repo_extras, "code_commit": code_commit}
    lock = _validate_dependency_lock_ref(dependency_lock_ref, "dependency_lock_ref")
    payload: dict[str, Any] = {
        "phase": _string(phase, "phase"), "command": list(command),
        "policy_sha256": _digest(policy_sha256, "policy_sha256"),
        "input_refs": list(input_refs), "dependencies": list(dependencies),
        "dependency_lock_ref": lock, "image": dict(image), "resources": dict(resources),
        "secrets": list(secrets), "runtime": dict(runtime), "submission": source,
        "submission_source": submission_source(source, lock),
        "output_prefix": _string(output_prefix, "output_prefix"), **bindings,
    }
    digest = canonical_sha256(payload)
    payload["plan_sha256"] = digest
    payload["job_id"] = f"desired-{phase}-{digest[:20]}"
    return validate_submission_job(payload)


def _deterministic_ids(job: Mapping[str, Any]) -> tuple[str, str]:
    plan = job.get("plan_sha256") or canonical_sha256(_logical_payload(job))
    policy = job.get("policy_sha256", "")
    seed = hashlib.sha256(f"{plan}:{policy}".encode()).hexdigest()
    return f"dr-batch-{seed[:20]}", f"dr-run-{seed[20:40]}"


def build_submission_request(job: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a planned job to the installed Stado submit_job fields."""
    clean = validate_submission_job(job)
    source = clean["submission"]
    resources = clean["resources"]
    batch_id, run_id = _deterministic_ids(clean)
    remote_command = " ".join(
        (
            "timeout",
            "--signal=TERM",
            f"--kill-after={RUNTIME_TIMEOUT_KILL_AFTER_SECONDS}s",
            "--",
            f"{resources['runtime_seconds']}s",
            shlex.join(clean["command"]),
        )
    )
    request: dict[str, Any] = {
        "command": remote_command,
        "provider": source["provider"],
        "batch_id": batch_id,
        "bucket": _queue_bucket(),
        "pin_to_provider": True,
        "repo": source["repo"],
        "repo_workdir": source["repo_workdir"],
        # Stado otherwise performs an unpinned editable install before pre_command.
        "repo_extras": "",
        "pre_command": build_pre_command(source, clean["dependency_lock_ref"]),
        "run_id": run_id,
    }
    if resources["accelerator"] == "none":
        if "--model" in clean["command"]:
            raise StadoSubmissionError("a model job cannot be safely forced onto Stado CPU")
    else:
        request["gpu_type"] = resources["accelerator"]
        # Round upward so byte-level policy limits are not weakened in translation.
        request["vram_gb"] = -(-resources["memory_bytes"] // (1024**3))
    expected = {"command"} | (SUPPORTED_SUBMIT_KWARGS - {"gpu_type", "vram_gb"})
    if resources["accelerator"] != "none":
        expected |= {"gpu_type", "vram_gb"}
    if set(request) != expected:
        raise AssertionError("adapter generated unsupported Stado fields")
    return request


def _bindings(job: Mapping[str, Any]) -> dict[str, str]:
    clean = validate_submission_job(job)
    plan = clean.get("plan_sha256") or canonical_sha256(_logical_payload(clean))
    return {
        "job_id": clean.get("job_id") or f"desired-job-{plan[:20]}",
        "plan_sha256": plan,
        "policy_sha256": _digest(clean.get("policy_sha256"), "job.policy_sha256"),
        "submission_source_sha256": submission_source_sha256(clean),
    }


def validate_submission_receipt(receipt: Mapping[str, Any], job: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = _mapping(receipt, "submission receipt")
    for key in RECEIPT_BINDING_KEYS:
        if key not in root:
            raise StadoSubmissionError(f"submission receipt is missing {key}")
    for key in ("plan_sha256", "policy_sha256", "submission_source_sha256"):
        _digest(root[key], f"submission receipt.{key}")
    _string(root["job_id"], "submission receipt.job_id")
    _string(root["scheduler_job_id"], "submission receipt.scheduler_job_id")
    direct = ("submission_source", "image", "dependency_lock_ref", "resources", "secrets", "runtime")
    for key in direct:
        if key not in root:
            raise StadoSubmissionError(f"submission receipt is missing immutable {key}")
    planned = validate_submission_job(root.get("planned_job") if job is None else job)
    expected = _bindings(planned)
    for key, value in expected.items():
        if root.get(key) != value:
            raise StadoSubmissionError(f"submission receipt {key} conflicts with the planned job")
    if root.get("planned_job") != planned:
        raise StadoSubmissionError("submission receipt does not retain the immutable planned job")
    if root.get("submitted") is not True:
        raise StadoSubmissionError("submission receipt must prove a live submission")
    if root.get("request") != build_submission_request(planned):
        raise StadoSubmissionError("submission receipt request differs from the exact enforced request")
    for key in direct:
        if root.get(key) != planned.get(key):
            raise StadoSubmissionError(f"submission receipt immutable {key} conflicts with the planned job")
    return json.loads(canonical_bytes(root))


def validate_worker_source_receipt(receipt: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    """Reject absent, ambient, stale, or different worker checkout evidence."""
    root = _mapping(receipt, "worker receipt")
    evidence = _mapping(root.get("submission_source"), "worker receipt.submission_source")
    planned = validate_submission_job(job)
    expected = submission_source(planned["submission"], planned["dependency_lock_ref"])
    if evidence.get("ambient_checkout") is not False:
        raise StadoSubmissionError("worker source evidence must prove a non-ambient checkout")
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise StadoSubmissionError(f"worker source evidence differs at {key}")
    if evidence.get("head_commit") != expected["code_commit"]:
        raise StadoSubmissionError("worker source evidence has a stale or different HEAD")
    if root.get("submission_source_sha256") != canonical_sha256(expected):
        raise StadoSubmissionError("worker source digest does not bind the planned submission")
    return json.loads(canonical_bytes(root))


def dispatch_jobs(jobs: Sequence[Mapping[str, Any]], submit: bool = False,
                  submit_job_fn: Callable[..., Any] | None = None) -> list[dict[str, Any]]:
    """Deterministically plan or live-submit jobs, deduplicating byte-identical IDs."""
    unique: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for raw in jobs:
        job = validate_submission_job(raw)
        binding = _bindings(job)
        encoded = canonical_bytes(job)
        prior = unique.get(binding["job_id"])
        if prior is not None:
            if prior[0] != encoded:
                raise StadoSubmissionError(f"conflicting duplicate logical job_id {binding['job_id']}")
            continue
        unique[binding["job_id"]] = (encoded, job)
    ordered = [item[1] for item in sorted(unique.values(), key=lambda item: _bindings(item[1])["job_id"])]
    dry: list[dict[str, Any]] = []
    for job in ordered:
        request = build_submission_request(job)
        dry.append({"submitted": False, **_bindings(job), "request": request, "planned_job": job})
    if not submit:
        return json.loads(canonical_bytes(dry))
    if os.environ.get("COMPUTE_API_KEY", "").strip():
        raise StadoSubmissionError("COMPUTE_API_KEY mode drops source/GPU controls; use direct GCS mode")
    if submit_job_fn is None:
        try:
            from stado.queue.submit import submit_job as submit_job_fn
        except ImportError as exc:
            raise StadoSubmissionError("the installed Stado SDK is required for live submission") from exc
    receipts: list[dict[str, Any]] = []
    for planned in dry:
            request = dict(planned["request"])
            job = planned["planned_job"]
            expected = {"command"} | (SUPPORTED_SUBMIT_KWARGS - {"gpu_type", "vram_gb"})
            if job["resources"]["accelerator"] != "none":
                expected |= {"gpu_type", "vram_gb"}
            actual = set(request)
            if actual != expected:
                unsupported = sorted(actual - expected)
                missing = sorted(expected - actual)
                detail = (
                    f"unsupported Stado kwargs: {unsupported}"
                    if unsupported
                    else f"missing Stado kwargs: {missing}"
                )
                raise StadoSubmissionError(f"submission request has {detail}")
            command = request.pop("command")
            submitted = submit_job_fn(command, **request)
            scheduler_id = getattr(submitted, "job_id", None)
            if not isinstance(scheduler_id, str) or not scheduler_id:
                raise StadoSubmissionError("Stado returned no immutable scheduler job_id")
            receipt = {
                **planned, "submitted": True, "scheduler_job_id": scheduler_id,
                **{key: job[key] for key in
                   ("submission_source", "image", "dependency_lock_ref", "resources", "secrets", "runtime")},
            }
            receipts.append(validate_submission_receipt(receipt, planned["planned_job"]))
    return receipts


def submit_jobs(jobs: Sequence[Mapping[str, Any]], *, submit: bool = False,
                submit_job_fn: Callable[..., Any] | None = None) -> list[dict[str, Any]]:
    return dispatch_jobs(jobs, submit=submit, submit_job_fn=submit_job_fn)


__all__ = [
    "CANONICAL_REPO", "DEFAULT_REPO_EXTRAS", "DEFAULT_REPO_WORKDIR",
    "DEPENDENCY_LOCK_FORMAT", "DEPENDENCY_LOCK_PATH", "DEPENDENCY_LOCK_SUFFIX", "IMAGE_KEYS",
    "IMMUTABLE_JOB_KEYS", "PROVIDER", "RECEIPT_BINDING_KEYS", "RESOURCE_KEYS", "RUNTIME_KEYS",
    "RUNTIME_PACKAGE", "RUNTIME_TIMEOUT_KILL_AFTER_SECONDS", "STADO_IMAGE_NAME",
    "STADO_IMAGE_PROJECT", "STADO_QUEUE_BUCKET", "SUBMISSION_KEYS", "SUBMISSION_SOURCE_KEYS",
    "SUPPORTED_SUBMIT_KWARGS",
    "StadoSubmissionError", "build_job", "build_pre_command", "build_submission_request",
    "canonical_bytes", "canonical_sha256", "dispatch_jobs", "submission_source",
    "submission_source_sha256", "submit_jobs", "validate_job", "validate_submission_job",
    "validate_submission_receipt", "validate_submission_source", "validate_worker_source_receipt",
]
