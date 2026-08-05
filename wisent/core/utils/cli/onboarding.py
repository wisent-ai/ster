"""Durable first-use journey for the Wisent command-line product surface."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

PRODUCT_ID = "wisent"
JOURNEY_ID = "first-use"
JOURNEY_VERSION = "2026-08-04.1"
SCHEMA_VERSION = 1
FIRST_SUCCESS_FACT = "representation_operation_completed"
CLIENT_ID = "python"
TOKEN_ENV = "WISENT_STADO_INTEGRATION_TOKEN"
BASE_URL_ENV = "STADO_INTEGRATION_API_URL"

_EVENT_NAMES = {
    "onboarding_started",
    "onboarding_step_viewed",
    "onboarding_step_completed",
    "onboarding_step_skipped",
    "onboarding_abandoned",
    "onboarding_resumed",
    "onboarding_reset",
    "onboarding_first_action_completed",
    "onboarding_first_success_observed",
    "onboarding_completed",
}
_STATUSES = {"in_progress", "skipped", "completed", "abandoned", "reset"}
_OPERATORS = {"present", "absent", "eq", "not_eq", "contains", "gt", "gte", "lt", "lte"}

_CONTENT = {
    "welcome.title": "Engineer a representation, end to end",
    "welcome.body": (
        "Wisent turns contrastive examples into inspectable model activations and steering vectors. "
        "This short journey follows the same workflow as the CLI rather than simulating a result."
    ),
    "workflow.title": "The representation-engineering workflow",
    "workflow.body": (
        "Build positive and negative examples, collect their activations, then derive and inspect a "
        "steering representation. Keep the generated pairs, enriched activations, and vector output: "
        "those artifacts are the evidence for each stage."
    ),
    "operation.title": "Produce your first real result",
    "operation.body": (
        "Run a representation command against a model and inspect its output. A full path starts with "
        "`wisent generate-vector-from-synthetic --help`; focused paths include `generate-pairs`, "
        "`get-activations`, and `create-steering-vector`. This journey completes only after one of "
        "those operations returns successfully—not when you continue past this screen."
    ),
}
_ALLOWED_ACTIONS = {"continue", "run_representation_operation"}

_REPRESENTATION_COMMANDS = frozenset(
    {
        "tasks",
        "generate-pairs",
        "generate-pairs-from-task",
        "get-activations",
        "create-steering-vector",
        "generate-vector-from-task",
        "generate-vector-from-synthetic",
        "synthetic",
        "generate-responses",
        "multi-steer",
        "modify-weights",
        "optimize-steering",
        "verify-steering",
        "discover-steering",
        "find-best-method",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _fallback_bundle() -> Dict[str, Any]:
    definition: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "product_id": PRODUCT_ID,
        "journey_id": JOURNEY_ID,
        "journey_version": JOURNEY_VERSION,
        "entry_screen_id": "welcome",
        "first_success_fact": FIRST_SUCCESS_FACT,
        "published_at": "2026-08-04T00:00:00Z",
        "source_revision": "wisent-cli-first-use-2026-08-04.1",
        "screens": [
            {
                "screen_id": "welcome",
                "screen_kind": "explanation",
                "title_key": "welcome.title",
                "body_key": "welcome.body",
                "required": False,
                "actions": ["continue"],
                "transitions": [
                    {
                        "next_screen_id": "workflow",
                        "reason_code": "workflow_requested",
                        "priority": 0,
                    }
                ],
                "presentation": {"surface": "cli"},
            },
            {
                "screen_id": "workflow",
                "screen_kind": "explanation",
                "title_key": "workflow.title",
                "body_key": "workflow.body",
                "required": False,
                "actions": ["continue"],
                "transitions": [
                    {
                        "next_screen_id": "operation",
                        "reason_code": "operation_requested",
                        "priority": 0,
                    }
                ],
                "presentation": {"surface": "cli"},
            },
            {
                "screen_id": "operation",
                "screen_kind": "first_success",
                "title_key": "operation.title",
                "body_key": "operation.body",
                "required": True,
                "completion_evidence": {
                    "kind": "fact",
                    "fact": FIRST_SUCCESS_FACT,
                    "operator": "eq",
                    "value": True,
                },
                "actions": ["run_representation_operation"],
                "transitions": [],
                "presentation": {"surface": "cli"},
            },
        ],
        "analytics_contract": {
            "contract_version": "1",
            "surface": "cli",
            "exposure_event": "onboarding_step_viewed",
            "primary_action_event": "onboarding_first_action_completed",
            "completion_event": "onboarding_completed",
            "first_success_event": "onboarding_first_success_observed",
        },
        "experiment_contract": {
            "experiment_id": "wisent-first-use-cli-2026-08-04",
            "control_variant_id": "control",
            "eligible_variant_ids": ["control"],
            "assignment_unit": "device",
            "reward_event": "onboarding_first_success_observed",
            "guardrail_events": ["onboarding_abandoned"],
            "owner": "wisent",
            "kill_switch": False,
        },
    }
    canonical_definition = _canonical(definition)
    return {
        "journey_version_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "wisent:first-use:2026-08-04.1")),
        "definition": definition,
        "canonical_definition": canonical_definition,
        "content_sha256": hashlib.sha256(canonical_definition.encode("utf-8")).hexdigest(),
        "source_revision": definition["source_revision"],
    }


def _validate_condition(condition: Any) -> None:
    if not isinstance(condition, dict):
        raise ValueError("journey condition must be an object")
    kind = condition.get("kind")
    if kind in {"all", "any"}:
        children = condition.get("conditions")
        if not isinstance(children, list):
            raise ValueError("journey condition group is invalid")
        for child in children:
            _validate_condition(child)
    elif kind == "not":
        _validate_condition(condition.get("condition"))
    elif kind == "fact":
        if not isinstance(condition.get("fact"), str) or condition.get("operator") not in _OPERATORS:
            raise ValueError("journey fact condition is invalid")
    else:
        raise ValueError("journey condition kind is invalid")


def _validate_bundle(bundle: Any) -> Dict[str, Any]:
    if not isinstance(bundle, dict):
        raise ValueError("journey bundle is not an object")
    try:
        uuid.UUID(str(bundle["journey_version_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("journey version id is invalid") from exc
    definition = bundle.get("definition")
    canonical_definition = bundle.get("canonical_definition")
    digest = bundle.get("content_sha256")
    if not isinstance(definition, dict) or not isinstance(canonical_definition, str):
        raise ValueError("journey definition is invalid")
    if _canonical(definition) != canonical_definition:
        raise ValueError("journey canonical definition does not match")
    if not isinstance(digest, str) or hashlib.sha256(canonical_definition.encode("utf-8")).hexdigest() != digest:
        raise ValueError("journey content hash does not match")
    if (
        definition.get("schema_version") != SCHEMA_VERSION
        or definition.get("product_id") != PRODUCT_ID
        or definition.get("journey_id") != JOURNEY_ID
        or definition.get("journey_version") != JOURNEY_VERSION
        or definition.get("first_success_fact") != FIRST_SUCCESS_FACT
    ):
        raise ValueError("journey identity is invalid")
    screens = definition.get("screens")
    if not isinstance(screens, list) or not 1 <= len(screens) <= 128:
        raise ValueError("journey screen graph is invalid")
    screen_ids = set()
    success_terminals = set()
    for screen in screens:
        if not isinstance(screen, dict):
            raise ValueError("journey screen is invalid")
        screen_id = screen.get("screen_id")
        if not isinstance(screen_id, str) or not screen_id or screen_id in screen_ids:
            raise ValueError("journey screen id is invalid")
        screen_ids.add(screen_id)
        if screen.get("title_key") not in _CONTENT or screen.get("body_key") not in _CONTENT:
            raise ValueError("journey content key is not owned by this product")
        actions = screen.get("actions")
        transitions = screen.get("transitions")
        if not isinstance(actions, list) or any(action not in _ALLOWED_ACTIONS for action in actions):
            raise ValueError("journey action is invalid")
        if not isinstance(transitions, list):
            raise ValueError("journey transitions are invalid")
        for key in ("entry_conditions", "completion_evidence"):
            if key in screen:
                _validate_condition(screen[key])
        completion_evidence = screen.get("completion_evidence")
        if (
            not transitions
            and _condition_mentions_success(completion_evidence)
            and _evaluate(completion_evidence, {FIRST_SUCCESS_FACT: True})
        ):
            success_terminals.add(screen_id)
    entry_screen_id = definition.get("entry_screen_id")
    if entry_screen_id not in screen_ids or not success_terminals:
        raise ValueError("journey entry or first-success terminal is missing")
    for screen in screens:
        fallback = screen.get("fallback_screen_id")
        if fallback is not None and fallback not in screen_ids:
            raise ValueError("journey fallback target is missing")
        for transition in screen["transitions"]:
            if (
                not isinstance(transition, dict)
                or transition.get("next_screen_id") not in screen_ids
                or not isinstance(transition.get("reason_code"), str)
                or not isinstance(transition.get("priority"), int)
            ):
                raise ValueError("journey transition is invalid")
            if "condition" in transition:
                _validate_condition(transition["condition"])
    graph = {
        screen["screen_id"]: [transition["next_screen_id"] for transition in screen["transitions"]]
        + ([screen["fallback_screen_id"]] if screen.get("fallback_screen_id") else [])
        for screen in screens
    }
    reachable = {entry_screen_id}
    frontier = [entry_screen_id]
    while frontier:
        for target in graph[frontier.pop()]:
            if target not in reachable:
                reachable.add(target)
                frontier.append(target)
    if not reachable.intersection(success_terminals):
        raise ValueError("journey first-success terminal is unreachable")
    return bundle


def _condition_mentions_success(condition: Any) -> bool:
    if not isinstance(condition, dict):
        return False
    if condition.get("kind") == "fact":
        return condition.get("fact") == FIRST_SUCCESS_FACT
    if condition.get("kind") in {"all", "any"}:
        return any(_condition_mentions_success(child) for child in condition.get("conditions", []))
    if condition.get("kind") == "not":
        return _condition_mentions_success(condition.get("condition"))
    return False


class _Storage:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or Path.home() / ".wisent" / "onboarding" / "first-use.json"

    def load(self) -> Dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("schema_version") == SCHEMA_VERSION:
                return value
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        return {"schema_version": SCHEMA_VERSION, "pending_events": []}

    def save(self, value: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(prefix=".first-use-", suffix=".json", dir=str(self.path.parent))
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(str(temporary), str(self.path))
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def identity(self) -> Tuple[Dict[str, Any], str]:
        state = self.load()
        installation_id = state.get("installation_id")
        try:
            installation_id = str(uuid.UUID(str(installation_id)))
        except (TypeError, ValueError, AttributeError):
            installation_id = str(uuid.uuid4())
            state["installation_id"] = installation_id
            self.save(state)
        subject_hash = hashlib.sha256(("wisent-device:" + installation_id).encode("utf-8")).hexdigest()
        return state, subject_hash


class _StadoTransport:
    def __init__(self, base_url: str, token: str) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("STADO_INTEGRATION_API_URL must be an HTTPS origin")
        if not token.strip():
            raise ValueError(f"{TOKEN_ENV} must not be empty")
        self.base_url = base_url.rstrip("/")
        self.token = token.strip()

    @classmethod
    def from_environment(cls) -> Optional["_StadoTransport"]:
        base_url = os.environ.get(BASE_URL_ENV, "").strip()
        token = os.environ.get(TOKEN_ENV, "").strip()
        if not base_url or not token:
            return None
        try:
            return cls(base_url, token)
        except ValueError:
            return None

    def post(self, operation: str, body: Mapping[str, Any]) -> Any:
        endpoint = (
            f"{self.base_url}/integration/{CLIENT_ID}/onboarding/"
            f"{urllib.parse.quote(PRODUCT_ID, safe='')}/{urllib.parse.quote(operation, safe='')}"
        )
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, urllib.error.URLError) as exc:
            raise RuntimeError("Stado onboarding transport failed") from exc
        if not isinstance(envelope, dict) or envelope.get("ok") is not True or "result" not in envelope:
            raise RuntimeError("Stado onboarding transport returned an invalid envelope")
        return envelope["result"]


class _Journey:
    def __init__(self) -> None:
        self.storage = _Storage()
        self.transport = _StadoTransport.from_environment()
        self.state: Dict[str, Any] = {}
        self.subject_hash = ""
        self.bundle: Dict[str, Any] = {}
        self.progress: Dict[str, Any] = {}

    def start(self) -> None:
        self.state, self.subject_hash = self.storage.identity()
        self._flush_events()
        self.bundle = self._load_bundle()
        stored = self._load_progress()
        is_new = stored is None
        if stored is None:
            stored = {
                "attempt_id": str(uuid.uuid4()),
                "product_id": PRODUCT_ID,
                "journey_version_id": self.bundle["journey_version_id"],
                "subject_hash": self.subject_hash,
                "scope_kind": "device",
                "current_screen_id": self.bundle["definition"]["entry_screen_id"],
                "completed_screen_ids": [],
                "status": "in_progress",
                "evidence_revision": "journey-started",
                "answers": [],
            }
            assignment = self._assign_experiment()
            if assignment is not None:
                stored["experiment_id"], stored["variant_id"] = assignment
        self.progress = stored
        self._save_progress()
        if is_new:
            self.emit("onboarding_started", {})
        elif self.progress["status"] == "in_progress":
            self.emit("onboarding_resumed", {})

    def _load_bundle(self) -> Dict[str, Any]:
        if self.transport is not None:
            try:
                remote = self.transport.post(
                    "bundle.read",
                    {
                        "product_id": PRODUCT_ID,
                        "journey_id": JOURNEY_ID,
                        "journey_version": JOURNEY_VERSION,
                        "if_none_match": None,
                    },
                )
                validated = _validate_bundle(remote)
                self.state["bundle"] = validated
                self.storage.save(self.state)
                return validated
            except (RuntimeError, ValueError, TypeError):
                pass
        cached = self.state.get("bundle")
        try:
            return _validate_bundle(cached)
        except (ValueError, TypeError):
            return _validate_bundle(_fallback_bundle())

    def _load_progress(self) -> Optional[Dict[str, Any]]:
        local = self.state.get("progress")
        valid_local = self._validated_progress(local)
        if self.transport is not None and valid_local is not None:
            try:
                remote = self.transport.post(
                    "state.read",
                    {
                        "product_id": PRODUCT_ID,
                        "attempt_id": valid_local["attempt_id"],
                        "subject_hash": self.subject_hash,
                    },
                )
                candidate = remote.get("progress", remote.get("attempt", remote)) if isinstance(remote, dict) else None
                valid_remote = self._validated_progress(candidate)
                if valid_remote is not None:
                    return valid_remote
            except (RuntimeError, TypeError, ValueError):
                pass
        return valid_local

    def _validated_progress(self, progress: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(progress, dict):
            return None
        try:
            uuid.UUID(str(progress["attempt_id"]))
        except (KeyError, TypeError, ValueError):
            return None
        screen_ids = {screen["screen_id"] for screen in self.bundle["definition"]["screens"]}
        if (
            progress.get("product_id") != PRODUCT_ID
            or progress.get("journey_version_id") != self.bundle["journey_version_id"]
            or progress.get("subject_hash") != self.subject_hash
            or progress.get("scope_kind") != "device"
            or progress.get("current_screen_id") not in screen_ids
            or progress.get("status") not in _STATUSES
            or not isinstance(progress.get("completed_screen_ids"), list)
            or not isinstance(progress.get("answers", []), list)
        ):
            return None
        return dict(progress)

    def _assign_experiment(self) -> Optional[Tuple[str, str]]:
        contract = self.bundle["definition"].get("experiment_contract")
        if self.transport is None or not isinstance(contract, dict) or contract.get("kill_switch") is True:
            return None
        try:
            result = self.transport.post(
                "experiments.assign",
                {
                    "product_id": PRODUCT_ID,
                    "app_id": "wisent",
                    "platform": "cli",
                    "surface": "command_line",
                    "subject": self.subject_hash,
                    "experiment_id": contract.get("experiment_id"),
                },
            )
            if not isinstance(result, dict):
                return None
            experiment_id = result.get("experimentId", result.get("experiment_id"))
            variant_id = result.get("variant", result.get("variant_id"))
            if isinstance(experiment_id, str) and isinstance(variant_id, str):
                return experiment_id, variant_id
        except RuntimeError:
            pass
        return None

    def _save_progress(self) -> None:
        self.state["progress"] = self.progress
        self.storage.save(self.state)

    @property
    def screen(self) -> Dict[str, Any]:
        screen_id = self.progress["current_screen_id"]
        return next(screen for screen in self.bundle["definition"]["screens"] if screen["screen_id"] == screen_id)

    def expose(self) -> None:
        self.emit("onboarding_step_viewed", {})

    def advance(self, evidence: Mapping[str, Any]) -> bool:
        decision = _select_next(self.bundle["definition"], self.screen, evidence)
        if decision is None:
            return False
        completed_screen = self.progress["current_screen_id"]
        completed = list(dict.fromkeys(self.progress["completed_screen_ids"] + [completed_screen]))
        self.progress.update(
            {
                "current_screen_id": decision[0],
                "completed_screen_ids": completed,
                "status": "in_progress",
                "evidence_revision": _evidence_revision(evidence),
            }
        )
        self._save_progress()
        self.emit(
            "onboarding_step_completed",
            {},
            screen_id=completed_screen,
            decision=decision,
        )
        return True

    def skip(self) -> None:
        self.progress["status"] = "skipped"
        self.progress["evidence_revision"] = "journey-skipped"
        self._save_progress()
        self.emit("onboarding_step_skipped", {})

    def resume(self) -> None:
        self.progress["status"] = "in_progress"
        self.progress["evidence_revision"] = "journey-resumed"
        self._save_progress()
        self.emit("onboarding_resumed", {})

    def reset(self) -> None:
        self.progress.update(
            {
                "attempt_id": str(uuid.uuid4()),
                "current_screen_id": self.bundle["definition"]["entry_screen_id"],
                "completed_screen_ids": [],
                "status": "in_progress",
                "evidence_revision": "journey-reset",
                "answers": [],
            }
        )
        self.progress.pop("experiment_id", None)
        self.progress.pop("variant_id", None)
        assignment = self._assign_experiment()
        if assignment is not None:
            self.progress["experiment_id"], self.progress["variant_id"] = assignment
        self._save_progress()
        self.emit("onboarding_reset", {})
        self.emit("onboarding_started", {})

    def observe_success(self, command: str) -> None:
        if self.progress.get("status") == "completed":
            return
        if self.progress.get("status") in {"skipped", "abandoned", "reset"}:
            self.resume()
        evidence = {FIRST_SUCCESS_FACT: True}
        revision = _evidence_revision(evidence, command)
        self.progress["evidence_revision"] = revision
        self._save_progress()
        self.emit("onboarding_first_action_completed", {"command": command})
        remaining = len(self.bundle["definition"]["screens"])
        while self.screen["transitions"] and remaining > 0:
            if not self.advance(evidence):
                return
            remaining -= 1
        screen = self.screen
        if screen["transitions"] or not _evaluate(screen.get("completion_evidence"), evidence):
            return
        completed_screen = self.progress["current_screen_id"]
        self.progress["completed_screen_ids"] = list(
            dict.fromkeys(self.progress["completed_screen_ids"] + [completed_screen])
        )
        self.progress["status"] = "completed"
        self.progress["evidence_revision"] = revision
        self._save_progress()
        properties = {"command": command, "first_success_fact": FIRST_SUCCESS_FACT}
        self.emit("onboarding_step_completed", properties, screen_id=completed_screen)
        self.emit("onboarding_first_success_observed", properties, screen_id=completed_screen)
        self.emit("onboarding_completed", properties, screen_id=completed_screen)

    def emit(
        self,
        event_name: str,
        properties: Mapping[str, Any],
        screen_id: Optional[str] = None,
        decision: Optional[Tuple[str, str]] = None,
    ) -> None:
        if event_name not in _EVENT_NAMES:
            raise ValueError("unknown onboarding event")
        event = {
            "event_id": str(uuid.uuid4()),
            "event_name": event_name,
            "attempt_id": self.progress["attempt_id"],
            "product_id": PRODUCT_ID,
            "journey_version_id": self.progress["journey_version_id"],
            "subject_hash": self.subject_hash,
            "scope_kind": "device",
            "screen_id": screen_id or self.progress["current_screen_id"],
            "occurred_at": _utc_now(),
            "evidence_revision": self.progress.get("evidence_revision", "unknown"),
            "properties": dict(properties),
            "answers": self.progress.get("answers", []),
        }
        if self.progress.get("experiment_id"):
            event["experiment_id"] = self.progress["experiment_id"]
        if self.progress.get("variant_id"):
            event["variant_id"] = self.progress["variant_id"]
        if decision is not None:
            event["selected_next_screen_id"], event["reason_code"] = decision
        pending = self.state.setdefault("pending_events", [])
        pending.append(event)
        self.storage.save(self.state)
        if self.transport is not None:
            try:
                self.transport.post("events.collect", event)
            except RuntimeError:
                return
            self.state["pending_events"] = [item for item in pending if item.get("event_id") != event["event_id"]]
            self.storage.save(self.state)

    def _flush_events(self) -> None:
        if self.transport is None:
            return
        pending = self.state.get("pending_events", [])
        if not isinstance(pending, list):
            self.state["pending_events"] = []
            self.storage.save(self.state)
            return
        remaining = list(pending)
        for event in pending:
            if not isinstance(event, dict):
                remaining.remove(event)
                continue
            try:
                self.transport.post("events.collect", event)
            except RuntimeError:
                break
            remaining = [item for item in remaining if item.get("event_id") != event.get("event_id")]
            self.state["pending_events"] = remaining
            self.storage.save(self.state)


def _evidence_revision(evidence: Mapping[str, Any], command: Optional[str] = None) -> str:
    payload = {"evidence": dict(evidence), "command": command, "observed_at": _utc_now()}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _evaluate(condition: Any, evidence: Mapping[str, Any]) -> bool:
    if condition is None:
        return True
    kind = condition.get("kind")
    if kind == "all":
        return all(_evaluate(child, evidence) for child in condition["conditions"])
    if kind == "any":
        return any(_evaluate(child, evidence) for child in condition["conditions"])
    if kind == "not":
        return not _evaluate(condition["condition"], evidence)
    actual = evidence.get(condition["fact"])
    present = condition["fact"] in evidence and actual is not None
    operator = condition["operator"]
    expected = condition.get("value")
    if operator == "present":
        return present
    if operator == "absent":
        return not present
    if operator == "eq":
        return actual == expected
    if operator == "not_eq":
        return actual != expected
    if operator == "contains":
        return isinstance(actual, (list, tuple)) and expected in actual
    if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)):
        if operator == "gt":
            return actual > expected
        if operator == "gte":
            return actual >= expected
        if operator == "lt":
            return actual < expected
        if operator == "lte":
            return actual <= expected
    return False


def _select_next(
    definition: Mapping[str, Any], screen: Mapping[str, Any], evidence: Mapping[str, Any]
) -> Optional[Tuple[str, str]]:
    if "completion_evidence" in screen and not _evaluate(screen["completion_evidence"], evidence):
        return None
    screens = {item["screen_id"]: item for item in definition["screens"]}
    transitions = sorted(screen["transitions"], key=lambda item: item.get("priority", 0))
    for transition in transitions:
        target = screens[transition["next_screen_id"]]
        if "condition" in transition and not _evaluate(transition["condition"], evidence):
            continue
        if "entry_conditions" in target and not _evaluate(target["entry_conditions"], evidence):
            continue
        return transition["next_screen_id"], transition["reason_code"]
    fallback_id = screen.get("fallback_screen_id")
    if fallback_id:
        fallback = screens[fallback_id]
        if "entry_conditions" not in fallback or _evaluate(fallback["entry_conditions"], evidence):
            return fallback_id, "fallback_evidence_unavailable"
    return None


def _render(journey: _Journey) -> None:
    screen = journey.screen
    title = _CONTENT[screen["title_key"]]
    body = _CONTENT[screen["body_key"]]
    print(f"\n{title}\n{'-' * len(title)}\n{body}")
    completed = len(journey.progress["completed_screen_ids"])
    total = len(journey.bundle["definition"]["screens"])
    print(f"\nJourney: {completed}/{total} steps complete ({journey.progress['status']}).")
    if journey.progress["status"] == "completed":
        print("First success observed: a representation operation completed successfully.")
    elif journey.progress["status"] == "skipped":
        print("Resume with `wisent onboarding continue`, or restart with `wisent onboarding reset`.")
    elif screen["transitions"]:
        print("Continue with `wisent onboarding continue`; skip with `wisent onboarding skip`.")
    else:
        print("Return here at any time with `wisent onboarding`; progress resumes across launches.")


def execute_onboarding(args: Any) -> None:
    """Execute the discoverable CLI journey command."""
    journey = _Journey()
    journey.start()
    action = getattr(args, "onboarding_action", "show")
    if action == "reset":
        journey.reset()
    elif action == "skip":
        journey.skip()
        _render(journey)
        return
    elif action == "continue":
        if journey.progress["status"] == "completed":
            _render(journey)
            return
        if journey.progress["status"] in {"skipped", "abandoned", "reset"}:
            journey.resume()
        journey.advance({})
    journey.expose()
    _render(journey)


def record_representation_operation(command: str) -> None:
    """Record first success only after a real representation handler returns successfully."""
    if command not in _REPRESENTATION_COMMANDS:
        return
    try:
        journey = _Journey()
        journey.start()
        journey.observe_success(command)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        # Onboarding persistence and its control plane must never turn a successful model operation into a failure.
        return
