# Serve API reference

`ster serve` is the loopback HTTP/JSON backend a desktop app spawns instead of
building argv for the CLI (`src/serve.rs`). It is deliberately minimal: a
`std::net` server with no HTTP dependency, one request per connection,
`connection: close` on every response. All examples below were executed
against a running `ster serve` on this machine.

## Lifecycle

```console
$ ster serve --port 0
{"port":61058,"ready":true}
```

- Binds **127.0.0.1 only**; `--port 0` (the default) selects an ephemeral
  port. There is no authentication — the boundary is the loopback interface
  and the OS user ([architecture](architecture.md)).
- Exactly one ready line lands on stdout; after that, stdout carries no
  protocol traffic. Failure to bind: `failed to bind the serve port`.
- The server runs until killed. Accept errors are logged to stderr
  (`serve accept failed: <error>`); a failed connection never stops the
  listener.
- Request bodies above 1 MiB are refused (`request body too large`).

## Endpoints

All under `/v1`. Anything else: `404` with
`{"error": "unknown endpoint: GET /v1/nope"}`.

| Method, path | Body | Mirrors |
|---|---|---|
| `GET /v1/health` | — | liveness: `200` `{"status": "ok"}` |
| `POST /v1/train` | TrainRequest | `ster train` |
| `POST /v1/optimize` | OptimizeRequest | `ster optimize` |
| `POST /v1/evaluate` | EvaluateRequest | `ster evaluate` |
| `POST /v1/generate` | GenerateRequest | `ster generate` |
| `POST /v1/extract` | ExtractRequest | `ster extract` |
| `POST /v1/inspect` | InspectRequest | `ster inspect` |

Every job handler reuses the exact functions the CLI commands use
(`workflow.rs`, `runtime.rs`, `artifact.rs`) and returns the same document
the CLI prints.

## Request shapes

Fields are **camelCase**. An empty or whitespace-only body is treated as
`{}`, and every field is defaulted, so validation — not deserialization —
reports what is missing. File-path fields (`pairs`, `output`, `vector`,
`input`, `artifact`) are paths on the server's filesystem, resolved relative
to the serve process's working directory.

Shared model fields (all jobs except `inspect`): `model` (required),
`revision` (optional), `device` (default `"cpu"`).

| Request | Fields beyond the model block | Defaults |
|---|---|---|
| Train | `pairs`*, `output`*, `layers`, `method` | `layers: "all"`, `method: "caa"` |
| Optimize | `pairs`*, `output`*, `layers` | `layers: "all"` |
| Evaluate | `pairs`*, `vector`* | — |
| Generate | `prompt`*, `vector`, `strength`, `maxNewTokens`, `temperature`, `topP`, `seed` | `strength: 1.0`, `maxNewTokens: 128`, `temperature: 0.0`, `seed: 42`; empty `vector` means unsteered |
| Extract | `input`*, `output`*, `layers` | `layers: "all"` |
| Inspect | `artifact`* | — |

\* validated as required. Each missing required field has one fixed refusal
sentence, e.g. `train requires a pairs file`, `generate requires a prompt`,
`inspect requires a steering artifact` — the full list is in the
[runbook](runbook.md).

## Failure envelope vs job stream

Failures **before** a job starts are plain non-2xx JSON envelopes:

```console
$ curl -si -X POST http://127.0.0.1:61058/v1/train -d '{"model":"toy-llama"}'
HTTP/1.1 400 Bad Request
...
{
  "error": "train requires a pairs file"
}
```

`400` covers a malformed request line, an oversized body, a body that is not
valid JSON (`request body is not valid JSON`), and failed field validation;
`404` covers unknown endpoints. The error string is the product's own refusal
sentence, verbatim.

## Job streaming (NDJSON)

Once validation passes, the response is `200` with
`content-type: application/x-ndjson` and the job streams:

```console
$ curl -s -X POST http://127.0.0.1:61058/v1/train \
    -d '{"model":"toy-llama","pairs":"pairs.json","output":"served.ster.json","layers":"2","method":"caa"}'
{"chunk":"reading pair 1/8\n","stream":"stderr","type":"log"}
{"chunk":"reading pair 2/8\n","stream":"stderr","type":"log"}
...
{"json":{"artifact":{"hidden_size":64,"layers":[{"layer":2,"train_accuracy":0.875,"train_margin":0.7613550424575806}],"method":"caa","model":"toy-llama","model_revision":null,"product":"ster","schema_version":1,"trait_name":"calm"}},"status":0,"type":"result"}
```

- Zero or more `{"type":"log","stream":"stderr","chunk":"..."}` events —
  the same progress lines the CLI writes to stderr, delivered through the
  workflow progress sink.
- Exactly one final `{"type":"result","status":<int>,"json":{...}}` event,
  where `json` is the document the CLI prints and `status` mirrors the CLI
  exit code.

A failure **mid-job** mirrors the CLI instead of switching to an HTTP error:
the refusal arrives as a stderr log event plus a status-1 result:

```console
$ curl -s -X POST http://127.0.0.1:61058/v1/inspect -d '{"artifact":"tampered.json"}'
{"chunk":"error: failed to inspect tampered.json: artifact belongs to product \"other\", not Ster\n","stream":"stderr","type":"log"}
{"json":{"error":"failed to inspect tampered.json: artifact belongs to product \"other\", not Ster"},"status":1,"type":"result"}
```

## Concurrency

Streamed jobs share one global progress sink, so **jobs run one at a time**:
each connection takes a job lock after validation and holds it until its
result event is written. Concurrent requests are accepted (each connection
gets a thread) but queue on the lock; each job's log events stay on its own
response. `GET /v1/health` does not take the lock and stays responsive while
a job runs.

## Generate over HTTP, executed

```console
$ curl -s -X POST http://127.0.0.1:61058/v1/generate -H 'content-type: application/json' \
    -d '{"model":"toy-llama","prompt":"describe the evening lake .","vector":"calm.ster.json","strength":1.0,"maxNewTokens":12}'
{"json":{"text":"white white white howls howls howls drifts , white white white white"},"status":0,"type":"result"}
```

Byte-for-byte the same text the CLI's `ster generate` printed for the same
arguments in [walkthrough-steering](walkthrough-steering.md). The full
session transcript is [walkthrough-serve](walkthrough-serve.md).
