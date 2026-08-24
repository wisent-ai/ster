# Walkthrough: one loopback serve session

This recorded session used the same offline toy checkpoint, pair file, and CPU
binary as [the steering walkthrough](walkthrough-steering.md). It exercised
health, validation, streamed training, generation, an artifact refusal, and an
unknown endpoint. The server performed no network model download.

## Start on an ephemeral port

```console
$ ster serve --port 0
{"port":61058,"ready":true}
```

The exact port varies. The one ready line is stdout; the process then listens
on `127.0.0.1:61058` until stopped.

```bash
P=61058
```

## Check health

```console
$ curl -s http://127.0.0.1:$P/v1/health
{
  "status": "ok"
}
```

Health does not load a model or take the serialized job lock.

## Observe pre-job validation

A train request with only a model is structurally valid JSON but lacks required
server-local paths:

```console
$ curl -s -X POST http://127.0.0.1:$P/v1/train -d '{"model":"toy-llama"}'
{
  "error": "train requires a pairs file"
}
```

The HTTP status was `400 Bad Request`, content type `application/json`. No
NDJSON response starts and no model loads because required-field validation
precedes the job.

Malformed JSON similarly returned
`{"error":"request body is not valid JSON"}` with 400. A blank body is treated
as `{}` and therefore gets the first action-specific missing-field sentence.

## Stream a training job

Paths are resolved in the serve process's working directory:

```console
$ curl -s -X POST http://127.0.0.1:$P/v1/train \
    -H 'content-type: application/json' \
    -d '{"model":"toy-llama","pairs":"pairs.json","output":"served.ster.json","layers":"2","method":"caa"}'
{"chunk":"reading pair 1/8\n","stream":"stderr","type":"log"}
{"chunk":"reading pair 2/8\n","stream":"stderr","type":"log"}
{"chunk":"reading pair 3/8\n","stream":"stderr","type":"log"}
{"chunk":"reading pair 4/8\n","stream":"stderr","type":"log"}
{"chunk":"reading pair 5/8\n","stream":"stderr","type":"log"}
{"chunk":"reading pair 6/8\n","stream":"stderr","type":"log"}
{"chunk":"reading pair 7/8\n","stream":"stderr","type":"log"}
{"chunk":"reading pair 8/8\n","stream":"stderr","type":"log"}
{"json":{"artifact":{"hidden_size":64,"layers":[{"layer":2,"train_accuracy":0.875,"train_margin":0.7613550424575806}],"method":"caa","model":"toy-llama","model_revision":null,"product":"ster","schema_version":1,"trait_name":"calm"}},"status":0,"type":"result"}
```

This was a `200 OK` `application/x-ndjson` response. The final event is always
a result, so clients read lines until it rather than treating connection close
as the success signal. `served.ster.json` contained the same layer-2 vector as
the CLI-trained artifact.

## Generate with the served artifact

```console
$ curl -s -X POST http://127.0.0.1:$P/v1/generate \
    -H 'content-type: application/json' \
    -d '{"model":"toy-llama","prompt":"describe the evening lake .","vector":"served.ster.json","strength":1.0,"maxNewTokens":12}'
{"json":{"text":"white white white howls howls howls drifts , white white white white"},"status":0,"type":"result"}
```

The text was byte-for-byte the deterministic CLI result for the same model,
prompt, vector, strength, and token limit.

## Observe a mid-job refusal

The valid request below points at a JSON artifact whose product was changed to
`other`:

```console
$ curl -s -X POST http://127.0.0.1:$P/v1/inspect -d '{"artifact":"tampered.json"}'
{"chunk":"error: failed to inspect tampered.json: artifact belongs to product \"other\", not Ster\n","stream":"stderr","type":"log"}
{"json":{"error":"failed to inspect tampered.json: artifact belongs to product \"other\", not Ster"},"status":1,"type":"result"}
```

The HTTP response remains 200 because streaming had already begun. Clients
must inspect the final event's `status`; HTTP status alone does not decide job
success.

## Unknown endpoint and shutdown

```console
$ curl -s http://127.0.0.1:$P/v1/nope
{
  "error": "unknown endpoint: GET /v1/nope"
}
```

That response was 404 JSON. The protocol has no shutdown endpoint; the parent
desktop process or operator terminates the child process. Complete field and
concurrency details: [serve API](serve-api.md).
