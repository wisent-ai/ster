# Architecture

Ster is one native Rust process with a thin CLI or loopback HTTP front end over
the same model, representation, and workflow code. It has no daemon database,
remote control plane, Python layer, or second inference implementation.

```text
CLI (main.rs) ─────────────┐
                           ├─ artifact.rs ─ pair/artifact JSON + validation
serve.rs (127.0.0.1 HTTP) ─┤
                           ├─ workflow.rs ─ capture/train/select/evaluate/export
                           ├─ representation.rs ─ CAA/PCA/logistic + measures
                           └─ runtime.rs ─ resolution/tokenization/generation
                                      │
                                      └─ model.rs ─ native Candle Llama loop
```

## Boundaries by module

### Front ends

`main.rs` parses seven subcommands. It loads inputs, calls library workflows,
and prints results; progress stays on stderr. `serve.rs` turns six workflows
into NDJSON jobs and exposes health. Field validation occurs before streaming;
a single job lock protects the process-global progress sink. The server does
not shell out to the CLI: each handler calls the same Rust functions as the
corresponding command.

### Data and representation

`artifact.rs` owns the pair-set and steering-artifact schemas and their
fail-closed validation. `representation.rs` is model-independent numerical
code: it accepts F32 activation rows, trains one normalized direction, and
scores pair ordering. It never sees tokens, model files, or HTTP.

### Workflow orchestration

`workflow.rs` captures both sides of each pair through `Runtime`, groups rows
by layer, trains and scores vectors, performs the 80/20 optimization split, and
exports activation JSON. It owns progress messages and summary/report wire
shapes. It does not own model kernels.

### Runtime and model

`runtime.rs` resolves local or Hub files, selects a device, parses the
tokenizer/config, maps Safetensors, gates artifacts against the loaded model,
and runs tokenization, sampling, EOS/context stopping, and KV-cache lifecycle.

`model.rs` owns the Llama decoder loop: embeddings; RMS-normalized attention;
rotary position embedding; grouped-query KV repetition; causal masking; gated
MLP; residual additions; final norm; and LM head. Two product-specific hooks
sit after each decoder block: copy the final-token residual state, then
optionally add the layer's steering vector. Owning this loop is what makes the
read and intervention points exact.

## Data flow

### Train or optimize

```text
pairs.json → validate → tokenize each side → model forward/capture
           → rows by layer → train normalized direction → evaluate
           → validate artifact → write *.ster.json + stdout summary
```

`optimize` captures once, evaluates every method/layer candidate on its ordered
holdout, retrains the winner on all rows, and writes one vector.

### Generate

```text
prompt → tokenize → optional artifact load/validate/model gate
       → steering plan on device → prompt forward + cached token forwards
       → sample/argmax until limit/EOS/context → decode generated suffix
```

No checkpoint weight is changed. The steering plan and KV cache die with the
call.

### Serve

```text
TCP connection → minimal HTTP parse → JSON/defaults/required fields
               → 200 NDJSON head → acquire job lock → shared workflow
               → progress log events → one status/result event → close
```

Health bypasses the job lock. Connections get threads; model jobs are still
serialized.

## Persistent and external state

Ster writes only paths explicitly requested for steering artifacts or
activation exports. `inspect`, `evaluate`, and `generate` do not mutate them.
For a Hub id, hf-hub maintains its normal disk cache and optional token file;
for a local directory, model resolution is offline. Ster has no application
configuration file, telemetry, usage ledger, or automatic upload.

## Trust boundaries

- **Model files and JSON inputs are untrusted.** Parse, architecture, width,
  finiteness, model identity, and layer gates stop before intervention.
- **Safetensors are memory-mapped with Candle's unsafe loader.** Only load
  checkpoints from a source you trust and keep file permissions local.
- **`serve` is unauthenticated but binds only `127.0.0.1`.** Any process under
  the same host/user boundary can reach it, supply server-local paths, and
  trigger expensive model work. Do not proxy or expose the port beyond
  loopback.
- **Hub access may send the model id/revision and token to Hugging Face.** Use a
  local directory for a no-network boundary.
- **Generated text is data, not an authorization decision.** Ster changes
  representations; it does not enforce policy or certify semantic outcomes.

## Product ownership

Ster owns local open-weight representation capture, direction learning,
evaluation, and residual steering. Brama owns hosted model routing. Stado owns
fleet placement and release delivery; Skarbiec owns credentials. Ster neither
calls those products nor substitutes for them.

## Concurrency and resource model

One `Runtime` owns one loaded model. Capturing one pair performs two full prompt
forwards; capturing more layers increases copied activation memory but not the
number of forwards. Training allocations scale with pair count × hidden width
per selected layer. Generation uses a KV cache up to the context limit.
Artifacts store `selected_layer_count × hidden_size` F32 numbers as JSON.

The HTTP front end accepts connections concurrently but serializes all jobs to
protect one global progress sink. Library callers using `set_progress_sink`
must provide the same serialization themselves.
