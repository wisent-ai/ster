# Rust library API reference

The `ster` package publishes no crates.io release (`publish = false`), but the
repository builds a library crate alongside the CLI. Depend on a pinned Git
revision or path. All fallible calls return `anyhow::Result`.

```toml
[dependencies]
ster = { git = "https://github.com/wisent-ai/ster", rev = "<commit>" }
```

The crate root re-exports `ContrastivePair`, `PairSet`, `SteeringArtifact`,
`TrainingMethod`, `DeviceChoice`, `GenerationOptions`, and `Runtime`. Its six
modules are also public. This page lists every public item in Ster 0.12.

## `artifact`

- `ARTIFACT_SCHEMA_VERSION: u32` — currently `1`.
- `ContrastivePair { positive: String, negative: String }` — serde JSON pair.
- `PairSet { trait_name: String, pairs: Vec<ContrastivePair> }` — serde JSON
  set; missing `trait_name` defaults to empty.
  - `PairSet::load(&Path)` reads JSON and checks a non-empty set and nonblank
    sides.
- `LayerVector { layer: usize, values: Vec<f32>, train_margin: f32,
  train_accuracy: f32 }` — one scored direction.
- `SteeringArtifact { schema_version, product, model, model_revision,
  trait_name, method, hidden_size, vectors, metadata }` — all fields public;
  `metadata` is `BTreeMap<String, String>` and serde-defaulted.
  - `new(model, model_revision, trait_name, method, hidden_size, vectors)`
    stamps the current schema/product and empty metadata.
  - `load(&Path)` reads and validates.
  - `save(&Path)` validates, creates parent directories, and writes compact
    JSON.
  - `validate()` checks schema, product, non-empty width/vectors, vector widths,
    and finite values.

Wire details: [steering artifact](concepts/steering-artifact.md).

## `representation`

- `TrainingMethod::{Caa, Pca, Logistic}` — `Debug + Clone + Copy + PartialEq +
  Eq`.
  - `parse(&str)` accepts `caa`/`mean-difference`, `pca`, and
    `logistic`/`probe`.
  - `name()` returns canonical `caa`, `pca`, or `logistic`.
- `train_direction(positive, negative, method) -> Result<Vec<f32>>` — validates
  matched rows, runs the selected algorithm, returns an L2-normalized vector.
- `evaluate_direction(positive, negative, direction) -> Result<(f32, f32)>` —
  returns `(accuracy, mean_margin)`.
- `cosine_similarity(left, right) -> Result<f32>` — equal non-empty non-zero
  vectors only.

Algorithm constants and measures: [training method](concepts/training-method.md)
and [evaluation](concepts/evaluation.md).

## `runtime`

- `DeviceChoice::{Cpu, Metal, Cuda}` — `Debug + Clone + Copy`.
  - `parse(&str)` accepts exactly lowercase `cpu`, `metal`, or `cuda`.
- `GenerationOptions { strength: f64, max_new_tokens: usize, temperature: f64,
  top_p: Option<f64>, seed: u64 }` — no `Default` implementation; callers set
  every field.
- `Runtime` — owns tokenizer, model, device, dtype, and EOS set. Public fields:
  `model_id: String`, `revision: Option<String>`.
  - `load(model, revision, device)` resolves and loads one F32 model.
  - `hidden_size()` and `layer_count()` expose config dimensions.
  - `activations(prompt, layers)` returns sorted `(layer, Vec<f32>)` final-token
    states.
  - `generate(prompt, artifact, options)` returns generated text; `None`
    artifact is baseline generation.

A `Runtime` is reusable across calls but methods take `&self`; each activation
or generation call allocates a fresh attention cache. It does not cache
captures or downloads beyond hf-hub's disk cache.

## `workflow`

- `train(runtime, pairs, layers, method) -> SteeringArtifact`
- `optimize(runtime, pairs, layers) -> SteeringArtifact`
- `evaluate(runtime, pairs, artifact) -> EvaluationReport`
- `extract(runtime, input, output, layers) -> ()`
- `parse_layers(value, count) -> Vec<usize>`
- `artifact_summary(artifact) -> serde_json::Value`
- `EvaluationReport { model, trait_name, method, pair_count, layers }` — serde
  serializable.
- `LayerEvaluation { layer, accuracy, margin }` — serde serializable.
- `set_progress_sink(Option<Box<dyn Fn(&str) + Send>>)` — process-global
  progress replacement used by `serve`; setting it affects all workflows and
  is not safe for concurrent independent jobs. Normal library clients should
  leave it unset so progress goes to stderr.

`extract`'s prompt and activation structs are intentionally private; its public
contract is file-to-file JSON described under [activation](concepts/activation.md).

## `model`

These low-level Candle-facing types are public because the module is public,
but most clients should use `Runtime`.

- `SteeringPlan::new(vectors, strength, hidden_size, device, dtype)` copies
  vectors to tensors and requires at least one correctly sized vector.
- `ForwardOutput { logits: Tensor, activations: BTreeMap<usize, Vec<f32>> }`.
- `Cache::new(use_kv_cache, dtype, &Config, &Device)` builds rotary tables and
  empty per-layer KV slots.
- `SteeringLlama::load(VarBuilder, Config)`, `config()`, and
  `forward(tokens, index_pos, cache, steering, capture_layers)` expose the
  native decoder. `capture_layers` must be sorted because lookup uses binary
  search.

## `serve`

- `serve::run(port: u16) -> Result<()>` binds loopback, prints the ready JSON,
  and blocks in the accept loop until the process is stopped. Protocol:
  [serve API](serve-api.md).

## Minimal library workflow

```rust
use std::path::Path;
use ster::{DeviceChoice, PairSet, Runtime, SteeringArtifact, TrainingMethod};
use ster::workflow;

let runtime = Runtime::load("./toy-llama", None, DeviceChoice::Cpu)?;
let pairs = PairSet::load(Path::new("pairs.json"))?;
let layers = workflow::parse_layers("1..3", runtime.layer_count())?;
let artifact = workflow::train(&runtime, &pairs, &layers, TrainingMethod::Caa)?;
artifact.save(Path::new("calm.ster.json"))?;
let checked = SteeringArtifact::load(Path::new("calm.ster.json"))?;
let report = workflow::evaluate(&runtime, &pairs, &checked)?;
# Ok::<(), anyhow::Error>(())
```

There are no async APIs, builder defaults, background workers, or global
runtime configuration beyond the optional progress sink.
