# Wisent

**Wisent is a Python representation-engineering toolkit for researchers who need
to measure, evaluate, and deliberately steer behavior inside language models.**

It combines activation extraction, contrastive representation analysis,
steering methods, benchmark integration, and reproducible evaluation in one
research package.

[Quick start](#quick-start) · [Core use cases](#core-use-cases) ·
[Package source](https://github.com/wisent-ai/wisent) ·
[Documentation](https://www.wisent.ai/documentation)

Current boundary: the package is classified as Beta and its Python API may still
change. A detector or steering result is experimental evidence, not a guarantee
that a model is safe, truthful, or suitable for deployment.

## Problem and intended users

Model behavior is difficult to inspect from generated text alone. Researchers
need repeatable ways to collect internal representations, compare contrasting
behaviors, train lightweight classifiers or steering directions, and measure
whether an intervention changes both the target behavior and unrelated model
capabilities.

Wisent serves:

- **representation-engineering researchers** studying model activations and
  causal interventions;
- **model evaluators** comparing detection and steering methods across tasks;
- **AI safety engineers** prototyping observable controls before deciding
  whether a technique belongs in a production boundary.

## Product boundaries

### Included

- activation and representation extraction from supported transformer models;
- contrastive-pair construction and representation analysis;
- classifier, steering-vector, REFT, and related intervention workflows exposed
  by the installed package;
- benchmark and evaluator integration, including the split-out Wisent namespace
  packages declared by `setup.py`;
- CLI access through the installed `wisent` command;
- parameter registries, task definitions, and research-oriented examples shipped
  as package data.

### Explicit non-goals

- Wisent does not prove that a model is safe or free from hallucinations.
- A benchmark score is not a production authorization decision.
- The package does not provide hosted inference, model weights, GPU capacity, or
  provider credentials.
- It does not replace application-level policy, access control, monitoring, or
  human review.
- Research methods may require model-specific tuning and can degrade unrelated
  capabilities; no intervention is universally transferable.
- The repository does not promise that every optional benchmark or large model
  runs on every machine.

### Supported environment and current capability

| Surface | Requirement | Current state |
|---|---|---|
| Python package and CLI | Python 3.8 or newer | Beta |
| Core model workflows | PyTorch, Transformers, Accelerate and declared dependencies | Available |
| Harness benchmarks | `wisent[harness]` | Optional |
| REFT workflows | `wisent[reft]` | Optional |
| CUDA acceleration | compatible CUDA environment and `wisent[cuda]` | Optional, host-dependent |
| Gradio application surface | `wisent[app]` and split-out `wisent-gradio` | Optional |
| Stable 1.0 API | — | Not published |

Large-model downloads, GPU memory, dataset licences, and benchmark-specific
runtime requirements remain operator responsibilities.

## Core use cases

### Compare representations for contrasting behavior

- **Actor:** a model researcher with a supported local or remote model runtime.
- **Initial state:** the researcher has explicit contrasting examples and a
  reproducible model revision.
- **Outcome:** Wisent extracts and compares internal representations and records
  the parameters needed to reproduce the analysis.
- **Boundary:** separation in one dataset is evidence for that setup, not proof
  of a universal semantic concept.

### Evaluate a detector or steering method

- **Actor:** an evaluator comparing methods across tasks.
- **Initial state:** model, task, evaluator, seeds, and intervention parameters
  are fixed.
- **Outcome:** the workflow produces task-level measurements that can be compared
  with a non-intervened baseline.
- **Boundary:** target improvement must be considered together with regressions,
  variance, and benchmark validity.

### Integrate a research workflow through Python or CLI

- **Actor:** a research engineer automating repeated experiments.
- **Initial state:** the exact package version and optional extras are installed.
- **Outcome:** the engineer uses the Python namespace or `wisent` CLI while
  retaining explicit control over models, data, hardware, and outputs.
- **Boundary:** Wisent never supplies external model or dataset authority on the
  caller's behalf.

## How Wisent works

```text
model revision + examples + task definition
                    │
                    ▼
         activation / representation extraction
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 detector or analysis      steering intervention
        │                       │
        └───────────┬───────────┘
                    ▼
       benchmark evaluation + baseline comparison
                    │
                    ▼
        reproducible metrics and research artifacts
```

The installed model library owns model execution. Wisent owns extraction,
analysis, intervention configuration, and evaluation orchestration. Split-out
packages such as `wisent-extractors`, `wisent-evaluators`, `wisent-optimizer`,
`wisent-gradio`, and `wisent-tools` contribute to the shared `wisent.*`
namespace and retain their own narrower contracts.

## Quick start

### Prerequisites

- Python 3.8 or newer;
- enough disk and memory for the selected model and datasets;
- model and dataset access accepted by the operator;
- optional GPU runtime for accelerated experiments.

Install the published package and inspect the available CLI without downloading
a model:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install wisent
wisent --help
```

Expected result: the final command prints the command surface provided by the
installed package. It does not establish model access or execute an experiment.

For development from source:

```bash
git clone https://github.com/wisent-ai/wisent.git
cd wisent
python -m pip install -e .
wisent --help
```

Before a real experiment, pin the model revision, dataset revision, package
version, task configuration, random seeds, and hardware assumptions. Record both
the intervention result and the non-intervened baseline.

## Primary interfaces

- **Python namespace:** canonical research and integration surface under
  `wisent.*`.
- **CLI:** `wisent`, backed by the model-interface entry point declared in
  `setup.py`.
- **Parameter and task registries:** package data under `wisent/support` and the
  model-interface primitives.
- **Sibling namespace packages:** extraction, evaluation, UI, optimization, and
  operational scripts distributed separately but imported under `wisent.*`.

## Operational model

- **Configuration:** experiment code and explicit parameter files own model,
  task, evaluator, intervention, and output choices.
- **State:** models, datasets, caches, and generated results are local to the
  selected libraries and paths; this repository does not provide a hosted store.
- **Credentials:** Hugging Face or other provider credentials remain with the
  operator and must not be committed to experiment artifacts.
- **Cost:** model downloads, inference, training, storage, and accelerator time
  are external costs. The package does not hide or meter them.
- **Observability:** retain exact versions, seeds, configurations, baseline
  results, and raw evaluation outputs before drawing conclusions.
- **Recovery:** an interrupted experiment is rerun from pinned inputs; partial
  output must not be presented as a completed benchmark.

## Project status and support

- **Maturity:** Beta Python package; stable 1.0 compatibility is not promised.
- **Python:** 3.8 or newer, as declared by `setup.py`.
- **Distribution:** PyPI package `wisent` and source repository
  [`wisent-ai/wisent`](https://github.com/wisent-ai/wisent).
- **Issues:** use the public repository issue tracker for non-sensitive defects.
- **Security:** use GitHub Security Advisories for vulnerabilities and never
  attach credentials, private models, or restricted datasets to an issue.
- **License:** MIT; see [`LICENSE`](LICENSE). Model, dataset, and dependency
  licences remain separate obligations.
