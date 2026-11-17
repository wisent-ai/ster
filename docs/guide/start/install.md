# Installing and where downloads land

How the binary reaches a machine inside Wisent, and where a `--model`
argument that is not a local directory puts its weights. The
[README](../../../README.md) covers building it from a checkout with kompilator.

## Installing through the release installer

Inside Wisent, the built binary is delivered by Stado, which builds it with
kompilator; no Cargo builds or installs it.
Ster ships the two files that make this work — `.wisent-release.json` and
`release/build.sh` — and Stado reads them. Every release Stado publishes
carries an installer beside its `release.json`:

```bash
# run the ster release installer at stado://releases/ster/<version>/<platform>/install.sh
# (read through `<release API>/api/release/object?uri=...` and run with
# WISENT_RELEASE_API naming that origin; it places `ster` in ~/.local/bin)
stado product status ster --surface cli
stado product rollback ster --surface cli
```

The installer downloads that release's manifest and archive, checks the
archive's size and SHA-256 against the manifest, and places `ster` in
`~/.local/bin`. To update, run the installer of the newer `<version>`; it also
removes a same-named `~/.stado/bin/ster` that the retired
`stado product install` left behind, so the new copy is the one found first.
`status` reports the recipe the binary was installed from (kind
`stado-release`, repository `wisent-ai/ster`, manifest `.wisent-release.json`)
and the `source_revision` it was built from; `rollback` restores the retained
bytes kept under `~/.stado/products/ster/backups/<timestamp>/ster`. A `status`
of `stale` means the installed revision is behind `origin/main`.

Without a published release for your platform, build from a checkout with
kompilator as the [README](../../../README.md) shows.

The build script leaves the bare binary beside the tarball on purpose: a
`stado-release` install copies individual `bin/` members out of `.wisent-output`
and never unpacks an archive. The dependency runs one way — Stado installs Ster,
and Ster never calls Stado.

## Which release computes on a GPU

`.wisent-release.json` declares four platforms, all built by
`release/build.sh`:

| Platform | Built on | Devices | Starts on |
|---|---|---|---|
| `darwin-arm64` | a Mac | `cpu`, `metal` | every Apple-silicon Mac |
| `linux-amd64` | a Linux host | `cpu` | every x86-64 Linux host |
| `linux-amd64-cuda-hopper` | a Linux host whose `nvcc` lists `compute_90` | `cpu`, `cuda` | x86-64 Linux hosts with the CUDA runtime (`libcudart`) and a compute capability 9.0 GPU: H100, H200, GH200 |
| `linux-amd64-cuda-blackwell` | a Linux host whose `nvcc` lists `compute_120` | `cpu`, `cuda` | x86-64 Linux hosts with the CUDA runtime (`libcudart`) and a compute capability 12.0 GPU: GeForce RTX 50 series, RTX PRO Blackwell |

The capabilities and the cards that have them are NVIDIA's
([CUDA GPU compute capability](https://developer.nvidia.com/cuda/gpus)).
`linux-amd64` stays CPU-only because a CUDA build links the CUDA runtime and
would not start on a host without it. There are two CUDA builds and not one
for both because Candle's kernel build takes one `CUDA_COMPUTE_CAP` and
compiles every kernel for its arch-specific target (`compute_90a`,
`compute_120a`); NVIDIA documents such code as neither forward nor backward
compatible
([Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/#application-compatibility-on-blackwell-architecture)),
so each binary runs on its own capability only. The Blackwell data-centre
parts (B200 and GB200 at 10.0, B300 and GB300 at 10.3) and GB10 (DGX Spark,
12.1) are served by neither; they run `linux-amd64` on the CPU. The release
installer of a CUDA platform reads the GPU's compute capability
(`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`) and refuses a
host whose GPU is not the one its platform names.

The platform names the capability: the build sets `CUDA_COMPUTE_CAP` to 90
or 120 and refuses, naming nvcc's version, when the builder's
`nvcc --list-gpu-arch` does not list `compute_90` or `compute_120`. nvcc
accepts only the host C++ compilers its release supports, so the build hands
Candle's kernel build (as `NVCC_CCBIN`) the first of `gcc`, `gcc-14`,
`gcc-13` and `gcc-12` that nvcc preprocesses with, and refuses, naming both
versions, when none does. The build also refuses to start without `nvcc`.
On a host with several GPUs, `--device cuda:N` picks card `N`; plain `cuda`
is card 0.

Before any of them is built, each platform's quality gate `error-pages` runs
`kompilator test --manifest-path Cargo.toml --test error-pages` with the
`private-sources` input: it compiles the library and the `ster` binary
(darwin-arm64 with `--features metal`, as its release builds) and checks
every page under `docs/guide/errors/` against the failure catalogue, so a
tree that does not compile, or whose error pages drift, is refused.

## Where checkpoint downloads land

A `--model` argument that is not a local directory is resolved against the
Hugging Face Hub, and the multi-gigabyte weights land in the hub cache:

- weights, `config.json` and `tokenizer.json`: `$HF_HOME/hub`, or
  `~/.cache/huggingface/hub` when `HF_HOME` is unset;
- an optional access token, read only if that file already exists:
  `$HF_HOME/token`, or `~/.cache/huggingface/token`.

Ster builds the hub client from the environment, as the Hub's own clients do:
`HF_HOME` moves the cache and `HF_ENDPOINT` the server. A Stado job that sets
`HF_HOME` to a directory of its own keeps every download there. Ster runs no
free-space preflight, so size the destination before a first fetch.

