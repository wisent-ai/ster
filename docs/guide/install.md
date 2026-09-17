# Installing and where downloads land

How the binary reaches a machine inside Wisent, and where a `--model`
argument that is not a local directory puts its weights. The
[README](../../README.md) covers the `cargo install` route.

## Installing through Stado

Inside Wisent, the built binary is delivered by Stado rather than by `cargo`.
Ster ships the two files that make this work — `.wisent-release.json` and
`release/build.sh` — and Stado reads them:

```bash
stado product install ster --surface cli
stado product status ster --surface cli
stado product update ster --surface cli
stado product rollback ster --surface cli
```

`status` reports the recipe it installed from (kind `stado-release`, repository
`wisent-ai/ster`, manifest `.wisent-release.json`), the `source_revision` it was
built from, and the two paths it wrote: `~/.stado/bin/ster` and
`~/.local/bin/ster`. Each `update` copies the binary it replaces under
`~/.stado/products/ster/backups/<timestamp>/ster`, which is what `rollback`
restores. A `status` of `stale` means the installed revision is behind
`origin/main`.

The build script leaves the bare binary beside the tarball on purpose: a
`stado-release` install copies individual `bin/` members out of `.wisent-output`
and never unpacks an archive. The dependency runs one way — Stado installs Ster,
and Ster never calls Stado. `cargo install` needs none of it.

## Where checkpoint downloads land

A `--model` argument that is not a local directory is resolved against the
Hugging Face Hub, and the multi-gigabyte weights land in the default hub cache:

- weights, `config.json` and `tokenizer.json`: `~/.cache/huggingface/hub`;
- an optional access token, read only if that file already exists:
  `~/.cache/huggingface/token`.

Ster builds the hub client with hf-hub's bare `Api::new` (`src/runtime/load/checkpoint.rs:57`)
rather than its environment-aware builder, so `HF_HOME` and `HF_ENDPOINT` do not
move that cache in 0.13. To download somewhere else — another disk, a shared
volume — fetch the repository yourself and pass the directory as `--model`. Ster
runs no free-space preflight, so size the destination before a first fetch.

