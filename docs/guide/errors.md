# How Ster fails

Every Ster functionality declares each way it can fail, in `src/failures`,
with the [wisent-errors](https://github.com/wisent-ai/wisent-errors)
catalogue. A declared cause states what failed, why, the state the refusal
shows, and what fixes it, so a refusal reads:

```text
Error: <what failed>: <cause>; observed: <state>; fix: <remedy>
```

with the actual paths, ids and values filled in. A step that adds context to
a declared refusal (an import that was undone, say) prints its sentence first
and the declared refusal under `Caused by:`. A functionality lists the
functionalities it uses; their causes are met through it and declared once,
where they belong.

## Exit status

| Status | When |
|---|---|
| `0` | The command did what it promised. |
| `1` | A declared cause refused it and retrying unchanged will not help, or a failure no cause declares yet. |
| `2` | A usage refusal: an argument or flag is wrong, including clap's own refusals. |
| `69` | A declared cause retrying unchanged can help (an unreachable or throttling dependency, a timeout). |

`ster request <operation>` keeps its own statuses (`0`, `1`, `2`, see
[desktop requests](desktop-requests.md)); when a declared cause refused the
operation, its result's `json.cause` carries the cause's `id`, `code`,
`exit` (the status the same CLI command exits with), `usage`, `retryable` and
`remedy`. A rejected or conflicting workspace import also carries `cause` in
its report on both surfaces.

## The catalogue, by area

| Page | Functionalities |
|---|---|
| [errors/catalogue.md](errors/catalogue.md) | `ster.errors` |
| [errors/pairs.md](errors/pairs.md) | `ster.pairs.file` |
| [errors/workspace.md](errors/workspace.md) | `ster.workspace.store`, `.set`, `.import`, `.show`, `.select`, `.remove`, `.active` |

Each page below its marker is what
`ster errors list --markdown --page docs/guide/errors/<area>.md` prints;
`tests/errors/catalogue.sh` refuses a page that differs from it.

## Reading and checking it

```sh
ster errors list                          # every functionality and its causes, as JSON
ster errors list --markdown               # every area page's sections
ster errors show ster.workspace.import    # its causes and every cause it meets
ster errors check --repository . \
  --uncatalogued tests/errors/uncatalogued.json \
  --untested tests/errors/untested.json
```

The same answers are `ster request errors/list` (no fields) and
`ster request errors/show` (`{"id": "<functionality>"}`).

`ster errors check` refuses (exit 1) a malformed declaration, a declared test
the checkout does not hold, and a command no functionality names as an entry.
Two baselines let the catalogue grow area by area, and both may only shrink:
`tests/errors/uncatalogued.json` lists the commands whose failures are not
declared yet, and `tests/errors/untested.json` the declared tests not written
yet, because no local state provokes their cause (a file appearing between two
steps of one import, a filesystem refusing a hard link). An entry that is no
longer needed is a finding too.
