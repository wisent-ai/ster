# Changelog

## Unreleased

- Every refusal of the workspace and of a pair-set file is a declared cause
  (wisent-errors): it says what failed, why, the state observed and the fix,
  and exits `2` for a usage refusal, `69` when retrying unchanged can help and
  `1` otherwise. An invalid `--name`, or a file name that gives no id, now
  exits `2`.
- `ster errors list|show|check` and `ster request errors/list|show` print and
  check the failure catalogue; `docs/guide/errors.md` and one page per area
  under `docs/guide/errors/` publish it, and `tests/errors/catalogue.sh`
  refuses a page that differs from the catalogue.
- A failed `ster request` result carries the declared cause in `json.cause`,
  and a rejected or conflicting workspace import carries it in its report.
- `ster workspace remove` deletes the copy before it changes the workspace: a
  copy that cannot be deleted leaves the set in place, where it used to leave
  a file no entry recorded that the next import of that id met as a conflict.
- A workspace write that fails says whether its temporary file was removed,
  and an import whose record cannot be written says it was undone.
