# How the workspace fails

Every declared way Ster's local workspace fails: locating and writing its
document, naming a set it holds, importing, showing, selecting and removing a
set, and reading the active set for a command run without `--pairs`. How a
refusal reads and which status it exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/workspace.md`;
`tests/errors/catalogue.sh` refuses a page that differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.workspace.store`

locate the workspace under XDG_DATA_HOME or HOME, read its document, and write it so a reader never sees half of it

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.workspace.store`: 6 declared cause(s) of its own.

<a id="ster-workspace-store-home-unset"></a>

### `ster.workspace.store.home-unset`

- **What failed:** locating the Ster workspace
- **Cause:** neither XDG_DATA_HOME nor HOME is set
- **Observed state shown:** XDG_DATA_HOME and HOME are unset or empty
- **Fix:** set XDG_DATA_HOME, or HOME, to the directory the workspace lives under: $XDG_DATA_HOME/ster, else ~/.local/share/ster
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::a_workspace_without_a_home_is_refused`

<a id="ster-workspace-store-workspace-unreadable"></a>

### `ster.workspace.store.workspace-unreadable`

- **What failed:** reading the Ster workspace {path}
- **Cause:** the file cannot be read: {error}
- **Observed state shown:** path: {path}
- **Fix:** make {path} a file this account can read
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::an_unreadable_workspace_is_refused`

<a id="ster-workspace-store-workspace-malformed"></a>

### `ster.workspace.store.workspace-malformed`

- **What failed:** reading the Ster workspace {path}
- **Cause:** it is not a Ster workspace document
- **Observed state shown:** invalid Ster workspace JSON in {path}: {error}
- **Fix:** restore {path} from a copy, or move it aside and import the pair sets again with `ster workspace import`
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::a_workspace_that_is_not_its_document_is_refused`

<a id="ster-workspace-store-workspace-schema"></a>

### `ster.workspace.store.workspace-schema`

- **What failed:** reading the Ster workspace {path}
- **Cause:** unsupported Ster workspace schema {found}; this ster reads {expected}
- **Observed state shown:** schema: {found}
- **Fix:** read it with the ster that wrote it, or move {path} aside and import the pair sets again with `ster workspace import`
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::a_workspace_of_another_schema_is_refused`

<a id="ster-workspace-store-workspace-directory"></a>

### `ster.workspace.store.workspace-directory`

- **What failed:** creating the Ster workspace directory {directory}
- **Cause:** {error}
- **Observed state shown:** the workspace lives under {root}
- **Fix:** point XDG_DATA_HOME, or HOME, at a directory this account can write
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::a_workspace_under_a_file_is_refused`

<a id="ster-workspace-store-workspace-unwritable"></a>

### `ster.workspace.store.workspace-unwritable`

- **What failed:** writing the Ster workspace {path}
- **Cause:** {step} failed: {error}
- **Observed state shown:** {left}
- **Fix:** make {directory} writable for this account and run the command again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::a_read_only_workspace_is_refused_and_the_import_undone`

## `ster.workspace.set`

find an imported pair set by the id `ster workspace show` prints

Started through: run only by the functionalities that use it. Uses: `ster.workspace.store`. Failure catalogue `ster.workspace.set`: 1 declared cause(s) of its own.

<a id="ster-workspace-set-set-unknown"></a>

### `ster.workspace.set.set-unknown`

- **What failed:** finding pair set {id}
- **Cause:** the Ster workspace holds no pair set named {id}; it holds {held}
- **Observed state shown:** workspace: {path}
- **Fix:** name a set the workspace holds, as `ster workspace show` prints them, or import it with `ster workspace import`
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::an_unknown_set_is_refused`

## `ster.workspace.import`

validate a pair-set file, copy it into the workspace under its id, and make it the active set; a repeat of the same content is the same import

Started through: `ster workspace import`, `ster onboarding --import-pairs`, `ster request workspace/import`. Uses: `ster.pairs.file`, `ster.workspace.store`. Failure catalogue `ster.workspace.import`: 8 declared cause(s) of its own.

<a id="ster-workspace-import-name-invalid"></a>

### `ster.workspace.import.name-invalid`

- **What failed:** naming the imported set {name}
- **Cause:** pair-set name must start with an ASCII letter or digit and contain only letters, digits, dots, underscores, or hyphens
- **Observed state shown:** --name {name}
- **Fix:** pass a --name such as my-pairs
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::an_invalid_name_is_refused`

<a id="ster-workspace-import-name-underivable"></a>

### `ster.workspace.import.name-underivable`

- **What failed:** deriving a pair-set id from {source}
- **Cause:** {source} {why}, so no pair-set id can be derived from it
- **Observed state shown:** file name: {file}
- **Fix:** name the set with --name
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::a_file_name_without_letters_needs_a_name`

<a id="ster-workspace-import-name-conflict"></a>

### `ster.workspace.import.name-conflict`

- **What failed:** importing {source} as {id}
- **Cause:** pair-set name conflicts with existing content
- **Observed state shown:** {id} holds {path}, a different set
- **Fix:** choose another --name to preserve {path}
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::a_taken_name_is_refused_and_the_first_set_kept`

<a id="ster-workspace-import-destination-exists"></a>

### `ster.workspace.import.destination-exists`

- **What failed:** importing {source} as {id}
- **Cause:** the destination already exists outside the Ster workspace index; it was not replaced
- **Observed state shown:** file: {path}
- **Fix:** move {path} out of the workspace, or import the set under another --name
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::a_file_outside_the_index_is_not_replaced`

<a id="ster-workspace-import-destination-appeared"></a>

### `ster.workspace.import.destination-appeared`

- **What failed:** importing {source} as {id}
- **Cause:** the destination appeared while the import was being committed; it was not replaced
- **Observed state shown:** file: {path}; {left}
- **Fix:** move {path} out of the workspace, or import the set under another --name
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::a_destination_appearing_mid_import_is_not_replaced`

<a id="ster-workspace-import-staging-left"></a>

### `ster.workspace.import.staging-left`

- **What failed:** staging the import of {source} as {id}
- **Cause:** a staging file of an earlier import that did not finish is in the way
- **Observed state shown:** staging file: {path}
- **Fix:** remove {path}: no workspace entry records it
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::a_leftover_staging_file_is_refused`

<a id="ster-workspace-import-commit-failed"></a>

### `ster.workspace.import.commit-failed`

- **What failed:** committing the import of {source} as {id}
- **Cause:** linking {staging} to {path} failed: {error}
- **Observed state shown:** {left}
- **Fix:** keep the workspace on a filesystem that supports hard links, writable for this account, then import again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::a_commit_the_filesystem_refuses_is_rolled_back`

<a id="ster-workspace-import-staging-unremovable"></a>

### `ster.workspace.import.staging-unremovable`

- **What failed:** finishing the import of {source} as {id}
- **Cause:** the staging file {staging} cannot be removed: {error}
- **Observed state shown:** {left}
- **Fix:** remove {staging} and import again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::a_staging_file_that_cannot_be_removed_is_rolled_back`

## `ster.workspace.show`

print every imported pair set and which one is active

Started through: `ster workspace show`, `ster request workspace/show`. Uses: `ster.workspace.store`. Failure catalogue `ster.workspace.show`: 0 declared cause(s) of its own.

## `ster.workspace.select`

make an imported pair set the one every command run without --pairs reads

Started through: `ster workspace select`, `ster request workspace/select`. Uses: `ster.workspace.set`, `ster.workspace.store`. Failure catalogue `ster.workspace.select`: 1 declared cause(s) of its own.

<a id="ster-workspace-select-already-active"></a>

### `ster.workspace.select.already-active`

- **What failed:** selecting pair set {id}
- **Cause:** pair set {id} is already the active one
- **Observed state shown:** active pair set: {id}
- **Fix:** nothing needs doing: commands run without --pairs already read {id}; name another set to change it
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::selecting_the_active_set_is_refused`

## `ster.workspace.remove`

delete the workspace's copy of an imported pair set and take it out of the workspace; the source file is not touched

Started through: `ster workspace remove`, `ster request workspace/remove`. Uses: `ster.workspace.set`, `ster.workspace.store`. Failure catalogue `ster.workspace.remove`: 1 declared cause(s) of its own.

<a id="ster-workspace-remove-copy-undeletable"></a>

### `ster.workspace.remove.copy-undeletable`

- **What failed:** removing pair set {id}
- **Cause:** its copy {path} cannot be deleted: {error}
- **Observed state shown:** the workspace still holds {id}; nothing was changed
- **Fix:** make {directory} writable for this account and run `ster workspace remove {id}` again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::a_copy_that_cannot_be_deleted_keeps_the_set`

## `ster.workspace.active`

name the active imported pair set's copy for a command run without --pairs

Started through: `ster pairs inspect`, `ster vector train`, `ster vector optimize`, `ster vector evaluate`, `ster vector curve`, `ster vector project`, `ster tune dpo`, `ster tune bipo`, `ster tune reward`. Uses: `ster.workspace.store`. Failure catalogue `ster.workspace.active`: 3 declared cause(s) of its own.

<a id="ster-workspace-active-no-active-set"></a>

### `ster.workspace.active.no-active-set`

- **What failed:** choosing the pair set to read
- **Cause:** no active Ster pair set
- **Observed state shown:** the workspace holds {held}
- **Fix:** pass --pairs, select an imported set with `ster workspace select <id>`, or import one with `ster workspace import`
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::a_run_without_pairs_and_no_active_set_is_refused`

<a id="ster-workspace-active-active-unknown"></a>

### `ster.workspace.active.active-unknown`

- **What failed:** choosing the pair set to read
- **Cause:** the Ster workspace names {id} active but holds no pair set by that name
- **Observed state shown:** it holds {held}
- **Fix:** select a set it holds with `ster workspace select <id>`, or pass --pairs
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::an_active_set_the_workspace_does_not_hold_is_refused`

<a id="ster-workspace-active-active-copy-missing"></a>

### `ster.workspace.active.active-copy-missing`

- **What failed:** reading the active pair set {id}
- **Cause:** active Ster pair set is missing: {path}
- **Observed state shown:** the workspace records {id} at {path}
- **Fix:** import it again with `ster workspace import`, select another with `ster workspace select <id>`, or take it out with `ster workspace remove {id}`
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/workspace/failures.sh::an_active_set_whose_copy_is_gone_is_refused`
