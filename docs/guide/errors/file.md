# How reading and writing one of Ster's files fails

Every declared way reading a file, reading a JSON document or writing,
linking or removing a file fails, whichever role the file plays: a pair set, a
steering artifact, a calibration, a checkpoint's state, a workspace record.
The functionality that reads the file names it (`reading the pair set …`) and
keeps its own cause for a file that is not there, since what to name instead
depends on what the file is; every other reason is one cause here, split by
the reason the system or the JSON reader gives. How a refusal reads and which
status it exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/file.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.file`

read one of Ster's files, JSON documents or safetensors files, or write, link or remove one, whichever role it plays

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.file`: 19 declared cause(s) of its own.

<a id="ster-file-file-absent"></a>

### `ster.file.file-absent`

- **What failed:** reading {role} {path}
- **Cause:** nothing is there: no file has the name {path}
- **Observed state shown:** path: {path} ({state}); its directory {parent}; the system answered: {error}
- **Fix:** name {role} where it is; `ls -ln {parent}` shows what that directory holds
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/generate.sh::a_system_turn_that_cannot_be_read_is_refused`

<a id="ster-file-file-read-denied"></a>

### `ster.file.file-read-denied`

- **What failed:** reading {role} {path}
- **Cause:** this account may not read it: the file's permission bits or owner keep this account out
- **Observed state shown:** path: {path} ({state}); the system answered: {error}
- **Fix:** give this account read access to {path} (`chmod u+r {path}` as its owner), or name a copy it may read
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/file.sh::a_pair_set_this_account_may_not_read_is_refused`

<a id="ster-file-file-is-directory"></a>

### `ster.file.file-is-directory`

- **What failed:** reading {role} {path}
- **Cause:** {path} is a directory, and {role} is one file
- **Observed state shown:** path: {path} ({state}); error kind {kind}; the system answered: {error}
- **Fix:** name the file that holds {role}, not a directory; `ls -ln {path}` shows what the directory holds
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/file.sh::a_directory_named_as_a_pair_set_is_refused`

<a id="ster-file-file-not-text"></a>

### `ster.file.file-not-text`

- **What failed:** reading {role} {path}
- **Cause:** it holds bytes that are not UTF-8, and {role} is read as UTF-8 text
- **Observed state shown:** path: {path} ({state}); the system answered: {error}
- **Fix:** convert {path} to UTF-8 (`iconv -t UTF-8`), or name {role} as it was first written
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::an_export_that_is_not_text_is_refused`

<a id="ster-file-file-below-file"></a>

### `ster.file.file-below-file`

- **What failed:** reading {role} {path}
- **Cause:** the path runs through a file: {blocker} is not a directory, so nothing can be found below it
- **Observed state shown:** path: {path}; {blocker}: {blocker_state}; the system answered: {error}
- **Fix:** name {role} by a path that runs through directories only; {blocker} is a file
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/file.sh::a_pair_set_below_a_file_is_refused`

<a id="ster-file-file-read-broken"></a>

### `ster.file.file-read-broken`

- **What failed:** reading {role} {path}
- **Cause:** the system broke off the read although this account may read the file: it was removed or replaced while Ster read it, or the disk failed
- **Observed state shown:** path: {path} ({state}); error kind {kind}; the system answered: {error}
- **Fix:** see what is at {path} with `ls -ln {path}`, and run the command again once the file is whole
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/pairs/imports/file.sh::a_read_the_system_breaks_off_is_refused`

<a id="ster-file-document-not-json"></a>

### `ster.file.document-not-json`

- **What failed:** reading {role} {path}
- **Cause:** it is not JSON, and {role} is {form}: it is another kind of file, or one broken by hand
- **Observed state shown:** path: {path} ({state}); the JSON reader stopped at line {line}, column {column}: {error}
- **Fix:** name {role}, which is {form}, or repair the JSON of {path} at line {line}, column {column}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/file.sh::a_pair_set_that_is_not_json_is_refused`

<a id="ster-file-document-truncated"></a>

### `ster.file.document-truncated`

- **What failed:** reading {role} {path}
- **Cause:** it ends before its JSON value closes: it was cut short while it was written (a full disk, a process stopped midway)
- **Observed state shown:** path: {path} ({state}); the JSON reader ran out at line {line}, column {column}: {error}
- **Fix:** write {role} again; it is {form}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/file.sh::a_pair_set_cut_short_is_refused`

<a id="ster-file-document-shape"></a>

### `ster.file.document-shape`

- **What failed:** reading {role} {path}
- **Cause:** it is JSON of another form than {role}, which is {form}: a field it requires is missing, a field holds another type or value, or it is a document of another kind
- **Observed state shown:** path: {path} ({state}); at line {line}, column {column} the reader answered: {error}
- **Fix:** name {role}, which is {form}; the reader's answer names the field that differs
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/file.sh::a_file_that_is_not_a_pair_set_is_refused`

<a id="ster-file-tensors-not-safetensors"></a>

### `ster.file.tensors-not-safetensors`

- **What failed:** reading {role} {path}
- **Cause:** {path} is not a safetensors file: its header or its table of tensors does not parse, so it was cut short while written or is a file of another format
- **Observed state shown:** path: {path} ({state}); the safetensors reader answered: {error}
- **Fix:** name {role} as Ster wrote it; a file cut short is written again by the command that made it
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::adapter_weights_that_are_not_safetensors_are_refused`

<a id="ster-file-tensors-dtype-unloadable"></a>

### `ster.file.tensors-dtype-unloadable`

- **What failed:** reading {role} {path}
- **Cause:** a tensor of {path} is stored in a dtype Candle does not load, such as bool, i8 or u64 (Candle loads the integer, float and 8-, 6- and 4-bit float dtypes)
- **Observed state shown:** path: {path} ({state}); Candle answered: {error}
- **Fix:** name {role} as Ster wrote it; a file another tool wrote must hold its tensors in a dtype Candle loads
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::adapter_weights_of_a_dtype_candle_does_not_load_are_refused`

<a id="ster-file-tensors-unplaced"></a>

### `ster.file.tensors-unplaced`

- **What failed:** reading {role} {path}
- **Cause:** the device did not take the tensors {path} holds: it has no memory left for them, or does not hold their dtype
- **Observed state shown:** path: {path} ({state}); the device answered: {error}
- **Fix:** run on a device with room for {role} (`--device cpu` uses host memory), or free the memory another process holds
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::adapter_weights_the_device_does_not_take_are_refused`

<a id="ster-file-tensors-unserialized"></a>

### `ster.file.tensors-unserialized`

- **What failed:** writing {role} {path}
- **Cause:** the tensors could not be laid out as safetensors: a tensor's data could not be copied off the device it lives on
- **Observed state shown:** path: {path}; {left}; Candle answered: {error}
- **Fix:** run the command again on `--device cpu`, where every tensor is in host memory
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::tensors_the_device_does_not_give_back_are_refused`

<a id="ster-file-file-write-denied"></a>

### `ster.file.file-write-denied`

- **What failed:** {action} {role} {path}
- **Cause:** this account may not change {directory} or {path}: their permission bits or owner keep this account out
- **Observed state shown:** path: {path} ({state}); directory {directory} ({directory_state}); {left}; the system answered: {error}
- **Fix:** give this account write access to {directory} (`chmod u+w {directory}` as its owner), or name a path in a directory it may write
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/file.sh::a_set_in_a_read_only_directory_is_refused`

<a id="ster-file-file-system-read-only"></a>

### `ster.file.file-system-read-only`

- **What failed:** {action} {role} {path}
- **Cause:** the file system holding {directory} is mounted read-only, so nothing on it can change
- **Observed state shown:** path: {path}; directory {directory} ({directory_state}); {left}; the system answered: {error}
- **Fix:** name a path on a file system mounted for writing (`mount` lists which are read-only)
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/file.sh::a_set_on_a_read_only_file_system_is_refused`

<a id="ster-file-file-disk-full"></a>

### `ster.file.file-disk-full`

- **What failed:** {action} {role} {path}
- **Cause:** the disk holding {directory} has no room left for {path}, or the file passes the largest size it allows
- **Observed state shown:** path: {path} ({state}); directory {directory}; {left}; the system answered: {error}
- **Fix:** free space on the disk holding {directory} (`df -h {directory}` shows it), then run the command again
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/pairs/imports/file.sh::a_set_on_a_full_disk_is_refused`

<a id="ster-file-file-path-through-file"></a>

### `ster.file.file-path-through-file`

- **What failed:** {action} {role} {path}
- **Cause:** the path runs through a file: {blocker} is not a directory, so nothing can be created below it
- **Observed state shown:** path: {path}; {blocker}: {blocker_state}; {left}; the system answered: {error}
- **Fix:** name a path that runs through directories only, or move {blocker} out of the way
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/file.sh::a_set_under_a_file_is_refused`

<a id="ster-file-file-target-directory"></a>

### `ster.file.file-target-directory`

- **What failed:** {action} {role} {path}
- **Cause:** {path} is a directory, and {role} is one file
- **Observed state shown:** path: {path} ({state}); {left}; the system answered: {error}
- **Fix:** name a file path for {role}, not the directory {path}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/file.sh::a_merge_into_a_directory_is_refused`

<a id="ster-file-file-write-broken"></a>

### `ster.file.file-write-broken`

- **What failed:** {action} {role} {path}
- **Cause:** the system broke off the change although this account may write {directory}: the disk failed, or another process changed {path} meanwhile
- **Observed state shown:** path: {path} ({state}); directory {directory} ({directory_state}); {left}; error kind {kind}; the system answered: {error}
- **Fix:** see {directory} with `ls -ln {directory}`, and run the command again once nothing else changes it
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/pairs/imports/file.sh::a_write_the_system_breaks_off_is_refused`
