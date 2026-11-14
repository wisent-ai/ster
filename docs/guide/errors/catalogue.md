# How the failure catalogue fails

Every declared way `ster errors list|show|check` and
`ster request errors/list|show` fail. How a refusal reads and which status it
exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/catalogue.md`;
`tests/errors/catalogue.sh` refuses a page that differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.errors`

list, show and check every declared way a Ster functionality fails

Started through: `ster errors list`, `ster errors show`, `ster errors check`, `ster request errors/list`, `ster request errors/show`. Uses: `ster.file`. Failure catalogue `ster.errors`: 4 declared cause(s) of its own.

<a id="ster-errors-functionality-unknown"></a>

### `ster.errors.functionality-unknown`

- **What failed:** showing the failure catalogue of {id}
- **Cause:** no functionality of this ster declares a catalogue under {id}
- **Observed state shown:** declared functionalities: {declared}
- **Fix:** name one of the declared functionalities; `ster errors list` prints them
- **Code:** `not_found`, exit 2; retrying unchanged will not help
- **Test:** `tests/errors/catalogue.sh::show_refuses_an_undeclared_functionality`

<a id="ster-errors-page-unknown"></a>

### `ster.errors.page-unknown`

- **What failed:** printing the catalogue sections of {page}
- **Cause:** no functionality is listed on {page}
- **Observed state shown:** pages: {pages}
- **Fix:** name one of those pages, or leave --page out for every section
- **Code:** `not_found`, exit 2; retrying unchanged will not help
- **Test:** `tests/errors/catalogue.sh::markdown_refuses_a_page_no_functionality_is_listed_on`

<a id="ster-errors-baseline-missing"></a>

### `ster.errors.baseline-missing`

- **What failed:** reading {role} {path}
- **Cause:** no file is there: nothing at {path} is the baseline named
- **Observed state shown:** path: {path} ({state}); its directory {parent}; the system answered: {error}
- **Fix:** name the baseline file the repository holds: tests/errors/uncatalogued.json for --uncatalogued, tests/errors/untested.json for --untested
- **Code:** `not_found`, exit 2; retrying unchanged will not help
- **Test:** `tests/errors/catalogue.sh::check_refuses_an_unreadable_baseline`

<a id="ster-errors-findings"></a>

### `ster.errors.findings`

- **What failed:** checking the failure catalogue
- **Cause:** {count} finding(s): a malformed declaration, a declared test the repository does not hold, a command no functionality names, or a baseline entry no longer needed
- **Observed state shown:** findings: {findings}
- **Fix:** fix each finding as it says: correct the declaration in src/failures, write the declared test, declare the command's failures, or remove the entry from its baseline
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/errors/catalogue.sh::check_finds_a_baseline_entry_no_longer_needed`
