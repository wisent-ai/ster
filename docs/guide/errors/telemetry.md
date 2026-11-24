# How telemetry fails

Every declared way `ster telemetry status|on|off|show|export|flush` fails:
through the telemetry of wisent-errors it runs, whose causes are declared and
tested in that package. How a refusal reads and which status it exits with is
on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/telemetry.md`;
`tests/errors/catalogue.sh` refuses a page that differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.telemetry`

read and change what this installation of Ster reports to its maintainers, and send it

Started through: `ster telemetry status`, `ster telemetry on`, `ster telemetry off`, `ster telemetry show`, `ster telemetry export`, `ster telemetry flush`, `ster request telemetry/status`, `ster request telemetry/set`, `ster request telemetry/show`, `ster request telemetry/flush`. Uses: `wisent-errors.telemetry.record`, `wisent-errors.telemetry.consent`, `wisent-errors.telemetry.deliver`. Failure catalogue `ster.telemetry`: 2 declared cause(s) of its own.

<a id="ster-telemetry-envelope-undeclared"></a>

### `ster.telemetry.envelope-undeclared`

- **What failed:** {step}
- **Cause:** wisent-errors' telemetry handed back a refusal naming no cause its functionalities declare, so its id, exit status and fix are unknown here
- **Observed state shown:** failure point {point}, code {code}, cause {named}: {detail}; its fix: {remedy}
- **Fix:** follow the fix it gives; build Ster against the wisent-errors revision whose telemetry declares that cause
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/telemetry/telemetry.sh::an_envelope_naming_no_declared_cause_is_refused`

<a id="ster-telemetry-program-unnamed"></a>

### `ster.telemetry.program-unnamed`

- **What failed:** starting the detached `ster telemetry flush` after `ster {command}`
- **Cause:** the system does not name this program's executable, so no detached flush can be started from it: the binary was removed or replaced while it ran
- **Observed state shown:** error kind {kind}; the system answered: {error}; the events stay queued and the next run's flush sends them
- **Fix:** run `ster telemetry flush` from an installed ster; `ster telemetry show` lists what waits
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/telemetry/telemetry.sh::a_flush_without_its_program_is_said`
