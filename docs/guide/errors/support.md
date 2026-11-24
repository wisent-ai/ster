# How support fails

Every declared way `ster support report|status` fails: through the support
client of wisent-errors it runs, whose causes are declared and tested in that
package. How a refusal reads and which status it exits with is on
[How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/support.md`;
`tests/errors/catalogue.sh` refuses a page that differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.support`

file a report with Ster's support from this installation and read the replies

Started through: `ster support report`, `ster support status`, `ster request support/report`, `ster request support/status`. Uses: `wisent-errors.support.report`, `wisent-errors.support.status`. Failure catalogue `ster.support`: 1 declared cause(s) of its own.

<a id="ster-support-envelope-undeclared"></a>

### `ster.support.envelope-undeclared`

- **What failed:** {step}
- **Cause:** wisent-errors' support client handed back a refusal naming no cause its functionalities declare, so its id, exit status and fix are unknown here
- **Observed state shown:** failure point {point}, code {code}, cause {named}: {detail}; its fix: {remedy}
- **Fix:** follow the fix it gives; build Ster against the wisent-errors revision whose support client declares that cause
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/support/support.sh::an_envelope_naming_no_declared_cause_is_refused`
