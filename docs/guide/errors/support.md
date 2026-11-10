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

Started through: `ster support report`, `ster support status`, `ster request support/report`, `ster request support/status`. Uses: `wisent-errors.support.report`, `wisent-errors.support.status`. Failure catalogue `ster.support`: 0 declared cause(s) of its own.
