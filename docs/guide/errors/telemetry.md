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

Started through: `ster telemetry status`, `ster telemetry on`, `ster telemetry off`, `ster telemetry show`, `ster telemetry export`, `ster telemetry flush`, `ster request telemetry/status`, `ster request telemetry/set`, `ster request telemetry/show`, `ster request telemetry/flush`. Uses: `wisent-errors.telemetry.record`, `wisent-errors.telemetry.consent`, `wisent-errors.telemetry.deliver`. Failure catalogue `ster.telemetry`: 0 declared cause(s) of its own.
