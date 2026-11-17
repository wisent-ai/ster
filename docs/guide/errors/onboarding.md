# How the first-use walkthrough fails

Every declared way `ster onboarding` fails: finding where this account keeps
its progress, and reading back a progress file that belongs to another
journey or stands at a screen this journey does not hold. The state file
failing to be read, parsed or written is on [file.md](file.md); importing a
pair set with `--import-pairs` is on [workspace.md](workspace.md). How a
refusal reads and which status it exits with is on
[How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/onboarding.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.onboarding`

show the first-use walkthrough from the screen the operator reached, keep the progress in $XDG_STATE_HOME/ster/first-use.json, and record its completion once a pair set is imported

Started through: `ster onboarding`. Uses: `ster.file`, `ster.workspace.import`. Failure catalogue `ster.onboarding`: 3 declared cause(s) of its own.

<a id="ster-onboarding-onboarding-home-unset"></a>

### `ster.onboarding.onboarding-home-unset`

- **What failed:** locating the onboarding state
- **Cause:** neither XDG_STATE_HOME nor HOME is set, so there is no directory of this account to keep the walkthrough's progress in
- **Observed state shown:** XDG_STATE_HOME: {xdg}; HOME: {home}
- **Fix:** set XDG_STATE_HOME, or HOME, to the directory the progress lives under: $XDG_STATE_HOME/ster, else ~/.local/state/ster
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/onboarding/state.sh::a_walkthrough_without_a_home_is_refused`

<a id="ster-onboarding-onboarding-state-foreign"></a>

### `ster.onboarding.onboarding-state-foreign`

- **What failed:** reading the onboarding state {path}
- **Cause:** the state records another journey than the one this Ster walks: its {field} is {found}, and this Ster's is {expected}
- **Observed state shown:** path: {path}; {field}: {found}
- **Fix:** run `ster onboarding --reset` to start this journey anew; the recorded progress belongs to another release
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/onboarding/state.sh::a_state_of_another_journey_is_refused`

<a id="ster-onboarding-onboarding-screen-unknown"></a>

### `ster.onboarding.onboarding-screen-unknown`

- **What failed:** reading the onboarding state {path}
- **Cause:** the state stands at screen {screen}, which this journey does not hold, so the walkthrough cannot go on from it
- **Observed state shown:** path: {path}; current_screen_id: {screen}; the journey's screens: {screens}
- **Fix:** run `ster onboarding --reset` to start the journey from its first screen
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/onboarding/state.sh::a_state_at_an_unknown_screen_is_refused`

## `ster.star`

read, dismiss or reset the GitHub star request the shared onboarding runtime keeps per user, and count each completed run of a model toward it

Started through: `ster star status`, `ster star dismiss`, `ster star reset`, `ster request star/status`, `ster request star/dismiss`, `ster request star/reset`. Uses: none. Failure catalogue `ster.star`: 4 declared cause(s) of its own.

<a id="ster-star-journey-storage"></a>

### `ster.star.journey-storage`

- **What failed:** {step} the GitHub star request kept in {file}
- **Cause:** the onboarding runtime could not keep the request's journey state, for the reason its answer below names
- **Observed state shown:** {file} ({state}); its directory {directory} ({directory_state}); XDG_STATE_HOME {xdg}; HOME {home}; USER {user}; the runtime answered: {said}
- **Fix:** do what the runtime's answer names for its reason; a file that holds no journey state is set aside by `ster star reset`
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/onboarding/star.sh::a_state_directory_that_cannot_be_made_is_refused`

<a id="ster-star-journey-invalid"></a>

### `ster.star.journey-invalid`

- **What failed:** {step} the GitHub star request kept in {file}
- **Cause:** the onboarding runtime refused the star-request journey: the journey it carries or the progress stored for it names a screen, stage or repository the runtime does not read
- **Observed state shown:** {file} ({state}); repository {repository}; the runtime answered: {said}
- **Fix:** `ster star reset` counts from zero with the journey this ster carries; if it refuses too, install a ster release built with a current wisent-onboarding-client
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/onboarding/star.sh::a_journey_the_runtime_refuses_is_refused`

<a id="ster-star-journey-transport"></a>

### `ster.star.journey-transport`

- **What failed:** {step} the GitHub star request kept in {file}
- **Cause:** the onboarding runtime's transport did not answer, which the offline transport the star request uses never does
- **Observed state shown:** {file} ({state}); the runtime answered: {said}
- **Fix:** run the command again; if it refuses again, install a ster release built with a current wisent-onboarding-client
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/onboarding/star.sh::a_transport_that_does_not_answer_is_refused`

<a id="ster-star-journey-not-started"></a>

### `ster.star.journey-not-started`

- **What failed:** {step} the GitHub star request kept in {file}
- **Cause:** the onboarding runtime holds no progress of the star-request journey right after opening it, so there is no stage to read
- **Observed state shown:** {file} ({state}); the runtime answered: {said}
- **Fix:** `ster star reset` starts the journey again from zero
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/onboarding/star.sh::a_journey_without_progress_is_refused`
