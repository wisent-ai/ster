# How writing a pair set with a model fails

Every declared way `ster pairs synthesize` and `ster request pairs/synthesize`
fail: the settings, each checked before any model loads or Stado's inference alternatives are
asked, and what the writing model answers. The near-duplicate settings and
writing the set are on [pairs.md](pairs.md). How a refusal reads and which
status it exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/pairs-synthesize.md`;
`tests/errors/catalogue.sh` refuses a page that differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.pairs.synthesize`

write a contrastive pair set with a local model or a route Stado's inference alternatives answer: the opposite trait, then a question and two answers per attempt, each answer judged for refusal and each pair checked for a near-duplicate as it lands

Started through: `ster pairs synthesize`, `ster request pairs/synthesize`. Uses: `ster.inference`, `ster.model.load`, `ster.pairs.dedupe`, `ster.pairs.file`. Failure catalogue `ster.pairs.synthesize`: 9 declared cause(s) of its own.

<a id="ster-pairs-synthesize-trait-blank"></a>

### `ster.pairs.synthesize.trait-blank`

- **What failed:** writing pairs
- **Cause:** pairs synthesize needs a description of the trait the positive side shows
- **Observed state shown:** --trait "{trait}"
- **Fix:** describe the trait in one sentence with --trait
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/synthesize.sh::a_blank_trait_is_refused_before_any_model_loads`

<a id="ster-pairs-synthesize-temperature-not-positive"></a>

### `ster.pairs.synthesize.temperature-not-positive`

- **What failed:** checking the sampling settings
- **Cause:** synthesis requires a temperature above zero; argmax generation repeats a single prompt
- **Observed state shown:** --temperature {temperature}
- **Fix:** give --temperature a number above zero
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/synthesize.sh::a_temperature_at_or_below_zero_is_refused_before_any_model_loads`

<a id="ster-pairs-synthesize-generator-unknown"></a>

### `ster.pairs.synthesize.generator-unknown`

- **What failed:** choosing the model that writes the pairs
- **Cause:** pairs synthesize writes with local or brama, and "{generator}" is neither
- **Observed state shown:** generator: "{generator}"
- **Fix:** give generator local, with a model, or brama, with generatorModel
- **Code:** `not_found`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/synthesize.sh::an_unknown_generator_is_refused`

<a id="ster-pairs-synthesize-model-missing"></a>

### `ster.pairs.synthesize.model-missing`

- **What failed:** choosing the model that writes the pairs
- **Cause:** pairs synthesize with --generator local requires --model
- **Observed state shown:** --generator local without --model
- **Fix:** name the Hugging Face model id or local model directory that writes the pairs with --model, or write them through Stado's inference alternatives with --generator brama --generator-model <route>
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/synthesize.sh::a_local_generator_without_a_model_is_refused`

<a id="ster-pairs-synthesize-route-missing"></a>

### `ster.pairs.synthesize.route-missing`

- **What failed:** choosing the model that writes the pairs
- **Cause:** pairs synthesize with --generator brama requires --generator-model
- **Observed state shown:** --generator brama without --generator-model
- **Fix:** name the route that writes the pairs with --generator-model: a Brama alias, a provider/model route or a selector
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/synthesize.sh::a_brama_generator_without_a_route_is_refused`

<a id="ster-pairs-synthesize-count-missing"></a>

### `ster.pairs.synthesize.count-missing`

- **What failed:** reading the pairs/synthesize request
- **Cause:** pairs synthesize requires count, a pair count above zero
- **Observed state shown:** the request names no count
- **Fix:** give count, the number of pairs to keep
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/synthesize.sh::a_request_without_a_count_is_refused`

<a id="ster-pairs-synthesize-opposite-empty"></a>

### `ster.pairs.synthesize.opposite-empty`

- **What failed:** asking {generator} for the opposite of {trait}
- **Cause:** {generator} returned no opposite trait for {trait}
- **Observed state shown:** its answer was empty
- **Fix:** state the opposite trait with --opposite
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/synthesize.sh::a_generator_naming_no_opposite_is_refused`

<a id="ster-pairs-synthesize-verdict-malformed"></a>

### `ster.pairs.synthesize.verdict-malformed`

- **What failed:** reading {generator}'s refusal judgement
- **Cause:** {generator} answered the refusal judgement with {text}, which is not exactly {"refuses": true} or {"refuses": false}
- **Observed state shown:** the parser stopped: {error}
- **Fix:** write the pairs with a generator that answers a one-line JSON instruction exactly: an instruction-tuned --model, or another --generator-model route
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/synthesize.sh::a_judgement_that_is_not_the_json_form_is_refused`

<a id="ster-pairs-synthesize-no-pairs"></a>

### `ster.pairs.synthesize.no-pairs`

- **What failed:** writing pairs with {generator}
- **Cause:** none of {attempts} attempts kept a pair, and a set without pairs is refused
- **Observed state shown:** rejected as empty: {empty}, as refusals: {refusals}, as duplicates: {duplicates}
- **Fix:** raise --retry-multiplier for more attempts, state --opposite, or write with another generator; the counts say which step rejected the attempts
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/synthesize.sh::a_run_that_keeps_no_pair_is_refused`
