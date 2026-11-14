# How the steering workflows fail

Every declared way the commands that read a model's states and fit or
measure directions fail: choosing the layers `--layers` names, reading a
prompt set, fitting a direction by `--method`, holding pairs out to rank on,
`vector train`, `optimize`, `curve` and `evaluate`, `vector compare`,
`vector project` and `vector score`, `extract` and `parity`. Reading a
steering artifact is on [steering.md](steering.md), loading a model on
[model-load.md](model-load.md), and how a refusal reads and which status it
exits with on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/workflow.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.vector.layers`

read a --layers selection (all, a comma list or a half-open range) as layer numbers the model has

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.vector.layers`: 6 declared cause(s) of its own.

<a id="ster-vector-layers-model-layerless"></a>

### `ster.vector.layers.model-layerless`

- **What failed:** choosing the layers --layers {selection} names
- **Cause:** the model's config names no decoder layer (num_hidden_layers is 0), so there is no layer to read or steer
- **Observed state shown:** --layers {selection}; the model has 0 layers
- **Fix:** name a checkpoint whose config.json gives num_hidden_layers above 0
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/layers.sh::a_model_without_layers_is_refused`

<a id="ster-vector-layers-range-bound-unparsed"></a>

### `ster.vector.layers.range-bound-unparsed`

- **What failed:** choosing the layers --layers {selection} names
- **Cause:** the range {segment} has a bound, "{bound}", that is not a whole number of zero or more, and a range is START..END of two layer numbers
- **Observed state shown:** --layers {selection}; segment {segment}; bound "{bound}": {error}; {model_layers}
- **Fix:** write the range as two layer numbers, such as {example}, or name the layers as a comma list
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/vectors/layers.sh::a_range_bound_that_is_not_a_number_is_refused`

<a id="ster-vector-layers-range-empty"></a>

### `ster.vector.layers.range-empty`

- **What failed:** choosing the layers --layers {selection} names
- **Cause:** layer range {segment} must have start < end: a range is half-open, so START..END reads START up to END-1 and {start}..{end} reads no layer
- **Observed state shown:** --layers {selection}; segment {segment}: start {start}, end {end}; {model_layers}
- **Fix:** give the smaller layer first, such as {end}..{after}, or name one layer as {start}
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/vectors/layers.sh::a_range_that_holds_no_layer_is_refused`

<a id="ster-vector-layers-layer-unparsed"></a>

### `ster.vector.layers.layer-unparsed`

- **What failed:** choosing the layers --layers {selection} names
- **Cause:** "{segment}" is not a layer number: --layers takes `all`, whole numbers of zero or more separated by commas, or a range START..END
- **Observed state shown:** --layers {selection}; segment "{segment}": {error}; {model_layers}
- **Fix:** name the layers as all, a comma list such as 0,1 or a range such as {example}
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/vectors/layers.sh::a_layer_that_is_not_a_number_is_refused`

<a id="ster-vector-layers-selection-empty"></a>

### `ster.vector.layers.selection-empty`

- **What failed:** choosing the layers --layers {selection} names
- **Cause:** no layers selected: the selection holds only commas and spaces, so it names no layer
- **Observed state shown:** --layers "{selection}"; {model_layers}
- **Fix:** name at least one layer, such as {example}, or every layer with all
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/vectors/layers.sh::a_selection_naming_no_layer_is_refused`

<a id="ster-vector-layers-layer-past"></a>

### `ster.vector.layers.layer-past`

- **What failed:** choosing the layers --layers {selection} names
- **Cause:** layer {layer} is outside the model's {count} layers, counted from the first, so the model has layers 0 to {last}
- **Observed state shown:** --layers {selection}; every layer named past the model: {past}
- **Fix:** name layers from 0 to {last}, such as {example}, or all
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/vectors/layers.sh::a_layer_past_the_model_is_refused`

## `ster.prompts`

read a JSON object whose "prompts" lists at least one prompt

Started through: run only by the functionalities that use it. Uses: `ster.file`. Failure catalogue `ster.prompts`: 2 declared cause(s) of its own.

<a id="ster-prompts-prompts-missing"></a>

### `ster.prompts.prompts-missing`

- **What failed:** reading the prompt set {path}
- **Cause:** no file is there: nothing at {path} is the prompt set named
- **Observed state shown:** path: {path} ({state}); its directory {parent}; the system answered: {error}
- **Fix:** name a prompt set, a JSON object whose "prompts" lists one prompt per string as docs/examples/prompts.json does; `ls -ln {parent}` shows what that directory holds
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/workflow.sh::a_missing_prompt_set_is_refused`

<a id="ster-prompts-prompts-none"></a>

### `ster.prompts.prompts-none`

- **What failed:** reading the prompt set {path}
- **Cause:** prompt set contains no prompts, so there is nothing to read the model with
- **Observed state shown:** path: {path} ({state}); prompts: []
- **Fix:** list at least one prompt under "prompts" in {path}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/workflow.sh::an_empty_prompt_set_is_refused`

## `ster.vector.direction`

fit a unit direction on one layer's reads of the pairs' positive and negative sides by caa, pca or logistic, and score a direction by how it orders them

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.vector.direction`: 7 declared cause(s) of its own.

<a id="ster-vector-direction-method-unknown"></a>

### `ster.vector.direction.method-unknown`

- **What failed:** choosing the direction estimator --method {value}
- **Cause:** unknown training method "{value}": Ster fits a direction by contrastive activation addition (caa), the leading principal direction (pca) or a logistic probe (logistic), and by nothing else
- **Observed state shown:** --method (a request's method) "{value}"; expected caa, pca, or logistic
- **Fix:** give --method caa, pca or logistic
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/vectors/layers.sh::an_unknown_method_is_refused`

<a id="ster-vector-direction-examples-unequal"></a>

### `ster.vector.direction.examples-unequal`

- **What failed:** {step}
- **Cause:** training requires the same non-zero number of positive and negative examples; got {positive} positive and {negative} negative
- **Observed state shown:** positive reads: {positive}; negative reads: {negative}
- **Fix:** give a pair set where every pair has both sides; `ster pairs inspect` counts its pairs
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/direction.sh::sides_of_unequal_counts_are_refused`

<a id="ster-vector-direction-read-empty"></a>

### `ster.vector.direction.read-empty`

- **What failed:** {step}
- **Cause:** the first {side} read holds no value, so there is no space to fit a direction in
- **Observed state shown:** {side} read 0: 0 values; the reads number {positive} positive and {negative} negative
- **Fix:** read the pairs with a model whose hidden_size is above 0
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/direction.sh::reads_without_values_are_refused`

<a id="ster-vector-direction-read-width"></a>

### `ster.vector.direction.read-width`

- **What failed:** {step}
- **Cause:** the {side} read {index} holds {found} values where the first read holds {width}, and every read of one layer has one width
- **Observed state shown:** {side} read {index}: {found} values; the first positive read: {width} values
- **Fix:** read every pair through one model at one layer
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/direction.sh::reads_of_two_widths_are_refused`

<a id="ster-vector-direction-read-not-finite"></a>

### `ster.vector.direction.read-not-finite`

- **What failed:** {step}
- **Cause:** the {side} read {index} holds a value that is not finite: the forward pass overflowed at this precision, or the weights hold one
- **Observed state shown:** {side} read {index}: value {position} of {width} is {value}
- **Fix:** read the pairs at a wider precision (--precision f32), or check the checkpoint's weights are finite
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/direction.sh::reads_that_are_not_finite_are_refused`

<a id="ster-vector-direction-direction-width"></a>

### `ster.vector.direction.direction-width`

- **What failed:** {step}
- **Cause:** direction width {direction} does not match activation width {width}: the direction was fitted on a model of another width
- **Observed state shown:** direction: {direction} values; each read: {width} values
- **Fix:** score the artifact with the model it was fitted on; `ster vector inspect` shows its model and width
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/direction.sh::a_direction_of_another_width_is_refused`

<a id="ster-vector-direction-direction-degenerate"></a>

### `ster.vector.direction.direction-degenerate`

- **What failed:** {step}
- **Cause:** training produced a zero or non-finite direction: its length is {length}, so the two sides read alike at this layer, or the fit overflowed
- **Observed state shown:** direction length {length} over {width} values; {positive} pairs
- **Fix:** give pairs whose two sides differ in the trait, read another layer, or fit with another --method
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/direction.sh::sides_that_read_alike_are_refused`

## `ster.vector.holdout`

cut a pair set into the pairs every candidate is fitted on and the held-out last fraction it is ranked on, each holding a pair

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.vector.holdout`: 2 declared cause(s) of its own.

<a id="ster-vector-holdout-fraction-outside"></a>

### `ster.vector.holdout.fraction-outside`

- **What failed:** holding out --holdout {holdout} of the {total} pairs of the "{trait}" pair set
- **Cause:** the held-out fraction must be above zero and below one: it is the share of the pairs every candidate is ranked on, and the rest are fitted on
- **Observed state shown:** --holdout (a request's holdout) {holdout}; the pair set holds {total} pairs
- **Fix:** give --holdout a fraction above 0 and below 1 that leaves a pair on each side, such as {example}
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/vectors/holdout.sh::a_fraction_outside_zero_and_one_is_refused`

<a id="ster-vector-holdout-side-empty"></a>

### `ster.vector.holdout.side-empty`

- **What failed:** holding out --holdout {holdout} of the {total} pairs of the "{trait}" pair set
- **Cause:** a held-out fraction of {holdout} over {total} contrastive pairs leaves {fit} pairs to fit and {held} to rank on; both need at least one
- **Observed state shown:** --holdout {holdout} × {total} pairs rounds to {held} held out; {fit} left to fit on
- **Fix:** give --holdout a fraction that leaves a pair on each side, such as {example}, or add pairs with `ster pairs add`
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/vectors/holdout.sh::a_fraction_that_empties_a_side_is_refused`

## `ster.vector.train`

read both sides of every pair through the model at the layers named, fit one direction per layer by the method named, and write the steering artifact

Started through: `ster vector train`, `ster request vector/train`. Uses: `ster.workspace.active`, `ster.pairs.file`, `ster.model.load`, `ster.chat.template`, `ster.vector.layers`, `ster.runtime.encode`, `ster.vector.direction`, `ster.steering.artifact`, `ster.file`. Failure catalogue `ster.vector.train`: 0 declared cause(s) of its own.

## `ster.vector.optimize`

fit every layer and method on part of the pairs, rank each on the held-out rest, refit the winner on every pair and write it with the ranking

Started through: `ster vector optimize`, `ster request vector/optimize`. Uses: `ster.workspace.active`, `ster.pairs.file`, `ster.model.load`, `ster.chat.template`, `ster.vector.layers`, `ster.vector.holdout`, `ster.runtime.encode`, `ster.vector.direction`, `ster.steering.artifact`, `ster.file`. Failure catalogue `ster.vector.optimize`: 0 declared cause(s) of its own.

## `ster.vector.curve`

fit one method at every layer on each named number of pairs and score each fit on the same held-out pairs

Started through: `ster vector curve`, `ster request vector/curve`. Uses: `ster.workspace.active`, `ster.pairs.file`, `ster.model.load`, `ster.chat.template`, `ster.vector.layers`, `ster.vector.holdout`, `ster.runtime.encode`, `ster.vector.direction`. Failure catalogue `ster.vector.curve`: 1 declared cause(s) of its own.

<a id="ster-vector-curve-size-past"></a>

### `ster.vector.curve.size-past`

- **What failed:** choosing the sizes vector curve fits {method} on
- **Cause:** a size of {size} pairs does not fit in the {fit} pairs left to fit on after holding out {held}
- **Observed state shown:** --sizes {sizes}; sizes past the fitting part: {past}; the "{trait}" pair set holds {total} pairs
- **Fix:** name sizes of at most {fit} pairs, give a smaller --holdout, or add pairs with `ster pairs add`
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/vectors/holdout.sh::a_size_past_the_fitting_part_is_refused`

## `ster.vector.evaluate`

check the artifact fits the model, read both sides of every pair at the layers it carries, and score how each layer's direction orders them

Started through: `ster vector evaluate`, `ster request vector/evaluate`. Uses: `ster.workspace.active`, `ster.pairs.file`, `ster.steering.artifact`, `ster.model.load`, `ster.chat.template`, `ster.runtime.steer`, `ster.runtime.encode`, `ster.vector.direction`, `ster.tune.strength`. Failure catalogue `ster.vector.evaluate`: 0 declared cause(s) of its own.

## `ster.vector.compare`

read two or more steering artifacts of one model, compare their unit directions at every shared layer, and cut them into the groups named

Started through: `ster vector compare`, `ster request vector/compare`. Uses: `ster.steering.artifact`. Failure catalogue `ster.vector.compare`: 6 declared cause(s) of its own.

<a id="ster-vector-compare-artifacts-few"></a>

### `ster.vector.compare.artifacts-few`

- **What failed:** comparing the steering artifacts {labels}
- **Cause:** vector compare needs at least two artifacts and got {count}: a comparison is between directions
- **Observed state shown:** artifacts named: {count} ({labels})
- **Fix:** name two or more steering artifacts fitted on one model, such as ster vector compare calm.json formal.json --clusters 1
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/workflow.sh::one_artifact_to_compare_is_refused`

<a id="ster-vector-compare-models-differ"></a>

### `ster.vector.compare.models-differ`

- **What failed:** comparing the steering artifacts {labels}
- **Cause:** vector compare compares directions of one model: {first} is {first_model} at width {first_width}, {other} is {other_model} at width {other_width}, and two models' residual streams share no basis
- **Observed state shown:** {first}: model {first_model}, hidden_size {first_width}; {other}: model {other_model}, hidden_size {other_width}
- **Fix:** compare artifacts fitted on one model; `ster vector inspect` shows each artifact's model and hidden_size
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/workflow.sh::artifacts_of_two_models_are_refused`

<a id="ster-vector-compare-clusters-past"></a>

### `ster.vector.compare.clusters-past`

- **What failed:** comparing the steering artifacts {labels}
- **Cause:** --clusters {clusters} cannot group {count} artifacts: at most one group per artifact
- **Observed state shown:** --clusters (a request's clusters) {clusters}; artifacts: {count}
- **Fix:** give --clusters from 1 to {count}
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/workflow.sh::more_groups_than_artifacts_are_refused`

<a id="ster-vector-compare-layer-uncarried"></a>

### `ster.vector.compare.layer-uncarried`

- **What failed:** comparing the steering artifacts {labels}
- **Cause:** layer {layer} is not in {label}'s artifact, which carries layers {carried}, and every compared layer must be carried by every artifact
- **Observed state shown:** --layers {named}; layers every artifact carries: {common}
- **Fix:** name only layers every artifact carries ({common}), or leave --layers out to compare all of them
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/workflow.sh::a_layer_an_artifact_lacks_is_refused`

<a id="ster-vector-compare-layers-unshared"></a>

### `ster.vector.compare.layers-unshared`

- **What failed:** comparing the steering artifacts {labels}
- **Cause:** the artifacts share no layer to compare at: no layer carries a direction in every one of them
- **Observed state shown:** layers each artifact carries: {listed}
- **Fix:** train the artifacts at a common layer (ster vector train --layers), or compare only the ones that share one
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/workflow.sh::artifacts_sharing_no_layer_are_refused`

<a id="ster-vector-compare-direction-lengthless"></a>

### `ster.vector.compare.direction-lengthless`

- **What failed:** comparing the steering artifacts {labels}
- **Cause:** {label}'s direction at layer {layer} has length {length}, so it has no angle to compare: every value is zero, or too small to measure
- **Observed state shown:** {label}: layer {layer}, {width} values, length {length}
- **Fix:** train the {label} vector again with ster vector train, or leave layer {layer} out with --layers
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/workflow.sh::a_direction_without_length_is_refused`

## `ster.vector.project`

read every side of every pair at one layer unsteered and steered, and project both onto the principal components of the unsteered reads

Started through: `ster vector project`, `ster request vector/project`. Uses: `ster.workspace.active`, `ster.pairs.file`, `ster.steering.artifact`, `ster.model.load`, `ster.chat.template`, `ster.runtime.steer`, `ster.runtime.layers`, `ster.runtime.encode`. Failure catalogue `ster.vector.project`: 1 declared cause(s) of its own.

<a id="ster-vector-project-reads-invariant"></a>

### `ster.vector.project.reads-invariant`

- **What failed:** projecting the {pairs} pairs of the "{trait}" pair set at layer {layer}
- **Cause:** the unsteered reads at layer {layer} do not vary, so they have no component to project onto: every side reads the same state there
- **Observed state shown:** {reads} unsteered reads at layer {layer}, {components} component(s) asked for, none with variance
- **Fix:** give pairs whose sides differ, or read another --layer; layer 0 reads little more than the token embeddings
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/workflow.sh::reads_that_do_not_vary_are_refused`

## `ster.vector.score`

read a text once at one layer and project each token's state onto the artifact's unit direction there

Started through: `ster vector score`, `ster request vector/score`. Uses: `ster.file`, `ster.steering.artifact`, `ster.model.load`, `ster.chat.template`, `ster.runtime.steer`, `ster.runtime.layers`, `ster.runtime.encode`. Failure catalogue `ster.vector.score`: 6 declared cause(s) of its own.

<a id="ster-vector-score-input-absent"></a>

### `ster.vector.score.input-absent`

- **What failed:** reading the text to score
- **Cause:** give the text to score with --input or --input-file: the command scores one text and none was named
- **Observed state shown:** neither --input nor --input-file given; --vector {vector} --layer {layer}
- **Fix:** give --input TEXT, or --input-file FILE holding the text
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/score.sh::no_text_is_refused`

<a id="ster-vector-score-input-twice"></a>

### `ster.vector.score.input-twice`

- **What failed:** reading the text to score
- **Cause:** --input cannot be used with --input-file: give the text once, so it is clear which one is scored
- **Observed state shown:** --input ({characters} characters) and --input-file {path} both given
- **Fix:** keep --input or --input-file, not both
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/score.sh::a_text_and_a_text_file_are_refused`

<a id="ster-vector-score-input-file-missing"></a>

### `ster.vector.score.input-file-missing`

- **What failed:** reading the text to score from {path}
- **Cause:** no file is there: nothing at {path} is the text file --input-file names
- **Observed state shown:** path: {path} ({state}); its directory {parent}; the system answered: {error}
- **Fix:** name the file that holds the text, or give the text itself with --input; `ls -ln {parent}` shows what that directory holds
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/score.sh::a_missing_text_file_is_refused`

<a id="ster-vector-score-text-blank"></a>

### `ster.vector.score.text-blank`

- **What failed:** reading the text to score from {source}
- **Cause:** the text is empty; give the text to score: it holds no character but whitespace, so there is no token to read
- **Observed state shown:** {source}: {bytes} bytes, all whitespace
- **Fix:** give a text that holds at least one word
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/score.sh::an_empty_text_is_refused`

<a id="ster-vector-score-layer-uncarried"></a>

### `ster.vector.score.layer-uncarried`

- **What failed:** scoring a text along the {trait} direction at layer {layer}
- **Cause:** the artifact carries no vector at layer {layer}; it carries layers {carried}, and a text is scored along the direction at the layer it is read at
- **Observed state shown:** --layer {layer}; the {trait} artifact carries layers {carried}
- **Fix:** give --layer one of {carried}, or train a vector at layer {layer} with ster vector train --layers {layer}
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/score.sh::a_layer_the_artifact_lacks_is_refused`

<a id="ster-vector-score-direction-lengthless"></a>

### `ster.vector.score.direction-lengthless`

- **What failed:** scoring a text along the {trait} direction at layer {layer}
- **Cause:** the artifact's direction at layer {layer} has no usable length ({length}): every value is zero, or too small to measure, so no token has a length along it
- **Observed state shown:** the {trait} artifact: layer {layer}, {width} values, length {length}
- **Fix:** train the vector again with ster vector train, or score at another of its layers ({carried})
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/score.sh::a_direction_without_length_is_refused`

## `ster.extract`

read the prompt set before any weight, read each prompt through the model at the layers named, and write every layer's last-position state as one JSON document

Started through: `ster extract`, `ster request extract`. Uses: `ster.prompts`, `ster.model.load`, `ster.chat.template`, `ster.vector.layers`, `ster.runtime.encode`, `ster.file`. Failure catalogue `ster.extract`: 1 declared cause(s) of its own.

<a id="ster-extract-states-not-finite"></a>

### `ster.extract.states-not-finite`

- **What failed:** reading prompt {index} of {prompts} at layer {layer}
- **Cause:** the state the model reached holds a value that is not finite, and JSON has no spelling for one, so it would be written as null: the forward pass overflowed at this precision, or the weights hold one
- **Observed state shown:** prompt {index} at layer {layer}: value {position} of {width} is {value}; precision {precision}; nothing was written to {output}
- **Fix:** read the prompts at a wider precision (--precision f32), or check the checkpoint's weights are finite
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/workflow.sh::states_that_are_not_finite_are_refused`

## `ster.parity`

read every record's token ids and reference states, run the model over the same ids and measure how far its final states lie from the reference's

Started through: `ster parity`, `ster request parity`. Uses: `ster.file`, `ster.model.load`. Failure catalogue `ster.parity`: 13 declared cause(s) of its own.

<a id="ster-parity-input-missing"></a>

### `ster.parity.input-missing`

- **What failed:** reading the parity file {path}
- **Cause:** no file is there: nothing at {path} is the parity file --input names
- **Observed state shown:** path: {path} ({state}); its directory {parent}; the system answered: {error}
- **Fix:** name the parity file that lists the records to compare; `ls -ln {parent}` shows what that directory holds
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/parity.sh::a_missing_parity_file_is_refused`

<a id="ster-parity-records-none"></a>

### `ster.parity.records-none`

- **What failed:** reading the parity file {input}
- **Cause:** {input} lists no records to compare, so there is no state to measure
- **Observed state shown:** path: {input} ({state}); records: []
- **Fix:** list at least one record under "records" in {input}, each naming the token ids (tokenIds) and the states file (hidden)
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/parity.sh::a_parity_file_without_records_is_refused`

<a id="ster-parity-ids-both"></a>

### `ster.parity.ids-both`

- **What failed:** comparing record {index} of the parity file {input}
- **Cause:** record {index} names both tokenIds and tokenIdsFile; give one, so it is clear which ids the reference read
- **Observed state shown:** record {index}: tokenIds holds {count} ids; tokenIdsFile: {file}
- **Fix:** keep tokenIds or tokenIdsFile in record {index}, not both
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/parity.sh::a_record_naming_two_id_sources_is_refused`

<a id="ster-parity-ids-neither"></a>

### `ster.parity.ids-neither`

- **What failed:** comparing record {index} of the parity file {input}
- **Cause:** record {index} names neither tokenIds nor tokenIdsFile, so there are no tokens to read
- **Observed state shown:** record {index}: hidden {hidden}; no tokenIds, no tokenIdsFile
- **Fix:** give record {index} the token ids the reference read, as tokenIds or as a file named by tokenIdsFile
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/parity.sh::a_record_naming_no_ids_is_refused`

<a id="ster-parity-ids-file-missing"></a>

### `ster.parity.ids-file-missing`

- **What failed:** reading {role} {path}
- **Cause:** no file is there: nothing at {path} is the token id file the record names; tokenIdsFile is read relative to the parity file's directory
- **Observed state shown:** path: {path} ({state}); its directory {parent}; the system answered: {error}
- **Fix:** name the token id file relative to the parity file's directory; `ls -ln {parent}` shows what it holds
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/parity.sh::a_missing_token_id_file_is_refused`

<a id="ster-parity-ids-none"></a>

### `ster.parity.ids-none`

- **What failed:** comparing record {index} of the parity file {input}
- **Cause:** record {index} holds no token ids, so there is no position to compare
- **Observed state shown:** record {index}: {source} holds []
- **Fix:** give record {index} the token ids the reference read
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/parity.sh::a_record_without_ids_is_refused`

<a id="ster-parity-token-past"></a>

### `ster.parity.token-past`

- **What failed:** comparing record {index} of the parity file {input}
- **Cause:** record {index} holds token id {id}, past the model's {vocabulary} tokens: the reference read another tokenizer's ids
- **Observed state shown:** record {index}: id {id} at position {position}; the model's embedding table holds ids 0 to {last}
- **Fix:** record the reference states with the tokenizer of the model --model names, or compare against the model the ids belong to
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/parity.sh::a_token_past_the_vocabulary_is_refused`

<a id="ster-parity-states-missing"></a>

### `ster.parity.states-missing`

- **What failed:** reading {role} {path}
- **Cause:** no file is there: nothing at {path} is the reference states file the record's hidden names; hidden is read relative to the parity file's directory
- **Observed state shown:** path: {path} ({state}); its directory {parent}; the system answered: {error}
- **Fix:** name the safetensors file the reference wrote, relative to the parity file's directory; `ls -ln {parent}` shows what it holds
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/parity.sh::a_missing_states_file_is_refused`

<a id="ster-parity-tensor-absent"></a>

### `ster.parity.tensor-absent`

- **What failed:** comparing record {index} of the parity file {input}
- **Cause:** {path} holds no tensor "{tensor}"; the record names it, or names none and hidden_states is read
- **Observed state shown:** {path}: tensors {names}
- **Fix:** name the tensor that holds the states with the record's tensor field, one of {names}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/parity.sh::a_tensor_the_file_lacks_is_refused`

<a id="ster-parity-tensor-shape"></a>

### `ster.parity.tensor-shape`

- **What failed:** comparing record {index} of the parity file {input}
- **Cause:** {path}'s "{tensor}" is {shape}; Ster compares [positions, hidden] states, or [1, positions, hidden] of one sequence
- **Observed state shown:** {path}: "{tensor}" of shape {shape}, {dtype}
- **Fix:** save the residual stream of one sequence after the final norm as [positions, hidden]
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/parity.sh::a_tensor_of_another_shape_is_refused`

<a id="ster-parity-positions-past"></a>

### `ster.parity.positions-past`

- **What failed:** comparing record {index} of the parity file {input}
- **Cause:** record {index}'s reference holds {positions} positions for {tokens} tokens: the reference read more tokens than the record lists
- **Observed state shown:** {path}: [{positions}, {width}]; record {index}: {tokens} token ids
- **Fix:** list every token id the reference read in record {index}, or save only the positions of those ids
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/parity.sh::more_positions_than_tokens_are_refused`

<a id="ster-parity-width-other"></a>

### `ster.parity.width-other`

- **What failed:** comparing record {index} of the parity file {input}
- **Cause:** record {index}'s reference states are {width} wide and this model's {ours}: they come from a model of another width
- **Observed state shown:** {path}: [{positions}, {width}]; {model}: hidden_size {ours}
- **Fix:** compare against states the reference computed for {model}, or name the model they came from with --model
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/parity.sh::states_of_another_width_are_refused`

<a id="ster-parity-positions-none"></a>

### `ster.parity.positions-none`

- **What failed:** comparing record {index} of the parity file {input}
- **Cause:** record {index}'s "{tensor}" in {path} holds no positions to compare
- **Observed state shown:** {path}: "{tensor}" of shape {shape}
- **Fix:** save the states of at least one position
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/vectors/parity.sh::a_tensor_without_positions_is_refused`
