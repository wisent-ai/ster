#!/usr/bin/env bash
# Real test of `ster vector compare` and `ster request vector/compare`
# through the built binary, on a toy checkpoint it writes under this run's
# directory.
#
# It fits three directions on the toy model from the checked-in pair set:
# calm itself, restless (the same pairs with their sides swapped, so the
# opposite direction), and calm-pca (the same pairs fitted another way). The
# comparison must say what those relations are: calm and restless point
# opposite ways, so their cosine is the most different pair and negative,
# while the sign-blind distance puts them on one axis; cut into as many groups
# as there are artifacts, each is alone. Then the refusals: one artifact, more
# groups than artifacts, a layer an artifact does not carry, and the request
# body with one artifact. Every command, whether it was accepted, and its
# answer go to the run's report.txt; a failed check stops the run.
#
# Usage: STER=target/debug/ster tests/vectors/compare.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/vectors/compare-$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $1" | tee -a "$REPORT" >/dev/stderr
  false
}
# run accepted|refused ARGS...: the command must end the way named. An accepted
# run's answer is its stdout; a refused run's answer is everything it wrote.
run() {
  expected=$1
  shift
  if [ "$expected" = accepted ]; then
    if "$BIN" "$@" >"$ROOT/output" </dev/null; then outcome=accepted; else outcome=refused; fi
  else
    if "$BIN" "$@" &>"$ROOT/output" </dev/null; then outcome=accepted; else outcome=refused; fi
  fi
  out=$(cat "$ROOT/output")
  printf '$ ster %s\noutcome: %s\noutput: %s\n\n' "$*" "$outcome" "$out" >>"$REPORT"
  [ "$outcome" = "$expected" ] || fail "$* was $outcome, expected $expected: $out"
}
check() {
  [ "$2" = "$3" ] || fail "$1: got '$2', expected '$3'"
  echo "ok: $1 = $2" >>"$REPORT"
}
refused() {
  case "$out" in
    *"$1"*) echo "ok: refused with: $1" >>"$REPORT" ;;
    *) fail "expected a refusal containing '$1', got: $out" ;;
  esac
}

TOY="$ROOT/toy-model"
CALM=docs/examples/pairs.json
RESTLESS="$ROOT/restless.json"
jq '.trait_name = "restless" | .pairs |= map({positive: .negative, negative: .positive})' "$CALM" >"$RESTLESS"
run accepted toy-model "$TOY"
# Settings are read from the run's inputs so none is a chosen number: the
# last layer, and the layer past the last one no artifact can carry.
LAST=$(jq '.num_hidden_layers - (.num_hidden_layers / .num_hidden_layers)' "$TOY/config.json")
PAST=$(jq .num_hidden_layers "$TOY/config.json")

run accepted vector train --model "$TOY" --pairs "$CALM" --output "$ROOT/calm.ster.json" --layers "$LAST"
run accepted vector train --model "$TOY" --pairs "$RESTLESS" --output "$ROOT/restless.ster.json" --layers "$LAST"
jq '.trait_name = "calm-pca"' "$CALM" >"$ROOT/calm-pca.json"
run accepted vector train --model "$TOY" --pairs "$ROOT/calm-pca.json" --method pca --output "$ROOT/calm-pca.ster.json" --layers "$LAST"
ARTIFACTS=("$ROOT/calm.ster.json" "$ROOT/restless.ster.json" "$ROOT/calm-pca.ster.json")
COUNT=${#ARTIFACTS[@]}

run accepted vector compare "${ARTIFACTS[@]}" --clusters "$COUNT"
cp "$ROOT/output" "$ROOT/comparison.json"
check "the artifacts are labelled by their traits, in order" \
  "$(jq -r '.labels | join(",")' "$ROOT/comparison.json")" "calm,restless,calm-pca"
check "the compared layer is the one every artifact carries" \
  "$(jq -r '.layers | map(tostring) | join(",")' "$ROOT/comparison.json")" "$LAST"
check "calm and restless are the most different pair" \
  "$(jq -r '[.most_different.a, .most_different.b] | join(",")' "$ROOT/comparison.json")" "calm,restless"
check "their cosine is negative: opposite directions" \
  "$(jq '.most_different.cosine < (.most_different.cosine - .most_different.cosine)' "$ROOT/comparison.json")" "true"
check "the sign-blind distance puts calm and restless on one axis, nearer than calm-pca" \
  "$(jq '.distance[0][1] <= .distance[0][2]' "$ROOT/comparison.json")" "true"
check "as many groups as artifacts leaves each alone" \
  "$(jq '.clusters | unique | length' "$ROOT/comparison.json")" "$COUNT"
check "each artifact alone in its group has no silhouette" \
  "$(jq '.silhouette' "$ROOT/comparison.json")" "null"
check "every artifact has a uniqueness score" \
  "$(jq '.uniqueness | length' "$ROOT/comparison.json")" "$COUNT"

run refused vector compare "$ROOT/calm.ster.json" --clusters "$COUNT"
refused "vector compare needs at least two artifacts and got 1"
run refused vector compare "${ARTIFACTS[@]}" --clusters "$((COUNT + COUNT))"
refused "cannot group $COUNT artifacts: at most one group per artifact"
run refused vector compare "${ARTIFACTS[@]}" --clusters "$COUNT" --layers "$PAST"
refused "layer $PAST is not in calm's artifact"

# The desktop's path: the same comparison through `ster request vector/compare`,
# and its refusal before anything runs.
BODY=$(jq -cn --arg a "${ARTIFACTS[0]}" --arg b "${ARTIFACTS[1]}" --arg c "${ARTIFACTS[2]}" --argjson clusters "$COUNT" \
  '{artifacts: [$a, $b, $c], clusters: $clusters}')
if echo "$BODY" | "$BIN" request vector/compare >"$ROOT/request.ndjson"; then outcome=accepted; else outcome=refused; fi
printf '$ ster request vector/compare <<< %s\noutcome: %s\noutput: %s\n\n' "$BODY" "$outcome" "$(tail -n 1 "$ROOT/request.ndjson")" >>"$REPORT"
check "the request completes" "$outcome" "accepted"
check "the request reports the CLI's most different pair" \
  "$(tail -n 1 "$ROOT/request.ndjson" | jq -c '.json.most_different')" "$(jq -c .most_different "$ROOT/comparison.json")"
SINGLE=$(echo "$BODY" | jq -c '.artifacts |= .[:1]')
if echo "$SINGLE" | "$BIN" request vector/compare >"$ROOT/request.ndjson"; then outcome=accepted; else outcome=refused; fi
out=$(cat "$ROOT/request.ndjson")
printf '$ ster request vector/compare <<< %s\noutcome: %s\noutput: %s\n\n' "$SINGLE" "$outcome" "$out" >>"$REPORT"
check "a body with one artifact is refused" "$outcome" "refused"
refused "vector/compare needs at least two artifacts and got 1"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
