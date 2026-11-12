#!/bin/sh
# Real test of how fitting a direction fails (`ster.vector.direction`),
# through `ster vector train`: pairs whose two sides read alike leave a
# direction of no length, and a fixture whose weights overflow at f16 reads
# states that are not finite. Neither writes an artifact. The run's home and
# fixture are tests/runtime/harness.sh's.
#
# Usage: tests/steering/vectors/direction.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=steering-direction . "$(dirname "$0")/../../runtime/harness.sh"
unwritten() {
  [ ! -e "$ROOT/out.json" ] || fail "a refused run wrote $ROOT/out.json"
}

sides_that_read_alike_are_refused() {
  jq '.trait_name = "alike" | .pairs = [.pairs[] | .negative = .positive]' docs/examples/pairs.json >"$ROOT/alike.json"
  run 1 vector train --model "$ROOT/base" --pairs "$ROOT/alike.json" --output "$ROOT/out.json" \
    --layers 1 --method caa
  said "fitting caa at layer 1 on 8 pairs: training produced a zero or non-finite direction: its length is 0"
  said "fix: give pairs whose two sides differ in the trait"
  unwritten
  passed sides_that_read_alike_are_refused
}

reads_that_are_not_finite_are_refused() {
  # Weights of 1e30 are finite in F32 and past F16's largest value, so the
  # same checkpoint mapped at f16 reads states that are not finite.
  jq '.initialization.embedding.scale = 1e30 | .initialization.projection.scale = 1e30' \
    docs/examples/model/fixture.json >"$ROOT/huge.json"
  "$BIN" model fixture create --from "$ROOT/huge.json" --output "$ROOT/huge" >>"$REPORT" 2>&1 \
    || fail "the overflowing fixture could not be written"
  run 1 vector train --model "$ROOT/huge" --pairs docs/examples/pairs.json --output "$ROOT/out.json" \
    --layers 1 --precision f16
  said "fitting caa at layer 1 on 8 pairs: the positive read 0 holds a value that is not finite"
  said "fix: read the pairs at a wider precision (--precision f32)"
  unwritten
  passed reads_that_are_not_finite_are_refused
}

sides_that_read_alike_are_refused
reads_that_are_not_finite_are_refused
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
