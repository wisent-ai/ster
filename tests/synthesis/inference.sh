#!/bin/sh
# Real test of the refusals of `ster.inference`, the layer every hosted
# completion goes through (`stado capabilities call inference --consumer
# ster --json`), that local state provokes, through the built binary:
#
#   * a blank route, refused (exit 2) before Stado is asked, by
#     `ster pairs synthesize --generator brama` and `ster decisions synthesize`;
#   * no `stado` at any place Stado says another program finds it (STADO_BIN,
#     else PATH, else ${STADO_BIN_DIR:-$HOME/.stado/bin}/stado), refused
#     (exit 1) naming the call it could not start and every place it looked;
#   * those places looked at in that order, an empty variable counting as
#     unset, shown by which `stado` a refusal names;
#   * a STADO_BIN naming nothing, a `stado` this account may not execute
#     and a `stado` whose interpreter is missing, each refused (exit 1).
#
# The files put where Stado's program would be are never run: each is
# refused by the system before anything in it executes.
#
# Nothing is written, and no model is fetched. Nothing answering is
# tests/synthesis/resolved.sh, which needs a real Stado.
#
# Usage: tests/synthesis/inference.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/synthesis/inference-$RUN"
mkdir -p "$ROOT/home" "$ROOT/empty"
REPORT="$ROOT/report.txt"
OUT="$ROOT/set.json"
export HOME="$ROOT/home" XDG_DATA_HOME="$ROOT/data" XDG_STATE_HOME="$ROOT/state"
unset STADO_BIN STADO_BIN_DIR
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
}
run() {
  expected=$1
  shift
  set +e
  out=$("$BIN" "$@" 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  # grep exits 1 when every line was a wait line; any other failure stops the run.
  err=$(grep -v '^czekam\|^koniec czekania' "$ROOT/stderr" || [ $? -eq 1 ])
  printf '$ ster %s\nexit: %s\nstdout: %s\nstderr: %s\n\n' "$*" "$status" "$out" "$err" >>"$REPORT"
  [ "$status" -eq "$expected" ] || fail "ster $* exited $status, expected $expected: $err"
  [ ! -e "$OUT" ] || fail "a refused run wrote $OUT"
}
said() {
  case "$err" in
    *"$1"*) ;;
    *) fail "expected '$1' in the refusal, got: $err" ;;
  esac
}
passed() { echo "ok: $1" >>"$REPORT"; }
SYNTH="pairs synthesize --generator brama --trait calm --count 1 --retry-multiplier 2 --dedupe-bits 3 --dedupe-bands 8 --max-new-tokens 16 --temperature 0.7 --top-p 0.9 --seed 1 --opposite tense --output $OUT"

a_blank_route_is_refused() {
  # shellcheck disable=SC2086
  run 2 $SYNTH --generator-model " "
  said "Error: choosing the route Ster asks Stado's inference alternatives for: the route is blank; observed: --generator-model \" \"; fix: name a Brama alias"
  run 2 decisions synthesize --schema "$ROOT/no-such-schema.json" --generator-model "" --per-option 1 \
    --retry-multiplier 2 --temperature 0.7 --max-tokens 16 --output "$OUT"
  said "the route is blank; observed: --generator-model \"\";"
  passed a_blank_route_is_refused
}

# Run the synthesis with the environment assignments given first, under
# this run's HOME, and keep its exit status and standard error.
synthesize() {
  set +e
  # shellcheck disable=SC2086
  out=$(env "$@" "$BIN" $SYNTH --generator-model wisent-backend/chat/primary 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  err=$(cat "$ROOT/stderr")
  printf '$ %s ster %s\nexit: %s\nstderr: %s\n\n' "$*" "$SYNTH" "$status" "$err" >>"$REPORT"
  [ "$status" -eq 1 ] || fail "ster with $* exited $status, expected 1: $err"
  [ ! -e "$OUT" ] || fail "a refused run wrote $OUT"
}

# A file named stado at `$1` that this account may not execute: whatever
# finds it is refused naming it, so the refusal shows which place won.
closed() {
  mkdir -p "$(dirname "$1")"
  : >"$1"
  chmod 644 "$1"
}

a_missing_stado_is_refused() {
  synthesize PATH="$ROOT/empty"
  said "Error: starting \`stado capabilities call inference --consumer ster --json\`: no \`stado\` is at any place Stado says another program finds it (\`stado --help\`)"
  said "; observed: STADO_BIN: unset; PATH: $ROOT/empty; $ROOT/home/.stado/bin/stado (from \$HOME/.stado/bin): absent; fix: install Stado with its install-stado.sh"
  said "or set STADO_BIN to the path of \`stado\`"
  # STADO_BIN_DIR, when set, takes the place of $HOME/.stado/bin.
  synthesize PATH="$ROOT/empty" STADO_BIN_DIR="$ROOT/empty"
  said "; observed: STADO_BIN: unset; PATH: $ROOT/empty; $ROOT/empty/stado (from STADO_BIN_DIR): absent; fix: "
  # An empty STADO_BIN or STADO_BIN_DIR counts as unset.
  synthesize PATH="$ROOT/empty" STADO_BIN= STADO_BIN_DIR=
  said "; observed: STADO_BIN: empty, which counts as unset; PATH: $ROOT/empty; $ROOT/home/.stado/bin/stado (from \$HOME/.stado/bin): absent; fix: "
  passed a_missing_stado_is_refused
}

a_stado_bin_naming_nothing_is_refused() {
  closed "$ROOT/order/path/stado"
  synthesize STADO_BIN="$ROOT/no-such-stado" PATH="$ROOT/order/path"
  said "Error: starting \`stado capabilities call inference --consumer ster --json\`: STADO_BIN names $ROOT/no-such-stado, where no file is, and STADO_BIN is the first place Stado says another program finds it"
  said "; observed: STADO_BIN: $ROOT/no-such-stado (absent); fix: set STADO_BIN to the path of the \`stado\` program"
  synthesize STADO_BIN="$ROOT/empty" PATH="$ROOT/order/path"
  said "observed: STADO_BIN: $ROOT/empty (a directory, "
  passed a_stado_bin_naming_nothing_is_refused
}

# STADO_BIN before PATH, PATH before ${STADO_BIN_DIR:-$HOME/.stado/bin}.
stado_is_looked_for_in_stados_order() {
  closed "$ROOT/order/named/stado"
  closed "$ROOT/order/path/stado"
  closed "$ROOT/order/dir/stado"
  closed "$ROOT/home/.stado/bin/stado"
  synthesize STADO_BIN="$ROOT/order/named/stado" PATH="$ROOT/order/path" STADO_BIN_DIR="$ROOT/order/dir"
  said "the \`stado\` Ster found through STADO_BIN may not be executed by this account; observed: $ROOT/order/named/stado (a file, "
  synthesize PATH="$ROOT/order/path" STADO_BIN_DIR="$ROOT/order/dir"
  said "the \`stado\` Ster found through PATH may not be executed by this account; observed: $ROOT/order/path/stado (a file, "
  synthesize PATH="$ROOT/empty" STADO_BIN_DIR="$ROOT/order/dir"
  said "the \`stado\` Ster found through STADO_BIN_DIR may not be executed by this account; observed: $ROOT/order/dir/stado (a file, "
  synthesize PATH="$ROOT/empty" STADO_BIN_DIR=
  said "the \`stado\` Ster found through \$HOME/.stado/bin may not be executed by this account; observed: $ROOT/home/.stado/bin/stado (a file, "
  rm "$ROOT/home/.stado/bin/stado"
  passed stado_is_looked_for_in_stados_order
}

# A script whose interpreter does not exist: the system answers "not found"
# for a file that is there.
a_stado_whose_interpreter_is_missing_is_refused() {
  mkdir -p "$ROOT/script-path"
  printf '#!%s/no-such-interpreter\n' "$ROOT" >"$ROOT/script-path/stado"
  chmod 755 "$ROOT/script-path/stado"
  synthesize PATH="$ROOT/script-path"
  said "Error: starting \`$ROOT/script-path/stado capabilities call inference --consumer ster --json\`: the system answered that the \`stado\` Ster found through PATH does not exist when Ster started it"
  said "; observed: $ROOT/script-path/stado (a file, "
  said "its interpreter: #!$ROOT/no-such-interpreter; the system's error kind: NotFound; it answered: "
  passed a_stado_whose_interpreter_is_missing_is_refused
}

# A file named stado on PATH that this account may not execute.
a_stado_this_account_may_not_execute_is_refused() {
  closed "$ROOT/closed-path/stado"
  synthesize PATH="$ROOT/closed-path"
  said "Error: starting \`$ROOT/closed-path/stado capabilities call inference --consumer ster --json\`: the \`stado\` Ster found through PATH may not be executed by this account"
  said "observed: $ROOT/closed-path/stado (a file, "
  said ", mode 644, owner uid $(id -u)); the system's error kind: PermissionDenied; it answered: "
  said "fix: make $ROOT/closed-path/stado executable for this account (\`chmod u+x $ROOT/closed-path/stado\` as its owner), or set STADO_BIN to the path of an executable \`stado\`"
  passed a_stado_this_account_may_not_execute_is_refused
}

a_blank_route_is_refused
a_missing_stado_is_refused
a_stado_bin_naming_nothing_is_refused
stado_is_looked_for_in_stados_order
a_stado_whose_interpreter_is_missing_is_refused
a_stado_this_account_may_not_execute_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
