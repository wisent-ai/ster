#!/bin/sh
# The product's own names, read from wisent-names.json at the checkout's top, for every
# shell test: PROGRAM is the command people type, and BIN the built binary a test runs
# (the one the variable <environment prefix>BIN names, else kompilator's debug build).
#
# Usage: . tests/support/names.sh   (from the checkout's top)
PROGRAM=$(jq -r '.own.program' wisent-names.json)
PRODUCT_VARIABLE="$(jq -r '.own.environment_prefix' wisent-names.json)BIN"
BIN=$(printenv "$PRODUCT_VARIABLE" || printf '%s' ".build/kompilator/debug/$PROGRAM")
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
