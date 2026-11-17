#!/bin/sh
# Real test of release/build.sh's CUDA platforms refusing a builder that names
# no host compiler: nvcc accepts only the host C++ compilers its release
# supports, so the builder names one in NVCC_CCBIN and the build chooses none
# for it. Without it each CUDA platform is refused before nvcc is looked for,
# with exit 65 (sysexits.h EX_DATAERR, as the script's other builder
# refusals) and nothing written under the output directory. It runs on any
# host: the refusal comes before the builder's platform is compared.
#
# Usage: tests/release/cuda.sh
set -eu
cd "$(dirname "$0")/../.."
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/release-cuda/$RUN"
mkdir -p "$ROOT"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1  # https://pubs.opengroup.org/onlinepubs/9799919799/functions/exit.html
}
passed() { echo "ok: $1" >>"$REPORT"; }

an_unnamed_host_compiler_is_refused() {
  for platform in linux-amd64-cuda-hopper linux-amd64-cuda-blackwell; do
    output="$ROOT/$platform"
    mkdir -p "$output"
    set +e
    err=$(env -u NVCC_CCBIN KOMPILATOR=kompilator WISENT_SOURCE_DIR="$PWD" WISENT_OUTPUT_DIR="$output" \
      WISENT_PLATFORM="$platform" WISENT_VERSION=0.0.0 bash release/build.sh 2>&1 >/dev/null)
    status=$?
    set -e
    printf '$ WISENT_PLATFORM=%s release/build.sh\nexit: %s\nstderr: %s\n\n' "$platform" "$status" "$err" >>"$REPORT"
    # sysexits.h EX_DATAERR: https://man.freebsd.org/cgi/man.cgi?query=sysexits
    [ "$status" -eq 65 ] || fail "$platform without NVCC_CCBIN exited $status, expected 65: $err"
    case "$err" in
      *"$platform needs NVCC_CCBIN, the host C++ compiler nvcc builds its kernels with"*) ;;
      *) fail "$platform without NVCC_CCBIN did not name it: $err" ;;
    esac
    [ -z "$(ls -A "$output")" ] || fail "$platform without NVCC_CCBIN wrote under $output"
  done
  passed an_unnamed_host_compiler_is_refused
}

an_unnamed_host_compiler_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
