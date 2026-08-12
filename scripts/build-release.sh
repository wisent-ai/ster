#!/bin/sh
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
TARGET=$(rustc -vV | sed -n 's/^host: //p')
case "$TARGET" in
    aarch64-apple-darwin) PLATFORM=darwin-arm64 ;;
    x86_64-unknown-linux-gnu) PLATFORM=linux-amd64 ;;
    *) printf 'unsupported release target: %s\n' "$TARGET" >&2; exit 1 ;;
esac

cd "$ROOT"
cargo build --release --locked
BINARY="$ROOT/target/release/ster"
[ -x "$BINARY" ] || { printf 'cargo did not produce %s\n' "$BINARY" >&2; exit 1; }

OUTPUT="$ROOT/.wisent-output/release"
STAGE="$OUTPUT/ster-$PLATFORM"
ARCHIVE="$OUTPUT/ster-$PLATFORM.tar.gz"
rm -rf "$STAGE" "$ARCHIVE" "$ARCHIVE.sha256"
mkdir -p "$STAGE"
cp "$BINARY" "$ROOT/LICENSE" "$ROOT/README.md" "$STAGE/"
tar -C "$OUTPUT" -czf "$ARCHIVE" "ster-$PLATFORM"
rm -rf "$STAGE"
(
    cd "$OUTPUT"
    shasum --algorithm 256 "$(basename "$ARCHIVE")" >"$(basename "$ARCHIVE").sha256"
)
printf '%s\n' "$ARCHIVE"
