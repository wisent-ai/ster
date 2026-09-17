#!/usr/bin/env bash
# The release build Stado runs from .wisent-release.json. It receives the
# source, output, version and platform through the WISENT_* variables the
# installer sets, builds the one binary, and leaves under $WISENT_OUTPUT_DIR
# exactly the members the manifest's stage map names: release/ster, the
# archive a person downloads, and its checksum.
set -euo pipefail

export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
source_dir=${WISENT_SOURCE_DIR:?WISENT_SOURCE_DIR is required}
output_dir=${WISENT_OUTPUT_DIR:?WISENT_OUTPUT_DIR is required}
platform=${WISENT_PLATFORM:?WISENT_PLATFORM is required}
version=${WISENT_VERSION:?WISENT_VERSION is required}

case "$platform" in
  darwin-arm64) expected_os=Darwin; expected_arch=arm64 ;;
  linux-amd64) expected_os=Linux; expected_arch=x86_64 ;;
  *) printf 'unsupported release platform: %s\n' "$platform" >&2; exit 64 ;;
esac
actual_os=$(uname -s)
actual_arch=$(uname -m)
if [[ "$actual_os" != "$expected_os" || "$actual_arch" != "$expected_arch" ]]; then
  printf 'builder %s/%s cannot produce %s\n' "$actual_os" "$actual_arch" "$platform" >&2
  exit 65
fi
declared=$(sed -n 's/^version = "\([^"]*\)"$/\1/p' "$source_dir/Cargo.toml" | sed -n '1p')
if [[ "$declared" != "$version" ]]; then
  printf 'WISENT_VERSION %s does not match Cargo.toml version %s\n' "$version" "$declared" >&2
  exit 65
fi

build_root="$output_dir/.build"
release_dir="$output_dir/release"
mkdir -p "$build_root" "$release_dir"
CARGO_TARGET_DIR="$build_root" cargo build --locked --release --bin ster --manifest-path "$source_dir/Cargo.toml"

# The bare binary is what a stado-release install copies out of the stage
# map; the archive beside it is what a person downloads.
install -m 0755 "$build_root/release/ster" "$release_dir/ster"
stage="$release_dir/ster-$platform"
archive="$release_dir/ster-$platform.tar.gz"
rm -rf "$stage" "$archive" "$archive.sha256"
mkdir -p "$stage"
install -m 0755 "$build_root/release/ster" "$stage/ster"
install -m 0644 "$source_dir/LICENSE" "$stage/LICENSE"
install -m 0644 "$source_dir/README.md" "$stage/README.md"
tar -C "$release_dir" -czf "$archive" "ster-$platform"
rm -rf "$stage"
(
  cd "$release_dir"
  shasum --algorithm 256 "$(basename "$archive")" >"$(basename "$archive").sha256"
)
printf '%s\n' "$archive"
