#!/usr/bin/env bash
# The release build Stado runs from .wisent-release.json. It receives the
# source, output, version and platform through the WISENT_* variables the
# installer sets, builds the one binary, and leaves under $WISENT_OUTPUT_DIR
# exactly the members the manifest's stage map names: release/ster, the
# archive a person downloads, and its checksum.
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
# Stado hands a release build the kompilator it compiles with as KOMPILATOR.
kompilator=${KOMPILATOR:?KOMPILATOR is required: Stado names the kompilator a release build compiles with}
source_dir=${WISENT_SOURCE_DIR:?WISENT_SOURCE_DIR is required}
output_dir=${WISENT_OUTPUT_DIR:?WISENT_OUTPUT_DIR is required}
platform=${WISENT_PLATFORM:?WISENT_PLATFORM is required}
version=${WISENT_VERSION:?WISENT_VERSION is required}

# The GPU each platform's release computes on besides the CPU: Apple's Metal
# on darwin-arm64, where every Mac has it. linux-amd64 builds for the CPU
# only, so it starts on any Linux host; the two linux-amd64-cuda platforms
# add NVIDIA's CUDA, which links the CUDA runtime and so starts only where it
# is installed, each for the one compute capability its name says.
case "$platform" in
  darwin-arm64) expected_os=Darwin; expected_arch=arm64; features=(--features metal) ;;
  linux-amd64) expected_os=Linux; expected_arch=x86_64; features=() ;;
  linux-amd64-cuda-hopper) expected_os=Linux; expected_arch=x86_64; features=(--features cuda); compute_cap=90 ;;
  linux-amd64-cuda-blackwell) expected_os=Linux; expected_arch=x86_64; features=(--features cuda); compute_cap=120 ;;
  *) printf 'unsupported release platform: %s\n' "$platform" >&2; exit 64 ;;
esac
# fallback: compute_cap is set only by the two CUDA platforms above; unset
# means the platform builds no CUDA kernels.
if [[ -n "${compute_cap:-}" ]]; then
  # 65 is sysexits.h's EX_DATAERR, the status this script's other builder
  # refusals (a builder of another platform, a version mismatch) exit with:
  # https://man.freebsd.org/cgi/man.cgi?query=sysexits
  command -v nvcc >/dev/null || { printf '%s needs nvcc, the CUDA compiler, on PATH\n' "$platform" >&2; exit 65; }
  # Compute capability 9.0 is Hopper (H100, H200, GH200) and 12.0 the
  # Blackwell RTX parts (GeForce RTX 50 series, RTX PRO Blackwell); the
  # Blackwell data-centre parts are 10.0 (B200, GB200) and 10.3 (B300,
  # GB300), and GB10 (DGX Spark) is 12.1:
  # https://developer.nvidia.com/cuda/gpus
  # Candle's kernel build (cudaforge) takes one CUDA_COMPUTE_CAP and compiles
  # every kernel for its arch-specific target, compute_90a or compute_120a,
  # one PTX text per kernel module (nvcc refuses --ptx for several
  # architectures), so one binary cannot carry both. Arch-specific PTX is
  # "not forward or backward compatible": each build runs on its own
  # capability only.
  # https://docs.nvidia.com/cuda/blackwell-compatibility-guide/#application-compatibility-on-blackwell-architecture
  # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
  # The platform names the capability; a CUDA_COMPUTE_CAP in the builder's
  # environment is replaced by it.
  nvcc --list-gpu-arch | grep -x "compute_$compute_cap" >/dev/null || {
    printf '%s compiles its kernels for compute_%s, which nvcc %s does not list: install a CUDA toolkit whose nvcc --list-gpu-arch lists it\n' \
      "$platform" "$compute_cap" "$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')" >&2
    exit 65  # sysexits.h EX_DATAERR: https://man.freebsd.org/cgi/man.cgi?query=sysexits
  }
  export CUDA_COMPUTE_CAP=$compute_cap
  # nvcc accepts only the host compilers its release was tested with. The
  # first of these it preprocesses with is the one Candle's kernel build is
  # handed (cudaforge reads NVCC_CCBIN); a builder stating NVCC_CCBIN keeps it.
  # fallback: an unset NVCC_CCBIN means the builder named no host compiler,
  # so the first one nvcc accepts is chosen below.
  if [[ -z "${NVCC_CCBIN:-}" ]]; then
    for candidate in gcc gcc-14 gcc-13 gcc-12; do
      if command -v "$candidate" >/dev/null &&
        # stderr: nvcc refusing this host compiler is the answer asked for;
        # the next candidate is tried, and when none is accepted the refusal
        # below names nvcc's and gcc's versions.
        nvcc -ccbin "$candidate" -x cu -E /dev/null -o /dev/null 2>/dev/null; then
        NVCC_CCBIN=$(command -v "$candidate")
        export NVCC_CCBIN
        break
      fi
    done
  fi
  # fallback: NVCC_CCBIN still unset means nvcc accepted no candidate, which
  # is refused here.
  [[ -n "${NVCC_CCBIN:-}" ]] || {
    # fallback: a builder without gcc has no gcc version to name, so the
    # refusal says it is absent.
    gcc_version=absent
    if command -v gcc >/dev/null; then
      gcc_version=$(gcc -dumpfullversion)
    fi
    printf 'nvcc %s accepts none of gcc, gcc-14, gcc-13, gcc-12 on this builder (gcc is %s): install a host compiler it supports, or a CUDA toolkit that supports this gcc\n' \
      "$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')" "$gcc_version" >&2
    exit 65  # sysexits.h EX_DATAERR: https://man.freebsd.org/cgi/man.cgi?query=sysexits
  }
  printf '%s: compute capability %s, host compiler %s\n' "$platform" "$CUDA_COMPUTE_CAP" "$NVCC_CCBIN" >&2
fi
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
# kompilator reads the locked private Git package (the onboarding runtime
# the GitHub star request runs on) from the manifest's private-sources input,
# whose mount Stado names as WISENT_INPUT_PRIVATE_SOURCES_DIR:
# a worker holds no Git credential.
built=$("$kompilator" build --manifest-path "$source_dir/Cargo.toml" --release --bin ster ${features[@]+"${features[@]}"} --out "$build_root/release")

# The bare binary is what a stado-release install copies out of the stage
# map; the archive beside it is what a person downloads.
install -m 0755 "$built" "$release_dir/ster"
stage="$release_dir/ster-$platform"
archive="$release_dir/ster-$platform.tar.gz"
rm -rf "$stage" "$archive" "$archive.sha256"
mkdir -p "$stage"
install -m 0755 "$built" "$stage/ster"
install -m 0644 "$source_dir/LICENSE" "$stage/LICENSE"
install -m 0644 "$source_dir/README.md" "$stage/README.md"
tar -C "$release_dir" -czf "$archive" "ster-$platform"
rm -rf "$stage"
(
  cd "$release_dir"
  shasum --algorithm 256 "$(basename "$archive")" >"$(basename "$archive").sha256"
)
printf '%s\n' "$archive"
