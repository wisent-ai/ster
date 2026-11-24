#!/bin/sh
# Written by Stado's release pass for ster, whose release carries no
# Rust program (`stado product updater adopt ster` writes the journey
# of one that does).
#
# Staging journey of ster: the release staging just unpacked in this
# account ($HOME, the staging account) must hold every file its manifest
# stages ('bin/ster' 'release/ster-darwin-arm64.tar.gz' 'release/ster-darwin-arm64.tar.gz.sha256' 'release/ster-linux-amd64.tar.gz' 'release/ster-linux-amd64.tar.gz.sha256'), and each program it places ('ster') must start as a
# person's copy would: answering --version, or --help when it has no
# --version. Every test of the released source runs after it, in this
# account, with these programs first on PATH. It needs no registry, store or
# credentials.
set -eu

release="$HOME/.local/share/ster/release"
for file in 'bin/ster' 'release/ster-darwin-arm64.tar.gz' 'release/ster-darwin-arm64.tar.gz.sha256' 'release/ster-linux-amd64.tar.gz' 'release/ster-linux-amd64.tar.gz.sha256'; do
  if [ ! -e "$release/$file" ]; then
    printf 'the staged release of ster holds no %s, which its manifest stages (platforms.<platform>.stage of .wisent-release.json)\n' "$file" >&2
    exit 1
  fi
done

for program in 'ster'; do
  placed="$HOME/.local/bin/$program"
  if [ ! -x "$placed" ]; then
    printf 'staging placed no runnable %s; the release archive must hold bin/%s\n' "$placed" "$program" >&2
    exit 1
  fi
  # stderr: only whether the program answers --version is asked; one that does not is asked --help below
  if "$placed" --version >/dev/null 2>&1; then
    printf 'staged %s answers --version\n' "$program"
    continue
  fi
  # stderr: only whether the program starts is asked here; the line below says it does not, when it does not
  if ! "$placed" --help >/dev/null 2>&1; then
    printf 'the staged %s answers neither --version nor --help, so it does not start as a copy would\n' "$placed" >&2
    exit 1
  fi
  printf 'staged %s answers --help\n' "$program"
done
printf 'staged ster holds what its manifest stages\n'
