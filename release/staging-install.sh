#!/bin/sh
# Written by Stado's release pass for ster, whose release carries no
# Rust program Stado's own installer places (`stado product updater adopt
# ster` writes that one instead).
#
# Staging installer of ster (the manifest's staging.install): unpacks
# the release archive the delivery worker names into the staging account
# ($HOME), at $HOME/.local/share/ster/release, and places everything the
# archive holds under bin/ (its programs and what they read beside them, a
# Swift resource bundle) in $HOME/.local/bin, where a person's copy runs them
# and where staging puts its tests' programs first on PATH. Each entry is
# placed through a sibling renamed over it, so a reader never runs half of
# one. A release with no bin/ (a site, a package) is unpacked and nothing
# else; staging then runs every test of its released source. It touches no
# unit and no service of the host.
set -eu

: "${WISENT_RELEASE_ARCHIVE:?the delivery worker names the release archive in WISENT_RELEASE_ARCHIVE; run this through stado release stage}"
: "${HOME:?HOME names no account to stage into}"

release="$HOME/.local/share/ster/release"
unpacked="$release.staging-$$"
rm -rf "$unpacked"
mkdir -p "$unpacked" "$HOME/.local/bin"
trap 'rm -rf "$unpacked"' EXIT
tar -xzf "$WISENT_RELEASE_ARCHIVE" -C "$unpacked"
rm -rf "$release"
mv "$unpacked" "$release"
trap - EXIT

placed=0
if [ -d "$release/bin" ]; then
  for entry in "$release/bin"/* "$release/bin"/.[!.]*; do
    [ -e "$entry" ] || continue
    name="$(basename "$entry")"
    target="$HOME/.local/bin/$name"
    rm -rf "$target.staging"
    cp -R "$entry" "$target.staging"
    rm -rf "$target"
    mv "$target.staging" "$target"
    placed=$((placed + 1))
  done
fi
printf 'staged ster unpacked in %s; %s entr(ies) of its bin/ placed in %s\n' "$release" "$placed" "$HOME/.local/bin"
