#!/usr/bin/env python3
"""Print the bump level for the next release: "major", "minor" or "patch".

The decision is NOT made here. It is made by the rule shared across every Wisent
product, https://github.com/lbartoszcze/AutoVersion, and this script supplies the two
things only this repository knows: the public surface it declares, and the version it
is at.

What changed, and why
---------------------

This script used to classify releases itself, and it disagreed with the rule the rest
of the fleet uses in two ways that produced wrong numbers:

- `minor` was triggered by *a new .py file appearing*. That is a proxy, not a
  contract: adding an internal helper bumped minor while nothing a caller can see had
  changed, and widening `__all__` without adding a file did not bump it at all. The
  trigger is now the public surface growing.

- a removed name advanced `major`. While the major slot is zero a product has no
  stable contract, so the minor slot carries the compatibility boundary; advancing to
  one is a deliberate declaration of stability, not a side effect of the first break.
  That is now written down in `product-guidelines/release-and-versioning-guidelines.md`.

The output contract is unchanged, so `publish_wisent.sh` needs no edit: the level
printed is whichever slot differs between the current version and the version the
shared rule derives. The shell then reproduces exactly that version.

`--diff-summary` is still accepted because the caller still passes it, and ignored
because file additions no longer decide anything. Removing it from the caller is a
separate change.

Args:
    --old-init <path>      wisent/__init__.py as released
    --current-init <path>  wisent/__init__.py proposed
    --diff-summary <path>  accepted and ignored

Output:
    one of "major", "minor", "patch"
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

try:
    from autoversion import decide
    from autoversion.rule import Version
    from autoversion.surfaces import python_all
except ImportError:  # pragma: no cover - the message is the whole point
    sys.exit(
        "the shared versioning rule is not installed. Install it with:\n"
        '  pip install "git+https://github.com/lbartoszcze/AutoVersion@v0.1.0"\n'
        "It is deliberately not vendored here: a copied rule drifts silently, and "
        "this repository is where that already happened once."
    )


def declared_version(init_path: str) -> str:
    """`__version__` from a module, read without importing it."""
    source = pathlib.Path(init_path)
    tree = ast.parse(source.read_text(), filename=str(source))
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if target.id == "__version__" and isinstance(value, ast.Constant):
                return str(value.value)
    sys.exit(f"{source}: no __version__ found")


def moved_slot(current: Version, following: Version) -> str:
    """Which slot the shared rule advanced, as the word the caller expects."""
    if following.major != current.major:
        return "major"
    if following.minor != current.minor:
        return "minor"
    return "patch"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the next bump level.")
    parser.add_argument("--old-init", required=True)
    parser.add_argument("--current-init", required=True)
    parser.add_argument("--diff-summary", required=False, help="accepted and ignored")
    args = parser.parse_args()

    current = declared_version(args.current_init)
    answer = decide(
        current,
        published=python_all(args.old_init),
        candidate=python_all(args.current_init),
    )
    print(moved_slot(Version.parse(answer["current"]), Version.parse(answer["next"])))


if __name__ == "__main__":
    main()
