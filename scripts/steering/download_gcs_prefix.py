#!/usr/bin/env python3
"""Download a GCS prefix with Application Default Credentials."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage


def download_prefix(uri: str, destination: Path) -> int:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"expected gs:// URI, got {uri!r}")

    prefix = parsed.path.lstrip("/").rstrip("/")
    if not prefix:
        raise ValueError("refusing to download an entire bucket")

    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for blob in storage.Client().list_blobs(parsed.netloc, prefix=f"{prefix}/"):
        relative = blob.name[len(prefix) :].lstrip("/")
        if not relative or blob.name.endswith("/"):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(target)
        count += 1

    if count == 0:
        raise FileNotFoundError(f"no objects found under {uri}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uri")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = download_prefix(args.uri, args.destination)
    print(f"downloaded {count} objects to {args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
