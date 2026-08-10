from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

EPOCH = 946684800


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required")
    return value


def source_dir() -> Path:
    path = Path(required("WISENT_SOURCE_DIR")).resolve()
    if not path.is_dir():
        raise SystemExit(f"WISENT_SOURCE_DIR is not a directory: {path}")
    return path


def output_dir() -> Path:
    path = Path(required("WISENT_OUTPUT_DIR")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def declared_version(root: Path) -> str:
    package = (root / "wisent" / "__init__.py").read_text(encoding="utf-8")
    matches = re.findall(r'^__version__\s*=\s*["\']([^"\']+)["\']', package, re.MULTILINE)
    if len(matches) != 1:
        raise SystemExit("wisent/__init__.py must contain exactly one literal version")
    return matches[0]


def quality() -> None:
    version = required("WISENT_VERSION")
    if declared_version(source_dir()) != version:
        raise SystemExit("WISENT_VERSION does not match wisent/__init__.py")
    required("WISENT_PLATFORM")
    required("WISENT_INPUTS_DIR")


def normalize_wheel(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        entries = [(item.filename, archive.read(item), item.external_attr) for item in archive.infolist()]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data, external_attr in sorted(entries):
            info = zipfile.ZipInfo(name, (2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = external_attr or (0o100644 << 16)
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def normalize_sdist(source: Path, destination: Path) -> None:
    import io

    with tarfile.open(source, "r:gz") as archive:
        members = [(member, archive.extractfile(member).read() if member.isfile() else None) for member in archive.getmembers()]
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for member, data in sorted(members, key=lambda item: item[0].name):
                    member.uid = member.gid = 0
                    member.uname = member.gname = ""
                    member.mtime = EPOCH
                    archive.addfile(member, io.BytesIO(data) if data is not None else None)


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    import io

    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = EPOCH
    archive.addfile(info, io.BytesIO(data))


def build() -> None:
    quality()
    root = source_dir()
    out = output_dir() / "release"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(dir=output_dir(), prefix="python-build-") as temporary:
        raw = Path(temporary) / "raw"
        normalized = Path(temporary) / "normalized"
        raw.mkdir()
        normalized.mkdir()
        environment = os.environ.copy()
        environment.update({"SOURCE_DATE_EPOCH": str(EPOCH), "PYTHONHASHSEED": "0"})
        subprocess.run(
            [sys.executable, "-m", "build", "--no-isolation", "--sdist", "--wheel", "--outdir", str(raw)],
            cwd=root,
            env=environment,
            check=True,
        )
        wheels = sorted(raw.glob("*.whl"))
        sdists = sorted(raw.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise SystemExit("build must produce exactly one wheel and one sdist")
        version = required("WISENT_VERSION")
        if version not in wheels[0].name or version not in sdists[0].name:
            raise SystemExit("distribution filenames do not contain WISENT_VERSION")
        wheel = normalized / wheels[0].name
        sdist = normalized / sdists[0].name
        normalize_wheel(wheels[0], wheel)
        normalize_sdist(sdists[0], sdist)
        with tarfile.open(out / "python-distributions.tar", "w", format=tarfile.PAX_FORMAT) as archive:
            for artifact in sorted((wheel, sdist), key=lambda path: path.name):
                add_bytes(archive, f"distributions/{artifact.name}", artifact.read_bytes())


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise SystemExit(f"unsafe archive member: {member.name}")
    archive.extractall(destination)


def find_distribution_bundle(release_archive: Path, destination: Path) -> Path:
    if release_archive.name == "python-distributions.tar":
        return release_archive
    destination.mkdir(parents=True)
    if tarfile.is_tarfile(release_archive):
        with tarfile.open(release_archive, "r:*") as archive:
            safe_extract(archive, destination)
    elif zipfile.is_zipfile(release_archive):
        with zipfile.ZipFile(release_archive) as archive:
            root = destination.resolve()
            for name in archive.namelist():
                target = (destination / name).resolve()
                if target != root and root not in target.parents:
                    raise SystemExit(f"unsafe archive member: {name}")
            archive.extractall(destination)
    else:
        raise SystemExit("WISENT_RELEASE_ARCHIVE is not a supported archive")
    matches = list(destination.rglob("python-distributions.tar"))
    if len(matches) != 1:
        raise SystemExit("canonical release must contain one python-distributions.tar")
    return matches[0]


def deliver_pypi() -> None:
    token = required("PYPI_TOKEN")
    release_archive = Path(required("WISENT_RELEASE_ARCHIVE")).resolve()
    if not release_archive.is_file():
        raise SystemExit("WISENT_RELEASE_ARCHIVE does not exist")
    with tempfile.TemporaryDirectory(prefix="wisent-pypi-") as temporary:
        temporary_path = Path(temporary)
        bundle = find_distribution_bundle(release_archive, temporary_path / "release")
        distributions = temporary_path / "distributions"
        distributions.mkdir()
        with tarfile.open(bundle, "r:") as archive:
            safe_extract(archive, distributions)
        files = sorted(path for path in distributions.rglob("*") if path.suffix == ".whl" or path.name.endswith(".tar.gz"))
        if len(files) != 2:
            raise SystemExit("canonical bundle must contain exactly one wheel and one sdist")
        environment = os.environ.copy()
        environment.update({"TWINE_USERNAME": "__token__", "TWINE_PASSWORD": token})
        subprocess.run(
            [
                sys.executable,
                "-m",
                "twine",
                "upload",
                "--non-interactive",
                "--repository-url",
                "https://upload.pypi.org/legacy/",
                *map(str, files),
            ],
            env=environment,
            check=True,
        )
        evidence = {
            "provider": "pypi",
            "product": required("WISENT_PRODUCT"),
            "version": required("WISENT_VERSION"),
            "release_uri": required("WISENT_RELEASE_URI"),
            "release_sha256": required("WISENT_RELEASE_SHA256"),
            "files": [path.name for path in files],
        }
        (output_dir() / "pypi-evidence.json").write_text(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("quality", "build", "deliver-pypi"))
    args = parser.parse_args()
    {"quality": quality, "build": build, "deliver-pypi": deliver_pypi}[args.command]()


if __name__ == "__main__":
    main()
