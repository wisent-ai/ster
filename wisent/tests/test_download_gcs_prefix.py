import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
DOWNLOADER_PATH = ROOT / "scripts" / "steering" / "download_gcs_prefix.py"
SPEC = importlib.util.spec_from_file_location("download_gcs_prefix", DOWNLOADER_PATH)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def _blob(name, contents):
    def download_to_filename(target):
        Path(target).write_text(contents)

    return SimpleNamespace(name=name, download_to_filename=download_to_filename)


def test_download_prefix_preserves_nested_paths_and_returns_file_count(monkeypatch, tmp_path):
    blobs = [
        _blob("calibration/winogrande/", "directory marker must not be downloaded"),
        _blob("calibration/winogrande/manifest.json", "manifest"),
        _blob("calibration/winogrande/splits/train/data.jsonl", "training data"),
    ]
    list_requests = []

    class FakeClient:
        def list_blobs(self, bucket, *, prefix):
            list_requests.append((bucket, prefix))
            return blobs

    monkeypatch.setattr(downloader.storage, "Client", FakeClient)
    destination = tmp_path / "download"

    count = downloader.download_prefix("gs://calibration-data/calibration/winogrande/", destination)

    assert count == 2
    assert list_requests == [("calibration-data", "calibration/winogrande/")]
    assert (destination / "manifest.json").read_text() == "manifest"
    assert (destination / "splits" / "train" / "data.jsonl").read_text() == "training data"
    assert sorted(path.relative_to(destination) for path in destination.rglob("*") if path.is_file()) == [
        Path("manifest.json"),
        Path("splits/train/data.jsonl"),
    ]


def test_download_prefix_rejects_non_gs_uri(tmp_path):
    with pytest.raises(ValueError, match=r"^expected gs:// URI"):
        downloader.download_prefix("https://calibration-data/calibration/winogrande", tmp_path)


@pytest.mark.parametrize("uri", ["gs://calibration-data", "gs://calibration-data/"])
def test_download_prefix_refuses_bucket_root(uri, tmp_path):
    with pytest.raises(ValueError, match="refusing to download an entire bucket"):
        downloader.download_prefix(uri, tmp_path)


def test_download_prefix_reports_empty_prefix(monkeypatch, tmp_path):
    class FakeClient:
        def list_blobs(self, bucket, *, prefix):
            return []

    monkeypatch.setattr(downloader.storage, "Client", FakeClient)

    with pytest.raises(
        FileNotFoundError,
        match=r"^no objects found under gs://calibration-data/calibration/winogrande$",
    ):
        downloader.download_prefix("gs://calibration-data/calibration/winogrande", tmp_path)
