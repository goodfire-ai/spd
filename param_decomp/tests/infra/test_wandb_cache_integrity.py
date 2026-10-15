"""A cached W&B file must match the remote size before it is trusted."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from param_decomp.infra import wandb as wandb_module
from param_decomp.infra.wandb import download_wandb_file


@dataclass
class _FakeFile:
    """Stands in for `wandb.apis.public.File`; records download calls."""

    name: str
    size: int | None
    content: bytes
    calls: list[dict[str, Any]] = field(default_factory=list)

    def download(self, *, root: str, replace: bool, exist_ok: bool) -> None:
        self.calls.append({"root": root, "replace": replace, "exist_ok": exist_ok})
        target = Path(root) / self.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.content)


@dataclass
class _FakeRun:
    file_obj: _FakeFile

    def file(self, name: str) -> _FakeFile:
        assert name == self.file_obj.name
        return self.file_obj


@pytest.fixture
def fake_file_cls(monkeypatch: pytest.MonkeyPatch) -> type[_FakeFile]:
    monkeypatch.setattr(wandb_module, "File", _FakeFile)
    return _FakeFile


def test_matching_size_is_trusted(fake_file_cls: type[_FakeFile], tmp_path: Path) -> None:
    content = b"checkpoint-bytes"
    remote = fake_file_cls(name="model_10.pth", size=len(content), content=content)
    (tmp_path / "model_10.pth").write_bytes(content)
    got = download_wandb_file(_FakeRun(remote), tmp_path, "model_10.pth")  # pyright: ignore[reportArgumentType]
    assert got == tmp_path / "model_10.pth"
    assert remote.calls == []  # cache hit, no download


def test_truncated_cache_is_redownloaded(fake_file_cls: type[_FakeFile], tmp_path: Path) -> None:
    content = b"checkpoint-bytes-that-got-cut"
    remote = fake_file_cls(name="model_10.pth", size=len(content), content=content)
    (tmp_path / "model_10.pth").write_bytes(content[:10])  # partial prior download
    got = download_wandb_file(_FakeRun(remote), tmp_path, "model_10.pth")  # pyright: ignore[reportArgumentType]
    assert got.read_bytes() == content
    assert remote.calls == [{"root": str(tmp_path), "replace": True, "exist_ok": True}]


def test_unknown_remote_size_falls_back_to_plain_download(
    fake_file_cls: type[_FakeFile], tmp_path: Path
) -> None:
    content = b"x"
    remote = fake_file_cls(name="cfg.yaml", size=None, content=content)
    download_wandb_file(_FakeRun(remote), tmp_path, "cfg.yaml")  # pyright: ignore[reportArgumentType]
    assert remote.calls == [{"root": str(tmp_path), "replace": False, "exist_ok": True}]
    assert (tmp_path / "cfg.yaml").read_bytes() == content


def test_cache_miss_downloads(fake_file_cls: type[_FakeFile], tmp_path: Path) -> None:
    content = b"fresh"
    remote = fake_file_cls(name="model_10.pth", size=len(content), content=content)
    download_wandb_file(_FakeRun(remote), tmp_path, "model_10.pth")  # pyright: ignore[reportArgumentType]
    assert remote.calls == [{"root": str(tmp_path), "replace": False, "exist_ok": True}]
    assert (tmp_path / "model_10.pth").read_bytes() == content
