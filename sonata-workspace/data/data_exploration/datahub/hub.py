
from __future__ import annotations

from dataclasses import dataclass

import fsspec
from upath import UPath

from .model import Dataset
from .storage import Storage


@dataclass(frozen=True, slots=True)
class Hub:
    storage: Storage
    root: UPath

    def dataset(self, name: str) -> Dataset:
        return Dataset(self.storage, self.root / name, name)

    def list_datasets(self) -> list[str]:
        base = self.root
        if not self.storage.exists(base):
            return []
        out = []
        for p in self.storage.listdir(base):
            name = p.name
            if name and self.storage.isdir(p):
                out.append(name)
        return sorted(out)


def open_hub(
    *,
    root: str | UPath = "3d-scenes/datasets",
    endpoint_url: str = "https://storage.yandexcloud.net",
    cache_storage: str | None = None,
    target_protocol: str = "s3",
    **target_options,
) -> Hub:
    opts = {"endpoint_url": endpoint_url, **target_options}
    if cache_storage:
        fs = fsspec.filesystem(
            "filecache",
            target_protocol=target_protocol,
            target_options=opts,
            cache_storage=cache_storage,
        )
    else:
        fs = fsspec.filesystem(target_protocol, **opts)
    return Hub(Storage(fs), UPath(str(root).rstrip("/")))