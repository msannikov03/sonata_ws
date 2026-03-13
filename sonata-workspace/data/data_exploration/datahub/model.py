from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterator

from .storage import Storage


def _join(*xs: str) -> str:
    return "/".join(x.strip("/") for x in xs if x and x.strip("/"))


def _name(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1]


def _batch_name(i: int | str) -> str:
    return f"{int(i):06d}.parquet" if isinstance(i, int) else str(i)


@dataclass(frozen=True, slots=True)
class Dataset:
    storage: Storage
    path: str
    name: str

    def scene(self, scene_id: str) -> Scene:
        return Scene(self.storage, _join(self.path, "scenes", scene_id), scene_id, self)

    def list_scenes(self) -> list[str]:
        base = _join(self.path, "scenes")
        if not self.storage.exists(base):
            return []
        return [_name(p) for p in self.storage.listdir(base) if self.storage.isdir(p)]

    def scenes(self) -> Iterator[Scene]:
        for scene_id in self.list_scenes():
            yield self.scene(scene_id)


@dataclass(frozen=True, slots=True)
class Scene:
    storage: Storage
    path: str
    scene_id: str
    dataset: Dataset

    @property
    def sequences_path(self) -> str:
        return _join(self.path, "sequences")

    @property
    def maps_path(self) -> str:
        return _join(self.path, "maps")

    def sequence(self, sequence_id: str) -> Sequence:
        return Sequence(self.storage, _join(self.sequences_path, sequence_id), sequence_id, self)

    def list_sequences(self) -> list[str]:
        base = self.sequences_path
        if not self.storage.exists(base):
            return []
        return [_name(p) for p in self.storage.listdir(base) if self.storage.isdir(p)]

    def sequences(self) -> Iterator[Sequence]:
        for sequence_id in self.list_sequences():
            yield self.sequence(sequence_id)

    def map(self, name: str) -> Asset:
        return Asset(self.storage, _join(self.maps_path, name), name, "map", self)

    def list_maps(self) -> list[str]:
        base = self.maps_path
        if not self.storage.exists(base):
            return []
        return [_name(p) for p in self.storage.listdir(base) if self.storage.isdir(p)]

    def maps(self) -> Iterator[Asset]:
        for name in self.list_maps():
            yield self.map(name)


@dataclass(frozen=True, slots=True)
class Sequence:
    storage: Storage
    path: str
    sequence_id: str
    scene: Scene

    def asset(self, name: str) -> Asset:
        return Asset(self.storage, _join(self.path, name), name, "sequence", self)

    def list_assets(self) -> list[str]:
        if not self.storage.exists(self.path):
            return []
        return [_name(p) for p in self.storage.listdir(self.path) if self.storage.isdir(p)]

    def assets(self) -> Iterator[Asset]:
        for name in self.list_assets():
            yield self.asset(name)

    sensor = asset


@dataclass(frozen=True, slots=True)
class Asset:
    storage: Storage
    path: str
    name: str
    scope: str
    owner: Sequence | Scene

    @property
    def batches_path(self) -> str:
        return _join(self.path, "batches")

    def batch(self, i: int | str) -> Batch:
        return Batch(self.storage, _join(self.batches_path, _batch_name(i)), self)

    def list_batches(self) -> list[str]:
        base = self.batches_path
        if not self.storage.exists(base):
            return []
        xs = self.storage.glob(_join(base, "*.parquet"))
        return [_name(x) for x in xs]

    def iter_batches(self) -> Iterator[Batch]:
        for name in self.list_batches():
            yield self.batch(name)

    def __getitem__(self, i: int | str) -> Batch:
        return self.batch(i)

    def head(self, n: int = 5):
        out = []
        for i, batch in enumerate(self.iter_batches()):
            if i >= n:
                break
            out.append(batch)
        return out


@dataclass(frozen=True, slots=True)
class Batch:
    storage: Storage
    path: str
    asset: Asset

    @cached_property
    def metadata(self):
        return self.storage.read_metadata(self.path)

    @property
    def name(self) -> str:
        return _name(self.path)

    @property
    def num_rows(self) -> int:
        return self.metadata.num_rows

    @property
    def num_row_groups(self) -> int:
        return self.metadata.num_row_groups

    @property
    def schema(self):
        return self.storage.parquet_file(self.path).schema_arrow

    def read_table(self, **kwargs):
        return self.storage.read_table(self.path, **kwargs)

    def read_pandas(self, **kwargs):
        return self.read_table(**kwargs).to_pandas()

    def read_columns(self, columns: list[str]):
        return self.read_table(columns=columns)

    def exists(self) -> bool:
        return self.storage.exists(self.path)

    def __repr__(self) -> str:
        return f"Batch(path={self.path!r}, rows={self.num_rows})"