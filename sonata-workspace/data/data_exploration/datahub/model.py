from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from upath import UPath

from .storage import Storage

def _batch_name(i: int | str) -> str:
    return f"{int(i):06d}.parquet" if isinstance(i, int) else str(i)


def _frame_id(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("frame_id must be an integer")
    frame_id = int(value)
    if frame_id != value:
        raise ValueError("frame_id must be an integer")
    return frame_id


def _frame_filters(filters, frame_id: int):
    frame_filter = ("frame_id", "==", frame_id)
    if filters is None:
        return [frame_filter]
    if isinstance(filters, list):
        if not filters:
            return [frame_filter]
        first = filters[0]
        if isinstance(first, tuple):
            return [frame_filter, *filters]
        if isinstance(first, list):
            return [[frame_filter, *group] for group in filters]
    return [frame_filter, filters]


@dataclass(frozen=True, slots=True)
class _FrameIndex:
    frame_to_batch: dict[int, UPath]
    batch_to_frames: dict[UPath, list[int]]


@dataclass(frozen=True, slots=True)
class Dataset:
    storage: Storage
    path: UPath
    name: str

    def scene(self, scene_id: str) -> Scene:
        return Scene(self.storage, self.path / "scenes" / scene_id, scene_id, self)

    def list_scenes(self) -> list[str]:
        base = self.path / "scenes"
        if not self.storage.exists(base):
            return []
        return [p.name for p in self.storage.listdir(base) if self.storage.isdir(p)]

    def scenes(self) -> Iterator[Scene]:
        for scene_id in self.list_scenes():
            yield self.scene(scene_id)


@dataclass(frozen=True, slots=True)
class Scene:
    storage: Storage
    path: UPath
    scene_id: str
    dataset: Dataset

    @property
    def sequences_path(self) -> UPath:
        return self.path / "sequences"

    @property
    def maps_path(self) -> UPath:
        return self.path / "maps"

    def sequence(self, sequence_id: str) -> Sequence:
        return Sequence(self.storage, self.sequences_path / sequence_id, sequence_id, self)

    def list_sequences(self) -> list[str]:
        base = self.sequences_path
        if not self.storage.exists(base):
            return []
        return [p.name for p in self.storage.listdir(base) if self.storage.isdir(p)]

    def sequences(self) -> Iterator[Sequence]:
        for sequence_id in self.list_sequences():
            yield self.sequence(sequence_id)

    def map(self, name: str) -> Asset:
        return Asset(self.storage, self.maps_path / name, name, "map", self)

    def list_maps(self) -> list[str]:
        base = self.maps_path
        if not self.storage.exists(base):
            return []
        return [p.name for p in self.storage.listdir(base) if self.storage.isdir(p)]

    def maps(self) -> Iterator[Asset]:
        for name in self.list_maps():
            yield self.map(name)


@dataclass(frozen=True, slots=True)
class Sequence:
    storage: Storage
    path: UPath
    sequence_id: str
    scene: Scene

    def asset(self, name: str) -> Asset:
        return Asset(self.storage, self.path / name, name, "sequence", self)

    def list_assets(self) -> list[str]:
        if not self.storage.exists(self.path):
            return []
        return [p.name for p in self.storage.listdir(self.path) if self.storage.isdir(p)]

    def assets(self) -> Iterator[Asset]:
        for name in self.list_assets():
            yield self.asset(name)

    sensor = asset


@dataclass(frozen=True, slots=True)
class Asset:
    storage: Storage
    path: UPath
    name: str
    scope: str
    owner: Sequence | Scene
    _frame_index: _FrameIndex | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def batches_path(self) -> UPath:
        return self.path / "batches"

    def batch(self, i: int | str) -> Batch:
        return Batch(self.storage, self.batches_path / _batch_name(i), self)

    def list_batches(self) -> list[str]:
        base = self.batches_path
        if not self.storage.exists(base):
            return []
        xs = self.storage.glob(base / "*.parquet")
        return [x.name for x in xs]

    def iter_batches(self) -> Iterator[Batch]:
        for name in self.list_batches():
            yield self.batch(name)

    def frame(self, frame_id: int) -> FrameRef:
        frame_id, batch_path = self._resolve_frame(frame_id)
        return FrameRef(self, frame_id, batch_path)

    def frames(self) -> Iterator[FrameRef]:
        index = self._get_frame_index()
        for frame_id in sorted(index.frame_to_batch):
            yield FrameRef(self, frame_id, index.frame_to_batch[frame_id])

    def batch_for_frame(self, frame_id: int) -> Batch:
        _, batch_path = self._resolve_frame(frame_id)
        return Batch(self.storage, batch_path, self)

    def _resolve_frame(self, frame_id: int) -> tuple[int, UPath]:
        frame_id = _frame_id(frame_id)
        batch_path = self._get_frame_index().frame_to_batch.get(frame_id)
        if batch_path is None:
            raise KeyError(f"frame {frame_id} not found in asset {self.name!r}")
        return frame_id, batch_path

    def _get_frame_index(self) -> _FrameIndex:
        if self.scope != "sequence":
            raise TypeError("frames are only available for sequence assets")
        index = self._frame_index
        if index is None:
            index = self._build_frame_index()
            object.__setattr__(self, "_frame_index", index)
        return index

    def _build_frame_index(self) -> _FrameIndex:
        frame_to_batch: dict[int, UPath] = {}
        batch_to_frames: dict[UPath, list[int]] = {}
        for batch in self.iter_batches():
            frame_ids = batch._read_frame_ids()
            batch_to_frames[batch.path] = frame_ids
            for frame_id in frame_ids:
                other = frame_to_batch.get(frame_id)
                if other is not None and other != batch.path:
                    raise ValueError(
                        f"frame {frame_id} is split across batches {other!r} and {batch.path!r}"
                    )
                frame_to_batch[frame_id] = batch.path
        return _FrameIndex(frame_to_batch=frame_to_batch, batch_to_frames=batch_to_frames)

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
class FrameRef:
    asset: Asset
    frame_id: int
    batch_path: UPath

    @property
    def storage(self) -> Storage:
        return self.asset.storage

    @property
    def batch(self) -> Batch:
        return Batch(self.storage, self.batch_path, self.asset)

    def read_table(self, **kwargs):
        filters = _frame_filters(kwargs.pop("filters", None), self.frame_id)
        return self.storage.read_table(self.batch_path, filters=filters, **kwargs)

    def read_pandas(self, **kwargs):
        return self.read_table(**kwargs).to_pandas()


@dataclass(frozen=True, slots=True)
class Batch:
    storage: Storage
    path: UPath
    asset: Asset
    _metadata: object | None = field(default=None, init=False, repr=False, compare=False)
    _schema: object | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def metadata(self):
        metadata = self._metadata
        if metadata is None:
            metadata = self.storage.read_metadata(self.path)
            object.__setattr__(self, "_metadata", metadata)
        return metadata

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def num_rows(self) -> int:
        return self.metadata.num_rows

    @property
    def num_row_groups(self) -> int:
        return self.metadata.num_row_groups

    @property
    def schema(self):
        schema = self._schema
        if schema is None:
            schema = self.storage.parquet_file(self.path).schema_arrow
            object.__setattr__(self, "_schema", schema)
        return schema

    def read_table(self, **kwargs):
        return self.storage.read_table(self.path, **kwargs)

    def read_pandas(self, **kwargs):
        return self.read_table(**kwargs).to_pandas()

    def read_columns(self, columns: list[str]):
        return self.read_table(columns=columns)

    def list_frames(self) -> list[int]:
        return list(self.asset._get_frame_index().batch_to_frames.get(self.path, []))

    def _read_frame_ids(self) -> list[int]:
        try:
            table = self.read_table(columns=["frame_id"])
        except Exception as exc:
            raise ValueError(
                f"sequence asset {self.asset.name!r} is missing 'frame_id' in batch {self.name!r}"
            ) from exc
        seen: set[int] = set()
        for value in table.column("frame_id").to_pylist():
            if value is None:
                raise ValueError(f"batch {self.name!r} contains null frame_id")
            seen.add(_frame_id(value))
        return sorted(seen)

    def exists(self) -> bool:
        return self.storage.exists(self.path)

    def __repr__(self) -> str:
        return f"Batch(path={str(self.path)!r}, rows={self.num_rows})"