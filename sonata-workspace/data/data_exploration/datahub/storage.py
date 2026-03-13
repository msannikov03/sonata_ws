from __future__ import annotations

from dataclasses import dataclass

import pyarrow.fs as pafs
import pyarrow.parquet as pq
from upath import UPath


@dataclass(frozen=True, slots=True)
class Storage:
    fs: object

    @property
    def arrow_fs(self):
        return pafs.PyFileSystem(pafs.FSSpecHandler(self.fs))

    def open(self, path: UPath, mode: str = "rb"):
        return self.fs.open(str(path), mode)

    def exists(self, path: UPath) -> bool:
        return self.fs.exists(str(path))

    def isdir(self, path: UPath) -> bool:
        return self.fs.isdir(str(path))

    def listdir(self, path: UPath) -> list[UPath]:
        xs = self.fs.ls(str(path), detail=False)
        return [UPath(str(x).rstrip("/")) for x in sorted(xs, key=str)]

    def glob(self, pattern: UPath) -> list[UPath]:
        xs = self.fs.glob(str(pattern))
        return [UPath(str(x).rstrip("/")) for x in sorted(xs, key=str)]

    def parquet_file(self, path: UPath) -> pq.ParquetFile:
        return pq.ParquetFile(str(path), filesystem=self.arrow_fs)

    def read_table(self, path: UPath, **kwargs):
        return pq.read_table(str(path), filesystem=self.arrow_fs, **kwargs)

    def read_metadata(self, path: UPath):
        return self.parquet_file(path).metadata