from __future__ import annotations

from dataclasses import dataclass

import pyarrow.parquet as pq


@dataclass(frozen=True, slots=True)
class Storage:
    fs: object

    def open(self, path: str, mode: str = "rb"):
        return self.fs.open(path, mode)

    def exists(self, path: str) -> bool:
        return self.fs.exists(path)

    def isdir(self, path: str) -> bool:
        return self.fs.isdir(path)

    def listdir(self, path: str) -> list[str]:
        xs = self.fs.ls(path, detail=False)
        return sorted(str(x).rstrip("/") for x in xs)

    def glob(self, pattern: str) -> list[str]:
        xs = self.fs.glob(pattern)
        return sorted(str(x).rstrip("/") for x in xs)

    def parquet_file(self, path: str) -> pq.ParquetFile:
        return pq.ParquetFile(self.open(path, "rb"))

    def read_table(self, path: str, **kwargs):
        with self.open(path, "rb") as f:
            return pq.read_table(f, **kwargs)

    def read_metadata(self, path: str):
        return self.parquet_file(path).metadata