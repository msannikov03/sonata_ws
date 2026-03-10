import shutil
import subprocess
from pathlib import Path
import os

import numpy as np

S3_ENDPOINT = "https://storage.yandexcloud.net"


def aws_s3(*args, capture_output=True):
    command = [str(x) for x in args]
    extra = []
    if command and command[0] in {"cp", "sync", "mv", "rm"}:
        extra += ["--no-progress", "--only-show-errors"]
    result = subprocess.run(
        [
            "aws",
            "s3",
            *command,
            *extra,
            "--endpoint-url",
            S3_ENDPOINT,
        ],
        capture_output=capture_output,
        text=True,
        check=True,
    )
    return result.stdout if capture_output else ""


def aws_ls(path):
    return aws_s3("ls", path).splitlines()


def aws_cp(src, dst):
    aws_s3("cp", src, dst, capture_output=False)


def aws_sync(src, dst, delete=False):
    args = ["sync", src, dst]
    if delete:
        args.append("--delete")
    aws_s3(*args, capture_output=False)


def list_batches(bucket_root, scene_id, map_id="lidar_pointcloud"):
    path = f"{bucket_root}/datasets/semantic_kitti/scenes/{scene_id}/maps/{map_id}/batches/"
    files = []
    for line in aws_ls(path):
        parts = line.split()
        if len(parts) == 4:
            files.append(parts[3])
    return sorted(files)


def _column(table, name, dtype=None):
    values = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    return values.astype(dtype, copy=False) if dtype is not None else values


def _load_batch(batch_path, only_static):
    import pyarrow.parquet as pq

    table = pq.read_table(batch_path)
    if not {"x", "y", "z"} <= set(table.column_names):
        raise ValueError(f"Batch {batch_path} has no xyz columns")

    xyz = np.column_stack(
        [
            _column(table, "x", np.float64),
            _column(table, "y", np.float64),
            _column(table, "z", np.float64),
        ]
    )
    intensity = _column(table, "intensity", np.float32) if "intensity" in table.column_names else None
    semantic = _column(table, "semantic_id", np.int32) if "semantic_id" in table.column_names else None

    if only_static and "is_static" in table.column_names:
        mask = _column(table, "is_static", bool)
        xyz = xyz[mask]
        if intensity is not None:
            intensity = intensity[mask]
        if semantic is not None:
            semantic = semantic[mask]

    return xyz, intensity, semantic


def _scale_intensity(intensity, size):
    if intensity is None:
        return np.zeros(size, dtype=np.uint16)
    values = np.nan_to_num(intensity.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    hi = float(values.max()) if len(values) else 0.0
    if hi > 0:
        values = values / hi
    return np.round(np.clip(values, 0.0, 1.0) * 65535.0).astype(np.uint16)


def _semantic_rgb(semantic, intensity, size):
    if semantic is None:
        gray = _scale_intensity(intensity, size)
        return gray, gray, gray

    labels = semantic.astype(np.uint32, copy=False)
    red = ((labels * 47 + 29) & 255).astype(np.uint16) * 257
    green = ((labels * 79 + 71) & 255).astype(np.uint16) * 257
    blue = ((labels * 131 + 113) & 255).astype(np.uint16) * 257
    zero = labels == 0
    if np.any(zero):
        red[zero] = 128 * 257
        green[zero] = 128 * 257
        blue[zero] = 128 * 257
    return red, green, blue


def _classification(semantic, size):
    if semantic is None:
        return np.ones(size, dtype=np.uint8)
    return np.clip(semantic, 0, 255).astype(np.uint8, copy=False)


def _write_las(batch_path, las_path, only_static):
    import laspy

    xyz, intensity, semantic = _load_batch(batch_path, only_static=only_static)
    if not len(xyz):
        return False

    las = laspy.create(file_version="1.4", point_format=7)
    las.header.offsets = xyz.min(axis=0)
    las.header.scales = np.array([0.001, 0.001, 0.001])
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.intensity = _scale_intensity(intensity, len(xyz))
    las.red, las.green, las.blue = _semantic_rgb(semantic, intensity, len(xyz))
    las.classification = _classification(semantic, len(xyz))
    las.write(las_path)
    return True


def build_viewer_tiles(
    local_root,
    bucket_root,
    scene_id,
    *,
    map_id="lidar_pointcloud",
    only_static=True,
    cleanup=True,
    jobs=None,
    cache_size_mb=1024,
    disable_processpool=False,
):
    if shutil.which("py3dtiles") is None:
        raise RuntimeError("py3dtiles is not installed")

    root = Path(local_root).resolve()
    work_dir = root / "viewer_tiles" / scene_id / map_id
    batches_dir = work_dir / "batches"
    las_dir = work_dir / "las"
    tiles_dir = work_dir / "tiles"

    shutil.rmtree(work_dir, ignore_errors=True)
    batches_dir.mkdir(parents=True, exist_ok=True)
    las_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    remote_batches_root = f"{bucket_root}/datasets/semantic_kitti/scenes/{scene_id}/maps/{map_id}/batches"
    remote_tiles_root = f"{bucket_root}/datasets/semantic_kitti/scenes/{scene_id}/maps/{map_id}/tiles"

    batch_names = list_batches(bucket_root, scene_id, map_id=map_id)
    if not batch_names:
        raise RuntimeError(f"No batches found for scene {scene_id} map {map_id}")

    las_files = []
    for name in batch_names:
        batch_path = batches_dir / name
        aws_cp(f"{remote_batches_root}/{name}", batch_path)
        las_path = las_dir / f"{Path(name).stem}.las"
        if _write_las(batch_path, las_path, only_static=only_static):
            las_files.append(las_path)

    if not las_files:
        raise RuntimeError(f"No points left after filtering for scene {scene_id}")

    if jobs is None:
        jobs = min(max((os.cpu_count() or 1) // 4, 1), 8)

    command = [
        "py3dtiles",
        "convert",
        *map(str, las_files),
        "--out",
        str(tiles_dir),
        "--jobs",
        str(jobs),
        "--cache_size",
        str(cache_size_mb),
    ]
    if disable_processpool:
        command.append("--disable-processpool")

    subprocess.run(command, check=True)
    aws_sync(tiles_dir, remote_tiles_root, delete=True)

    if cleanup:
        shutil.rmtree(batches_dir, ignore_errors=True)
        shutil.rmtree(las_dir, ignore_errors=True)

    return {
        "scene_id": scene_id,
        "map_id": map_id,
        "batch_count": len(batch_names),
        "las_count": len(las_files),
        "local_tiles": tiles_dir,
        "remote_tiles": remote_tiles_root,
    }
