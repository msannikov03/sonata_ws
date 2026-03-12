import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import pyarrow.parquet as pq
from py3dtiles.convert import convert
from tqdm.auto import tqdm
from cloudpathlib import S3Path

S3_ENDPOINT = "https://storage.yandexcloud.net"


# AWS S3 utils
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

# Remote datasets utils
def list_remote_sequences(remote_root: str) -> list[str]:
    out = aws_s3("ls", remote_root)
    seqs = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].endswith("/"):
            name = parts[1].strip("/")
            if name.isdigit():
                seqs.append(name)
    return sorted(seqs)

def list_batches(bucket_root, scene_id, map_id="lidar_pointcloud"):
    path = f"{bucket_root}/datasets/semantic_kitti/scenes/{scene_id}/maps/{map_id}/batches/"
    files = []
    for line in aws_ls(path):
        parts = line.split()
        if len(parts) == 4:
            files.append(parts[3])
    return sorted(files)


# Other utils
def _column(table, name, dtype=None):
    values = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    return values.astype(dtype, copy=False) if dtype is not None else values


def _load_batch(batch_path, only_static):
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

def load_times(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)

def decode_labels(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not path.exists():
        return None, None
    labels = np.fromfile(path, dtype=np.uint32)
    semantic = (labels & 0xFFFF).astype(np.int32)
    instance = (labels >> 16).astype(np.int32)
    return semantic, instance


def write_poses_parquet(poses: list[np.ndarray], times: np.ndarray, out_path: Path) -> None:
    n = len(poses)
    table = pa.table({
        "frame_id": np.array([f"{i:06d}" for i in range(n)], dtype=object),
        "timestamp": times,
        "m00": np.array([p[0, 0] for p in poses], dtype=np.float32),
        "m01": np.array([p[0, 1] for p in poses], dtype=np.float32),
        "m02": np.array([p[0, 2] for p in poses], dtype=np.float32),
        "m03": np.array([p[0, 3] for p in poses], dtype=np.float32),
        "m10": np.array([p[1, 0] for p in poses], dtype=np.float32),
        "m11": np.array([p[1, 1] for p in poses], dtype=np.float32),
        "m12": np.array([p[1, 2] for p in poses], dtype=np.float32),
        "m13": np.array([p[1, 3] for p in poses], dtype=np.float32),
        "m20": np.array([p[2, 0] for p in poses], dtype=np.float32),
        "m21": np.array([p[2, 1] for p in poses], dtype=np.float32),
        "m22": np.array([p[2, 2] for p in poses], dtype=np.float32),
        "m23": np.array([p[2, 3] for p in poses], dtype=np.float32),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")


def make_frame_table(points, semantic, instance, is_static, frame_id):
    n = len(points)

    data = {
        "frame_id": np.full(n, frame_id, dtype=object),
        "x": points[:, 0].astype(np.float32),
        "y": points[:, 1].astype(np.float32),
        "z": points[:, 2].astype(np.float32),
        "intensity": points[:, 3].astype(np.float32),
        "is_static": is_static.astype(bool),
    }

    if semantic is not None:
        data["semantic_id"] = semantic.astype(np.int32)

    if instance is not None:
        data["instance_id"] = instance.astype(np.int32)

    return pa.table(data)


def concat_tables(tables):
    if not tables:
        return None
    return pa.concat_tables(tables, promote_options="default")


def flush_batch(tables, out_path: Path, remote_path: str):
    table = concat_tables(tables)
    if table is None:
        return
    pq.write_table(table, out_path, compression="zstd")
    upload_file(out_path, remote_path)
    out_path.unlink(missing_ok=True)


def make_gt_table(
    xyz: np.ndarray,
    intensity: np.ndarray,
    frame_id: str,
    timestamp: float,
    semantic: np.ndarray | None,
    instance: np.ndarray | None,
) -> pa.Table:
    n = len(xyz)
    data = {
        "x": xyz[:, 0].astype(np.float32),
        "y": xyz[:, 1].astype(np.float32),
        "z": xyz[:, 2].astype(np.float32),
        "intensity": intensity.astype(np.float32),
        "frame_id": np.full(n, frame_id, dtype=object),
        "timestamp": np.full(n, timestamp, dtype=np.float64),
    }
    if semantic is not None:
        data["semantic_id"] = semantic
    if instance is not None:
        data["instance_id"] = instance
    return pa.table(data)



def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1), dtype=np.float32)
    pts_h = np.hstack([points[:, :3], ones])
    return (pose @ pts_h.T).T[:, :3]


def upload_file(local_path: Path, remote_path: str) -> None:
    aws_s3("cp", str(local_path), remote_path, capture_output=False)


def download_sequence(remote_path: str, local_path: Path) -> None:
    local_path.mkdir(parents=True, exist_ok=True)
    aws_s3("sync", remote_path, str(local_path), capture_output=False)

def compute_is_static(
    points: np.ndarray,
    semantic: np.ndarray | None,
) -> np.ndarray:
    MOVING_MIN = 252
    MOVING_MAX = 259
    MIN_DISTANCE = 3.5
    is_static = np.ones(len(points), dtype=bool)

    if semantic is not None:
        is_static &= (semantic < MOVING_MIN) | (semantic > MOVING_MAX)

    dist = np.linalg.norm(points[:, :3], axis=1)
    is_static &= dist > MIN_DISTANCE

    return is_static

def transfer_all_frames(
    local_dataset_root: str,
    bucket_root: str,
    sequences=None,
    batch_size: int = 100,
):
    local_root = Path(local_dataset_root).resolve()

    local_raw_root = local_root / "sequences"
    remote_raw_root = f"{bucket_root}/raw/semantic_kitti/sequences"
    remote_dataset_root = f"{bucket_root}/datasets/semantic_kitti/scenes"

    if sequences is None:
        local_sequences = sorted(
            p.name for p in local_raw_root.glob("*")
            if p.is_dir() and p.name.isdigit()
        )
        sequences = local_sequences or list_remote_sequences(remote_raw_root)

    if not sequences:
        raise RuntimeError("No sequences found")

    for seq in sequences:
        print(f"Sequence {seq}")

        downloaded = False
        local_seq = local_raw_root / seq

        if not local_seq.exists():
            download_sequence(f"{remote_raw_root}/{seq}", local_seq)
            downloaded = True

        try:
            velodyne_dir = local_seq / "velodyne"
            labels_dir = local_seq / "labels"

            calib_path = local_seq / "calib.txt"
            poses_path = local_seq / "poses.txt"
            times_path = local_seq / "times.txt"

            poses = load_poses(str(calib_path), str(poses_path))
            times = load_times(times_path)
            frame_files = sorted(velodyne_dir.glob("*.bin"))

            if len(frame_files) != len(poses) or len(frame_files) != len(times):
                raise ValueError(
                    f"Length mismatch in sequence {seq}: "
                    f"frames={len(frame_files)}, poses={len(poses)}, times={len(times)}"
                )

            scene_id = seq
            sequence_id = "00"

            remote_frames_root = (
                f"{remote_dataset_root}/{scene_id}/sequences/{sequence_id}"
                "/sensors/lidar.velodyne/batches"
            )
            remote_map_root = f"{remote_dataset_root}/{scene_id}/maps/lidar_pointcloud/batches"

            with tempfile.TemporaryDirectory(prefix="skitti_") as tmp_dir:
                tmp_root = Path(tmp_dir)

                frames_batch_tables = []
                maps_batch_tables = []
                batch_idx = 0

                frames_batch_tmp = tmp_root / "frames_batch.parquet"
                maps_batch_tmp = tmp_root / "maps_batch.parquet"

                bar = tqdm(
                    frame_files,
                    total=len(frame_files),
                    desc=f"Frames {seq}",
                    unit="frame",
                    dynamic_ncols=True,
                    miniters=1,
                    mininterval=0.2,
                )

                for i, frame_path in enumerate(bar):
                    frame_id = frame_path.stem
                    pose = poses[i]

                    points = np.fromfile(frame_path, dtype=np.float32).reshape(-1, 4)
                    semantic, instance = decode_labels(labels_dir / f"{frame_id}.label")
                    is_static = compute_is_static(points, semantic)

                    frame_table = make_frame_table(
                        points=points,
                        semantic=semantic,
                        instance=instance,
                        is_static=is_static,
                        frame_id=frame_id,
                    )
                    frames_batch_tables.append(frame_table)

                    xyz_world = transform_points(points, pose)
                    gt_points = np.empty_like(points)
                    gt_points[:, :3] = xyz_world
                    gt_points[:, 3] = points[:, 3]

                    gt_table = make_frame_table(
                        points=gt_points,
                        semantic=semantic,
                        instance=instance,
                        is_static=is_static,
                        frame_id=frame_id,
                    )
                    maps_batch_tables.append(gt_table)

                    should_flush = (
                        len(frames_batch_tables) >= batch_size
                        or i == len(frame_files) - 1
                    )

                    if should_flush:
                        batch_name = f"{batch_idx:06d}.parquet"

                        flush_batch(
                            frames_batch_tables,
                            frames_batch_tmp,
                            f"{remote_frames_root}/{batch_name}",
                        )
                        flush_batch(
                            maps_batch_tables,
                            maps_batch_tmp,
                            f"{remote_map_root}/{batch_name}",
                        )

                        frames_batch_tables.clear()
                        maps_batch_tables.clear()
                        batch_idx += 1

                    bar.set_postfix(frame=frame_id, batch=batch_idx)

        finally:
            if downloaded and local_seq.exists():
                shutil.rmtree(local_seq, ignore_errors=True)


@dataclass(frozen=True, slots=True)
class DataBatch:
    sensor_name: str
    local_path: Path
    remote_path: S3Path

    @property
    def name(self) -> str:
        return self.remote_path.name

    @property
    def stem(self) -> str:
        return self.remote_path.stem

    def has_local(self) -> bool:
        return self.local_path.exists()

    def to_local(self) -> bool:
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        if self.local_path.exists():
            return True
        if not self.remote_path.exists():
            return False
        self.remote_path.download_to(self.local_path)
        return self.local_path.exists()

def get_dataset_batches_loader(
    dataset_local_root,
    dataset_remote_root,
    scene_id,
    sequence_id,
    sensor_id
):


    dataset_local_root = Path(dataset_local_root).resolve()
    dataset_remote_root = dataset_remote_root.rstrip("/")
    if not dataset_remote_root.startswith("s3://"):
        dataset_remote_root = f"s3://{dataset_remote_root.lstrip('/')}"

    if sequence_id is None:
        local_batches_root = dataset_local_root / scene_id / "maps" / sensor_id / "batches"
        remote_batches_root = S3Path(
            f"{dataset_remote_root}/{scene_id}/maps/{sensor_id}/batches"
        )
    else:
        local_batches_root = (
            dataset_local_root
            / scene_id
            / "sequences"
            / sequence_id
            / "sensors"
            / sensor_id
            / "batches"
        )
        remote_batches_root = S3Path(
            f"{dataset_remote_root}/{scene_id}/sequences/{sequence_id}/sensors/{sensor_id}/batches"
        )

    names = {
        path.name
        for path in local_batches_root.glob("*.parquet")
        if path.is_file()
    }
    try:
        for line in aws_ls(str(remote_batches_root)):
            parts = line.split()
            if len(parts) == 4:
                names.add(parts[3])
    except subprocess.CalledProcessError:
        if not names:
            raise

    for name in sorted(names):
        yield SensorBatch(
            sensor_name=sensor_id,
            local_path=local_batches_root / name,
            remote_path=remote_batches_root / name,
        )
    

def points_to_viewer(
    dataset_local_root,
    dataset_remote_root,
    viewer_local_root,
    viewer_remote_root,
    scene_id,
    map_id="lidar_pointcloud",
    only_static=True,
    cleanup=True,
    jobs=None,
    cache_size_mb=1024,
    disable_processpool=False,
    py3dtiles_verbose=1,
):
    root = Path(viewer_local_root).resolve()
    work_dir = root / scene_id / map_id
    las_dir = work_dir / "las"
    tiles_dir = work_dir / "tiles"

    las_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    batches = list(
        get_dataset_batches_loader(
            dataset_local_root=dataset_local_root,
            dataset_remote_root=dataset_remote_root,
            scene_id=scene_id,
            sequence_id=None,
            sensor_id=map_id,
        )
    )
    if not batches:
        raise RuntimeError(f"No batches found for scene {scene_id} map {map_id}")
    remote_tiles_root = f"{viewer_remote_root.rstrip('/')}/{scene_id}/{map_id}/tiles"

    las_files = []
    for batch in tqdm(batches, desc=f"{scene_id}/{map_id}", unit="batch"):
        las_path = las_dir / f"{batch.stem}.las"
        if las_path.exists():
            las_files.append(las_path)
            continue
        if _write_las(batch.ensure_local(), las_path, only_static=only_static):
            las_files.append(las_path)

    if not las_files:
        raise RuntimeError(f"No points left after filtering for scene {scene_id}")

    if jobs is None:
        jobs = min(max((os.cpu_count() or 1) // 4, 1), 8)

    convert(
        files=las_files,
        outfolder=tiles_dir,
        overwrite=True,
        jobs=jobs,
        cache_size=cache_size_mb,
        use_process_pool=not disable_processpool,
        rgb=True,
        verbose=py3dtiles_verbose,
    )
    aws_sync(tiles_dir, remote_tiles_root, delete=True)

    if cleanup:
        shutil.rmtree(las_dir, ignore_errors=True)

    return {
        "scene_id": scene_id,
        "map_id": map_id,
        "batch_count": len(batches),
        "las_count": len(las_files),
        "local_tiles": tiles_dir,
        "remote_tiles": remote_tiles_root,
    }

