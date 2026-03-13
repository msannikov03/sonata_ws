# datahub

Minimal object storage data loader for normalized scene datasets.

`datahub` gives a small, strict API over datasets stored as:

- `dataset`
- `scene`
- `sequence`
- `asset`
- `batch`

It does not hardcode dataset-specific logic into the public interface.  
It reflects the storage layout directly and keeps reading lazy until the final batch access.

---

## Storage layout

A dataset is expected to follow this structure:

```text
<root>/
  <dataset_name>/
    scenes/
      <scene_id>/
        sequences/
          <sequence_id>/
            <asset_name>/
              batches/
                000000.parquet
                000001.parquet
                ...
        maps/
          <asset_name>/
            batches/
              000000.parquet
              000001.parquet
              ...
````

Example:

```text
3d-scenes/datasets/
  semantic_kitti/
    scenes/
      01/
        sequences/
          00/
            lidar.velodyne/
              batches/
                000011.parquet
            poses/
              batches/
                000000.parquet
            lidar.velodyne.image_fov_points/
              batches/
                000000.parquet
        maps/
          lidar_pointcloud/
            batches/
              000000.parquet
```

---

## Core model

The library exposes a small domain model:

* `Hub`
* `Dataset`
* `Scene`
* `Sequence`
* `Asset`
* `Batch`
* `FrameRef`

Semantics:

* a **dataset** contains scenes
* a **scene** contains:

  * sequence-scoped assets under `sequences/`
  * scene-scoped assets under `maps/`
* an **asset** contains parquet batches
* a **batch** is the final readable object
* a **frame** is a filtered view over one batch

Both sequence data and map data are represented as `Asset`.
The only difference is scope and storage location.

---

## Installation

```bash
pip install fsspec pyarrow s3fs universal_pathlib
```

If you use local cache:

```bash
pip install fsspec pyarrow s3fs filecache universal_pathlib
```

---

## Quick start

```python
from datahub import open_hub

hub = open_hub(
    root="3d-scenes/datasets",
    endpoint_url="https://storage.yandexcloud.net",
    cache_storage="/workspace/datasets/cache",
)

ds = hub.dataset("semantic_kitti")
scene = ds.scene("01")

seq_asset = scene.sequence("00").asset("lidar.velodyne")
batch = seq_asset.batch(11)

print(batch.num_rows)
table = batch.read_table()

frame = seq_asset.frame(123)
frame_table = frame.read_table()
```

Scene-level asset:

```python
pc_map = scene.map("lidar_pointcloud")
batch = pc_map.batch(0)
df = batch.read_pandas()
```

---

## API

### Open hub

```python
hub = open_hub(
    root="3d-scenes/datasets",
    endpoint_url="https://storage.yandexcloud.net",
    cache_storage="/workspace/datasets/cache",
)
```

Parameters:

* `root`: dataset root prefix
* `endpoint_url`: S3-compatible endpoint
* `cache_storage`: optional local cache directory
* `target_protocol`: storage protocol, default is `s3`

---

### List datasets

```python
hub.list_datasets()
```

---

### Access dataset

```python
ds = hub.dataset("semantic_kitti")
ds.list_scenes()
```

---

### Access scene

```python
scene = ds.scene("01")

scene.list_sequences()
scene.list_maps()
```

---

### Access sequence asset

```python
asset = scene.sequence("00").asset("lidar.velodyne")
asset.list_batches()
```

`Sequence.sensor(...)` is available as an alias of `Sequence.asset(...)`.

```python
asset = scene.sequence("00").sensor("lidar.velodyne")
```

---

### Access map asset

```python
asset = scene.map("lidar_pointcloud")
asset.list_batches()
```

---

### Access batch

```python
batch = asset.batch(11)

print(batch.path)
print(batch.num_rows)
print(batch.num_row_groups)
print(batch.schema)
```

Read data:

```python
table = batch.read_table()
df = batch.read_pandas()
xyz = batch.read_columns(["x", "y", "z"])
```

---

### Access frame in sequence asset

Sequence assets may expose frame-level access via `frame_id`.

Invariants:

* `frame_id` is integer
* each frame belongs to exactly one batch
* a frame is never split across batches

API:

```python
asset = scene.sequence("00").asset("lidar.velodyne")

frame = asset.frame(123)
table = frame.read_table()
df = frame.read_pandas()

batch = asset.batch_for_frame(123)
frames = asset.batch(11).list_frames()
```

You can also iterate:

```python
for frame in asset.frames():
    print(frame.frame_id, frame.batch_path)
```

`FrameRef.read_table()` reads only rows with this `frame_id` inside its batch.

---

## Design principles

### 1. Storage-first model

The API mirrors the real storage hierarchy instead of hiding it behind a large abstract loader.

This keeps the system predictable:

```python
hub.dataset("semantic_kitti").scene("01").sequence("00").asset("lidar.velodyne").batch(11)
```

Frame access stays on top of the same storage model:

```python
hub.dataset("semantic_kitti").scene("01").sequence("00").asset("lidar.velodyne").frame(123)
```

---

### 2. Assets everywhere

The library does not distinguish between “raw sensor streams”, “precomputed features”, “filtered point clouds”, or “maps” at the type level.

They are all `Asset`.

Examples:

* `lidar.velodyne`
* `poses`
* `camera.left.rgb`
* `lidar.velodyne.image_fov_points`
* `lidar_pointcloud`
* `mesh`
* `3dgs`

This keeps the public model stable when new derived data is added later.

---

### 3. Lazy reads

Navigation is cheap.
Actual parquet reading happens only at `Batch`.

---

### 4. No unnecessary framework logic

No registries, no hidden dataset inference layer, no forced metadata schema.

The code is intentionally small and direct.

---

## Example workflow

```python
from datahub import open_hub

hub = open_hub(
    root="3d-scenes/datasets",
    endpoint_url="https://storage.yandexcloud.net",
    cache_storage="/workspace/datasets/cache",
)

ds = hub.dataset("semantic_kitti")

for scene in ds.scenes():
    print(scene.scene_id)

    for seq in scene.sequences():
        for asset in seq.assets():
            print("sequence asset:", seq.sequence_id, asset.name)

    for asset in scene.maps():
        print("map asset:", asset.name)
```

---

## Expected naming conventions

The library does not enforce asset naming, but these conventions work well:

### Sequence assets

Stored under:

```text
scenes/<scene_id>/sequences/<sequence_id>/<asset_name>/batches/
```

Examples:

* `lidar.velodyne`
* `poses`
* `camera.left.rgb`
* `lidar.velodyne.image_fov_points`

### Map assets

Stored under:

```text
scenes/<scene_id>/maps/<asset_name>/batches/
```

Examples:

* `lidar_pointcloud`
* `mesh`
* `occupancy`
* `3dgs`

---

## Why this structure

This layout separates two scopes cleanly:

* **sequence scope** — data tied to one pass through the scene
* **scene scope** — data aggregated for the whole scene

That distinction is structural, not semantic.
In both cases the final object is still just an asset with batched parquet data.

---

## Minimal example

```python
from datahub import open_hub

hub = open_hub(
    root="3d-scenes/datasets",
    endpoint_url="https://storage.yandexcloud.net",
    cache_storage="/workspace/datasets/cache",
)

rows = (
    hub.dataset("semantic_kitti")
       .scene("01")
       .sequence("00")
       .asset("lidar.velodyne")
       .batch(11)
       .num_rows
)

print(rows)
```

---

## Current scope

This project currently provides:

* navigation over normalized dataset storage
* lazy parquet access
* S3-compatible access via `fsspec`
* optional local file cache
* uniform sequence/map asset model
* optional frame-level access for sequence assets

It intentionally does not yet provide:

* dataset manifests
* typed decoders
* schema registry
* train/val split logic
* dataset-specific adapters

These can be added later without changing the core public model.

---

## License

Internal / project-specific unless stated otherwise