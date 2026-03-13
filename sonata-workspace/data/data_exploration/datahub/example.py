from datahub import open_hub

hub = open_hub(
    root="3d-scenes/datasets",
    endpoint_url="https://storage.yandexcloud.net",
    cache_storage="/workspace/datasets/semantic_kitti_cache",
)

ds = hub.dataset("semantic_kitti")
scene = ds.scene("01")

seq_asset = scene.sequence("00").asset("lidar.velodyne")
b = seq_asset.batch(11)
print(b.num_rows)
print(seq_asset.batch_for_frame(123).name)
print(seq_asset.frame(123).read_table())

map_asset = scene.map("lidar_pointcloud")
print(map_asset.list_batches()[:3])

tbl = b.read_table()
print(tbl.schema)