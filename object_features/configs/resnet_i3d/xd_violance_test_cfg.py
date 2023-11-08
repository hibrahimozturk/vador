_base_ = "../xd_violence_cfg.py"

extractor_type = "resnet_i3d"

model = dict(
    path="models/resnet_i3d/i3d_r50_kinetics.pth",
)

input_processor = dict(
    temporal_stride=16,
    input_length=16,
    batch_size=20,
    input_size=(224, 224),
    num_threads=1,
    max_len=None,
    flip_frame=False
)

output_writer = dict(
    clip_folder="../data/object_features/xd_violance_resnet/all",
    json_path="../data/object_features/xd_violance_resnet/labels_i3d.json",
    half_precision=True
)

extractor = dict(
    temporal_annotions="../data/xd_violance/test.json",
    video_folder="../data/xd_violance/videos",
    num_producers=1,
    dry_run=False,
    # top_k=5,
    filter_pattern='.*'
)
