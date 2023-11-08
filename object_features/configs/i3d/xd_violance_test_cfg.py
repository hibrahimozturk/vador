_base_ = "../xd_violence_cfg.py"

extractor_type = "i3d"

model = dict(
    path="models/i3d/model_rgb.pth",
    end_point='mixed_5c'
)

input_processor = dict(
    temporal_stride=5,
    input_length=16,
    batch_size=16,
    input_size=(224, 224),
    num_threads=1,
    max_len=None,
    flip_frame=False
)

output_writer = dict(
    clip_folder="../data/object_features/xd_violance_test/all",
    json_path="../data/object_features/xd_violance_test/labels_i3d.json",
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
