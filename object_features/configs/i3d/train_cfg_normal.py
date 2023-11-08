_base_ = "../with_extras_cfg.py"

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
    num_threads=2,
    max_len=7500,
    flip_frame=True
)

output_writer = dict(
    clip_folder="../data/object_features/train_5fps_flip/normal",
    json_path="../data/object_features/train_5fps_flip/normal_labels_i3d_post.json",
    half_precision=True
)

extractor = dict(
    temporal_annotions="../data/Annotations/Splits/normal/train.json",
    pre_json="../data/object_features/train_5fps_flip/normal_labels_i3d.json",
    video_folder="../data/Videos",
    num_producers=1,
    dry_run=False,
    top_k=400,
    filter_pattern='.*'
)
