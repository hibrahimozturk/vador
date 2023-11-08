_base_ = "../with_extras_cfg.py"

extractor_type = "faster_rcnn"

model = dict(
    net="res101",
    dataset="vg",
    load_dir="models/faster_rcnn_vg/pretrained_weights",
    cuda=True,
    cfg_file="models/faster_rcnn_vg/cfgs/res101.yml",
    classes_dir="models/faster_rcnn_vg/data/genome/1600-400-20",
    class_agnostic=False,
)

input_processor = dict(
    batch_size=16,
    input_size=None,
    frame_sample_rate=5,
    num_threads=1,
    max_len=None,
    flip_frame=True
)

output_writer = dict(
    clip_folder="../data/object_features/val_5fps_flip/normal",
    json_path="../data/object_features/val_5fps_flip/normal_labels.json",
    half_precision=True,
)

extractor = dict(
    temporal_annotions="../data/Annotations/Splits/normal/val.json",
    video_folder="../data/Videos",
    num_producers=1,
    dry_run=False,
    # top_k=5,
    filter_pattern='.*'
)
