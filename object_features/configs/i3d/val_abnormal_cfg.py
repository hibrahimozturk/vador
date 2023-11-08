_base_ = "../with_extras_cfg.py"

extractor_type = "i3d"

sub_class = ''
# Abuse Arrest Arson Assault Banner Burglary Explosion Fighting MolotovBomb RoadAccidents  Robbery Shooting Shoplifting Stealing Vandalism

model = dict(
    path="models/i3d/model_rgb.pth",
    end_point='mixed_5c'
)

input_processor = dict(
    temporal_stride=5,
    input_length=16,
    batch_size=64,
    input_size=(224, 224),
    num_threads=1,
    max_len=None,
    flip_frame=True
)

output_writer = dict(
    clip_folder="../data/object_features/val_5fps_flip/{}/abnormal".format(sub_class),
    json_path="../data/object_features/val_5fps_flip/{}/abnormal_labels_i3d.json".format(sub_class),
    half_precision=True
)

extractor = dict(
    temporal_annotions="../data/Annotations/Splits/abnormal/val.json",
    video_folder="../data/Videos",
    num_producers=1,
    dry_run=False,
    # top_k=10,
    filter_pattern='.*{}.*'.format(sub_class)
)
