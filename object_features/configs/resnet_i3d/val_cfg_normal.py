_base_ = "../with_extras_cfg.py"

extractor_type = "resnet_i3d"

sub_class = ''
# Abuse Arrest Arson Assault Banner Burglary Explosion Fighting MolotovBomb RoadAccidents  Robbery Shooting Shoplifting Stealing Vandalism

model = dict(
    path="models/resnet_i3d/i3d_r50_kinetics.pth",
)

input_processor = dict(
    temporal_stride=16,
    input_length=16,
    batch_size=16,
    input_size=None,
    num_threads=1,
    flip_frame=False,
    max_len=None
)

output_writer = dict(
    clip_folder="../data/object_features/val_resnet/{}/normal".format(sub_class),
    json_path="../data/object_features/val_resnet/{}/normal_labels_i3d.json".format(sub_class),
    half_precision=True
)

extractor = dict(
    temporal_annotions="../data/Annotations/Splits/normal/val.json",
    video_folder="../data/Videos",
    num_producers=1,
    dry_run=False,
    # top_k=10,
    filter_pattern='.*{}.*'.format(sub_class)
)
