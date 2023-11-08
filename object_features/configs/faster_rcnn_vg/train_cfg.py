_base_ = "../with_extras_cfg.py"

extractor_type = "faster_rcnn"

sub_class = ''
# Abuse Arrest Arson Assault Banner Burglary Explosion Fighting MolotovBomb RoadAccidents  Robbery Shooting Shoplifting Stealing Vandalism

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
    flip_frame=True,
    max_len=7500
)

output_writer = dict(
    clip_folder="../data/object_features/train_5fps_flip/{}/abnormal__".format(sub_class),
    json_path="../data/object_features/train_5fps_flip/{}/abnormal_labels_train_val_v1.json".format(sub_class),
    half_precision=True
)

extractor = dict(
    temporal_annotions="../data/Annotations/Splits/abnormal/train_val_v1_new.json",
    # pre_json="../data/object_features/train_5fps_flip/{}/abnormal_labels.json".format(sub_class),
    video_folder="../data/Videos",
    num_producers=16,
    dry_run=True,
    # top_k=3,
    filter_pattern='.*{}.*'.format(sub_class)
    # filter_pattern='.*{}.*'.format('Abuse015')
)
