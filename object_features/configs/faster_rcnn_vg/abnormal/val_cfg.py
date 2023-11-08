_base_ = "../../with_extras_cfg.py"

extractor_type = "faster_rcnn"

sub_class = 'Arrest'
# 'Robbery' 'Shoplifting' 'Stealing' 'Vandalism'

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
    batch_size=1,
    input_size=None,
    frame_sample_rate=10
)

output_writer = dict(
    clip_folder="../data/object_features/val/{}/abnormal_".format(sub_class),
    json_path="../data/object_features/val/{}/abnormal_labels_.json".format(sub_class)
)

extractor = dict(
    temporal_annotions="../data/Annotations/Splits/abnormal/val.json",
    video_folder="../data/Videos",
    num_producers=1,
    dry_run=True,
    # top_k=5,
    filter_pattern='.*{}.*'.format(sub_class)
)

del sub_class
