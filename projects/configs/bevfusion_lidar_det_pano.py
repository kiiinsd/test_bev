_base_ = ['./_base_/default_runtime.py']
custom_imports = dict(
    imports = [
        "projects.mmdet3d.datasets.pano_dataset",
        "projects.mmdet3d.models.bevfusion_l",
        "projects.mmdet3d.models.transfusion_simple"
    ],
    allow_failed_imports = False,
)

root_path = '/home/kinsd/test_bev/'
pretrained_path = root_path + 'pretrained/'
dataset_type = 'PanoDataset'
dataset_root = root_path + 'data/panosim/'

gt_paste_stop_epoch = -1
reduce_beams = 32
load_dim = 5
use_dim = 5
load_augmented = False
max_epoch = 24
sequential = False
adj_frame_num = 0

voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
object_classes = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
map_classes = [
    "drivable_area",
    # "drivable_area*",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    # "road_divider",
    # "lane_divider",
    "divider",
]
input_modality = dict(
    use_lidar = True,
    use_camera = False,
    use_radar = False,
    use_map = False,
    use_external = False,
)

model = dict(
    type = "BEVFusion_lidar",
    encoders = dict(
        lidar = dict(
            voxelize = dict(
                max_num_points = 10,
                point_cloud_range = point_cloud_range,
                voxel_size = voxel_size,
                max_voxels = [120000, 160000],
            ),
            backbone = dict(
                type = "SparseEncoder",
                in_channels = 5,
                sparse_shape = [1440, 1440, 41],
                output_channels = 128,
                order = [
                    "conv",
                    "norm",
                    "act",
                ],
                encoder_channels = [
                    [16, 16, 32],
                    [32, 32, 64],
                    [64, 64, 128],
                    [128, 128],
                ],
                encoder_paddings = [
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, [1, 1, 0]],
                    [0, 0],
                ],
                block_type = "basicblock",
            ),
        ),
    ),
    decoder = dict(
        backbone = dict(
            type = "SECOND",
            in_channels = 256,
            out_channels = [128, 256],
            layer_nums = [5, 5],
            layer_strides = [1, 2],
            norm_cfg = dict(
                type = "BN",
                eps = 1.0e-3,
                momentum = 0.01,
            ),
            conv_cfg = dict(
                type = "Conv2d",
                bias = False,
            ),
        ),
        neck = dict(
            type = "SECONDFPN",
            in_channels = [128, 256],
            out_channels = [256, 256],
            upsample_strides = [1, 2],
            norm_cfg = dict(
                type = "BN",
                eps = 1.0e-3,
                momentum = 0.01,
            ),
            upsample_cfg = dict(
                type = "deconv",
                bias = False,
            ),
            use_conv_for_no_stride = True,
        ),
    ),
    heads = dict(
        object = dict(
            type = "TransFusion_simple",
            num_proposals = 200,
            auxiliary = True,
            in_channels = 512,
            hidden_channel = 128,
            num_classes = 10,
            num_decoder_layers = 1,
            num_heads = 8,
            nms_kernel_size = 3,
            ffn_channel = 256,
            dropout = 0.1,
            bn_momentum = 0.1,
            activation = "relu",
            train_cfg = dict(
                dataset = "PanoSim",
                point_cloud_range = point_cloud_range,
                grid_size = [1440, 1440, 41],
                voxel_size = voxel_size,
                out_size_factor = 8,
                gaussian_overlap = 0.1,
                min_radius = 2,
                pos_weight = -1,
                code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                assigner = dict(
                    type = "HungarianAssigner3D",
                    iou_calculator = dict(
                        type = "BboxOverlaps3D",
                        coordinate = "lidar",
                    ),
                ),
                cls_cost = dict(
                    type = "FocalLossCost",
                    gamma = 2.0,
                    alpha = 0.25,
                    weight = 0.15,
                ),
                reg_cost = dict(
                    type = "BBoxBEVL1Cost",
                    weight = 0.25,
                ),
                iou_cost = dict(
                    type = "IoU3DCost",
                    weight = 0.25,
                ),
            ),
            test_cfg = dict(
                dataset = "PanoSim",
                grid_size = [1440, 1440, 41],
                out_size_factor = 8,
                voxel_size = voxel_size[:2],
                pc_range = point_cloud_range[:2],
                nms_type = None,
            ),
            common_heads = dict(
                center = [2, 2],
                height = [1, 2],
                dim = [3, 2],
                rot = [2, 2],
                vel = [2, 2],
            ),
            bbox_coder = dict(
                type = "TransFusionBBoxCoder",
                pc_range = point_cloud_range[:2],
                post_center_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                score_threshold = 0.0,
                out_size_factor = 8,
                voxel_size = voxel_size[:2],
                code_size = 10,
            ),
            loss_cls = dict(
                type = "FocalLoss",
                use_sigmoid = True,
                gamma = 2.0,
                alpha = 0.25,
                reduction = "mean",
                loss_weight = 1.0,
            ),
            loss_heatmap = dict(
                type = "GaussianFocalLoss",
                reduction = "mean",
                loss_weight = 1.0,
            ),
            loss_bbox = dict(
                type = "L1Loss",
                reduction = "mean",
                loss_weight = 0.25,
            ),
        ),
    ),
    fuser = None,
)

test_pipeline = [
    dict(
        type = "LoadPointsFromFile",
        coord_type = "LIDAR",
        load_dim = load_dim,
        use_dim = use_dim,
        reduce_beams = reduce_beams,
        load_augmented = load_augmented,
    ),
    dict(
        type = "DefaultFormatBundle3D",
        classes = object_classes,
        sequential = sequential,
        with_gt = False,
        with_label = False,
    ),
    dict(
        type = "Collect3D",
        sequential = sequential,
        keys = [
            "points",
            "points_num",
            
        ],
        meta_keys = [
        ],
    ),
]

data = dict(
    samples_per_gpu = 2,
    workers_per_gpu = 1,
    train = dict(
        type = "CBGSDataset",
        dataset = dict(
            type = dataset_type,
            dataset_root = dataset_root,
            ann_file = dataset_root + "pano_infos_train.pkl",
            pipeline = test_pipeline,
            object_classes = object_classes,
            map_classes = map_classes,
            modality = input_modality,
            test_mode = False,
            box_type_3d = "LiDAR",
            sequential = sequential,
            adj_frame_num = adj_frame_num,
        ),
    ),
    val = dict(
        type = dataset_type,
        dataset_root = dataset_root,
        ann_file = dataset_root + "pano_infos_val.pkl",
        pipeline = test_pipeline,
        object_classes = object_classes,
        map_classes = map_classes,
        modality = input_modality,
        test_mode = False,
        box_type_3d = "LiDAR",
        sequential = sequential,
        adj_frame_num = adj_frame_num,
    ),
    test = dict(
        type = dataset_type,
        dataset_root = dataset_root,
        ann_file = dataset_root + "pano_infos_val.pkl",
        pipeline = test_pipeline,
        object_classes = object_classes,
        map_classes = map_classes,
        modality = input_modality,
        test_mode = True,
        box_type_3d = "LiDAR",
        sequential = sequential,
        adj_frame_num = adj_frame_num,
    ),
)

evaluation = dict(
    interval = 1,
    pipeline = test_pipeline,
)

optimizer = dict(
    type = "AdamW",
    lr = 1.0e-4,
    weight_decay = 0.01,
)

optimizer_config = dict(
    grad_clip = dict(
        max_norm = 35,
        norm_type = 2,
    ),
)

lr_config = dict(
    policy = "cyclic",
)

momentum_config = dict(
    policy = "cyclic",
)