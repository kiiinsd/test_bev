from typing import Any, Dict
import mmcv
import numpy as np
from pyquaternion import Quaternion

from mmdet3d.datasets import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes

@DATASETS.register_module()
class PanoDataset(Custom3DDataset):

    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        # "construction_vehicle",
        # "bicycle",
        # "motorcycle",
        # "pedestrian",
        # "traffic_cone",
        # "barrier",
    )

    NameMapping = {
        'Car': 'car',
        'Truck': 'truck',
        'Bus': 'bus',
        'Pedestrain': 'pedestrain',
        'Bicycle': 'bicycle',
        'Trailer': 'trailer',
        'Construction': 'construction_vehicle',
        'Motocycle': 'motocycle'
    }

    def __init__(
        self, 
        ann_file,
        dataset_root=None,
        pipeline=None, 
        object_classes=None, 
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None, 
        box_type_3d='LiDAR', 
        filter_empty_gt=True, 
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        sequential=False,
        adj_frame_num=0,
    ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            dataset_root=dataset_root, 
            ann_file=ann_file, 
            pipeline=pipeline, 
            classes=object_classes, 
            modality=modality, 
            box_type_3d=box_type_3d, 
            filter_empty_gt=filter_empty_gt, 
            test_mode=test_mode
        )
        self.with_velocity = with_velocity
        self.sequential = sequential
        self.adj_frame_num = adj_frame_num
        self.map_classes = map_classes

    def get_cat_ids(self, idx):
        '''Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        '''
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids
    
    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda x: x['timestamp']))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index:int) -> Dict[str, Any]:
        input_dict = dict()
        annos = self.get_anno_info(index)
        input_dict['ann_info'] = annos
        info = self.data_infos[index]
        data = dict(
            token = info['token'],
            sample_idx = info['token'],
            lidar_path = info['lidar_path'],
            timestamp = info['timestamp'],
            sweeps = info['sweeps'],
        )

        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = info['ego2global_translation']
        data['ego2global'] = ego2global

        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = info['lidar2ego_translation']
        data['lidar2ego'] = lidar2ego

        if self.modality['use_camera']:
            data['image_paths'] = []
            data['lidar2camera'] = []
            data['lidar2image'] = []
            data['camera2ego'] = []
            data['camera_intrinsics'] = []
            data['camera2lidar'] = []

            for _, camera_info in info['cams'].items():
                data['image_paths'].append(camera_info['data_path'])

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(camera_info['sensor2lidar_rotation'])
                lidar2camera_t = (
                    camera_info['sensor2lidar_translation'] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data['lidar2camera'].append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = camera_info['camera_intrinsics']
                data['camera_intrinsics'].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                data['lidar2image'].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    camera_info['sensor2ego_rotation']
                ).rotation_matrix
                camera2ego[:3, 3] = camera_info['sensor2ego_translation']
                data['camera2ego'].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = camera_info['sensor2lidar_rotation']
                camera2lidar[:3, 3] = camera_info['sensor2lidar_translation']
                data['camera2lidar'].append(camera2lidar)

        input_dict['curr'] = data
        return input_dict

    def get_anno_info(self, index):
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results

    