from typing import Any, Dict
import mmcv
import numpy as np
from mmdet3d.datasets import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset

@DATASETS.register_module()
class PanoDataset(Custom3DDataset):
    def __init__(
        self, 
        dataset_root, 
        ann_file, 
        pipeline=None, 
        object_classes=None, 
        map_classes=None,
        modality=None, 
        box_type_3d="LiDAR", 
        filter_empty_gt=False, 
        test_mode=False,
        use_valid_flag=False,
        sequential=False,
        adj_frame_num=0,
    ):
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
        self.use_valid_flag = use_valid_flag
        self.sequential = sequential
        self.adj_frame_num = adj_frame_num
        self.map_classes = map_classes

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids
    
    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda x: x["timestamp"]))
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos

    def get_data_info(self, index:int) -> Dict[str, Any]:
        input_dict = dict()
        annos = self.get_anno_info(index)
        input_dict["ann_info"] = annos
        info = self.data_infos[index]
        data = dict(
            lidar_path = info["lidar_path"],
            timestamp = info["timestamp"],
            sweeps = info["sweeps"],
        )

        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = info["ego2global_rotation"]
        ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global

        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = info["lidar2ego_rotation"]
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            for _, camera_info in info["cams"].items():
                data["image_paths"].append(camera_info["data_path"])

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
                lidar2camera_t = (
                    camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data["lidar2camera"].append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
                data["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                data["lidar2image"].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = camera_info["sensor2ego_rotation"]
                camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                data["camera2ego"].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar)

        input_dict["curr"] = data
        return input_dict

    def get_anno_info(self, index):
        anns_results = dict()
        return anns_results

    