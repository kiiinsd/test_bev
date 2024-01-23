from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch._tensor import Tensor
from torch.nn import functional as F

from mmdet3d.models import FUSIONMODELS
from mmdet3d.models.fusion_models.bevfusion import BEVFusion

__all__ = ["My_BEVFusion"]

@FUSIONMODELS.register_module()
class My_BEVFusion(BEVFusion):
    def __init__(
            self, 
            encoders: Dict[str, Any], 
            fuser: Dict[str, Any], 
            decoder: Dict[str, Any], 
            heads: Dict[str, Any],
            sequential,
            adj_frame_num, 
            **kwargs
        ) -> None:
        super().__init__(encoders, fuser, decoder, heads, **kwargs)
        self.num_frames = adj_frame_num + 1
        self.sequential = sequential

    def extract_camera_features(
            self, 
            x, 
            points, 
            lidar2image, 
            camera_intrinsics, 
            camera2lidar, 
            img_aug_matrix, 
            lidar_aug_matrix, 
        ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
        )
        
        return x

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        points_num,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if self.sequential:
            features = self.extract_bev_features_sequential(
                img,
                points,
                points_num,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                **kwargs
            )
        
        return
        

    @auto_fp16(apply_to=("img", "points"))
    def extract_bev_features_sequential(
        self, 
        img, 
        points, 
        points_num,
        lidar2image, 
        camera_intrinsics, 
        camera2lidar, 
        img_aug_matrix, 
        lidar_aug_matrix, 
        **kwargs
    ):
        img_list , points_list, lidar2image_list, camera_intrinsics_list, camera2lidar_list = \
        self.prepare_inputs(
            img,
            points,
            points_num,
            lidar2image,
            camera_intrinsics,
            camera2lidar
        )

        fused_feature_list = []
        for img, points, lidar2image, camera_intrinsics, camera2lidar in \
        zip(img_list, points_list, lidar2image_list, camera_intrinsics_list, camera2lidar_list):
            features = []
            for sensor in (
                self.encoders if self.training else list(self.encoders.keys())[::-1]
            ):
                if sensor == "camera":
                    feature = self.extract_camera_features(
                        img,
                        points,
                        lidar2image,
                        camera_intrinsics,
                        camera2lidar,
                        img_aug_matrix,
                        lidar_aug_matrix
                    )
                elif sensor == "lidar":
                    feature = self.extract_lidar_features(points)
                else: 
                    raise ValueError(f"unsupported sensor: {sensor}")
                features.append(feature)

            if not self.training:
                # avoid OOM
                features = features[::-1]

            if self.fuser is not None:
                x = self.fuser(features)
            else:
                assert len(features) == 1, features
                x = features[0]

            fused_feature_list.append(x)


        return x
        
    def prepare_inputs(
        self,
        img,
        points,
        points_num,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
    ):
        B, N, C, H, W = img.size()
        N = N // self.num_frames
        img = img.view(B, N, self.num_frames, C, H, W)
        img = torch.split(img, 1, dim=2)
        img_list = [t.squeeze(2) for t in img]

        num_list = points_num.squeeze().tolist()
        points = torch.split(points, num_list, dim=1)
        points_list = [p.squeeze(1) for p in points]

        lidar2image = torch.split(lidar2image, 1, dim=1)
        lidar2image_list = [t.squeeze(1) for t in lidar2image]

        camera_intrinsics = torch.split(camera_intrinsics, 1, dim=1)
        camera_intrinsics_list = [t.squeeze(1) for t in camera_intrinsics]

        camera2lidar = torch.split(camera2lidar, 1, dim=1)
        camera2lidar_list = [t.squeeze(1) for t in camera2lidar]

        return img_list, points_list, lidar2image_list, camera_intrinsics_list, camera2lidar_list