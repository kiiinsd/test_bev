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
        ego2global,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if self.sequential:
            x = self.extract_bev_features_sequential(
                img,
                points,
                points_num,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                ego2global,
                img_aug_matrix,
                lidar_aug_matrix,
                **kwargs
            )
        
        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
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
        ego2global,
        img_aug_matrix, 
        lidar_aug_matrix, 
        **kwargs
    ):
        img_list , points_list, lidar2image_list, camera_intrinsics_list, \
        camera2lidar_list, ego2global_list = \
        self.prepare_inputs(
            img,
            points,
            points_num,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            ego2global
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

        x = self.align_feature(fused_feature_list, ego2global_list)

        return x
        
    def align_feature(
        self,
        feature_list,
        ego2global_list
    ):
        feature = torch.cat(feature_list, dim=1)
        return feature

    def prepare_inputs(
        self,
        img,
        points,
        points_num,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        ego2global,
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

        ego2global = torch.split(ego2global, 1, dim=1)
        ego2global_list = [t.squeeze(1) for t in ego2global]

        return img_list, points_list, lidar2image_list, camera_intrinsics_list, \
            camera2lidar_list, ego2global_list