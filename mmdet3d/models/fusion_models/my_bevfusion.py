from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch._tensor import Tensor
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)

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
        if self.sequential:
            decoder_backbone_cfg = decoder["backbone"]
            decoder_backbone_cfg["in_channels"] = 256 * self.num_frames
            self.decoder["backbone"] = build_backbone(decoder_backbone_cfg)
        self.grid = None
        self.xbound = encoders["camera"]["vtransform"]["xbound"]
        self.ybound = encoders["camera"]["vtransform"]["ybound"]
        self.downsample = encoders["camera"]["vtransform"]["downsample"]

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
        lidar2ego,
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
            feature_list = []
            imgs, points_list, lidar2images, lidar2egos, intrins, camera2lidars, ego2globals = \
            self.prepare_inputs(
                img, points, points_num, lidar2image, lidar2ego, camera_intrinsics, camera2lidar, ego2global
            )
            key_frame = True
            for img, points, lidar2image, intrin, camera2lidar in \
            zip(imgs, points_list, lidar2images, intrins, camera2lidars):
                if key_frame:
                    feature = self.extract_bev_feature(
                                img,
                                points,
                                lidar2image,
                                camera_intrinsics,
                                camera2lidar,
                                img_aug_matrix,
                                lidar_aug_matrix,
                                **kwargs
                            )
                    key_frame = False
                else:
                    with torch.no_grad():
                        feature = self.extract_bev_feature(
                                img,
                                points,
                                lidar2image,
                                camera_intrinsics,
                                camera2lidar,
                                img_aug_matrix,
                                lidar_aug_matrix,
                                **kwargs
                            )
                feature_list.append(feature)
            for frame in range(1, self.num_frames):
                feature_list[frame] = self.align_feature(feature_list[frame],
                                                         [lidar2ego[0], lidar2ego[frame]],
                                                         [ego2global[0], ego2global[frame]])
            x = torch.cat(feature_list, dim=1)
                    
        else:
            x = self.extract_bev_feature(
                img,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
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
        
        return outputs
    
    @auto_fp16(apply_to=("img", "points"))
    def extract_bev_feature(
        self, 
        img, 
        points, 
        lidar2image, 
        camera_intrinsics, 
        camera2lidar, 
        img_aug_matrix, 
        lidar_aug_matrix, 
        **kwargs
    ):
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

        return x
        
        
    @force_fp32()
    def align_feature(
        self,
        feature,
        lidar2egos,
        ego2globals
    ):
        n, c, h, w = feature.shape
        if self.grid is None:
            # generate grid
            xs = torch.linspace(
                0, w - 1, w, dtype=feature.dtype,
                device=feature.device).view(1, w).expand(h, w)
            ys = torch.linspace(
                0, h - 1, h, dtype=feature.dtype,
                device=feature.device).view(h, 1).expand(h, w)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
            self.grid = grid
        else:
            grid = self.grid
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)

        curr_l2e, adj_l2e = lidar2egos
        curr_e2g, adj_e2g = ego2globals
 
        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.xbound[2] * self.downsample
        feat2bev[1, 1] = self.ybound[2] * self.downsample
        feat2bev[0, 2] = self.xbound[0]
        feat2bev[1, 2] = self.ybound[0]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)

        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=feature.dtype,
                                        device=feature.device)

        curr2adj = torch.inverse(adj_l2e).matmul(torch.inverse(adj_e2g))\
            .matmul(curr_e2g).matmul(curr_l2e).view(n, 1, 1, 4, 4)
        curr2adj = curr2adj[:, :, :,
                    [True, True, False, True], :][:, :, :, :,
                                                [True, True, False, True]]
        tf = torch.inverse(feat2bev).matmul(curr2adj).matmul(feat2bev)
        adj_grid = tf.matmul(grid)
        adj_grid = adj_grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 
                                                        2) * 2.0 - 1.0
        
        feature= F.grid_sample(feature, adj_grid.to(feature), 
                                            align_corners=True)

        return feature

    def prepare_inputs(
        self,
        imgs,
        points,
        points_num,
        lidar2images,
        lidar2egos,
        camera_intrinsics,
        camera2lidars,
        ego2globals,
    ):
        B, N, C, H, W = imgs.shape
        N = N // self.num_frames
        imgs = imgs.view(B, N, self.num_frames, C, H, W)
        imgs = torch.split(imgs, 1, dim=2)
        imgs = [t.squeeze(2) for t in imgs]

        for b in range(B):
            num_list = points_num[b].squeeze().tolist()
            points[b] = torch.split(points[b], num_list, dim=0)

        points_list = []
        for frame in range(self.num_frames):
            points_list.append([])
            for b in range(B):
                points_list[frame].append(points[b][frame])

        extra = [
            lidar2images,
            lidar2egos,
            camera_intrinsics,
            camera2lidars,
            ego2globals
        ]
        extra = [torch.split(t, 1, dim=1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        # lidar2image = torch.split(lidar2image, 1, dim=1)
        # lidar2image_list = [t.squeeze(1) for t in lidar2image]

        # lidar2ego = torch.split(lidar2ego, 1, dim=1)
        # lidar2ego_list = [t.squeeze(1) for t in lidar2ego]

        # camera_intrinsics = torch.split(camera_intrinsics, 1, dim=1)
        # camera_intrinsics_list = [t.squeeze(1) for t in camera_intrinsics]

        # camera2lidar = torch.split(camera2lidar, 1, dim=1)
        # camera2lidar_list = [t.squeeze(1) for t in camera2lidar]

        # ego2global = torch.split(ego2global, 1, dim=1)
        # ego2global_list = [t.squeeze(1) for t in ego2global]
        
        lidar2images, lidar2egos, camera_intrinsics, camera2lidars, ego2globals = extra

        return imgs, points_list, lidar2images, lidar2egos, camera_intrinsics, \
            camera2lidars, ego2globals