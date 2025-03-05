from typing import Any, Dict
from mmdet3d.models.fusion_models.bevfusion import BEVFusion
from mmcv.runner import auto_fp16, force_fp32
from mmdet3d.models import FUSIONMODELS

__all__ = ["BEVFusion_lidar"]

@FUSIONMODELS.register_module()
class BEVFusion_lidar(BEVFusion):
    def __init__(
        self, 
        encoders: Dict[str, Any], 
        fuser: Dict[str, Any], 
        decoder: Dict[str, Any], 
        heads: Dict[str, Any], 
        **kwargs
    ) -> None:
        super().__init__(encoders, fuser, decoder, heads, **kwargs)
        self.training = False
    
    @auto_fp16(apply_to=("points"))
    def forward(
        self,
        points,
        **kwargs
    ):
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)
        
        if not self.training:
            features = features[::-1]
        
        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]
        
        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            pass
            # outputs = {}
            # for type, head in self.heads.items():
            #     if type == "object":
            #         pred_dict = head(x, metas)
            #         losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
            #     elif type == "map":
            #         losses = head(x, gt_masks_bev)
            #     else:
            #         raise ValueError(f"unsupported head: {type}")
            #     for name, val in losses.items():
            #         if val.requires_grad:
            #             outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
            #         else:
            #             outputs[f"stats/{type}/{name}"] = val
            # return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x)
                    bboxes = head.get_bboxes(pred_dict)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                # elif type == "map":
                #     logits = head(x)
                #     for k in range(batch_size):
                #         outputs[k].update(
                #             {
                #                 "masks_bev": logits[k].cpu(),
                #                 "gt_masks_bev": gt_masks_bev[k].cpu(),
                #             }
                #         )
                else:
                    raise ValueError(f"unsupported head: {type}")
        return outputs 