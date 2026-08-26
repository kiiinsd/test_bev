import argparse
import copy
import os

import mmcv
import numpy as np
import torch
import cv2
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import load_checkpoint
# from torchpack.utils.tqdm import tqdm
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
# from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from projects.mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def main() -> None:
    #dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred", "both"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    args, opts = parser.parse_known_args()

    # configs.load(args.config, recursive=True)
    # configs.update(opts)

    cfg = Config.fromfile(args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    # torch.cuda.set_device(dist.local_rank())

    distributed = False

    # build the dataloader
    dataset = build_dataset(cfg.data["test"])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred" or args.mode == "both":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        # if distributed:
        #     model = MMDistributedDataParallel(
        #         model.cuda(),
        #         device_ids=[torch.cuda.current_device()],
        #         broadcast_buffers=False,
        #     )
        model = MMDataParallel(model, device_ids=[0])
        model.eval()

    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        name = "{}".format(metas["timestamp"])
        # ego_vel = metas["ego_vel"]

        if args.mode == "pred" or args.mode == "both":
            with torch.inference_mode():
                outputs = model(**data)

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            # bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes[..., 2] = -1.55
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "both":
            if "gt_bboxes_3d" in data:
                gt_bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
                gt_labels = data["gt_labels_3d"].data[0][0].numpy()
                if args.bbox_classes is not None:
                    indices = np.isin(labels, args.bbox_classes)
                    gt_bboxes = gt_bboxes[indices]
                    gt_labels = gt_labels[indices]
            else:
                gt_bboxes = None
                gt_labels = None
            
            if "boxes_3d" in outputs[0]:
                bboxes = outputs[0]["boxes_3d"].tensor.numpy()
                scores = outputs[0]["scores_3d"].numpy()
                labels = outputs[0]["labels_3d"].numpy()
                if args.bbox_score is not None:
                    indices = scores >= args.bbox_score
                    bboxes = bboxes[indices]
                    scores = scores[indices]
                    labels = labels[indices]
                if args.bbox_classes is not None:
                    indices = np.isin(labels, args.bbox_classes)
                    bboxes = bboxes[indices]
                    labels = labels[indices]
            else:
                bboxes = None
                labels = None

            bboxes[..., 2] = -1.55
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
            gt_bboxes[..., 2] -= gt_bboxes[..., 5] / 2
            gt_bboxes = LiDARInstance3DBoxes(gt_bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                if args.mode == "both":
                    image = visualize_camera(
                        image=image,
                        bboxes=gt_bboxes,
                        labels=gt_labels,
                        transform=metas["lidar2image"][k],
                        color=(51, 238, 0),
                        classes=cfg.object_classes,
                        thickness=2
                    )
                image = visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image=image,
                    bboxes=bboxes,
                    labels=labels,
                    # ego_vel=ego_vel,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                    thickness=2
                )
                # if k == 0:
                #     cv2.imshow("cam", image)
                

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            if cfg.sequential:
                num = data['points_num'].data[0][0][0]
                lidar = lidar[0:num]
            if args.mode == "both":
                lidar_img = visualize_lidar(
                    os.path.join(args.out_dir, "lidar", f"{name}.png"),
                    lidar=lidar,
                    bboxes=bboxes,
                    labels=labels,
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels,
                    xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                    ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                    color=np.array((51, 238, 0))/255,
                    classes=cfg.object_classes,
                    thickness=15
                )
            else:
                lidar_img = visualize_lidar(
                    os.path.join(args.out_dir, "lidar", f"{name}.png"),
                    lidar=lidar,
                    bboxes=bboxes,
                    labels=labels,
                    xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                    ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                    classes=cfg.object_classes,
                    thickness=15
                )
            # cv2.imshow('lidar', lidar_img)
            # cv2.waitKey(0)

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()
