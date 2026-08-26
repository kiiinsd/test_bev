import argparse
import copy
import os
from pickle import OBJ
from typing import Tuple

import mmcv
import numpy as np
import torch
import cv2

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from projects.deep_sort.deep_sort import DeepSort
from projects.deep_sort.sort.iou_matching import iou

OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    args, opts = parser.parse_known_args()

    cfg = Config.fromfile(args.config)
    distributed = False
    classes=cfg.object_classes

    dataset = build_dataset(cfg.data["test"])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )
    sort = DeepSort("projects/deep_sort/ckpt.t7", min_confidence=0.2)

    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        model = MMDataParallel(model, device_ids=[0])
        model.eval()

    last_bboxes = None
    last_tracks = []
    last_metas = {}
    for data in dataflow:
        metas = data["metas"].data[0][0]
        
        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)
        
        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()
            scores = np.ones(labels.shape)

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
        else:
            bboxes = None
        
        image = mmcv.imread(metas["filename"][0])
        height, width = image.shape[:2]
        transform = metas["lidar2image"][0]
        bbox_xywh = []
        bbox_idx = np.array([i for i in range(bboxes.tensor.shape[0])])
        indices = []
        if bboxes:
            coords, bbox_idx = lidar2image_transform(bboxes, bbox_idx, transform)
            for coord in coords:
                min_x = np.min(coord[:,0])
                max_x = np.max(coord[:,0])
                min_y = np.min(coord[:,1])
                max_y = np.max(coord[:,1])
                w = max_x - min_x
                h = max_y - min_y
                x = (min_x + max_x)/2
                y = (min_y + max_y)/2
                if min_x > (width-1) or max_x < 0 or min_y > (height-1) or max_y < 0:
                    indices.append(False)
                else:
                    bbox_xywh.append([x,y,w,h])
                    # image = cv2.rectangle(image, (int(min_x), int(min_y)), (int(min_x+w), int(min_y+h)), (0,0,255), 2, cv2.LINE_AA)
                    indices.append(True)
            bbox_xywh = np.array(bbox_xywh)
            bbox_tlwh = sort._xywh_to_tlwh(bbox_xywh)
            indices = np.array(indices, dtype=np.bool)
            coords = coords[indices]
            bbox_idx = bbox_idx[indices]
            if len(bbox_xywh) > 0:
                confs = np.array([scores[i] for i in bbox_idx])
                tracks = sort.update(bbox_xywh, confs, bbox_idx, image)
                bbox_id_list = tracks[:,-1].tolist() if len(tracks) else []
                last_track_id_list = last_tracks[:, 4].tolist() if len(last_tracks) else []
            for index in range(coords.shape[0]):
                bbox_id = bbox_idx[index]
                if bbox_id in bbox_id_list: # bbox in tracks
                    track = tracks[bbox_id_list.index(bbox_id)]
                    track_id = track[-2]
                    if track_id in last_track_id_list: # track in last_tracks
                        last_bbox_id = last_tracks[last_track_id_list.index(track_id)][-1]
                        last_center = last_bboxes.bottom_center[last_bbox_id]
                        curr_center = bboxes.bottom_center[bbox_id]
                        last_ts = last_metas['timestamp']
                        curr_ts = metas['timestamp']
                        vx = (curr_center[0]-last_center[0]) / (curr_ts-last_ts) * 1000
                        vy = (curr_center[1]-last_center[1]) / (curr_ts-last_ts) * 1000
                
                else:
                    track_id = -1
                    vx, vy = bboxes.tensor[bbox_id][7:]

                name = classes[labels[index]]
                for start, end in [
                    (0, 1),
                    (0, 3),
                    (0, 4),
                    (1, 2),
                    (1, 5),
                    (3, 2),
                    (3, 7),
                    (4, 5),
                    (4, 7),
                    (2, 6),
                    (5, 6),
                    (6, 7),
                ]:
                    cv2.line(
                        image,
                        coords[index, start].astype(np.int),
                        coords[index, end].astype(np.int),
                        OBJECT_PALETTE[name],
                        2,
                        cv2.LINE_AA,
                    )
                point_tl = bbox_tlwh[index].astype(np.int)
                cv2.putText(image, "%d vx=%.2f vy=%.2f" % (track_id, vx, vy), point_tl[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.75, OBJECT_PALETTE[name], 0.75, cv2.LINE_AA)
                
        else:
            sort.increment_ages()
        
        last_bboxes = bboxes
        last_tracks = tracks
        last_metas = metas

        cv2.imshow("track", image)
        cv2.waitKey(0)
        

    cv2.destroyAllWindows()


def lidar2image_transform(bboxes, bbox_idx, transform):
        corners = bboxes.corners
        num_bboxes = corners.shape[0]
        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        bbox_idx = bbox_idx[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        bbox_idx = bbox_idx[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)

        return coords, bbox_idx

if __name__ == "__main__":
    main()