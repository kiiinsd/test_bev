import mmcv
import torch
import torchvision
import json
import argparse
import numpy as np

from pyquaternion import Quaternion
from hashlib import md5
from PIL import Image
from torch import Tensor
from typing import Dict, List, Any, Tuple

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmcv.parallel import DataContainer as DC
from mmdet3d.models import build_model
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet.datasets.pipelines import to_tensor

mean = [0.3289, 0.3207, 0.3098]
std = [0.1627, 0.1589, 0.1724]
data_root = 'data/panosim/'
cam_name = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

class ImageNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, ori_imgs: List[np.ndarray]) -> Tuple[List, Dict]:
        imgs = [self.compose(img) for img in ori_imgs]
        cfg = dict(mean=self.mean, std=self.std)
        return imgs, cfg

def load_data(timestamp:int):
    images = []
    for cam in cam_name:
        filepath = data_root + 'samples/{}/{}.jpg'.format(cam, timestamp)
        img = np.array(Image.open(filepath))
        images.append(img)
    
    filepath = data_root + 'samples/LIDAR_TOP/{}.pcd.bin'.format(timestamp)
    points = to_tensor(np.fromfile(filepath, dtype=np.float32).reshape(-1,5))

    return images, points

def load_transforms():
    trans = dict(
        camera2ego = [],
        camera2lidar = [],
        lidar2camera = [],
        lidar2image = [],
        camera_intrinsics = [],
        img_aug_matrix = []
    )
    filepath = data_root + 'v1.0/calibrated_sensor.json'
    with open(filepath, 'r') as f:
        cs_record = json.load(f)

    l2e_r = Quaternion(cs_record[0]['rotation']).rotation_matrix
    l2e_t = np.array(cs_record[0]['translation'])
    l2e_matrix = np.eye(4)
    l2e_matrix[:3, :3] = l2e_r
    l2e_matrix[:3, 3] = l2e_t
    trans['lidar2ego'] = l2e_matrix
    lidar_aug_matrix = np.eye(4)
    trans['lidar_aug_matrix'] = lidar_aug_matrix
    
    for name in cam_name:
        sensor_token = md5(name.encode()).hexdigest()
        for record in cs_record:
            if record['sensor_token'] == sensor_token:
                c2e_r = Quaternion(record['rotation']).rotation_matrix
                c2e_t = np.array(record['translation'])
                c2e_matrix = np.eye(4)
                c2e_matrix[:3, :3] = c2e_r
                c2e_matrix[:3, 3] = c2e_t
                trans['camera2ego'].append(c2e_matrix)

                camera_intrinsics = np.eye(4)
                camera_intrinsics[:3, :3] = np.array(record['camera_intrinsic'])
                trans['camera_intrinsics'].append(camera_intrinsics)

                c2l_matrix = np.linalg.inv(l2e_matrix) @ c2e_matrix
                trans['camera2lidar'].append(c2l_matrix)
                trans['lidar2camera'].append(np.linalg.inv(c2l_matrix))

                l2i_matrix = camera_intrinsics @ l2e_matrix.T
                trans['lidar2image'].append(l2i_matrix)

                image_aug_matrix = np.eye(4)
                trans['img_aug_matrix'].append(image_aug_matrix)
                break

    return trans


def prep_data(imgs:List[Tensor], points:Tensor, trans:Dict, metas:Dict):
    data = dict(
        img = DC([torch.stack(imgs).unsqueeze(dim=0)], stack=True),
        points = DC([[points]]),
        metas = DC([[metas]], cpu_only=True)
    )
    for key in trans.keys():
        val = np.array(trans[key])
        if isinstance(trans[key], list):
            data[key] = DC([to_tensor(val)], stack=True)
        else:
            data[key] = DC([to_tensor(val)], stack=True, pad_dims=1)

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--bbox-score', type=float, default=None)
    args, opts = parser.parse_known_args()

    cfg = Config.fromfile(args.config)

    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    norm = ImageNormalize(mean, std)
    timestamp = 1752728178062
    imgs, points = load_data(timestamp)
    imgs, norm_cfg = norm(imgs)
    metas = dict(
        img_norm_cfg = norm_cfg
    )
    trans = load_transforms()
    data = prep_data(imgs, points, trans, metas)

    with torch.inference_mode():
        outputs = model(**data)
    
    if 'boxes_3d' in outputs[0]:
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
        labels = None

    if 'img' in data:
        pass
    if 'points' in data:
        pass