import torch
import torchvision
import time
import socket
import json
import cv2
import queue
import threading
import argparse
import copy
import math
import numpy as np

from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
from collections import deque
from hashlib import md5
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel, collate
from mmcv.parallel import DataContainer as DC
# from data.panosim.anno_adjust import boxes
from mmdet3d.models import build_model
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet.datasets.builder import build_dataloader
from mmdet.datasets.pipelines import to_tensor
# from mmdet3d.core.utils.visualize import *

class MultiSensorStreamDataset(IterableDataset):
    """支持多摄像头的流式数据集"""
    data_root = 'data/panosim/'
    cam_name = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK"]
    UbuntuIp = '0.0.0.0'
    Start_TcpPort = 14321
    LidarBeams = 32
    LidarMeasurements = 1080
    MonoHeight = 900
    MonoWidth = 1600
    BufferSizeMono = 4 + 4 + MonoWidth * MonoHeight * 3
    
    def __init__(
            self, 
            camera_id=[0, 1, 2], 
            img_aug_cfg=None, 
            cam_transform=None, 
            buffer_size=30, 
            use_lidar=True,
            use_lane=True,
        ):
        """
        Args:
            sensor_id: 摄像头编号[0, 1, 2, ...]
            img_aug_cfg: 图像增强参数
            cam_transform: 图像归一化器
            buffer_size: 每个传感器的缓冲大小
            use_lidar: lidar使能
        """
        self.camera_id = camera_id
        sensor_id = copy.deepcopy(camera_id)
        self.transform = cam_transform
        self.buffer_size = buffer_size

        self.use_lidar = use_lidar
        self.use_lane = use_lane
        if use_lidar and use_lane:
            self.lidar_id = camera_id[-1]+1
            self.lane_id = camera_id[-1]+2
            sensor_id.extend([self.lidar_id, self.lane_id])
        elif use_lane:
            self.lane_id = camera_id[-1]+1
            sensor_id.extend([self.lane_id])
        elif use_lidar:
            self.lidar_id=camera_id[-1]+1
            sensor_id.extend([self.lidar_id])
        
        self.sensor_id = sensor_id

        self.receive_barrier = threading.Barrier(len(sensor_id))
        if img_aug_cfg:
            self.img_aug = ImageAug3D(**img_aug_cfg)

        self.TcpPort = [self.Start_TcpPort + i for i in sensor_id]
        
        # 为每个传感器创建队列
        self.queues = [deque(maxlen=buffer_size) for _ in sensor_id]
        self.locks = [threading.Lock() for _ in sensor_id]
        self.frame_counts = [0 for _ in sensor_id]
        self.running = False
        self.threads = []
        self.lanes = {}
        self.trans = self._load_transforms()
        
    def _camera_worker(self, camera_idx):
        print('PanoSim--mono.{}--listen({}:{})'.format(camera_idx, self.UbuntuIp, self.TcpPort[camera_idx]))
        sock = socket.socket()
        sock.bind((self.UbuntuIp, self.TcpPort[camera_idx]))
        sock.listen(5)

        client_sock, client_info = sock.accept()
        print('PanoSim--mono.{}--connected:{}'.format(camera_idx, client_info))

        try:
            while self.running:
                recv_data = client_sock.recv(self.BufferSizeMono, socket.MSG_WAITALL)
                if recv_data:
                    timestamp = int.from_bytes(recv_data[0:4], byteorder="little")
                    #print('recv mono.{} timestamp:{}'.format(camera_idx, timestamp))
                    data_width = int.from_bytes(recv_data[4:8], byteorder="little")
                    mono_data = recv_data[8:(data_width * 3 + 8)]
                    img = Image.frombuffer('RGB', (self.MonoWidth,self.MonoHeight), mono_data, 'raw', 'RGB', 0, 1)
                    #img = np.frombuffer(mono_data, dtype=np.uint8).reshape((self.MonoHeight, self.MonoWidth, 3))
                    # 添加到队列
                    with self.locks[camera_idx]:
                        if len(self.queues[camera_idx]) < self.buffer_size:
                            self.queues[camera_idx].append({
                                'frame': img,
                                'timestamp': timestamp,
                                'sensor_id': camera_idx,
                                'frame_id': self.frame_counts[camera_idx]
                            })
                            self.frame_counts[camera_idx] += 1
                    self.receive_barrier.wait()
                else:
                    break
        finally:
            client_sock.close()
            print('PanoSim--mono.{}--disconnect:{}'.format(camera_idx, client_info))
        

        print('PanoSim--thread_recv_mono_data quit')

    def _lidar_worker(self):
        print('PanoSim--lidar--listen({}:{})'.format(self.UbuntuIp, self.TcpPort[self.lidar_id]))
        sock = socket.socket()
        sock.bind((self.UbuntuIp, self.TcpPort[self.lidar_id]))
        sock.listen(5)

        client_sock, client_info = sock.accept()
        print('PanoSim--lidar--connected:{}'.format(client_info))

        try:
            while self.running:
                recv_data = client_sock.recv(8, socket.MSG_WAITALL)
                if recv_data:
                    timestamp = int.from_bytes(recv_data[0:4], byteorder="little")
                    data_width = int.from_bytes(recv_data[4:8], byteorder="little")
                    #print('recv lidar data(total bytes):', data_width)
                    lidar_data = client_sock.recv(data_width * 16, socket.MSG_WAITALL)
                    if lidar_data:
                        points = np.frombuffer(lidar_data, np.float32).reshape(-1, 4)
                        points_ = points.copy()
                        points_[:,:2] = points[:,:2] * (-1)
                        # 添加到队列
                        with self.locks[self.lidar_id]:
                            if len(self.queues[self.lidar_id]) < self.buffer_size:
                                self.queues[self.lidar_id].append({
                                    'frame': points_,
                                    'timestamp': timestamp,
                                    'sensor_id': self.lidar_id,
                                    'frame_id': self.frame_counts[self.lidar_id]
                                })
                                self.frame_counts[self.lidar_id] += 1
                        
                        self.receive_barrier.wait()
                    else:
                        break
                else:
                    break
        finally:
            client_sock.close()
            print('PanoSim-lidar--disconnect:{}'.format(client_info))
        

        print('PanoSim--thread_recv_lidar_data quit')
    
    def _lane_worker(self):
        print('PanoSim--lane--listen({}:{})'.format(self.UbuntuIp, self.TcpPort[self.lane_id]))
        sock = socket.socket()
        sock.bind((self.UbuntuIp, self.TcpPort[self.lane_id]))
        sock.listen(5)

        client_sock, client_info = sock.accept()
        print('PanoSim--lane--connected:{}'.format(client_info))

        try:
            while self.running:
                recv_data = client_sock.recv(4, socket.MSG_WAITALL)
                data_width = int.from_bytes(recv_data, byteorder='little')
                if recv_data:
                    lane_data = client_sock.recv(data_width, socket.MSG_WAITALL)
                    if lane_data:
                        lanes = json.loads(lane_data)
                        with self.locks[self.lane_id]:
                            if len(self.queues[self.lane_id]) < self.buffer_size:
                                self.queues[self.lane_id].append({
                                    'frame': lanes,
                                    'timestamp': '',
                                    'sensor_id': self.lane_id,
                                    'frame_id': self.frame_counts[self.lane_id]
                                })
                                self.frame_counts[self.lane_id] += 1
                        self.receive_barrier.wait()
                    else:
                        break
                else:
                    break
        finally:
            client_sock.close()
            print('PanoSim-lane--disconnect:{}'.format(client_info))


    def _load_transforms(self):
        trans = dict(
            camera2ego = [],
            camera2lidar = [],
            lidar2camera = [],
            lidar2image = [],
            camera_intrinsics = [],
            img_aug_matrix = []
        )
        filepath = self.data_root + 'v1.0/calibrated_sensor.json'
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
        
        for name in self.cam_name:
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

                    l2i_matrix = camera_intrinsics @ np.linalg.inv(c2l_matrix)
                    trans['lidar2image'].append(l2i_matrix)

                    image_aug_matrix = np.eye(4)
                    trans['img_aug_matrix'].append(image_aug_matrix)
                    break
            
        return trans

    def start(self):
        self.running = True
        """启动所有摄像头线程"""
        for i in self.camera_id:
            thread = threading.Thread(
                target=self._camera_worker, 
                args=(i,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        """启动激光雷达线程"""
        if self.use_lidar:
            thread = threading.Thread(
                target=self._lidar_worker,
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        """启动车道线接收线程"""
        if self.use_lane:
            thread = threading.Thread(
                target=self._lane_worker,
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
    
    def __iter__(self):
        """迭代器实现"""
        worker_info = get_worker_info()
        
        if worker_info is None:  # 单进程
            # 返回所有摄像头的迭代器
            return self._generate_frames()
        else:  # 多进程
            # 分配摄像头给不同的worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            cameras_for_worker = [
                i for i in range(len(self.sensor_id)) 
                if i % num_workers == worker_id
            ]
            return self._generate_frames(cameras_for_worker)
    
    def _generate_frames(self, sensor_indices=None):
        """生成帧数据"""
        if sensor_indices is None:
            sensor_indices = [i for i in range(len(self.sensor_id))]
            if self.use_lidar and self.use_lane:
                camera_indices = sensor_indices[:-2]
            elif self.use_lane or self.use_lidar:
                camera_indices = sensor_indices[:-1]
            else:
                camera_indices = sensor_indices
        
        while self.running:
            valid = True
            for sensor_idx in sensor_indices:
                with self.locks[sensor_idx]:
                    if len(self.queues[sensor_idx]) == 0:
                        valid = False
                        break
            if valid:
                ori_imgs = []
                imgs = []
                for cam_idx in camera_indices:
                    with self.locks[cam_idx]:
                        frame_data = self.queues[cam_idx].popleft()
                    frame = frame_data['frame']
                    ori_imgs.append(frame)
                    imgs.append(frame)

                if self.use_lane:
                    with self.locks[self.lane_id]:
                        frame_data = self.queues[self.lane_id].popleft()
                        lanes = frame_data['frame']['centerlines']
                        e2g = frame_data['frame']['e2g']
                        ego_vel = frame_data['frame']['ego_vel']
                        
                if self.use_lidar:
                    with self.locks[self.lidar_id]:
                        frame_data = self.queues[self.lidar_id].popleft()
                    frame = frame_data['frame']
                    points = frame
                    c0 = np.zeros((points.shape[0], 1), dtype=np.float32)
                    points = np.column_stack((points, c0))
                if hasattr(self, 'img_aug'):
                    img_aug_matrix, imgs = self.img_aug(imgs)
                else:
                    img_aug_matrix = np.eye(4)
                for cam_idx in camera_indices:
                    imgs[cam_idx] = self.transform(imgs[cam_idx])
                self.trans['img_aug_matrix'] = img_aug_matrix
                data = dict(
                    img = DC(torch.stack(imgs), stack=True),
                    metas = DC(dict(
                                box_type_3d=LiDARInstance3DBoxes,
                                lidar2image=self.trans['lidar2image'],
                                ), 
                                cpu_only=True)
                )
                if self.use_lane:
                    data['metas']['lanes'] = lanes
                    data['metas']['ego_vel'] = ego_vel
                    data['metas']['ego2global'] = e2g
                if self.use_lidar:
                    data['points'] = DC(to_tensor(points)),
                for key in self.trans.keys():
                    val = np.array(self.trans[key], dtype=np.float32)
                    if isinstance(self.trans[key], list):
                        data[key] = DC(to_tensor(val), stack=True)
                    else:
                        data[key] = DC(to_tensor(val), stack=True, pad_dims=1)
                
                yield DC(ori_imgs), data
            
            else:
                time.sleep(0.001)
    
    def stop(self):
        """停止数据流"""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1.0)

class ImageAug3D:
    def __init__(
        self, ori_dim, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
    ):
        self.ori_dim = ori_dim
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim

    def sample_augmentation(self):
        W, H = self.ori_dim
        fH, fW = self.final_dim
        resize = np.mean(self.resize_lim)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(
        self, img, rotation, translation, resize, resize_dims, crop, flip, rotate
    ):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def __call__(self, imgs) -> Dict[str, Any]:
        transforms = []
        for img_id in range(len(imgs)):
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                imgs[img_id],
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            imgs[img_id] = new_img
            
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            transforms.append(transform.numpy())

        return transforms, imgs

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
def visualize_camera(
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 2,
):
    canvas = image.copy()
    if bboxes is not None and len(bboxes) > 0:
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
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
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
                    canvas,
                    coords[index, start].astype(int),
                    coords[index, end].astype(int),
                    color or OBJECT_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    return canvas

# def visualize_lidar(
#     lidar: Optional[np.ndarray] = None,
#     *,
#     bboxes: Optional[LiDARInstance3DBoxes] = None,
#     labels: Optional[np.ndarray] = None,
#     classes: Optional[List[str]] = None,
#     xlim: Tuple[float, float] = (-50, 50),
#     ylim: Tuple[float, float] = (-50, 50),
#     color: Optional[Tuple[int, int, int]] = None,
#     radius: float = 15,
#     thickness: float = 10,
# ) -> None:
#     fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

#     ax = plt.gca()
#     ax.set_xlim(*xlim)
#     ax.set_ylim(*ylim)
#     ax.set_aspect(1)
#     ax.set_axis_off()

#     if lidar is not None:
#         plt.scatter(
#             lidar[:, 0],
#             lidar[:, 1],
#             s=radius,
#             c="white",
#         )

#     if bboxes is not None and len(bboxes) > 0:
#         coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
#         for index in range(coords.shape[0]):
#             name = classes[labels[index]]
#             plt.plot(
#                 coords[index, :, 0],
#                 coords[index, :, 1],
#                 linewidth=thickness,
#                 color=np.array(color or OBJECT_PALETTE[name]) / 255,
#             )
#     fig.canvas.draw()
#     buf = fig.canvas.tostring_rgb()
#     ncols, nrows = fig.canvas.get_width_height()
#     image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
#     plt.close()
#     return image

def get_evenly_spaced_points(x1, y1, x2, y2, d):
    """
    在两点确定的线段上以间距d均匀取点（包括起点，不包括终点，最后一段不足d不取）
    
    参数:
    x1, y1: 第一个点的坐标
    x2, y2: 第二个点的坐标
    d: 取点间距
    
    返回:
    list: 包含取点坐标的列表，每个点为元组(x, y)
    """
    # 计算两点间欧氏距离
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx*dx + dy*dy)
    
    # 如果两点间距小于d，返回空列表
    if dist < d:
        return []
    
    # 计算单位向量
    u_x = dx / dist
    u_y = dy / dist
    
    points = []
    # 从起点开始，以d为步长取点，直到不超过终点
    k = 0
    while k * d < dist:
        x = x1 + k * d * u_x
        y = y1 + k * d * u_y
        points.append([x, y])
        k += 1
    
    return points

def draw_lanes(
    img: np.ndarray,
    lanes: np.ndarray,
    transform: np.ndarray,
    color: Tuple = (255, 0, 0),
    thickness: float = 4,
):
    canvas = img.copy()
    transform = copy.deepcopy(transform).reshape(4, 4)
    for ori_lane in lanes:
        lane = copy.deepcopy(ori_lane)
        lane = lane @ transform.T
        idx = lane[:, 2] > 0
        # lane[:, 2] = np.clip(lane[:, 2], a_min=1e-5, a_max=1e5)
        lane = lane[idx, :]
        ori_lane = ori_lane[idx, :]
        lane[:, 0] /= lane[:, 2]
        lane[:, 1] /= lane[:, 2]
        lane = lane[:, :2]
        last_point = lane[0]
        for i, point in enumerate(lane[1:]):
            cv2.line(
                    canvas,
                    last_point.astype(int),
                    point.astype(int),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            if abs(ori_lane[i+1][0]) > 70 or abs(ori_lane[i+1][1]) > 70:
                break
            
            last_point = point
        
    canvas = canvas.astype(np.uint8)

    return canvas

def draw_lane_lidar(
    lanes: np.ndarray,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Tuple = (0, 0, 255),
    thickness: float = 10,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]), dpi=10)

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    # ax.set_axis_off()

    for lane in lanes:
        plt.plot(
            lane[:,0],
            lane[:,1],
            linewidth=thickness,
            color=np.array(color)/255,
        )
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    image = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    plt.close()
    return image

# 使用示例
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE')
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--bbox-score', type=float, default=None)
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    args, opts = parser.parse_known_args()

    # 数据变换
    mean = np.array([0.3289, 0.3207, 0.3098])
    std = np.array([0.1627, 0.1589, 0.1724])
    transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
    img_aug_cfg = dict(
        ori_dim=[900, 1600],
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False
    )
    
    # 创建数据集
    dataset = MultiSensorStreamDataset(
        camera_id=[i for i in range(6)],
        img_aug_cfg=img_aug_cfg,
        cam_transform=transform,
        buffer_size=20,
        use_lidar=True,
        use_lane=False
    )
    
    # 启动数据流
    dataset.start()
    
    # 创建数据加载器
    # dataloader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=1,
    #     dist=False,
    #     shuffle=False,
    # )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,  # 实时流通常用0或1个worker
        pin_memory=True,
        collate_fn=collate
    )

    cfg = Config.fromfile(args.config)

    if args.mode == 'pred':
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
    
    try:
        for batch_idx, batch_data in enumerate(dataloader):
            imgs, data = batch_data
            data['points'] = data['points'][0]
            metas = data["metas"].data[0][0]
            # lanes = metas['lanes']
            # homo_lanes = []
            # e2g = np.array(metas['ego2global'], dtype=np.float32)
            # ego_vel = metas['ego_vel']
            # l2e = data['lidar2ego'].data[0][0].cpu().detach().numpy()
            # for lane in lanes:
            #     new_lane = []
            #     for i in range(len(lane)-1):
            #         point_list = get_evenly_spaced_points(lane[i][0], lane[i][1], lane[i+1][0], lane[i+1][1], 1)
            #         new_lane.extend(point_list)
            #     lane = new_lane
            #     lane = np.column_stack([lane, np.zeros(len(lane)), np.ones(len(lane))])
            #     lane = lane @ (np.linalg.inv(e2g)).T @ (np.linalg.inv(l2e)).T
            #     homo_lanes.append(lane)
            
            # img = draw_lanes(
            #     img = np.array(imgs.data[0][0][0]),
            #     lanes = homo_lanes,
            #     transform = metas['lidar2image'][0],
            # )
            # imgs.data[0][0][0] = img
            # img = draw_lane_lidar(
            #     homo_lanes
            # )
            # cv2.imshow('', img)
            # cv2.waitKey(100)

            #print(f"批次 {batch_idx}: {imgs[0].shape} {data['points'].shape}")
            #print(imgs[0].shape)
            # row1 = np.concatenate([imgs[1], imgs[0], imgs[2]], axis=2)
            # row2 = np.concatenate([imgs[4], imgs[5], imgs[3]], axis=2)
            # full = np.concatenate([row1, row2], axis=1)
            # full = full.squeeze()
            # #print(full.shape)
            # img = cv2.cvtColor(full, cv2.COLOR_RGB2BGR)
            # img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            
            # cv2.imshow('', img)
            # cv2.waitKey(100)

                
            
            if args.mode == 'pred':
                with torch.inference_mode():
                    outputs = model(**data)
            
            if args.mode =='gt':
                bboxes = None
                labels = None
            
            elif 'boxes_3d' in outputs[0] and args.mode == 'pred':
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
                bboxes[..., 2] = -1.5
                bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
                print(len(bboxes))
            else:
                bboxes = None
                labels = None

            display = []
            for i in dataset.camera_id:
                new_img = visualize_camera(
                    image=np.array(imgs.data[0][0][i]),
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][i],
                    classes=cfg.object_classes,
                )
                display.append(new_img)
            # row1 = cv2.hconcat([display[1], display[0], display[2]])
            # row2 = cv2.hconcat([display[4], display[5], display[3]])
            # full = cv2.vconcat([row1, row2])
            # full = cv2.resize(full, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            full = display[0]

            # lidar = visualize_lidar(
            #     lidar=data['points'].data[0][0].numpy(),
            #     bboxes=bboxes,
            #     labels=labels,
            #     xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
            #     ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
            #     classes=cfg.object_classes
            # )
            cv2.imshow('camera', full)
            # cv2.imshow('lidar', lidar)
            cv2.waitKey(100)
            # if batch_idx >= 50:  # 演示限制
            #     break
                
    except KeyboardInterrupt:
        print("中断")
    finally:
        dataset.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()