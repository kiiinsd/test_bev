import copy
import os
from tkinter import RIDGE
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from mmdet3d.core import bbox
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from .warning import Risk, warning

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]


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

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}

WARNING_PALETTE = {
    Risk.NO_RISK: (0, 255, 0),
    Risk.LOW_RISK: (255, 255, 0),
    Risk.HIGH_RISK: (255, 0, 0)
}

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3.0,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return text_size

def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    ego_vel: float = 0.0,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        centers = bboxes.gravity_center
        bottom_centers = bboxes.bottom_center
        bottom_dims = bboxes.dims[:, :2]
        velocitys = bboxes.tensor[:, 7:9]
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        centers = np.concatenate(
            [centers, np.ones((num_bboxes, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)
        centers = centers @ transform.T
        centers = centers.reshape(-1, 1, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        centers = centers[indices]
        bottom_centers = bottom_centers[indices]
        bottom_dims = bottom_dims[indices]
        velocitys = velocitys[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        centers = centers[indices]
        velocitys = velocitys[indices]
        bottom_centers = bottom_centers[indices]
        bottom_dims = bottom_dims[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]
        centers = centers.reshape(-1, 4)
        centers[:, 2] = np.clip(centers[:, 2], a_min=1e-5, a_max=1e5)
        centers[:, 0] /= centers[:, 2]
        centers[:, 1] /= centers[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        centers = centers[..., :2]
        for index in range(coords.shape[0]):
            warning_level = warning(
                bottom_centers[index][0],
                bottom_centers[index][1],
                velocitys[index][0],
                velocitys[index][1]-ego_vel,
                0 if velocitys[index][1] >= 0 else 1,
                2.0,
                1.25 + bottom_dims[index][0],
                2.2 + bottom_dims[index][1]
            )
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
                    coords[index, start].astype(np.int),
                    coords[index, end].astype(np.int),
                    WARNING_PALETTE[warning_level],
                    thickness,
                    cv2.LINE_AA,
                )
            
            dist = np.linalg.norm(bottom_centers[index])
            _, text_h = draw_text(
                canvas,
                '{:.2f}, warning:{}'.format(dist, warning_level),
                cv2.FONT_HERSHEY_PLAIN,
                (int(centers[index][0]), int(centers[index][1])),
                1,
                1,
                (0,0,0),
                WARNING_PALETTE[warning_level]
            )
            draw_text(
                canvas,
                'vx:{:.2f} vy:{:.2f}'.format(velocitys[index][0], velocitys[index][1] - ego_vel),
                cv2.FONT_HERSHEY_PLAIN,
                (int(centers[index][0]), int(centers[index][1])+text_h+1),
                1,
                1,
                (0,0,0),
                WARNING_PALETTE[warning_level]
            )

        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def visualize_lidar_overlap(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 1,
) -> None:
    image = mmcv.imread(fpath)
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    width, height, _ = canvas.shape

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            coords[index, : , 0] = (coords[index, :, 0]+xlim[1]) / (xlim[1]-xlim[0]) * width
            coords[index, : , 1] = (ylim[1]-coords[index, :, 1]) / (ylim[1]-ylim[0]) * height
            for start, end in [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4)
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].numpy().astype(np.int),
                    coords[index, end].numpy().astype(np.int),
                    color or OBJECT_PALETTE[name],
                    thickness,
                    cv2.LINE_AA
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 10,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=20,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)
