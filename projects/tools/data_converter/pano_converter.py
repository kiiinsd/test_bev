import os
import mmcv
import json
import numpy as np
from os import path as osp

def create_pano_infos(
    root_path, info_prefix
):
    assert osp.exists(root_path)
    lidar_path = root_path + "/samples/LIDAR"
    val_set = set(
        [
            file.strip(".pcd.bin") for file in os.listdir(lidar_path)
        ]
    )
    train_set = set({})

    tran_infos, val_infos = _fill_trainval_infos(root_path, train_set, val_set)
    metadata = dict(version="lidar-test")
    print("train_samples: {}, val_samples: {}".format(len(tran_infos), len(val_infos)))
    
    data = dict(infos = tran_infos, metadata=metadata)
    info_path = osp.join(root_path, "{}_infos_train.pkl".format(info_prefix))
    mmcv.dump(data, info_path)
    data["infos"] = val_infos
    info_path = osp.join(root_path, "{}_infos_val.pkl".format(info_prefix))
    mmcv.dump(data, info_path)


def _fill_trainval_infos(
    root_path, train_set, val_set, test=True, max_sweeps=10
):
    train_infos = []
    val_infos = []
    sample_path = osp.join(root_path, "samples")
    sweep_path = osp.join(root_path, "sweeps")

    cs_path = osp.join(root_path, "calibrated_sensor.json")
    assert osp.exists(cs_path)
    with open(cs_path, "r", encoding="utf-8") as f:
        cs_record = json.load(f)

    pose_path = osp.join(root_path, "ego_pose.json")
    assert osp.exists(pose_path)
    with open(pose_path, "r", encoding="utf-8") as f:
        pose_record = json.load(f)

    timestamps = set.union(train_set, val_set)
    for ts in timestamps:
        info = {
            "lidar_path": osp.join(sample_path, "LIDAR", ts+".pcd.bin"),
            "timestamp": int(ts),
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": cs_record["LIDAR"]["translation"],
            "lidar2ego_rotation": np.matrix(cs_record["LIDAR"]["rotation"]),
            "ego2global_translation": pose_record[ts]["translation"],
            "ego2global_rotation": np.matrix(pose_record[ts]["rotation"]),
        }
        l2e_r_mat = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r_mat = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]

        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        for cam in camera_types:
            camera_intrinsics = cs_record[cam]["cam_intrinsics"]
            cam_info = obtain_sensor2top(
                l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam, sample_path, ts, cs_record, pose_record
            )
            cam_info.update(camera_intrinsics=camera_intrinsics)
            info["cams"].update({cam:cam_info})

        sweeps = []
        while len(sweeps) < max_sweeps:
            curr_ts = ts
            prev_ts = str(pose_record[curr_ts]["prev"])
            if not prev_ts == "0":
                sweep = obtain_sensor2top(
                    l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, "LIDAR", sweep_path, prev_ts, cs_record, pose_record
                )
                sweeps.append(sweep)
                curr_ts = prev_ts
            else:
                break
        info["sweeps"] = sweeps

        if ts in train_set:
            train_infos.append(info)
        if ts in val_set:
            val_infos.append(info)
    
    return train_infos, val_infos

def obtain_sensor2top(
    l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type, root_path, ts, cs_record, pose_record
):
    suffix = ".pcd.bin" if sensor_type == "LIDAR" else ".jpg"
    data_path = osp.join(root_path, sensor_type, ts+suffix)
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sensor2ego_translation": cs_record[sensor_type]["translation"],
        "sensor2ego_rotation": np.matrix(cs_record[sensor_type]["rotation"]),
        "ego2global_translation": pose_record[ts]["translation"],
        "ego2global_rotation": np.matrix(pose_record[ts]["rotation"]),
        "timestamp": int(ts)
    }
    l2e_r_s_mat = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s_mat = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep    
