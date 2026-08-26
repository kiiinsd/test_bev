import os
from os import path as osp

import mmcv
import json
import numpy as np
from pyquaternion import Quaternion

from projects.mmdet3d.datasets import PanoDataset
from projects.panosim import PanoSim
from projects.panosim.utils.splits import create_splits_scenes

def create_pano_infos(
    root_path, info_prefix, version='v1.0', max_sweeps=9
):
    pano = PanoSim(data_root=root_path, version=version)
    available_scenes = get_available_scenes(pano)
    available_scene_names = [s['name'] for s in available_scenes]
    
    split = create_splits_scenes()
    train_scenes = split['train']
    val_scenes = split['val']

    train_scenes = set(
        [available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes]
    )

    tran_infos, val_infos = _fill_trainval_infos(pano, train_scenes, val_scenes)
    metadata = dict(version='v1.0')
    print('train_samples: {}, val_samples: {}'.format(len(tran_infos), len(val_infos)))
    
    data = dict(infos = tran_infos, metadata=metadata)
    info_path = osp.join(root_path, '{}_infos_train.pkl'.format(info_prefix))
    mmcv.dump(data, info_path)
    data['infos'] = val_infos
    info_path = osp.join(root_path, '{}_infos_val.pkl'.format(info_prefix))
    mmcv.dump(data, info_path)

def get_available_scenes(pano):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print("total scene num: {}".format(len(pano.scene)))
    for scene in pano.scene:
        scene_token = scene["token"]
        scene_rec = pano.get("scene", scene_token)
        sample_rec = pano.get("sample", scene_rec["first_sample_token"])
        sd_rec = pano.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = pano.get_sample_data(sd_rec["token"])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num: {}".format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(
    pano: PanoSim, train_scenes, val_scenes, test=False, max_sweeps=9
):
    train_info = []
    val_info = []

    prev_sample = pano.sample[0]
    for sample in mmcv.track_iter_progress(pano.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = pano.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = pano.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = pano.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = pano.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            # 'ego_vel': pose_record['vel'],
            'timestamp': sample['timestamp'],
            # 'location': location,
            'scene_token': sample['scene_token'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, camera_intrinsics = pano.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(
                pano, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
            )
            cam_info.update(camera_intrinsics=camera_intrinsics)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = pano.get("sample_data", sample["data"]["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec["prev"] == "":
                sweep = obtain_sensor2top(
                    pano, sd_rec["prev"], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, "lidar"
                )
                sweeps.append(sweep)
                sd_rec = pano.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps

        # obtain annotation
        if not test:
            annotations = [
                pano.get("sample_annotation", token) for token in sample["anns"]
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
                -1, 1
            )
            velocity = np.array(
                [anno['vel'] for anno in annotations]
            )
            trans = np.zeros((len(annotations), 3))
            if prev_sample['scene_token'] == sample['scene_token']:
                prev_annos = [
                    pano.get('sample_annotation', token) for token in prev_sample['anns']
                ]
                for i, anno in enumerate(annotations):
                    for prev_anno in prev_annos:
                        if anno['instance_token'] == prev_anno['instance_token']:
                            trans[i] = np.array(anno['translation']) - np.array(prev_anno['translation'])
                            trans[i] = trans[i] @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                            break
            prev_sample = sample
                
            valid_flag = np.array([anno["num_lidar_pts"] > 0 for anno in annotations],
                                  dtype=bool,).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in PanoDataset.NameMapping:
                    names[i] = PanoDataset.NameMapping[names[i]]
            names = np.array(names)
            # we need to convert rot to SECOND format.
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                annotations
            ), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["gt_trans"] = trans[:,:2]
            info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
            # info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
            # info["valid_flag"] = np.array([True for _ in annotations], dtype=bool,).reshape(-1)
            info["valid_flag"] = valid_flag

        if sample["scene_token"] in train_scenes:
            train_info.append(info)
        if sample['scene_token'] in val_scenes:
            val_info.append(info)
    return train_info, val_info

def obtain_sensor2top(
    pano, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        pano (class): Dataset class in the PanoSim dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = pano.get("sample_data", sensor_token)
    cs_record = pano.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = pano.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(pano.get_sample_data_path(sd_rec["token"]))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
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