import os
import mmcv
from os import path as osp

def create_pano_infos(
    root_path, info_prefix
):
    assert osp.exists(root_path)
    lidar_path = root_path + "/lidar"
    val_set = set(
        [
            file for file in os.listdir(lidar_path)
        ]
    )
    train_set = set({})

    tran_infos, val_infos = _fill_trainval_infos(lidar_path, train_set, val_set)
    metadata = dict(version="lidar-test")
    print("train_samples: {}, val_samples: {}".format(len(tran_infos), len(val_infos)))
    
    data = dict(infos = tran_infos, metadata=metadata)
    info_path = osp.join(root_path, "{}_infos_train.pkl".format(info_prefix))
    mmcv.dump(data, info_path)
    data["infos"] = val_infos
    info_path = osp.join(root_path, "{}_infos_val.pkl".format(info_prefix))
    mmcv.dump(data, info_path)


def _fill_trainval_infos(
    root_path, train_set, val_set, test=True
):
    train_infos = []
    val_infos = []

    available_scenes = set.union(train_set, val_set)
    for scene in available_scenes:
        info = {
            "lidar_path": osp.join(root_path, scene),
            "timestamp": int(scene.strip(".pcd.bin"))
        }

        if scene in train_set:
            train_infos.append(info)
        if scene in val_set:
            val_infos.append(info)
    
    return train_infos, val_infos
