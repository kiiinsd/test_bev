# projects/Meg_dataset/bash_runner/create_data.sh
ROOT_PATH_PROJ='/home/kinsd/test_bev/'
ROOT_PATH_DATASET=${ROOT_PATH_PROJ}'data/panosim'
echo ${ROOT_PATH_DATASET}
python projects/tools/create_data.py pano --root-path ${ROOT_PATH_DATASET} --out-dir ${ROOT_PATH_DATASET} --extra-tag pano