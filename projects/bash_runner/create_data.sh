# projects/Meg_dataset/bash_runner/create_data.sh
new_data=0
ROOT_PATH_PROJ='/home/workstation/test_bev/'
ROOT_PATH_DATASET=${ROOT_PATH_PROJ}'data/nuscenes'
echo ${ROOT_PATH_DATASET}
if [ $new_data -eq 1 ]; then
    echo 'remove previous data'
    rm -rf ~/panosim
    echo 'copy new data'
    mkdir ~/panosim
    cd /mnt/e/panosim
    cp -r $(ls | grep -v '.conda') ~/panosim/
    cd ~/test_bev
fi
python tools/create_data.py nuscenes --root-path ${ROOT_PATH_DATASET} --out-dir ${ROOT_PATH_DATASET} --extra-tag nuscenes