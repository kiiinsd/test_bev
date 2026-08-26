# projects/Meg_dataset/bash_runner/train.sh
DATE=$(date '+%Y-%m-%d_%H-%M-%S')
TRAIN_PY='projects/tools/train.py'
CONFIG_FILE='projects/configs/bevfusion_det_pano_seq_align.py'
WORK_DIR="runs/${DATE}/"

python ${TRAIN_PY} ${CONFIG_FILE} --run-dir ${WORK_DIR} #--model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
# torchpack dist-run -np 1 python -m debugpy --listen 8531 --wait-for-client ${TRAIN_PY} ${CONFIG_FILE}
