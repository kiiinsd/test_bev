# projects/bash_runner/vis.shlidar_det_pano='projects/tools/visualize.py'
VIS_PY='projects/tools/visualize.py'
CONFIG_FILE='projects/configs/bevfusion_lidar_det_pano.py'
CHECK_POINT='pretrained/lidar_only_det.pth'
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'

python ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT} --mode pred
# python ${DEBUG_PY} ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT}
