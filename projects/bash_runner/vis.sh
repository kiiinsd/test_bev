# projects/bash_runner/vis.shlidar_det_pano='projects/tools/visualize.py'
VIS_PY='projects/tools/visualize.py'
CONFIG_FILE='projects/configs/bevfusion_lidar_det_pano.py'
CHECK_POINT='run-d6cb076f-9c530828/latest.pth'
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'

python ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT} --mode pred --bbox-score 0.2
# python ${DEBUG_PY} ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT}
