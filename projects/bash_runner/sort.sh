# projects/bash_runner/vis.shlidar_det_pano='projects/tools/visualize.py'
SORT_PY='projects/tools/sort.py'
CONFIG_FILE='projects/configs/bevfusion_det_pano.py'
CHECK_POINT='runs/2026-02-07_04-41-06-single-pretrained/latest.pth'

python ${SORT_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT} --mode pred --bbox-score 0.2