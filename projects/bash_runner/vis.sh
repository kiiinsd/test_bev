# projects/bash_runner/vis.shlidar_det_pano='projects/tools/visualize.py'
VIS_PY='projects/tools/visualize.py'
CONFIG_FILE='projects/configs/bevfusion_det_pano.py'
CHECK_POINT='runs/pano_bevfusion/latest.pth'
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'

rm -rf viz-single
python ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT} --mode both --bbox-score 0.2 --out-dir viz-single
# python ${DEBUG_PY} ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT}
