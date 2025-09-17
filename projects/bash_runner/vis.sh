# projects/bash_runner/vis.shlidar_det_pano='projects/tools/visualize.py'
VIS_PY='projects/tools/visualize.py'
CONFIG_FILE='projects/configs/bevfusion_det_pano_seq_con.py'
CHECK_POINT='runs/2025-07-30_22-35-33/latest.pth'
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'

rm -rf viz
python ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT} --mode pred --bbox-score 0.2
# python ${DEBUG_PY} ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT}
