# projects/bash_runner/vis.shlidar_det_pano='projects/tools/visualize.py'
VIS_PY='projects/tools/visualize.py'
CONFIG_FILE='projects/configs/bevfusion_det_pano_seq_align.py'
CHECK_POINT='runs/pano_seq_align/latest.pth'
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'

rm -rf viz-gt
python ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT} --mode gt --bbox-score 0.2 --out-dir viz-gt
# python ${DEBUG_PY} ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT}
