TEST_PY='projects/tools/test.py'
CONFIG_FILE='projects/configs/bevfusion_det_seq_nuscenes_align.py'
PTH='runs/2026-09-03_09-43-22/latest.pth'
EVAL='object'

python ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval ${EVAL}
# torchpack dist-run -np 1 python -m debugpy --listen 8531 --wait-for-client ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval ${EVAL}