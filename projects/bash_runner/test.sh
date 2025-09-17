TEST_PY='projects/tools/test.py'
CONFIG_FILE='projects/configs/bevfusion_det_pano_seq_con.py'
PTH='runs/2025-08-03_09-58-04/latest.pth'
EVAL='object'

python ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval ${EVAL}
# torchpack dist-run -np 1 python -m debugpy --listen 8531 --wait-for-client ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval ${EVAL}