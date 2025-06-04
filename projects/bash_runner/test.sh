TEST_PY='projects/tools/test.py'
CONFIG_FILE='projects/configs/bevfusion_det_pano.py'
PTH='runs/2025-05-15_00-27-12/latest.pth'
EVAL='object'

python ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval ${EVAL}
# torchpack dist-run -np 1 python -m debugpy --listen 8531 --wait-for-client ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval ${EVAL}