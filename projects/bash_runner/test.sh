TEST_PY='projects/tools/test.py'
CONFIG_FILE='projects/configs/bevfusion_det_pano_seq_align.py'
PTH='runs/pano_seq_align/latest.pth'
EVAL='object'

python ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval ${EVAL}
# torchpack dist-run -np 1 python -m debugpy --listen 8531 --wait-for-client ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval ${EVAL}