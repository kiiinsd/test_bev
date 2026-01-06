CONFIG=projects/configs/bevfusion_det_pano.py
PTH=runs/pano_single/latest.pth

python projects/tools/realtime/realtime_dataset_.py ${CONFIG} --checkpoint ${PTH} --bbox-score 0.2 --mode pred