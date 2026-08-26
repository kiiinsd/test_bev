CONFIG=projects/configs/bevfusion_det_pano.py
PTH=runs/2026-05-11_16-27-53-single/latest.pth

python projects/tools/realtime/realtime_dataset_.py ${CONFIG} --checkpoint ${PTH} --bbox-score 0.2 --mode pred