#!/bin/bash

python traffic_control.py \
  --input data/videos/test2.mp4 \
  --zones configs/zones.json \
  --model yolo26x-obb.pt \
  --device 0 \
  --conf 0.05 \
  --imgsz 1024 \
  --show \
  --save-video \
  --initial-green NS \
  --ns-zones North,South \
  --ew-zones East,West \
  --min-green 8 \
  --switch-margin 2 \
  --sec-per-vehicle 1.5