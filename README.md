# YOLO26n-OBB Top-view Intersection Vehicle Counter

This version uses `yolo26n-obb.pt` by default.

## Install

```bash
pip install -U ultralytics
pip install -r requirements.txt
```

## Prepare video

Put your video here:

```text
data/videos/intersection_input.mp4
```

## Draw intersection zones

```bash
python zone_editor.py --input data/videos/intersection_input.mp4 --zone-names North,East,South,West
```

## Run YOLO26n-OBB

```bash
python main.py --input data/videos/intersection_input.mp4 --zones configs/zones.json --device 0 --show --save-video --debug
```

For night/top-view/tiny vehicles:

```bash
python main.py --input data/videos/intersection_input.mp4 --zones configs/zones.json --device 0 --conf 0.03 --imgsz 1024 --enhance-night --tile --tile-size 1024 --tile-overlap 160 --show --save-video --debug --save-debug-frames
```

## Important class names

YOLO26n-OBB is normally DOTA-pretrained. DOTA vehicle classes are usually:

```text
small vehicle
large vehicle
```

This script filters those classes by default. If you want to see every OBB detection for debugging:

```bash
python main.py --input data/videos/intersection_input.mp4 --zones configs/zones.json --device 0 --show --no-class-filter --debug
```

## Output

```text
outputs/{run_time}/
├── traffic_counter_yolo26_obb_result.mp4
├── traffic_counts_yolo26_obb.csv
└── summary_yolo26_obb.json
```
