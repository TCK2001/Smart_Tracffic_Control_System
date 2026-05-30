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
data/videos/test2.mp4
```
[Sample Video Link] https://pixabay.com/videos/linear-city-holy-spirit-center-296284/

## Draw intersection zones

```bash
python zone_editor.py --input data/videos/test2.mp4 --zone-names North,East,South,West
```
or 
Execute the `zone.sh` shell file

## Run YOLO26n-OBB
Run `traffic_control_start.sh` shell file the quick start

## Important class names

YOLO26n-OBB is normally DOTA-pretrained. DOTA vehicle classes are usually:

```text
small vehicle
large vehicle
```

This script filters those classes by default. If you want to see every OBB detection for debugging:

## Output

```text
outputs/{run_time}/
├── outputs\20260530_143344\traffic_counts.csv
├── outputs\20260530_143344\traffic_result.mp4
```
