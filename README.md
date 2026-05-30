# YOLO26n-OBB Top-view Intersection Vehicle Counter

This version uses `yolo26x-obb.pt` by default.

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

<img width="1275" height="748" alt="image" src="https://github.com/user-attachments/assets/e7f678c6-5882-4f20-bdab-e88af3cc1aa9" />

## Run YOLO26n-OBB
Run `traffic_control_start.sh` shell file the quick start

## Shell File Description
```python traffic_control.py	Runs the main traffic control Python script.
--input data/videos/test2.mp4	Specifies the input traffic video.
--zones configs/zones.json	Loads the predefined intersection zones, such as North, South, East, and West.
--model yolo26x-obb.pt	Uses the YOLO26x OBB model for top-view vehicle detection.
--device 0	Runs the model on GPU 0. Use cpu if you do not have a GPU.
--conf 0.05	Sets the confidence threshold for detection. A lower value helps detect small vehicles in top-view footage.
--imgsz 1024	Sets the image size used for YOLO inference. Larger values may improve small object detection but require more GPU memory.
--show	Displays the real-time detection window.
--save-video	Saves the output video with detection boxes, zone counts, and signal timing information.
--initial-green NS	Sets the initial green-light direction to North-South.
--ns-zones North,South	Groups the North and South zones as one traffic direction.
--ew-zones East,West	Groups the East and West zones as one traffic direction.
--min-green 8	Keeps the current green light for at least 8 seconds before allowing a signal change.
--switch-margin 2	Changes the signal only when the red-light direction has at least 2 more vehicles than the green-light direction.
```

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
