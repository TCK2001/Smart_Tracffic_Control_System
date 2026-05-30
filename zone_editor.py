import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


class ZoneEditor:
    def __init__(self, video_path: str, zone_names: List[str], output_path: str,
                 display_width: int = 1280, display_height: int = 720):
        self.video_path = video_path
        self.zone_names = zone_names
        self.output_path = output_path
        self.display_width = display_width
        self.display_height = display_height
        self.display_scale = 1.0
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame = None
        self.paused = False
        self.current_points: List[Tuple[int, int]] = []
        self.saved_zones = []
        self.zone_index = 0
        self.window = "Zone Editor - SPACE pause, click polygon, ENTER next, S save"

    def current_zone_name(self) -> str:
        if self.zone_index < len(self.zone_names):
            return self.zone_names[self.zone_index]
        return f"Zone_{self.zone_index + 1}"

    def display_to_original_point(self, x: int, y: int) -> Tuple[int, int]:
        """Mouse coordinates come from the resized preview; convert them back to original video coords."""
        if self.display_scale <= 0:
            self.display_scale = 1.0
        ox = int(round(x / self.display_scale))
        oy = int(round(y / self.display_scale))
        ox = max(0, min(self.width - 1, ox))
        oy = max(0, min(self.height - 1, oy))
        return ox, oy

    def mouse_callback(self, event, x, y, flags, param):
        if not self.paused:
            return
        ox, oy = self.display_to_original_point(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((ox, oy))
        elif event == cv2.EVENT_RBUTTONDOWN and self.current_points:
            self.current_points.pop()

    def resize_for_display(self, frame):
        if self.display_width <= 0 or self.display_height <= 0:
            self.display_scale = 1.0
            return frame

        h, w = frame.shape[:2]
        self.display_scale = min(self.display_width / w, self.display_height / h, 1.0)
        if self.display_scale >= 1.0:
            self.display_scale = 1.0
            return frame

        new_w = int(w * self.display_scale)
        new_h = int(h * self.display_scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def draw(self, frame):
        canvas = frame.copy()

        for z in self.saved_zones:
            pts = np.array(z["points"], dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
            cv2.polylines(canvas, [pts], True, (0, 255, 255), 2)
            cv2.putText(canvas, z["name"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if self.current_points:
            pts = np.array(self.current_points, dtype=np.int32)
            for p in self.current_points:
                cv2.circle(canvas, p, 4, (0, 0, 255), -1)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, (0, 0, 255), 2)
            if len(pts) >= 3:
                cv2.polylines(canvas, [pts], True, (0, 0, 255), 1)

        status = "PAUSED - draw polygon" if self.paused else "PLAYING - press SPACE to pause"
        lines = [
            status,
            f"Current zone: {self.current_zone_name()} ({self.zone_index + 1}/{len(self.zone_names)})",
            "Left click: add point | Right click: undo | R: reset | ENTER: next | S: save | Q/ESC: quit",
        ]
        y = 28
        for line in lines:
            cv2.rectangle(canvas, (10, y - 22), (10 + 900, y + 7), (0, 0, 0), -1)
            cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            y += 30

        return canvas

    def finish_current_zone(self):
        if len(self.current_points) < 3:
            print("Need at least 3 points for a polygon.")
            return
        zone = {
            "name": self.current_zone_name(),
            "points": [[int(x), int(y)] for x, y in self.current_points],
        }
        self.saved_zones.append(zone)
        print(f"Saved zone: {zone['name']} -> {zone['points']}")
        self.current_points = []
        self.zone_index += 1

    def save(self):
        data = {
            "video_size": {"width": self.width, "height": self.height},
            "zones": self.saved_zones,
        }
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Zones saved to: {self.output_path}")

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.mouse_callback)

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Could not read the first frame.")
        self.frame = frame

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                else:
                    self.paused = True

            canvas = self.draw(self.frame)
            preview = self.resize_for_display(canvas)
            cv2.imshow(self.window, preview)
            key = cv2.waitKey(30 if not self.paused else 10) & 0xFF

            if key == ord(" "):
                self.paused = not self.paused
            elif key in (ord("q"), 27):
                break
            elif key in (ord("r"), ord("R")) and self.paused:
                self.current_points = []
            elif key in (13, 10) and self.paused:
                self.finish_current_zone()
                if self.zone_index >= len(self.zone_names):
                    print("All requested zones are drawn. Press S to save, or continue adding more zones.")
            elif key in (ord("s"), ord("S")):
                if self.current_points:
                    print("You still have an unfinished polygon. Press ENTER to save it first, or R to reset it.")
                else:
                    self.save()
                    break

        self.cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Draw intersection/lane polygons for traffic counting.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--zone-names", default="North,East,South,West", help="Comma-separated zone names")
    parser.add_argument("--output", default="configs/zones.json", help="Output JSON path")
    parser.add_argument("--display-width", type=int, default=1280, help="Max preview window width. Use 0 to disable resize.")
    parser.add_argument("--display-height", type=int, default=720, help="Max preview window height. Use 0 to disable resize.")
    return parser.parse_args()


def main():
    args = parse_args()
    zone_names = [z.strip() for z in args.zone_names.split(",") if z.strip()]
    if not zone_names:
        raise ValueError("No zone names given.")
    editor = ZoneEditor(args.input, zone_names, args.output, args.display_width, args.display_height)
    editor.run()


if __name__ == "__main__":
    main()
