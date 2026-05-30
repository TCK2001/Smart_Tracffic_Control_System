import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


VEHICLE_NAMES = {
    "small vehicle",
    "large vehicle",
    "car",
    "van",
    "truck",
    "bus",
    "motor",
    "motorcycle",
    "tricycle",
    "awning-tricycle",
}


@dataclass
class Zone:
    name: str
    points: np.ndarray  # shape: (N, 2), int32


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    conf: float
    cls_id: int
    cls_name: str
    polygon: Optional[np.ndarray] = None  # shape: (4, 2), int32 for OBB


def normalize_name(name: str) -> str:
    return " ".join(str(name).lower().replace("_", "-").split())


def load_zones(path: str) -> List[Zone]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    zones = []
    for item in data.get("zones", []):
        pts = np.array(item["points"], dtype=np.int32)
        if len(pts) < 3:
            raise ValueError(f"Zone {item.get('name')} has fewer than 3 points.")
        zones.append(Zone(name=item["name"], points=pts))

    if not zones:
        raise ValueError("No zones found in the zones JSON file.")
    return zones


def make_run_dir(base: str = "outputs") -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def point_in_zone(point: Tuple[int, int], zone: Zone) -> bool:
    return cv2.pointPolygonTest(zone.points, point, False) >= 0


def draw_transparent_polygon(frame: np.ndarray, points: np.ndarray, alpha: float = 0.18) -> None:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], (0, 255, 255))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [points], True, (0, 255, 255), 2)


def draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.7,
    thickness: int = 2,
    fg=(255, 255, 255),
    bg=(0, 0, 0),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    x = max(0, min(int(x), frame.shape[1] - 1))
    y = max(th + baseline + 4, min(int(y), frame.shape[0] - 1))
    cv2.rectangle(frame, (x - 4, y - th - baseline - 4), (x + tw + 4, y + baseline + 4), bg, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, fg, thickness, cv2.LINE_AA)


# ── zone crop inference ───────────────────────────────────────────────────────

def zone_bounding_box(zone: Zone, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) bounding box of a zone, clamped to frame."""
    h, w = frame_shape[:2]
    x1 = int(np.clip(np.min(zone.points[:, 0]), 0, w - 1))
    y1 = int(np.clip(np.min(zone.points[:, 1]), 0, h - 1))
    x2 = int(np.clip(np.max(zone.points[:, 0]), 0, w - 1))
    y2 = int(np.clip(np.max(zone.points[:, 1]), 0, h - 1))
    return x1, y1, x2, y2


def predict_zone_crops(
    model: YOLO,
    frame_for_model: np.ndarray,
    zones: List[Zone],
    conf: float,
    imgsz: int,
    device: str,
    class_ids: Optional[List[int]],
) -> List[Detection]:
    """Crop each zone's bounding box, run YOLO, restore original coordinates,
    then keep only detections whose center falls inside the zone polygon."""
    all_detections: List[Detection] = []
    seen_boxes: List[List[int]] = []

    for zone in zones:
        bx1, by1, bx2, by2 = zone_bounding_box(zone, frame_for_model.shape)
        if bx2 <= bx1 or by2 <= by1:
            continue

        crop = frame_for_model[by1:by2, bx1:bx2]

        result = model.predict(
            source=crop,
            conf=conf,
            imgsz=imgsz,
            classes=class_ids,
            device=device,
            verbose=False,
        )[0]

        raw = parse_model_result(result, model, x_offset=bx1, y_offset=by1)

        for det in raw:
            # keep only detections whose center is inside this zone polygon
            if not point_in_zone((det.cx, det.cy), zone):
                continue
            # simple dedup: skip if nearly identical box already added
            box = [det.x1, det.y1, det.x2, det.y2]
            duplicate = False
            for sb in seen_boxes:
                if (abs(box[0] - sb[0]) < 8 and abs(box[1] - sb[1]) < 8 and
                        abs(box[2] - sb[2]) < 8 and abs(box[3] - sb[3]) < 8):
                    duplicate = True
                    break
            if not duplicate:
                all_detections.append(det)
                seen_boxes.append(box)

    return all_detections


# ── signal grouping / timing helpers ─────────────────────────────────────────

def split_zone_names(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def resolve_zone_names(available_names: List[str], requested: List[str]) -> List[str]:
    lookup = {normalize_name(name): name for name in available_names}
    resolved = []
    for name in requested:
        key = normalize_name(name)
        if key in lookup:
            resolved.append(lookup[key])
        else:
            print(f"WARNING: requested zone '{name}' not found. Available={available_names}")
    return resolved


def count_group(counts: Dict[str, int], zone_names: List[str]) -> int:
    return int(sum(counts.get(name, 0) for name in zone_names))


def mean_queue(q) -> float:
    return float(sum(q) / len(q)) if q else 0.0


def decide_signal_change(
    current_green: str,
    ns_smooth: float,
    ew_smooth: float,
    elapsed_green_sec: float,
    min_green_sec: float,
    switch_margin: float,
    sec_per_vehicle: float,
    max_gain_time: float,
) -> Tuple[bool, str, float, float, str]:
    current_green = current_green.upper()
    if current_green == "EW":
        green_demand, red_demand, target_green = ew_smooth, ns_smooth, "NS"
    else:
        green_demand, red_demand, target_green = ns_smooth, ew_smooth, "EW"

    demand_diff = red_demand - green_demand
    gain_time_sec = min(max_gain_time, max(0.0, demand_diff) * sec_per_vehicle)

    if elapsed_green_sec < min_green_sec:
        return False, current_green, demand_diff, gain_time_sec, "KEEP_MIN_GREEN"
    if demand_diff >= switch_margin:
        return True, target_green, demand_diff, gain_time_sec, "CHANGE_TIMING"
    return False, current_green, demand_diff, gain_time_sec, "KEEP"


def draw_signal_panel(
    frame: np.ndarray,
    current_green: str,
    ns_count: int,
    ew_count: int,
    ns_smooth: float,
    ew_smooth: float,
    elapsed_green_sec: float,
    should_switch: bool,
    target_green: str,
    demand_diff: float,
    gain_time_sec: float,
    reason: str,
) -> None:
    current_green = current_green.upper()
    signal_text = "NS GREEN / EW RED" if current_green == "NS" else "EW GREEN / NS RED"
    bg = (0, 110, 0) if current_green == "NS" else (0, 90, 160)

    draw_text_with_bg(frame, f"Signal: {signal_text}", (18, 30), font_scale=0.75, thickness=2, bg=bg)
    draw_text_with_bg(
        frame,
        f"NS={ns_count} avg={ns_smooth:.1f} | EW={ew_count} avg={ew_smooth:.1f} | elapsed={elapsed_green_sec:.1f}s",
        (18, 62),
        font_scale=0.62,
        thickness=2,
        bg=(30, 30, 30),
    )

    if should_switch:
        banner = f"SIGNAL CHANGE: {current_green} -> {target_green} | gain={gain_time_sec:.1f}s | diff={demand_diff:.1f}"
        cv2.rectangle(frame, (0, 82), (frame.shape[1], 132), (0, 0, 180), -1)
        cv2.putText(frame, banner, (18, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        draw_text_with_bg(
            frame,
            f"Decision: {reason} | potential_gain={gain_time_sec:.1f}s | diff={demand_diff:.1f}",
            (18, 94),
            font_scale=0.58,
            thickness=1,
            bg=(45, 45, 45),
        )


def resize_for_display(frame: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    if max_width <= 0 or max_height <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def enhance_night_frame(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return cv2.convertScaleAbs(enhanced, alpha=1.15, beta=12)


def get_class_ids(model: YOLO, wanted_names: set) -> List[int]:
    wanted = {normalize_name(x) for x in wanted_names}
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return [int(idx) for idx, name in items if normalize_name(str(name)) in wanted]


def polygon_to_xyxy(poly: np.ndarray) -> Tuple[int, int, int, int]:
    return (int(np.min(poly[:, 0])), int(np.min(poly[:, 1])),
            int(np.max(poly[:, 0])), int(np.max(poly[:, 1])))


def parse_obb_result(result, model: YOLO, x_offset: int = 0, y_offset: int = 0) -> List[Detection]:
    detections: List[Detection] = []
    if getattr(result, "obb", None) is None or result.obb is None or len(result.obb) == 0:
        return detections

    polys = result.obb.xyxyxyxy.cpu().numpy()
    confs = result.obb.conf.cpu().numpy()
    cls_ids = result.obb.cls.cpu().numpy().astype(int)

    for poly, conf, cls_id in zip(polys, confs, cls_ids):
        poly = np.asarray(poly, dtype=np.float32)
        poly[:, 0] += x_offset
        poly[:, 1] += y_offset
        poly_i = np.round(poly).astype(np.int32)
        x1, y1, x2, y2 = polygon_to_xyxy(poly_i)
        center = poly_i.mean(axis=0)
        detections.append(Detection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            cx=int(center[0]), cy=int(center[1]),
            conf=float(conf), cls_id=int(cls_id),
            cls_name=str(model.names[int(cls_id)]),
            polygon=poly_i,
        ))
    return detections


def parse_box_result(result, model: YOLO, x_offset: int = 0, y_offset: int = 0) -> List[Detection]:
    detections: List[Detection] = []
    if getattr(result, "boxes", None) is None or result.boxes is None:
        return detections

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
        conf = float(box.conf[0].cpu().item())
        cls_id = int(box.cls[0].cpu().item())
        detections.append(Detection(
            x1=x1 + x_offset, y1=y1 + y_offset,
            x2=x2 + x_offset, y2=y2 + y_offset,
            cx=int((x1 + x2) / 2) + x_offset,
            cy=int((y1 + y2) / 2) + y_offset,
            conf=conf, cls_id=cls_id,
            cls_name=str(model.names[cls_id]),
            polygon=None,
        ))
    return detections


def parse_model_result(result, model: YOLO, x_offset: int = 0, y_offset: int = 0) -> List[Detection]:
    obb = parse_obb_result(result, model, x_offset=x_offset, y_offset=y_offset)
    return obb if obb else parse_box_result(result, model, x_offset=x_offset, y_offset=y_offset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zone-crop YOLO traffic counter with signal timing.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--zones", default="configs/zones.json", help="Zone JSON made by zone_editor.py")
    parser.add_argument("--model", default="yolo26n-obb.pt", help="OBB or detection model path")
    parser.add_argument("--device", default="cpu", help="cpu or GPU index e.g. 0")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="YOLO inference image size")
    parser.add_argument("--show", action="store_true", help="Show realtime OpenCV window")
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument("--save-video", action="store_true", help="Save annotated output video")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Only process first N seconds. 0=full.")
    parser.add_argument("--no-class-filter", action="store_true", help="Do not filter vehicle classes")
    parser.add_argument("--enhance-night", action="store_true", help="Apply CLAHE/brightness enhancement")
    parser.add_argument("--debug", action="store_true", help="Print detection debug info")
    parser.add_argument("--save-debug-frames", action="store_true", help="Save first annotated frame images")
    # signal
    parser.add_argument("--ns-zones", default="North,South")
    parser.add_argument("--ew-zones", default="East,West")
    parser.add_argument("--initial-green", choices=["NS", "EW"], default="EW")
    parser.add_argument("--min-green", type=float, default=8.0)
    parser.add_argument("--switch-margin", type=float, default=2.0)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--sec-per-vehicle", type=float, default=1.5)
    parser.add_argument("--max-gain-time", type=float, default=30.0)
    parser.add_argument("--advice-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zones = load_zones(args.zones)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(args.max_seconds * fps) if args.max_seconds > 0 else None

    out_dir = make_run_dir("outputs")
    csv_path = out_dir / "traffic_counts.csv"
    video_path = out_dir / "traffic_result.mp4"

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    model = YOLO(args.model)
    class_ids = None if args.no_class_filter else get_class_ids(model, VEHICLE_NAMES)
    if class_ids == []:
        print("Model classes:", model.names)
        raise RuntimeError(
            "Could not find vehicle classes. Try --no-class-filter to inspect all detections."
        )

    print(f"Input: {width}x{height} @ {fps:.2f} fps")
    print(f"Model: {args.model} | Class filter: {class_ids if class_ids is not None else 'OFF'}")
    print(f"conf={args.conf}, imgsz={args.imgsz}, enhance_night={args.enhance_night}")
    print(f"Zone-crop inference: only {[z.name for z in zones]} regions will be processed")

    available_zone_names = [z.name for z in zones]
    ns_zone_names = resolve_zone_names(available_zone_names, split_zone_names(args.ns_zones))
    ew_zone_names = resolve_zone_names(available_zone_names, split_zone_names(args.ew_zones))
    if not ns_zone_names or not ew_zone_names:
        raise RuntimeError(
            f"NS/EW zone names invalid. Available={available_zone_names}, "
            f"NS={ns_zone_names}, EW={ew_zone_names}"
        )
    print(f"Signal groups — NS={ns_zone_names}, EW={ew_zone_names}, initial_green={args.initial_green}")

    # ── CSV: only the columns we care about ──────────────────────────────────
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_idx", "time_sec",
        "NS_count", "EW_count",
        "recommended_green", "gain_time_sec",
    ])

    frame_idx = 0
    ns_history = deque(maxlen=max(1, int(args.smooth_window)))
    ew_history = deque(maxlen=max(1, int(args.smooth_window)))
    current_green = args.initial_green.upper()
    current_green_start_sec = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            frame_for_model = enhance_night_frame(frame) if args.enhance_night else frame

            # ── Zone-crop inference ───────────────────────────────────────
            detections = predict_zone_crops(
                model=model,
                frame_for_model=frame_for_model,
                zones=zones,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                class_ids=class_ids,
            )

            # Count per zone (center already confirmed inside zone by predict_zone_crops)
            counts: Dict[str, int] = {z.name: 0 for z in zones}
            detections_with_zone: List[Tuple[Detection, Optional[str]]] = []
            for det in detections:
                for zone in zones:
                    if point_in_zone((det.cx, det.cy), zone):
                        counts[zone.name] += 1
                        detections_with_zone.append((det, zone.name))
                        break
                else:
                    detections_with_zone.append((det, None))

            if args.debug and frame_idx < 10:
                print(f"[DEBUG] frame={frame_idx}, detections={len(detections)}, counts={counts}")

            time_sec = frame_idx / fps
            ns_count = count_group(counts, ns_zone_names)
            ew_count = count_group(counts, ew_zone_names)
            ns_history.append(ns_count)
            ew_history.append(ew_count)
            ns_smooth = mean_queue(ns_history)
            ew_smooth = mean_queue(ew_history)
            elapsed_green_sec = max(0.0, time_sec - current_green_start_sec)

            should_switch, target_green, demand_diff, gain_time_sec, decision = decide_signal_change(
                current_green=current_green,
                ns_smooth=ns_smooth,
                ew_smooth=ew_smooth,
                elapsed_green_sec=elapsed_green_sec,
                min_green_sec=args.min_green,
                switch_margin=args.switch_margin,
                sec_per_vehicle=args.sec_per_vehicle,
                max_gain_time=args.max_gain_time,
            )

            recommended_green = target_green if should_switch else current_green

            if should_switch and not args.advice_only:
                current_green = target_green
                current_green_start_sec = time_sec

            # ── CSV row ───────────────────────────────────────────────────
            csv_writer.writerow([
                frame_idx,
                f"{time_sec:.2f}",
                ns_count,
                ew_count,
                recommended_green,
                f"{gain_time_sec:.2f}",
            ])

            # ── Annotation ───────────────────────────────────────────────
            annotated = frame.copy()

            for zone in zones:
                draw_transparent_polygon(annotated, zone.points)
                label_pos = tuple(zone.points[0].tolist())
                draw_text_with_bg(annotated, f"{zone.name}: {counts[zone.name]}", label_pos)

            for det, matched_zone in detections_with_zone:
                color = (0, 255, 0) if matched_zone else (160, 160, 160)
                if det.polygon is not None:
                    cv2.polylines(annotated, [det.polygon], True, color, 2)
                else:
                    cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2)
                cv2.circle(annotated, (det.cx, det.cy), 4, (0, 0, 255), -1)

            draw_signal_panel(
                annotated,
                current_green=current_green,
                ns_count=ns_count, ew_count=ew_count,
                ns_smooth=ns_smooth, ew_smooth=ew_smooth,
                elapsed_green_sec=elapsed_green_sec,
                should_switch=should_switch,
                target_green=target_green,
                demand_diff=demand_diff,
                gain_time_sec=gain_time_sec,
                reason=decision,
            )

            # compact zone summary
            y = 150
            draw_text_with_bg(annotated, f"Total in zones: {sum(counts.values())}", (18, y),
                               font_scale=0.58, thickness=1, bg=(30, 30, 30))
            y += 25
            for name, cnt in counts.items():
                draw_text_with_bg(annotated, f"{name}: {cnt}", (18, y),
                                   font_scale=0.58, thickness=1, bg=(30, 30, 30))
                y += 25

            if args.save_debug_frames and frame_idx in (0, 1, 2):
                cv2.imwrite(str(out_dir / f"debug_frame_{frame_idx:04d}.jpg"), annotated)

            if writer is not None:
                writer.write(annotated)

            if args.show:
                preview = resize_for_display(annotated, args.display_width, args.display_height)
                cv2.imshow("Traffic Counter", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            frame_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        csv_file.close()
        cv2.destroyAllWindows()

    print("Done.")
    print(f"Processed frames : {frame_idx}")
    print(f"CSV saved to     : {csv_path}")
    if args.save_video:
        print(f"Video saved to   : {video_path}")


if __name__ == "__main__":
    main()