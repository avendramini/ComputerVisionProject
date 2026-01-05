"""
Linear interpolation for missing YOLO labels.
Reads labels from dataset/infer_video/labels_outX/ and fills frames where
a class is missing using linear interpolation from previous/next frames.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import json


def load_labels_for_camera(labels_dir_or_json):
    """
    Load all labels for a camera from a directory of .txt files or a JSON file.

    Args:
        labels_dir_or_json: Path to labels_outX/ directory or labels_outX.json file

    Returns:
        frames: dict {frame_idx: {class_id: (x, y, w, h, conf)}}
        max_frame: int, last frame index
    """
    path = Path(labels_dir_or_json)
    frames = {}
    max_frame = -1
    
    # Try to load as JSON
    if path.is_file() and path.suffix == '.json':
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        for fi_str, items in data.get("frames", {}).items():
            fi = int(fi_str)
            max_frame = max(max_frame, fi)
            frame_dets = {}
            for item in items:
                cls_id = int(item["class_id"])
                x = float(item["x"])
                y = float(item["y"])
                w = float(item["w"])
                h = float(item["h"])
                conf = float(item.get("conf", 1.0))
                frame_dets[cls_id] = (x, y, w, h, conf)
            frames[fi] = frame_dets
        return frames, max_frame
    
    # Otherwise, load from .txt directory
    labels_path = Path(labels_dir_or_json)
    if not labels_path.exists():
        raise FileNotFoundError(f"Directory/file not found: {labels_dir_or_json}")
    
    # Read all frame_XXXXX.txt files
    for label_file in sorted(labels_path.glob("frame_*.txt")):
        frame_idx = int(label_file.stem.split('_')[1])
        max_frame = max(max_frame, frame_idx)
        frame_dets = {}
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    conf = float(parts[5])
                    frame_dets[cls_id] = (x, y, w, h, conf)
        frames[frame_idx] = frame_dets
    
    return frames, max_frame


def interpolate_missing_detections(frames, max_frame, max_gap=10):
    """
    Interpolates missing detections for each class.
    
    Args:
        frames: dict {frame_idx: {class_id: [x, y, w, h, conf]}}
        max_frame: last frame index
        max_gap: maximum number of consecutive missing frames to interpolate
    
    Returns:
        interpolated_frames: complete dict with interpolated detections
    """
    interpolated = defaultdict(dict)
    
    # Copy existing frames
    for frame_idx, detections in frames.items():
        interpolated[frame_idx] = detections.copy()
    
    # Find all present classes
    all_classes = set()
    for detections in frames.values():
        all_classes.update(detections.keys())
    
    print(f"[INFO] Found classes: {sorted(all_classes)}")
    
    # For each class, interpolate missing frames
    total_interpolated = 0
    for cls_id in sorted(all_classes):
        # Find all frames where this class is present
        frames_with_class = sorted([f for f in frames.keys() if cls_id in frames[f]])
        if len(frames_with_class) < 2:
            continue
        interpolated_count = 0
        # Interpolate only between consecutive frames with gap <= max_gap
        for i in range(len(frames_with_class) - 1):
            frame_start = frames_with_class[i]
            frame_end = frames_with_class[i + 1]
            gap = frame_end - frame_start - 1
            if gap == 0 or gap > max_gap:
                continue
            det_start = np.array(frames[frame_start][cls_id][:4])
            det_end = np.array(frames[frame_end][cls_id][:4])
            conf_start = frames[frame_start][cls_id][4]
            conf_end = frames[frame_end][cls_id][4]
            for j in range(1, gap + 1):
                frame_idx = frame_start + j
                alpha = j / (gap + 1)
                det_interp = (1 - alpha) * det_start + alpha * det_end
                conf_interp = (1 - alpha) * conf_start + alpha * conf_end
                # Only interpolate if this class is missing in this frame
                if cls_id not in interpolated[frame_idx]:
                    interpolated[frame_idx][cls_id] = [
                        det_interp[0], det_interp[1], det_interp[2], det_interp[3], conf_interp
                    ]
                    interpolated_count += 1
        total_interpolated += interpolated_count
    
    print(f"[INFO] Interpolated {total_interpolated} detections for this camera.")
    return interpolated


def save_interpolated_json(frames, output_file: Path, camera_id: int | None = None, coord_mode: str = "pixel"):
    """
    Saves a single JSON file with all detections for a camera.

    Schema:
    {
      "camera": <int|null>,
      "coord_mode": "pixel"|"normalized",
      "frames": {
         "0": [ {"class_id": int, "x": float, "y": float, "w": float, "h": float, "conf": float}, ...],
         ...
      }
    }
    """
    payload = {
        "camera": int(camera_id) if camera_id is not None else None,
        "coord_mode": coord_mode,
        "frames": {}
    }
    for frame_idx, dets in frames.items():
        items = []
        for cls_id, vals in dets.items():
            x, y, w, h, conf = vals
            items.append({
                "class_id": int(cls_id),
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "conf": float(conf)
            })
        payload["frames"][str(frame_idx)] = items
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[INFO] JSON saved: {output_file}")



