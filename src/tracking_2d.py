"""
Lightweight 2D tracking on YOLO labels per frame (per-camera) with IoU assignment.

Offers reusable functions for the pipeline and (optional) a main for video inference.
"""

import random
import argparse
import re
from pathlib import Path
import cv2
from ultralytics import YOLO
import json
from typing import Dict, List, Tuple

Det = Tuple[float, float, float, float, float]  # x,y,w,h,conf
PerFrame = Dict[int, Dict[int, Det]]  # frame -> {class_id: (x,y,w,h,conf)}


def _xywh_to_xyxy(b):
    x, y, w, h = b
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return x1, y1, x2, y2


def _iou(b1, b2) -> float:
    x11, y11, x12, y12 = _xywh_to_xyxy(b1)
    x21, y21, x22, y22 = _xywh_to_xyxy(b2)
    xi1 = max(x11, x21)
    yi1 = max(y11, y21)
    xi2 = min(x12, x22)
    yi2 = min(y12, y22)
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter = inter_w * inter_h
    a1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
    a2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def track_labels_for_camera(frames: PerFrame, iou_thresh: float = 0.3, max_age: int = 15) -> Dict[int, Dict[int, Dict]]:
    """
    Assigns track_id for each class on sequential frames using IoU matching (greedy).

    Args:
        frames: {frame_idx: {class_id: (x,y,w,h,conf)}}
        iou_thresh: minimum threshold to associate with an existing track
        max_age: maximum age (frames) to keep a track alive without match

    Returns:
        tracks_by_frame: {frame_idx: {class_id: {x,y,w,h,conf,track_id}}}
    """
    next_id = 1
    # Active tracks per class: {class_id: List[dict(track_id, bbox, age)]}
    active: Dict[int, List[Dict]] = {}
    out: Dict[int, Dict[int, Dict]] = {}

    for fi in sorted(frames.keys()):
        out[fi] = {}
        dets = frames[fi]  # {cls: [x,y,w,h,conf]}
        # Age tracks
        for cls in list(active.keys()):
            for tr in active[cls]:
                tr["age"] += 1
        # For each class separately
        for cls_id, vals in dets.items():
            x, y, w, h, conf = vals
            bbox = (x, y, w, h)
            if cls_id not in active:
                active[cls_id] = []
            # Find best match by IoU
            best_iou = 0.0
            best_tr = None
            for tr in active[cls_id]:
                iou = _iou(tr["bbox"], bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tr = tr
            if best_tr is not None and best_iou >= iou_thresh:
                # update existing track
                best_tr["bbox"] = bbox
                best_tr["age"] = 0
                track_id = best_tr["id"]
            else:
                # new track
                track_id = next_id
                next_id += 1
                active[cls_id].append({"id": track_id, "bbox": bbox, "age": 0})
            out[fi][cls_id] = {
                "x": float(x), "y": float(y), "w": float(w), "h": float(h),
                "conf": float(conf), "track_id": int(track_id)
            }
        # remove too old tracks
        for cls in list(active.keys()):
            active[cls] = [tr for tr in active[cls] if tr["age"] <= max_age]
    return out


def save_tracks_json(tracks_by_frame: Dict[int, Dict[int, Dict]], output_file: Path, camera_id: int | None = None, coord_mode: str = "pixel"):
    payload = {
        "camera": int(camera_id) if camera_id is not None else None,
        "coord_mode": coord_mode,
        "frames": {}
    }
    for fi in sorted(tracks_by_frame.keys()):
        items = []
        for cls_id, it in tracks_by_frame[fi].items():
            item = {"class_id": int(cls_id), **it}
            items.append(item)
        payload["frames"][str(fi)] = items
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[TRACK-2D] JSON saved: {output_file}")


def run_video_inference(video_path: Path, model_path: str, labels_base: Path, device: str = 'auto', conf_thres: float = 0.25, imgsz: int = 1920):
    """
    Runs YOLO inference on a video and saves labels in a single JSON per camera.
    Extracts camera number from video name (e.g. out13.mp4 -> cam 13).
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLO model weights
        labels_base: Directory where to save labels JSON
        device: Inference device ('auto', 'cpu', '0', etc.)
        conf_thres: Confidence threshold to filter predictions (default: 0.25)
        imgsz: Image size for inference (default: 1920)
    """
    print(f"[INFERENCE] Loading model {model_path}...")
    model = YOLO(model_path)
    
    video_stem = video_path.stem  # es. out13
    m = re.search(r"(\d+)", video_stem)
    if m:
        cam_num = int(m.group(1))
    else:
        cam_num = None
    
    print(f"[INFERENCE] Processing video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    frame_idx = 1  # Start from 1 to match dataset conventions (frame_0001)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO predict con confidence threshold configurabile
        results = model.predict(source=frame, device=device, conf=conf_thres, verbose=False, imgsz=imgsz)
        result = results[0]
        
        # Filter best per class
        best_per_class = {}
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                if box.cls is None or box.conf is None:
                    continue
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if cls_id not in best_per_class or conf > best_per_class[cls_id][0]:
                    best_per_class[cls_id] = (conf, box)
        
        # Aggregate per frame
        items = []
        for cls_id, (conf, box) in best_per_class.items():
            xywh = box.xywh[0].cpu().numpy()
            x, y, w, h = xywh
            items.append({
                "class_id": int(cls_id),
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "conf": float(conf)
            })
        all_frames[str(frame_idx)] = items
        
        if frame_idx % 30 == 0:
            print(f"[INFERENCE] Frame {frame_idx}/{total_frames} ({video_path.name})")
        
        frame_idx += 1
    
    cap.release()
    
    # Save single JSON
    labels_base.mkdir(parents=True, exist_ok=True)
    if cam_num is not None:
        out_json = labels_base / f"labels_out{cam_num}.json"
    else:
        out_json = labels_base / f"labels_{video_stem}.json"
    
    payload = {
        "camera": cam_num,
        "coord_mode": "pixel",
        "frames": all_frames
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    
    print(f"[INFERENCE] Completed: {frame_idx} frames saved in {out_json}")
    return cam_num if m else None


def run_images_inference(images_dir: Path, model_path: str, labels_base: Path, device: str = 'auto', conf_thres: float = 0.25):
    """
    Runs YOLO inference on an image directory and saves labels in JSON per camera.
    Supports filename pattern: out{cam}_frame_{num}_...
    """
    print(f"[INFERENCE] Loading model {model_path}...")
    model = YOLO(model_path)
    
    print(f"[INFERENCE] Scanning images in: {images_dir}")
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    if not image_files:
        print("[INFERENCE] No images found.")
        return []

    # Structure: {cam_id: {frame_idx: [detections]}}
    per_cam_results = {}
    
    # Regex to extract cam and frame
    pattern = re.compile(r"out(\d+)_frame_(\d+)_")
    
    count = 0
    for img_path in image_files:
        match = pattern.search(img_path.name)
        if not match:
            continue
            
        cam_id = int(match.group(1))
        frame_idx = int(match.group(2))
        
        if cam_id not in per_cam_results:
            per_cam_results[cam_id] = {}
            
        # Inference
        results = model.predict(source=str(img_path), device=device, conf=conf_thres, verbose=False, imgsz=1920)
        result = results[0]
        
        # Filter best per class
        best_per_class = {}
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                if box.cls is None or box.conf is None:
                    continue
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if cls_id not in best_per_class or conf > best_per_class[cls_id][0]:
                    best_per_class[cls_id] = (conf, box)
        
        items = []
        for cls_id, (conf, box) in best_per_class.items():
            xywh = box.xywh[0].cpu().numpy()
            x, y, w, h = xywh
            items.append({
                "class_id": int(cls_id),
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "conf": float(conf)
            })
            
        per_cam_results[cam_id][str(frame_idx)] = items
        count += 1
        if count % 50 == 0:
            print(f"[INFERENCE] Processed {count} images...")

    # Save results
    labels_base.mkdir(parents=True, exist_ok=True)
    saved_cams = []
    
    for cam_id, frames in per_cam_results.items():
        out_file = labels_base / f"labels_out{cam_id}.json"
        payload = {
            "camera": cam_id,
            "coord_mode": "pixel",
            "frames": frames
        }
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"[INFERENCE] Saved {out_file} with {len(frames)} frames")
        saved_cams.append(cam_id)
    
    return saved_cams


# Colors mapped by class index (BGR OpenCV)
CLASS_COLORS = [
    (255, 0, 0),       # blue
    (238, 130, 238),   # violet
    (235, 206, 135),   # skyblue
    (255, 0, 255),     # fuchsia
    (211, 0, 148),     # darkviolet
    (0, 128, 0),       # green
    (42, 42, 165),     # brown
    (203, 192, 255),   # pink
    (0, 255, 255),     # yellow
    (144, 238, 144),   # lightgreen
    (0, 0, 255),       # red
    (0, 165, 255),     # orange
    (255, 255, 0)      # cyan
]

# Default class names (must match training order)
DEFAULT_CLASS_NAMES = ['Ball', 'Red_0', 'Red_11', 'Red_12', 'Red_16', 'Red_2', 'Refree_F', 'Refree_M', 'White_13', 'White_16', 'White_25', 'White_27', 'White_34']

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a random frame from dataset")
    parser.add_argument('--model', type=str, default='fine_tuned_yolo.pt', help='Path to trained YOLO model (.pt)')
    parser.add_argument('--images', type=str, default='dataset/train/images', help='Images directory to sample from')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--show', action='store_true', help='Show window with result (requires GUI)')
    parser.add_argument('--save', action='store_true', help='Save annotated image to runs/infer_random')
    parser.add_argument('--device', type=str, default='auto', help="Device: 'auto', 'cpu', '0', '0,1'")
    parser.add_argument('--fit', action='store_true', help='Resize to fit screen (keeps aspect)')
    parser.add_argument('--maxw', type=int, default=1920, help='Max window width (if --fit)')
    parser.add_argument('--maxh', type=int, default=1080, help='Max window height (if --fit)')
    parser.add_argument('--scale-display', type=float, default=0.5, help='Manual scale (e.g. 0.5 = half). Overrides --fit if set')
    # Video options
    parser.add_argument('--video', type=str, default=None, help='MP4 video path (if provided ignores --images random)')
    parser.add_argument('--out-video', type=str, default='runs/infer_video/out.mp4', help='Annotated video output path')
    parser.add_argument('--frame-skip', type=int, default=0, help='Skip N frames between processed ones')
    parser.add_argument('--limit-frames', type=int, default=None, help='Process max N frames (debug)')
    parser.add_argument('--save-json', action='store_true', help='Save an aggregated JSON file with detections per frame')
    parser.add_argument('--json-dir', type=str, default='dataset/infer_video_json', help='Directory for aggregated JSONs')
    parser.add_argument('--labels-base', type=str, default='dataset/infer_video', help='Base dir where to save labels per frame (labels_outX)')
    return parser.parse_args()

def pick_random_image(images_dir: str) -> Path:
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [p for p in Path(images_dir).glob('*') if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return random.choice(files)

def draw_detections(image, result, class_names):
    """Draws only the bounding box with max conf for each class.

    If there are multiple boxes of the same class, only the one with highest confidence is kept.
    """
    annot = image.copy()
    if not (hasattr(result, 'boxes') and result.boxes is not None):
        return annot

    best_per_class = {}
    for box in result.boxes:
        if box.cls is None or box.conf is None:
            continue
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id not in best_per_class or conf > best_per_class[cls_id][0]:
            best_per_class[cls_id] = (conf, box)

    for cls_id, (conf, box) in best_per_class.items():
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        label = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)] if CLASS_COLORS else (0,255,0)
        cv2.rectangle(annot, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annot, f"{label} {conf:.2f}", (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return annot

def summarize(result, class_names):
    """Returns only classes with their best confidence (one per class)."""
    per_class = {}
    if hasattr(result, 'boxes') and result.boxes is not None:
        for box in result.boxes:
            if box.cls is None or box.conf is None:
                continue
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            if cls_id not in per_class or conf > per_class[cls_id][0]:
                per_class[cls_id] = (conf, box)
    summary = {}
    for cid, (conf, _) in per_class.items():
        name = class_names[cid] if 0 <= cid < len(class_names) else f"cls_{cid}"
        summary[name] = conf
    return summary


def save_tracked_video_from_images(frames_data: Dict[int, Dict[int, Dict]], images_dir: Path, output_path: Path, fps: int = 25, camera_id: int = None):
    """
    Generates an MP4 video from images in the directory, drawing bounding boxes and track IDs.
    
    Args:
        frames_data: Dict {frame_idx: {cls_id: {x, y, w, h, track_id, ...}}}
        images_dir: Directory containing source images (out{cam}_frame_{num}...)
        output_path: Output video file path (.mp4)
        fps: Output video frame rate (default 25)
        camera_id: Camera ID to filter images
    """
    import cv2
    import re
    import numpy as np
    
    images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not images:
        print(f"[TRACK-VIDEO] No images found in {images_dir}")
        return

    # Filter images by camera
    cam_images = []
    pattern = re.compile(r"out(\d+)_frame_(\d+)_")
    
    for img_path in images:
        m = pattern.search(img_path.name)
        if m:
            c_id = int(m.group(1))
            f_idx = int(m.group(2))
            if camera_id is not None and c_id != camera_id:
                continue
            cam_images.append((f_idx, img_path))
    
    # Sort by frame index
    cam_images.sort(key=lambda x: x[0])
    
    # Remove duplicates (keep only first image for each frame index)
    unique_cam_images = []
    seen_frames = set()
    for f_idx, img_path in cam_images:
        if f_idx not in seen_frames:
            unique_cam_images.append((f_idx, img_path))
            seen_frames.add(f_idx)
    cam_images = unique_cam_images
    
    if not cam_images:
        print(f"[TRACK-VIDEO] No images found for camera {camera_id}")
        return

    # Read first image for dimensions
    first_img = cv2.imread(str(cam_images[0][1]))
    if first_img is None:
        print(f"[TRACK-VIDEO] Error reading image {cam_images[0][1]}")
        return
    
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    print(f"[TRACK-VIDEO] Generating video {output_path} ({len(cam_images)} frames, {fps} fps)...")
    
    # Colors mapped by class index (BGR OpenCV)
    # Use CLASS_COLORS if available, otherwise fallback
    colors = CLASS_COLORS if 'CLASS_COLORS' in globals() else []
    
    for f_idx, img_path in cam_images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
            
        # Draw detections
        if f_idx in frames_data:
            for cls_id, det in frames_data[f_idx].items():
                x, y, bw, bh = det['x'], det['y'], det['w'], det['h']
                tid = det.get('track_id', -1)
                
                # Pixel coordinates (x,y center)
                x1 = int(x - bw/2)
                y1 = int(y - bh/2)
                x2 = int(x + bw/2)
                y2 = int(y + bh/2)
                
                # Use color based on class, not track_id
                cls_id_int = int(cls_id)
                if colors:
                    color = colors[cls_id_int % len(colors)]
                else:
                    color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"ID:{tid} C:{cls_id_int}"
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out.write(frame)
    
    out.release()
    print(f"[TRACK-VIDEO] Video saved: {output_path}")


def save_tracked_video_from_video(frames_data: Dict[int, Dict[int, Dict]], video_path: Path, output_path: Path):
    """
    Generates an MP4 video starting from a source video, drawing bounding boxes and track IDs.
    
    Args:
        frames_data: Dict {frame_idx: {cls_id: {x, y, w, h, track_id, ...}}}
        video_path: Input source video path
        output_path: Output video file path (.mp4)
    """
    import cv2
    import numpy as np
    
    if not video_path.exists():
        print(f"[TRACK-VIDEO] Source video not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[TRACK-VIDEO] Cannot open source video: {video_path}")
        return
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    print(f"[TRACK-VIDEO] Generating video {output_path} from {video_path.name}...")
    
    # Colors mapped by class index (BGR OpenCV)
    # Use CLASS_COLORS if available, otherwise fallback
    colors = CLASS_COLORS if 'CLASS_COLORS' in globals() else []
    
    frame_idx = 1  # Start from 1 to match inference indexing
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw detections if present for this frame
        if frame_idx in frames_data:
            for cls_id, det in frames_data[frame_idx].items():
                x, y, bw, bh = det['x'], det['y'], det['w'], det['h']
                tid = det.get('track_id', -1)
                
                # Pixel coordinates (x,y center)
                x1 = int(x - bw/2)
                y1 = int(y - bh/2)
                x2 = int(x + bw/2)
                y2 = int(y + bh/2)
                
                # Use color based on class, not track_id
                cls_id_int = int(cls_id)
                if colors:
                    color = colors[cls_id_int % len(colors)]
                else:
                    color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"ID:{tid} C:{cls_id_int}"
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"[TRACK-VIDEO] Processed {frame_idx}/{total_frames} frames", end='\r')
            
    print("")
    cap.release()
    out.release()
    print(f"[TRACK-VIDEO] Video saved: {output_path}")
