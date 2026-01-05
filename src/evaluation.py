"""
Evaluation module for YOLO models on test datasets and for label-vs-label comparison.

Main features:
    1. YOLO inference + IoU calculation against ground truth (compute_iou_metrics)
    2. Comparison of pre-existing labels (no inference) against GT (compute_iou_metrics_from_predictions)
    3. Support for video evaluation with flexible file patterns
    4. Export metrics as JSON for analysis
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

def load_yolo_labels(label_path):
    """
    Loads bounding boxes from YOLO txt file (YOLOv8 format: class x y w h).

    Args:
        label_path: Path to .txt file with labels (one per line)

    Returns:
        boxes: List of tuples (class_id, x, y, w, h) normalized to [0,1]

    File format:
        0 0.5 0.5 0.1 0.2
        1 0.3 0.7 0.15 0.25
        ...

    Example:
        boxes = load_yolo_labels('dataset/val/labels/out13_frame_0001.txt')
        # boxes = [(0, 0.192, 0.443, 0.005, 0.009), (1, 0.357, 0.534, 0.026, 0.095), ...]
    """
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            boxes.append((cls, x, y, w, h))
    return boxes

def box_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two normalized bounding boxes.

    Args:
        box1: Tuple (x, y, w, h) normalized coordinates [0,1]
        box2: Tuple (x, y, w, h) normalized coordinates [0,1]

    Returns:
        iou: Float in [0,1], 0=no overlap, 1=perfect match

    Note:
        - x,y are bbox center coordinates
        - w,h are normalized width and height
        - Converts internally to xyxy format for intersection calculation

    Example:
        box1 = (0.5, 0.5, 0.2, 0.2)  # center (0.5,0.5), size 20%x20%
        box2 = (0.52, 0.51, 0.18, 0.19)  # partially overlapping
        iou = box_iou(box1, box2)
        # iou ≈ 0.75 (high overlap)
    """
    # Internal function for xywh -> xyxy conversion
    def to_xyxy(x, y, w, h):
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return x1, y1, x2, y2
    
    # Convert both boxes to xyxy format
    x1_1, y1_1, x2_1, y2_1 = to_xyxy(*box1)
    x1_2, y1_2, x2_2, y2_2 = to_xyxy(*box2)
    
    # Calculate intersection rectangle
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    # Calculate areas of both boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Union = sum - intersection
    union = area1 + area2 - inter_area
    if union == 0:
        return 0.0
    return inter_area / union

def per_class_iou(gt_boxes, pred_boxes, num_classes=13):
    """
    Calculates IoU for each class. Optimized for max 1 object per class per frame.

    Args:
        gt_boxes: List of ground truth [(class, x, y, w, h), ...]
        pred_boxes: List of predictions [(class, x, y, w, h), ...]
        num_classes: Total number of classes in the dataset (default 13)

    Returns:
        iou_per_class: Dict {class_id: [iou_values]} with single IoU value per class

    Logic (for max 1 object per class per frame):
        - If GT exists and pred exists for same class → IoU between them
        - If only GT exists (pred missing) → IoU = 0.0 (false negative - PENALTY)
        - If only pred exists (GT missing) → IoU = 0.0 (false positive)
        - If neither exists → skip (no entry for that class)

    Example:
        # Case 1: Perfect match (1 ball detected)
        gt = [(0, 0.5, 0.5, 0.1, 0.1)]
        pred = [(0, 0.51, 0.49, 0.09, 0.11)]
        iou_pc = per_class_iou(gt, pred)
        # iou_pc = {0: [0.85], 1: [], 2: [], ...}

        # Case 2: False negative (1 ball missed by model)
        gt = [(0, 0.5, 0.5, 0.1, 0.1)]
        pred = []
        iou_pc = per_class_iou(gt, pred)
        # iou_pc = {0: [0.0], 1: [], ...}  # penalize missed detection

        # Case 3: False positive (model detected when no GT)
        gt = []
        pred = [(0, 0.5, 0.5, 0.1, 0.1)]
        iou_pc = per_class_iou(gt, pred)
        # iou_pc = {0: [0.0], 1: [], ...}  # penalize false positive
    """
    iou_per_class = {c: [] for c in range(num_classes)}
    
    # Build class dictionaries: class_id → single box or None
    gt_by_class = {}
    pred_by_class = {}
    
    for g in gt_boxes:
        gt_by_class[g[0]] = g[1:]  # store (x, y, w, h)
    
    for p in pred_boxes:
        pred_by_class[p[0]] = p[1:]  # store (x, y, w, h)
    
    # For each class with GT, calculate IoU
    for c in gt_by_class:
        gt_box = gt_by_class[c]
        pred_box = pred_by_class.get(c)
        
        if pred_box is not None:
            # Both GT and pred exist: calculate IoU
            iou = box_iou(pred_box, gt_box)
        else:
            # Only GT exists: false negative (IoU = 0)
            iou = 0.0
        
        iou_per_class[c].append(iou)
    
    # NOTE: False Positives (Pred without GT) are NOT added to IoU list
    # This ensures Mean IoU denominator is based on GT count only.
    # TP/FP/FN counts are handled separately in per_class_tp_fp_fn.
    
    return iou_per_class


def per_class_tp_fp_fn(gt_boxes, pred_boxes, num_classes=13, iou_threshold=0.5):
    """
    Calculates True Positives, False Positives, and False Negatives per class using greedy matching.
    
    Args:
        gt_boxes: List of ground truth [(class, x, y, w, h), ...]
        pred_boxes: List of predictions [(class, x, y, w, h), ...]
        num_classes: Total number of classes (default 13)
        iou_threshold: Minimum IoU to count as a match (default 0.5)
    
    Returns:
        metrics: Dict {class_id: {'tp': int, 'fp': int, 'fn': int}}
        
    Example:
        gt = [(0, 0.5, 0.5, 0.1, 0.1), (0, 0.3, 0.7, 0.1, 0.1)]
        pred = [(0, 0.51, 0.49, 0.09, 0.11)]
        metrics = per_class_tp_fp_fn(gt, pred, iou_threshold=0.5)
        # metrics = {0: {'tp': 1, 'fp': 0, 'fn': 1}, 1: {'tp': 0, 'fp': 0, 'fn': 0}, ...}
    """
    metrics = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in range(num_classes)}
    
    # For each class separately
    for c in range(num_classes):
        # Filter GT and predictions for this class
        gt_c = [g for g in gt_boxes if g[0] == c]
        pred_c = [p for p in pred_boxes if p[0] == c]
        
        # If no GT for this class
        if not gt_c:
            # All predictions are FP
            metrics[c]['fp'] = len(pred_c)
            continue
        
        # Track matched GTs
        used = set()
        
        # For each prediction, find best GT match
        for p in pred_c:
            px, py, pw, ph = p[1:]
            best_iou = 0.0
            best_idx = -1
            
            # Find GT with max IoU not yet used
            for idx, g in enumerate(gt_c):
                if idx in used:
                    continue
                gx, gy, gw, gh = g[1:]
                iou = box_iou((px, py, pw, ph), (gx, gy, gw, gh))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            # Check if IoU exceeds threshold
            if best_iou >= iou_threshold and best_idx >= 0:
                metrics[c]['tp'] += 1
                used.add(best_idx)
            else:
                metrics[c]['fp'] += 1
        
        # For each unmatched GT -> FN
        for idx in range(len(gt_c)):
            if idx not in used:
                metrics[c]['fn'] += 1
    
    return metrics


def compute_precision_recall_f1(tp, fp, fn):
    """
    Computes precision, recall and F1-score from TP, FP, FN counts.
    
    Args:
        tp: True positives count
        fp: False positives count
        fn: False negatives count
    
    Returns:
        metrics: Dict with 'precision', 'recall', 'f1' (or None if undefined)
        
    Example:
        metrics = compute_precision_recall_f1(tp=10, fp=2, fn=3)
        # metrics = {'precision': 0.833, 'recall': 0.769, 'f1': 0.800}
    """
    precision = None
    recall = None
    f1 = None
    
    # Precision = TP / (TP + FP)
    if tp + fp > 0:
        precision = float(tp) / (tp + fp)
    
    # Recall = TP / (TP + FN)
    if tp + fn > 0:
        recall = float(tp) / (tp + fn)
    
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_eval_json(metrics: dict, out_json: Path):
    """
    Saves evaluation metrics to JSON file.
    
    Args:
        metrics: Dict containing metrics to save
        out_json: Path to output JSON file
    
    Example:
        metrics = compute_iou_metrics(...)
        save_eval_json(metrics, Path('runs/eval/eval_cam13_raw.json'))
        # File saved: runs/eval/eval_cam13_raw.json
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open('w', encoding='utf-8') as f:
        json.dump(metrics, f)
    print(f"[EVAL] Saved metrics to {out_json}")


# -----------------------
# Label-vs-label evaluation (no model) for video label directories
# -----------------------

def load_labels_dir_map(labels_dir: Path, camera_id: int | None = None) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """
    Loads YOLO labels from txt directory organized by frame.
    Supports two naming patterns:
    - Standard: frame_XXXXX.txt
    - Roboflow: out{cam}_frame_{num}_png.rf.{hash}.txt
    
    Args:
        labels_dir: Directory path with .txt files (e.g. Path('dataset/val/labels'))
        camera_id: If specified, filters only frames for that camera (Roboflow pattern)
    
    Returns:
        frames: Dict {frame_idx: [(class_id, x, y, w, h), ...]}
        
    Standard Example:
        # labels_dir contains: frame_00001.txt, frame_00002.txt, ...
        frames = load_labels_dir_map(Path('dataset/gt_video/labels'))
        print(frames[1])  # [(0, 0.5, 0.5, 0.1, 0.1), (5, 0.3, 0.4, 0.08, 0.09)]
    
    Roboflow Example:
        # labels_dir contains: out13_frame_0001_png.rf.{hash}.txt, out4_frame_0012_png.rf.{hash}.txt, ...
        frames_cam13 = load_labels_dir_map(Path('dataset/val/labels'), camera_id=13)
        frames_cam4 = load_labels_dir_map(Path('dataset/val/labels'), camera_id=4)
        print(len(frames_cam13))  # 110
    """
    frames: dict[int, list[tuple[int, float, float, float, float]]] = {}
    if not labels_dir.exists():
        return frames
    
    # Regex pattern for Roboflow: out{cam}_frame_{num}_png.rf.{hash} or out{cam}_frame_{num}
    import re
    
    for p in sorted(labels_dir.glob('*.txt')):
        stem = p.stem
        fi = None  # frame index
        cam = None  # camera id
        
        # Roboflow pattern: out{cam}_frame_{num}_png.rf.{hash}
        m = re.match(r"out(\d+)_frame_(\d+)", stem)
        if m:
            cam = int(m.group(1))
            fi = int(m.group(2))
        # Standard pattern: frame_00012
        elif stem.startswith('frame_') and '_' not in stem[6:]:
            if stem[6:].isdigit():
                fi = int(stem[6:])
        else:
            # Fallback: take final numbers
            m2 = re.search(r"(\d+)$", stem)
            if m2:
                fi = int(m2.group(1))
        
        if fi is None:
            continue
        
        # Filter by camera if specified
        if camera_id is not None and cam is not None and cam != camera_id:
            continue
        
        # Load bounding boxes from file
        boxes = load_yolo_labels(str(p))  # [(cls,x,y,w,h)]
        frames[fi] = [(int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])) for b in boxes]
    
    return frames


def frames_from_perframe(perframe: dict[int, dict[int, tuple[float, float, float, float, float]]]) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """
    Converts PerFrame structure {frame: {cls: (x,y,w,h,conf)}} to evaluation format {frame: [(cls,x,y,w,h),...]}.
    
    Args:
        perframe: Dict {frame_idx: {class_id: (x, y, w, h, conf)}}
                  Internal pipeline structure with confidence
    
    Returns:
        frames: Dict {frame_idx: [(class_id, x, y, w, h), ...]}
                Evaluation format (without confidence)
    
    Example:
        # Input from pipeline after interpolation
        perframe = {
            1: {0: (0.5, 0.5, 0.1, 0.1, 0.95), 5: (0.3, 0.4, 0.08, 0.09, 0.87)},
            2: {0: (0.51, 0.52, 0.1, 0.1, 0.93)}
        }
        frames = frames_from_perframe(perframe)
        print(frames)
        # {1: [(0, 0.5, 0.5, 0.1, 0.1), (5, 0.3, 0.4, 0.08, 0.09)], 
        #  2: [(0, 0.51, 0.52, 0.1, 0.1)]}
    """
    out: dict[int, list[tuple[int, float, float, float, float]]] = {}
    
    # For each frame
    for fi, dets in perframe.items():
        lst = []
        # For each detection (class_id: (x,y,w,h,conf))
        for cls_id, (x, y, w, h, _conf) in dets.items():
            # Discard confidence, keep only bbox
            lst.append((int(cls_id), float(x), float(y), float(w), float(h)))
        out[int(fi)] = lst
    
    return out


def compute_iou_metrics_from_predictions(pred_frames: dict[int, list[tuple[int, float, float, float, float]]],
                                         gt_frames: dict[int, list[tuple[int, float, float, float, float]]],
                                         num_classes: int | None = None) -> dict:
    """
    Calculates IOU metrics by comparing ready-made predictions (no model) with GT per frame/class.
    
    Args:
        pred_frames: Dict {frame_idx: [(class_id, x, y, w, h), ...]} normalized predictions
        gt_frames: Dict {frame_idx: [(class_id, x, y, w, h), ...]} normalized ground truth
        num_classes: Number of classes (auto-detect if None)
    
    Returns:
        metrics: Dict with:
          - 'mode': 'labels-vs-labels'
          - 'frames_evaluated': int number of frames evaluated
          - 'per_frame': [{'frame': int, 'average_iou': float}, ...]
          - 'per_class': {class_id: {'mean_iou': float|None, 'count': int}, ...}
          - 'mean_iou': float (average over all frames)
          - 'num_classes': int
    
    Note:
        Used by pipeline.py for RAW and INTERP evaluations (without re-inference).
    
    Example:
        # Predictions after tracking
        pred_frames = {1: [(0, 0.5, 0.5, 0.1, 0.1), (5, 0.3, 0.4, 0.08, 0.09)],
                      2: [(0, 0.51, 0.52, 0.1, 0.1)]}
        
        # Ground truth from dataset/val/labels
        gt_frames = {1: [(0, 0.49, 0.51, 0.11, 0.09), (5, 0.31, 0.39, 0.09, 0.1)],
                    2: [(0, 0.50, 0.53, 0.10, 0.11)]}
        
        metrics = compute_iou_metrics_from_predictions(pred_frames, gt_frames, num_classes=13)
        print(f"Mean IOU: {metrics['mean_iou']:.3f}")  # 0.823
        print(f"Frames: {metrics['frames_evaluated']}")  # 2
    """
    # Find all frames to evaluate (union of pred and GT)
    frames_to_evaluate = sorted(set(pred_frames.keys()) | set(gt_frames.keys()))

    # Find all GT frames (for evaluation on all GT)
    frames_gt = sorted(gt_frames.keys())

    # Auto-detect number of classes if not specified
    if num_classes is None:
        max_cls = -1
        for d in (pred_frames, gt_frames):
            for lst in d.values():
                for it in lst:
                    max_cls = max(max_cls, int(it[0]))
        num_classes = max(0, max_cls + 1)

    # --- STANDARD METRIC: on all frames ---
    all_iou_per_class = {c: [] for c in range(num_classes)}
    all_tp = {c: 0 for c in range(num_classes)}
    all_fp = {c: 0 for c in range(num_classes)}
    all_fn = {c: 0 for c in range(num_classes)}
    
    per_frame = []
    for fi in frames_to_evaluate:
        pred = pred_frames.get(fi, [])
        gt = gt_frames.get(fi, [])
        iou_pc = per_class_iou(gt, pred, num_classes=num_classes)
        
        # Calculate TP/FP/FN
        tp_fp_fn = per_class_tp_fp_fn(gt, pred, num_classes=num_classes, iou_threshold=0.5)
        for c, metrics in tp_fp_fn.items():
            all_tp[c] += metrics['tp']
            all_fp[c] += metrics['fp']
            all_fn[c] += metrics['fn']
        
        for c, ious in iou_pc.items():
            all_iou_per_class[c].extend(ious)
        avg_iou = float(np.mean([v for lst in iou_pc.values() for v in lst])) if any(iou_pc.values()) else 0.0
        per_frame.append({"frame": int(fi), "average_iou": avg_iou})
    
    # Build per_class with complete metrics
    per_class = {}
    for c in range(num_classes):
        ious = all_iou_per_class[c]
        prec_recall_f1 = compute_precision_recall_f1(all_tp[c], all_fp[c], all_fn[c])
        per_class[c] = {
            "mean_iou": float(np.mean(ious)) if ious else None,
            "count": int(all_tp[c] + all_fn[c]),
            "tp": int(all_tp[c]),
            "fp": int(all_fp[c]),
            "fn": int(all_fn[c]),
            "precision": prec_recall_f1['precision'],
            "recall": prec_recall_f1['recall'],
            "f1": prec_recall_f1['f1'],
        }
    
    all_ious = [iou for ious in all_iou_per_class.values() for iou in ious]
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    
    # Global metrics for frame matching
    global_tp = sum(all_tp.values())
    global_fp = sum(all_fp.values())
    global_fn = sum(all_fn.values())
    global_prec_recall_f1 = compute_precision_recall_f1(global_tp, global_fp, global_fn)

    # --- NEW METRIC: on all GT frames (assign IoU=0 if prediction missing) ---
    all_iou_per_class_gt = {c: [] for c in range(num_classes)}
    all_tp_gt = {c: 0 for c in range(num_classes)}
    all_fp_gt = {c: 0 for c in range(num_classes)}
    all_fn_gt = {c: 0 for c in range(num_classes)}
    
    per_frame_gt = []
    for fi in frames_gt:
        pred = pred_frames.get(fi, [])  # if missing, empty list
        gt = gt_frames.get(fi, [])
        iou_pc = per_class_iou(gt, pred, num_classes=num_classes)
        
        # Calculate TP/FP/FN
        tp_fp_fn_gt = per_class_tp_fp_fn(gt, pred, num_classes=num_classes, iou_threshold=0.5)
        for c, metrics in tp_fp_fn_gt.items():
            all_tp_gt[c] += metrics['tp']
            all_fp_gt[c] += metrics['fp']
            all_fn_gt[c] += metrics['fn']
        
        for c, ious in iou_pc.items():
            all_iou_per_class_gt[c].extend(ious)
        avg_iou = float(np.mean([v for lst in iou_pc.values() for v in lst])) if any(iou_pc.values()) else 0.0
        per_frame_gt.append({"frame": int(fi), "average_iou": avg_iou})
    
    # Build per_class_gt with complete metrics
    per_class_gt = {}
    for c in range(num_classes):
        ious = all_iou_per_class_gt[c]
        prec_recall_f1_gt = compute_precision_recall_f1(all_tp_gt[c], all_fp_gt[c], all_fn_gt[c])
        per_class_gt[c] = {
            "mean_iou": float(np.mean(ious)) if ious else None,
            "count": int(all_tp_gt[c] + all_fn_gt[c]),
            "tp": int(all_tp_gt[c]),
            "fp": int(all_fp_gt[c]),
            "fn": int(all_fn_gt[c]),
            "precision": prec_recall_f1_gt['precision'],
            "recall": prec_recall_f1_gt['recall'],
            "f1": prec_recall_f1_gt['f1'],
        }
    
    all_ious_gt = [iou for ious in all_iou_per_class_gt.values() for iou in ious]
    mean_iou_gt = float(np.mean(all_ious_gt)) if all_ious_gt else 0.0
    
    # Global metrics for all GT frames
    global_tp_gt = sum(all_tp_gt.values())
    global_fp_gt = sum(all_fp_gt.values())
    global_fn_gt = sum(all_fn_gt.values())
    global_prec_recall_f1_gt = compute_precision_recall_f1(global_tp_gt, global_fp_gt, global_fn_gt)

    return {
        "mode": "labels-vs-labels",
        "frames_evaluated": len(frames_to_evaluate),
        "per_frame": per_frame,
        "per_class": per_class,
        "mean_iou": mean_iou,
        "global_metrics": {
            "tp": int(global_tp),
            "fp": int(global_fp),
            "fn": int(global_fn),
            "precision": global_prec_recall_f1['precision'],
            "recall": global_prec_recall_f1['recall'],
            "f1": global_prec_recall_f1['f1'],
        },
        "frames_evaluated_gt": len(frames_gt),
        "per_frame_gt": per_frame_gt,
        "per_class_gt": per_class_gt,
        "mean_iou_gt": mean_iou_gt,
        "global_metrics_gt": {
            "tp": int(global_tp_gt),
            "fp": int(global_fp_gt),
            "fn": int(global_fn_gt),
            "precision": global_prec_recall_f1_gt['precision'],
            "recall": global_prec_recall_f1_gt['recall'],
            "f1": global_prec_recall_f1_gt['f1'],
        },
        "num_classes": num_classes,
    }
