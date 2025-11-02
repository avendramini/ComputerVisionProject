"""
Calcola l'average IOU per ogni frame nella cartella dataset/test confrontando le predizioni YOLO con le label YOLOv8.

Uso:
    python eval_iou_per_frame.py --model fine_tuned_yolo.pt --images dataset/test/images --labels dataset/test/labels

Se non specifichi --model usa i pesi di default (fine_tuned_yolo.pt).
"""
import os
import argparse
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Calcola average IOU per frame su un dataset di test")
    parser.add_argument('--model', type=str, default='fine_tuned_yolo.pt', help='YOLO model weights')
    parser.add_argument('--images', type=str, default='dataset/test/images', help='Cartella immagini')
    parser.add_argument('--labels', type=str, default='dataset/test/labels', help='Cartella label YOLOv8')
    parser.add_argument('--imgsz', type=int, default=1920, help='Dimensione input YOLO')
    parser.add_argument('--device', type=str, default='auto', help='Device YOLO')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='Confidence threshold per predizioni')
    return parser.parse_args()

def load_yolo_labels(label_path):
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
    # box: (x, y, w, h) normalizzati
    # converte in xyxy
    def to_xyxy(x, y, w, h):
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return x1, y1, x2, y2
    x1_1, y1_1, x2_1, y2_1 = to_xyxy(*box1)
    x1_2, y1_2, x2_2, y2_2 = to_xyxy(*box2)
    # Intersezione
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter_area
    if union == 0:
        return 0.0
    return inter_area / union

def per_class_iou(gt_boxes, pred_boxes, num_classes=13):
    # Restituisce: {class_id: [iou, iou, ...]} solo per frame dove c'è almeno una GT di quella classe
    iou_per_class = {c: [] for c in range(num_classes)}
    for c in range(num_classes):
        gt_c = [g for g in gt_boxes if g[0] == c]
        pred_c = [p for p in pred_boxes if p[0] == c]
        if not gt_c:
            continue  # ignora frame per questa classe
        used = set()
        for p in pred_c:
            px, py, pw, ph = p[1:]
            best_iou = 0.0
            best_idx = -1
            for idx, g in enumerate(gt_c):
                if idx in used:
                    continue
                gx, gy, gw, gh = g[1:]
                iou = box_iou((px, py, pw, ph), (gx, gy, gw, gh))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0:
                used.add(best_idx)
            iou_per_class[c].append(best_iou)
    return iou_per_class


def compute_iou_metrics(model_path: str, images_dir: str, labels_dir: str, imgsz: int = 1920, device: str = 'auto', conf_thres: float = 0.01, num_classes: int = 13) -> dict:
    """
    Esegue inferenza YOLO su un dataset di immagini e calcola IOU rispetto alle label YOLOv8.

    Ritorna:
      {
        'per_image': [ { 'image': str, 'average_iou': float }, ... ],
        'per_class': { class_id: { 'mean_iou': float|None, 'count': int }, ... },
        'mean_iou': float,
        'images_count': int,
        'images_dir': str,
        'labels_dir': str,
        'model': str,
        'imgsz': int,
        'device': str,
        'conf_thres': float,
      }
    """
    model = YOLO(model_path)
    images = sorted([p for p in Path(images_dir).glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    per_image = []
    all_iou_per_class = {c: [] for c in range(num_classes)}
    for img_path in images:
        label_path = Path(labels_dir) / (img_path.stem + '.txt')
        gt_boxes = load_yolo_labels(label_path)
        pred = model.predict(source=str(img_path), device=device, verbose=False, imgsz=imgsz, conf=conf_thres)
        pred_boxes = []
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2] if img is not None else (1, 1)
        if pred and hasattr(pred[0], 'boxes') and pred[0].boxes is not None:
            for box in pred[0].boxes:
                cls = int(box.cls.item())
                xywh = box.xywh[0].cpu().numpy()
                x, y, bw, bh = xywh
                x /= w; y /= h; bw /= w; bh /= h
                pred_boxes.append((cls, x, y, bw, bh))
        iou_pc = per_class_iou(gt_boxes, pred_boxes, num_classes=num_classes)
        for c, ious in iou_pc.items():
            all_iou_per_class[c].extend(ious)
        avg_iou_img = float(np.mean([v for lst in iou_pc.values() for v in lst])) if any(iou_pc.values()) else 0.0
        per_image.append({"image": img_path.name, "average_iou": avg_iou_img})

    per_class = {}
    for c, ious in all_iou_per_class.items():
        per_class[c] = {"mean_iou": float(np.mean(ious)) if ious else None, "count": int(len(ious))}
    mean_iou = float(np.mean([it["average_iou"] for it in per_image])) if per_image else 0.0
    return {
        "per_image": per_image,
        "per_class": per_class,
        "mean_iou": mean_iou,
        "images_count": len(per_image),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "model": str(model_path),
        "imgsz": imgsz,
        "device": device,
        "conf_thres": conf_thres,
    }


def save_eval_json(metrics: dict, out_json: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open('w', encoding='utf-8') as f:
        json.dump(metrics, f)
    print(f"[EVAL] Salvate metriche in {out_json}")


# -----------------------
# Label-vs-label evaluation (no model) for video label directories
# -----------------------

def load_labels_dir_map(labels_dir: Path, camera_id: int | None = None) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """Carica tutte le label YOLOv8 in una mappa {frame_idx: [(cls,x,y,w,h), ...]}.
    Accetta file nominati come frame_00000.txt o out{cam}_frame_{num}_png.rf.{hash}.txt.
    Se camera_id è fornito, filtra solo i file per quella camera.
    """
    frames: dict[int, list[tuple[int, float, float, float, float]]] = {}
    if not labels_dir.exists():
        return frames
    for p in sorted(labels_dir.glob('*.txt')):
        stem = p.stem
        # prova a estrarre indice frame e camera
        fi = None
        cam = None
        import re
        # pattern: out{cam}_frame_{num}_png.rf.{hash} oppure out{cam}_frame_{num}
        m = re.match(r"out(\d+)_frame_(\d+)", stem)
        if m:
            cam = int(m.group(1))
            fi = int(m.group(2))
        # pattern frame_00012
        elif stem.startswith('frame_') and '_' not in stem[6:]:
            # solo se non ci sono altri underscore dopo frame_
            if stem[6:].isdigit():
                fi = int(stem[6:])
        else:
            # fallback: prendi numeri finali
            m2 = re.search(r"(\d+)$", stem)
            if m2:
                fi = int(m2.group(1))
        if fi is None:
            continue
        # Filtra per camera se specificato
        if camera_id is not None and cam is not None and cam != camera_id:
            continue
        boxes = load_yolo_labels(str(p))  # [(cls,x,y,w,h)]
        frames[fi] = [(int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])) for b in boxes]
    return frames


def frames_from_perframe(perframe: dict[int, dict[int, tuple[float, float, float, float, float]]]) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """Converte la struttura PerFrame {frame: {cls: (x,y,w,h,conf)}} in {frame: [(cls,x,y,w,h),...]}"""
    out: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for fi, dets in perframe.items():
        lst = []
        for cls_id, (x, y, w, h, _conf) in dets.items():
            lst.append((int(cls_id), float(x), float(y), float(w), float(h)))
        out[int(fi)] = lst
    return out


def compute_iou_metrics_from_predictions(pred_frames: dict[int, list[tuple[int, float, float, float, float]]],
                                         gt_frames: dict[int, list[tuple[int, float, float, float, float]]],
                                         num_classes: int | None = None) -> dict:
    """Calcola metriche IOU confrontando predizioni già pronte (no modello) con GT per frame/class.

    pred_frames / gt_frames: {frame: [(cls,x,y,w,h), ...]} in coordinate normalizzate YOLO.
    """
    frames = sorted(set(pred_frames.keys()) & set(gt_frames.keys()))
    if num_classes is None:
        max_cls = -1
        for d in (pred_frames, gt_frames):
            for lst in d.values():
                for it in lst:
                    max_cls = max(max_cls, int(it[0]))
        num_classes = max(0, max_cls + 1)
    all_iou_per_class = {c: [] for c in range(num_classes)}
    per_frame = []
    for fi in frames:
        pred = pred_frames.get(fi, [])
        gt = gt_frames.get(fi, [])
        iou_pc = per_class_iou(gt, pred, num_classes=num_classes)
        for c, ious in iou_pc.items():
            all_iou_per_class[c].extend(ious)
        avg_iou = float(np.mean([v for lst in iou_pc.values() for v in lst])) if any(iou_pc.values()) else 0.0
        per_frame.append({"frame": int(fi), "average_iou": avg_iou})
    per_class = {c: {"mean_iou": float(np.mean(v)) if v else None, "count": int(len(v))} for c, v in all_iou_per_class.items()}
    mean_iou = float(np.mean([it["average_iou"] for it in per_frame])) if per_frame else 0.0
    return {
        "mode": "labels-vs-labels",
        "frames_evaluated": len(frames),
        "per_frame": per_frame,
        "per_class": per_class,
        "mean_iou": mean_iou,
        "num_classes": num_classes,
    }

def main():
    args = parse_args()
    model = YOLO(args.model)
    images = sorted([p for p in Path(args.images).glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    results = []
    all_iou_per_class = {c: [] for c in range(13)}
    for img_path in images:
        label_path = Path(args.labels) / (img_path.stem + '.txt')
        gt_boxes = load_yolo_labels(label_path)
        # Predict
        pred = model.predict(source=str(img_path), device=args.device, verbose=False, imgsz=args.imgsz, conf=args.conf_thres)
        pred_boxes = []
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2] if img is not None else (1, 1)
        if pred and hasattr(pred[0], 'boxes') and pred[0].boxes is not None:
            for box in pred[0].boxes:
                cls = int(box.cls.item())
                xywh = box.xywh[0].cpu().numpy()  # in pixel
                # Normalizza rispetto a dimensione immagine
                x, y, bw, bh = xywh
                x /= w
                y /= h
                bw /= w
                bh /= h
                pred_boxes.append((cls, x, y, bw, bh))
        # Print predizioni per debug
        print(f"{img_path.name} PREDICTIONS:")
        for pb in pred_boxes:
            print(f"  class={pb[0]} x={pb[1]:.4f} y={pb[2]:.4f} w={pb[3]:.4f} h={pb[4]:.4f}")
        # Print ground truth box per debug
        print(f"{img_path.name} GROUND TRUTH:")
        for gb in gt_boxes:
            print(f"  class={gb[0]} x={gb[1]:.4f} y={gb[2]:.4f} w={gb[3]:.4f} h={gb[4]:.4f}")
        # Per-class IOU
        iou_per_class = per_class_iou(gt_boxes, pred_boxes, num_classes=13)
        for c, ious in iou_per_class.items():
            all_iou_per_class[c].extend(ious)
        # (opzionale: average IOU su tutte le box)
        iou = np.mean([iou for ious in iou_per_class.values() for iou in ious]) if any(iou_per_class.values()) else 0.0
        results.append((img_path.name, iou))
        print(f"{img_path.name}: average IOU = {iou:.3f}")

    # Statistica finale per classe
    print("\nAverage IOU per classe (solo frame con GT):")
    for c, ious in all_iou_per_class.items():
        if ious:
            print(f"Classe {c}: {np.mean(ious):.3f} su {len(ious)} box")
        else:
            print(f"Classe {c}: N/A (nessuna GT)")
    # Statistica finale
    if results:
        mean_iou = np.mean([r[1] for r in results])
        print(f"\nMean IOU su {len(results)} frame: {mean_iou:.3f}")
    else:
        print("Nessuna immagine trovata.")

if __name__ == "__main__":
    main()
