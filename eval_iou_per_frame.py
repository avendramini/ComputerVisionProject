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
