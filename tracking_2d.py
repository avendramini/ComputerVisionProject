"""
Tracking 2D leggero su etichette YOLO per frame (per-camera) con assegnamento per IoU.

Offre funzioni riusabili dalla pipeline e (opzionale) un main per inference video.
"""

import os
import random
import argparse
import re
from pathlib import Path
import cv2
from ultralytics import YOLO
import json
from typing import Dict, List, Tuple
import numpy as np

# ----------------------------
# Tracking 2D minimale per labels già pronte
# ----------------------------

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
    Assegna track_id per ogni classe su frame sequenziali usando matching per IoU (greedy).

    Args:
        frames: {frame_idx: {class_id: (x,y,w,h,conf)}}
        iou_thresh: soglia minima per associare ad una traccia esistente
        max_age: età massima (frame) per tenere viva una traccia senza match

    Returns:
        tracks_by_frame: {frame_idx: {class_id: {x,y,w,h,conf,track_id}}}
    """
    next_id = 1
    # Tracce attive per classe: {class_id: List[dict(track_id, bbox, age)]}
    active: Dict[int, List[Dict]] = {}
    out: Dict[int, Dict[int, Dict]] = {}

    for fi in sorted(frames.keys()):
        out[fi] = {}
        dets = frames[fi]  # {cls: [x,y,w,h,conf]}
        # Invecchia tracce
        for cls in list(active.keys()):
            for tr in active[cls]:
                tr["age"] += 1
        # Per ogni classe separatamente
        for cls_id, vals in dets.items():
            x, y, w, h, conf = vals
            bbox = (x, y, w, h)
            if cls_id not in active:
                active[cls_id] = []
            # Trova miglior match per IoU
            best_iou = 0.0
            best_tr = None
            for tr in active[cls_id]:
                iou = _iou(tr["bbox"], bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tr = tr
            if best_tr is not None and best_iou >= iou_thresh:
                # aggiorna traccia esistente
                best_tr["bbox"] = bbox
                best_tr["age"] = 0
                track_id = best_tr["id"]
            else:
                # nuova traccia
                track_id = next_id
                next_id += 1
                active[cls_id].append({"id": track_id, "bbox": bbox, "age": 0})
            out[fi][cls_id] = {
                "x": float(x), "y": float(y), "w": float(w), "h": float(h),
                "conf": float(conf), "track_id": int(track_id)
            }
        # rimuovi tracce troppo vecchie
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
    print(f"[TRACK-2D] JSON salvato: {output_file}")


def run_video_inference(video_path: Path, model_path: str, labels_base: Path, device: str = 'auto', conf_thres: float = 0.25, imgsz: int = 1920):
    """
    Esegue inferenza YOLO su un video e salva le labels in un unico JSON per camera.
    Estrae il numero di camera dal nome del video (es. out13.mp4 -> cam 13).
    
    Args:
        video_path: Path al video input
        model_path: Path ai pesi del modello YOLO
        labels_base: Directory dove salvare il JSON delle labels
        device: Device per inferenza ('auto', 'cpu', '0', etc.)
        conf_thres: Confidence threshold per filtrare predizioni (default: 0.25)
        imgsz: Dimensione immagine per inferenza (default: 1920)
    """
    print(f"[INFERENCE] Carico modello {model_path}...")
    model = YOLO(model_path)
    
    video_stem = video_path.stem  # es. out13
    m = re.search(r"(\d+)", video_stem)
    if m:
        cam_num = int(m.group(1))
    else:
        cam_num = None
    
    print(f"[INFERENCE] Processo video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire video: {video_path}")
    
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
        
        # Filtra best per classe
        best_per_class = {}
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                if box.cls is None or box.conf is None:
                    continue
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if cls_id not in best_per_class or conf > best_per_class[cls_id][0]:
                    best_per_class[cls_id] = (conf, box)
        
        # Aggrega per frame
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
    
    # Salva unico JSON
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
    
    print(f"[INFERENCE] Completato: {frame_idx} frame salvati in {out_json}")
    return cam_num if m else None


def run_images_inference(images_dir: Path, model_path: str, labels_base: Path, device: str = 'auto', conf_thres: float = 0.25):
    """
    Esegue inferenza YOLO su una directory di immagini e salva le labels in JSON per camera.
    Supporta pattern filename: out{cam}_frame_{num}_...
    """
    print(f"[INFERENCE] Carico modello {model_path}...")
    model = YOLO(model_path)
    
    print(f"[INFERENCE] Scansione immagini in: {images_dir}")
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    if not image_files:
        print("[INFERENCE] Nessuna immagine trovata.")
        return []

    # Struttura: {cam_id: {frame_idx: [detections]}}
    per_cam_results = {}
    
    # Regex per estrarre cam e frame
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
        
        # Filtra best per classe
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
            print(f"[INFERENCE] Processate {count} immagini...")

    # Salva i risultati
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
        print(f"[INFERENCE] Salvato {out_file} con {len(frames)} frame")
        saved_cams.append(cam_id)
    
    return saved_cams


# Colori mappati per indice di classe (BGR OpenCV)
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
    parser.add_argument('--fit', action='store_true', help='Ridimensiona per adattare allo schermo (mantiene aspect)')
    parser.add_argument('--maxw', type=int, default=1920, help='Larghezza massima finestra (se --fit)')
    parser.add_argument('--maxh', type=int, default=1080, help='Altezza massima finestra (se --fit)')
    parser.add_argument('--scale-display', type=float, default=0.5, help='Scala manuale (es. 0.5 = metà). Sovrascrive --fit se impostato')
    # Video options
    parser.add_argument('--video', type=str, default=None, help='Percorso video mp4 (se fornito ignora --images random)')
    parser.add_argument('--out-video', type=str, default='runs/infer_video/out.mp4', help='Percorso output video annotato')
    parser.add_argument('--frame-skip', type=int, default=0, help='Salta N frame tra uno processato e il successivo')
    parser.add_argument('--limit-frames', type=int, default=None, help='Processa al massimo N frame (debug)')
    parser.add_argument('--save-json', action='store_true', help='Salva un file JSON aggregato con le detections per frame')
    parser.add_argument('--json-dir', type=str, default='dataset/infer_video_json', help='Directory per i JSON aggregati')
    parser.add_argument('--labels-base', type=str, default='dataset/infer_video', help='Base dir dove salvare le label per frame (labels_outX)')
    return parser.parse_args()

def pick_random_image(images_dir: str) -> Path:
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [p for p in Path(images_dir).glob('*') if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"Nessuna immagine trovata in {images_dir}")
    return random.choice(files)

def draw_detections(image, result, class_names):
    """Disegna solo la bounding box con conf massima per ogni classe.

    Se ci sono più box della stessa classe, viene mantenuta solo quella con confidence maggiore.
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
    """Ritorna solo le classi con la loro migliore confidence (una per classe)."""
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

def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Modello non trovato: {args.model}. Assicurati che 'fine_tuned_yolo.pt' sia stato salvato.")

    model = YOLO(args.model)

    # Modalità VIDEO
    if args.video:
        if not os.path.isfile(args.video):
            raise FileNotFoundError(f"Video non trovato: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Impossibile aprire video: {args.video}")
        os.makedirs(os.path.dirname(args.out_video), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))
        frame_idx = 0
        processed = 0
        json_frames = {}
        print(f"[INFO] Inizio elaborazione video: {args.video} -> {args.out_video}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.limit_frames is not None and processed >= args.limit_frames:
                break
            if args.frame_skip > 0 and (frame_idx % (args.frame_skip + 1)) != 0:
                frame_idx += 1
                continue

            # YOLO predict su frame (usa direttamente array BGR)
            results = model.predict(source=frame, device=args.device, verbose=False, imgsz=1920)
            result = results[0]
            annotated = draw_detections(frame, result, DEFAULT_CLASS_NAMES)

            # Salva etichette YOLOv8 (solo best per classe) per ogni frame
            video_stem = Path(args.video).stem  # es. out13
            # Prova a estrarre camera id per salvare in dataset/infer_video/labels_out{cam}
            import re
            cam_num = None
            m = re.search(r"(\d+)", video_stem)
            if m:
                cam_num = int(m.group(1))
            if cam_num is not None:
                labels_dir = Path(args.labels_base) / f"labels_out{cam_num}"
            else:
                # fallback: usa il nome video
                labels_dir = Path(args.labels_base) / f"labels_{video_stem}"
            labels_dir.mkdir(parents=True, exist_ok=True)
            label_file = labels_dir / f"frame_{frame_idx:05d}.txt"
            # Filtra best per classe (come draw_detections)
            best_per_class = {}
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    if box.cls is None or box.conf is None:
                        continue
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    if cls_id not in best_per_class or conf > best_per_class[cls_id][0]:
                        best_per_class[cls_id] = (conf, box)
            with open(label_file, "w") as f:
                for cls_id, (conf, box) in best_per_class.items():
                    # YOLOv8: x_center, y_center, width, height normalizzati [0,1]
                    xywh = box.xywh[0].cpu().numpy()
                    x, y, w, h = xywh
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")

            # Aggrega per JSON
            if args.save_json:
                items = []
                for cls_id, (conf, box) in best_per_class.items():
                    xywh = box.xywh[0].cpu().numpy()
                    x, y, w, h = [float(v) for v in xywh]
                    items.append({
                        "class_id": int(cls_id),
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "conf": float(conf)
                    })
                json_frames[str(frame_idx)] = items

            writer.write(annotated)

            # Log minimale ogni 30 frame
            if processed % 30 == 0:
                summary = summarize(result, DEFAULT_CLASS_NAMES)
                print(f"[FRAME {frame_idx}] {summary}")
            processed += 1
            frame_idx += 1
        cap.release()
        writer.release()
        # Salva JSON aggregato se richiesto
        if args.save_json:
            import re, json as _json
            video_stem = Path(args.video).stem  # es. out13
            # prova a estrarre camera id da nome
            cam_num = None
            m = re.search(r"(\d+)", video_stem)
            if m:
                cam_num = int(m.group(1))
            out_dir = Path(args.json_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_json = out_dir / f"labels_{video_stem}.json"
            payload = {
                "camera": cam_num,
                "coord_mode": "pixel",
                "frames": json_frames
            }
            with out_json.open("w", encoding="utf-8") as jf:
                _json.dump(payload, jf)
            print(f"[INFO] JSON aggregato salvato in {out_json}")
        print(f"[INFO] Video completato. Frame salvati: {processed}")
        return

    # Modalità IMMAGINE (random da directory)
    img_path = pick_random_image(args.images)
    print(f"[INFO] Immagine scelta: {img_path}")
    results = model.predict(source=str(img_path), device=args.device, verbose=False, imgsz=1920)
    result = results[0]
    image = cv2.imread(str(img_path))
    if image is None:
        raise RuntimeError(f"Impossibile leggere immagine: {img_path}")
    annotated = draw_detections(image, result, DEFAULT_CLASS_NAMES)
    counts = summarize(result, DEFAULT_CLASS_NAMES)
    if counts:
        print("[BEST PER CLASSE]", counts)
    else:
        print("[DETEZIONI] Nessun oggetto rilevato")
    if args.save:
        out_dir = Path('runs/infer_random')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(out_file), annotated)
        print(f"[INFO] Salvato: {out_file}")
    if args.show:
        to_show = annotated
        if args.scale_display is not None:
            if args.scale_display <= 0:
                raise ValueError('--scale-display deve essere > 0')
            w = int(to_show.shape[1] * args.scale_display)
            h = int(to_show.shape[0] * args.scale_display)
            to_show = cv2.resize(to_show, (w, h), interpolation=cv2.INTER_AREA)
        elif args.fit:
            h0, w0 = to_show.shape[:2]
            scale = min(args.maxw / w0, args.maxh / h0, 1.0)
            if scale < 1.0:
                new_w = int(w0 * scale)
                new_h = int(h0 * scale)
                to_show = cv2.resize(to_show, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imshow('detections', to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save_tracked_video_from_images(frames_data: Dict[int, Dict[int, Dict]], images_dir: Path, output_path: Path, fps: int = 25, camera_id: int = None):
    """
    Genera un video MP4 dalle immagini nella directory, disegnando le bounding box e i track ID.
    
    Args:
        frames_data: Dict {frame_idx: {cls_id: {x, y, w, h, track_id, ...}}}
        images_dir: Directory contenente le immagini sorgente (out{cam}_frame_{num}...)
        output_path: Path del file video output (.mp4)
        fps: Frame rate del video output (default 25)
        camera_id: ID della camera per filtrare le immagini
    """
    import cv2
    import re
    import numpy as np
    
    images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not images:
        print(f"[TRACK-VIDEO] Nessuna immagine trovata in {images_dir}")
        return

    # Filtra immagini per camera
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
    
    # Ordina per frame index
    cam_images.sort(key=lambda x: x[0])
    
    # Rimuovi duplicati (mantieni solo la prima immagine per ogni frame index)
    unique_cam_images = []
    seen_frames = set()
    for f_idx, img_path in cam_images:
        if f_idx not in seen_frames:
            unique_cam_images.append((f_idx, img_path))
            seen_frames.add(f_idx)
    cam_images = unique_cam_images
    
    if not cam_images:
        print(f"[TRACK-VIDEO] Nessuna immagine trovata per camera {camera_id}")
        return

    # Leggi prima immagine per dimensioni
    first_img = cv2.imread(str(cam_images[0][1]))
    if first_img is None:
        print(f"[TRACK-VIDEO] Errore lettura immagine {cam_images[0][1]}")
        return
    
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    print(f"[TRACK-VIDEO] Generazione video {output_path} ({len(cam_images)} frames, {fps} fps)...")
    
    # Colori mappati per indice di classe (BGR OpenCV)
    # Usa CLASS_COLORS se disponibile, altrimenti fallback
    colors = CLASS_COLORS if 'CLASS_COLORS' in globals() else []
    
    for f_idx, img_path in cam_images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
            
        # Disegna detections
        if f_idx in frames_data:
            for cls_id, det in frames_data[f_idx].items():
                x, y, bw, bh = det['x'], det['y'], det['w'], det['h']
                tid = det.get('track_id', -1)
                
                # Coordinate pixel (x,y centro)
                x1 = int(x - bw/2)
                y1 = int(y - bh/2)
                x2 = int(x + bw/2)
                y2 = int(y + bh/2)
                
                # Usa colore basato sulla classe, non sul track_id
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
    print(f"[TRACK-VIDEO] Video salvato: {output_path}")


def save_tracked_video_from_video(frames_data: Dict[int, Dict[int, Dict]], video_path: Path, output_path: Path):
    """
    Genera un video MP4 partendo da un video sorgente, disegnando le bounding box e i track ID.
    
    Args:
        frames_data: Dict {frame_idx: {cls_id: {x, y, w, h, track_id, ...}}}
        video_path: Path del video sorgente input
        output_path: Path del file video output (.mp4)
    """
    import cv2
    import numpy as np
    
    if not video_path.exists():
        print(f"[TRACK-VIDEO] Video sorgente non trovato: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[TRACK-VIDEO] Impossibile aprire video sorgente: {video_path}")
        return
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    print(f"[TRACK-VIDEO] Generazione video {output_path} da {video_path.name}...")
    
    # Colori mappati per indice di classe (BGR OpenCV)
    # Usa CLASS_COLORS se disponibile, altrimenti fallback
    colors = CLASS_COLORS if 'CLASS_COLORS' in globals() else []
    
    frame_idx = 1  # Start from 1 to match inference indexing
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Disegna detections se presenti per questo frame
        if frame_idx in frames_data:
            for cls_id, det in frames_data[frame_idx].items():
                x, y, bw, bh = det['x'], det['y'], det['w'], det['h']
                tid = det.get('track_id', -1)
                
                # Coordinate pixel (x,y centro)
                x1 = int(x - bw/2)
                y1 = int(y - bh/2)
                x2 = int(x + bw/2)
                y2 = int(y + bh/2)
                
                # Usa colore basato sulla classe, non sul track_id
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
            print(f"[TRACK-VIDEO] Processati {frame_idx}/{total_frames} frames", end='\r')
            
    print("")
    cap.release()
    out.release()
    print(f"[TRACK-VIDEO] Video salvato: {output_path}")


if __name__ == '__main__':
    main()

# ----------------------------------------
# How to run (PowerShell)
# ----------------------------------------
# Immagine casuale dal train, salva l'annotazione:
#   python inference_random_frame.py --save
# Specifica una cartella immagini diversa (es. validation):
#   python inference_random_frame.py --images dataset/val/images --seed 42 --save
# Video: salva video annotato e JSON aggregato per frame (best per classe):
#   python inference_random_frame.py --video dataset/infer_video/out13.mp4 --out-video runs/infer_video/out13_annot.mp4 --save-json --json-dir dataset/infer_video_json
