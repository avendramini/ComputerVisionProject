"""
Esempi di utilizzo (da eseguire nella root del progetto):

1) Inference semplice su un'immagine casuale del train e salvataggio risultato:
    python inference_random_frame.py --save

2) Mostrare a schermo (richiede un ambiente con GUI disponibile):
    python inference_random_frame.py --show

3) Specificare directory immagini (ad esempio la validation) e seed riproducibile:
    python inference_random_frame.py --images dataset/val/images --seed 42 --save

4) Forzare la GPU 0 esplicitamente (invece di 'auto'):
    python inference_random_frame.py --device 0 --save

5) Usare un modello differente (es. best.pt dentro runs):
    python inference_random_frame.py --model runs/detect/train2/weights/best.pt --save

"""

import os
import random
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

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
            video_stem = Path(args.video).stem
            labels_dir = Path(args.out_video).parent / f"labels_{video_stem}"
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

            writer.write(annotated)

            # Log minimale ogni 30 frame
            if processed % 30 == 0:
                summary = summarize(result, DEFAULT_CLASS_NAMES)
                print(f"[FRAME {frame_idx}] {summary}")
            processed += 1
            frame_idx += 1
        cap.release()
        writer.release()
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

if __name__ == '__main__':
    main()
