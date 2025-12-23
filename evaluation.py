"""
Modulo di valutazione per modelli YOLO su dataset di test e per confronto label-vs-label.

Funzionalità principali:
  1. Inferenza YOLO + calcolo IOU contro ground truth (compute_iou_metrics)
  2. Confronto labels già pronte (no inferenza) contro GT (compute_iou_metrics_from_predictions)
  3. Supporto per valutazione video con pattern file flessibili
  4. Export metriche JSON per analisi

Uso tipico:
    # Con modello YOLO:
    python evaluation.py --model fine_tuned_yolo.pt --images dataset/test/images --labels dataset/test/labels
    
    # Da pipeline (labels-vs-labels):
    from evaluation import load_labels_dir_map, compute_iou_metrics_from_predictions
    gt = load_labels_dir_map(Path('dataset/val/labels'), camera_id=13)
    pred = {...}  # da inferenza o interpolazione
    metrics = compute_iou_metrics_from_predictions(pred, gt)
"""
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import json
from config import get_args, PipelineConfig

def load_yolo_labels(label_path):
    """
    Carica bounding boxes da file YOLO txt (formato YOLOv8: class x y w h).
    
    Args:
        label_path: Path file .txt con labels (una per riga)
    
    Returns:
        boxes: Lista di tuple (class_id, x, y, w, h) normalizzate [0,1]
    
    Formato file:
        0 0.5 0.5 0.1 0.2
        1 0.3 0.7 0.15 0.25
        ...
    
    Esempio:
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
    Calcola Intersection over Union (IoU) tra due bounding box normalizzate.
    
    Args:
        box1: Tuple (x, y, w, h) coordinate normalizzate [0,1]
        box2: Tuple (x, y, w, h) coordinate normalizzate [0,1]
    
    Returns:
        iou: Float in [0,1], 0=nessuna sovrapposizione, 1=perfetta coincidenza
    
    Note:
        - x,y sono coordinate del centro bbox
        - w,h sono larghezza e altezza normalizzate
        - Converte internamente in formato xyxy per calcolo intersezione
    
    Esempio:
        box1 = (0.5, 0.5, 0.2, 0.2)  # centro (0.5,0.5), size 20%x20%
        box2 = (0.52, 0.51, 0.18, 0.19)  # parzialmente sovrapposta
        iou = box_iou(box1, box2)
        # iou ≈ 0.75 (alta sovrapposizione)
    """
    # Funzione interna per conversione xywh → xyxy
    def to_xyxy(x, y, w, h):
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return x1, y1, x2, y2
    
    # Converte entrambe le box in formato xyxy
    x1_1, y1_1, x2_1, y2_1 = to_xyxy(*box1)
    x1_2, y1_2, x2_2, y2_2 = to_xyxy(*box2)
    
    # Calcola rettangolo di intersezione
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    # Calcola aree delle due box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Union = somma - intersezione
    union = area1 + area2 - inter_area
    if union == 0:
        return 0.0
    return inter_area / union

def per_class_iou(gt_boxes, pred_boxes, num_classes=13):
    """
    Calcola IOU per ogni classe usando matching greedy (best IoU per prediction).
    Penalizza falsi negativi (GT non matchate) aggiungendo IoU=0.
    
    Args:
        gt_boxes: Lista di ground truth [(class, x, y, w, h), ...]
        pred_boxes: Lista di predictions [(class, x, y, w, h), ...]
        num_classes: Numero totale di classi nel dataset (default 13)
    
    Returns:
        iou_per_class: Dict {class_id: [iou_values]} con IOU per ogni prediction matchata + GT persa
    
    Algoritmo:
        1. Per ogni classe, filtra GT e predictions
        2. Per ogni prediction, trova GT con massimo IoU (greedy)
        3. Marca GT usate per evitare match multipli
        4. Se nessun match, IOU=0 per quella prediction (falso positivo debole)
        5. Per ogni GT non matchata, aggiungi IOU=0 (falso negativo - PENALIZZAZIONE)
        6. Ignora classi senza GT in questo frame
    
    Esempio:
        # Caso 1: Perfetto match
        gt = [(0, 0.5, 0.5, 0.1, 0.1), (1, 0.3, 0.7, 0.15, 0.2)]
        pred = [(0, 0.51, 0.49, 0.09, 0.11), (1, 0.32, 0.68, 0.14, 0.21)]
        iou_pc = per_class_iou(gt, pred)
        # iou_pc = {0: [0.85], 1: [0.92], 2: [], ...}  # alta sovrapposizione
        
        # Caso 2: Falso negativo (GT persa)
        gt = [(0, 0.5, 0.5, 0.1, 0.1)]  # 1 palla
        pred = []                       # modello non la vede
        iou_pc = per_class_iou(gt, pred)
        # iou_pc = {0: [0.0], 1: [], ...}  # GT persa = IoU 0 (penalizzata)
    """
    iou_per_class = {c: [] for c in range(num_classes)}
    
    # Per ogni classe separatamente
    for c in range(num_classes):
        # Filtra GT e predictions per questa classe
        gt_c = [g for g in gt_boxes if g[0] == c]
        pred_c = [p for p in pred_boxes if p[0] == c]
        
        # Se nessuna GT per questa classe, salta
        if not gt_c:
            continue
        
        # Traccia GT già usate (un GT può matchare solo una prediction)
        used = set()
        
        # Per ogni prediction, trova best GT match
        for p in pred_c:
            px, py, pw, ph = p[1:]
            best_iou = 0.0
            best_idx = -1
            
            # Trova GT con massimo IoU non ancora usata
            for idx, g in enumerate(gt_c):
                if idx in used:
                    continue
                gx, gy, gw, gh = g[1:]
                iou = box_iou((px, py, pw, ph), (gx, gy, gw, gh))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            # Marca GT come usata
            if best_idx >= 0:
                used.add(best_idx)
            
            # Registra IOU (anche se 0 per falso positivo)
            iou_per_class[c].append(best_iou)
        
        # ⭐ NUOVO: Penalizza GT non matchate (falsi negativi)
        # Per ogni GT di questa classe che non è stata usata
        for idx in range(len(gt_c)):
            if idx not in used:
                # GT persa dal modello → aggiungi IoU=0
                iou_per_class[c].append(0.0)
    
    return iou_per_class


def compute_iou_metrics(model_path: str, images_dir: str, labels_dir: str, imgsz: int = 1920, device: str = 'auto', conf_thres: float = 0.01, num_classes: int = 13) -> dict:
    """
    Esegue inferenza YOLO su dataset e calcola metriche IOU contro ground truth.
    
    Args:
        model_path: Path al modello YOLO .pt
        images_dir: Directory con immagini (.jpg, .png, .bmp)
        labels_dir: Directory con label YOLO .txt (stesso nome immagini)
        imgsz: Dimensione input YOLO (default 1920)
        device: Device inferenza 'auto', 'cpu', '0', '0,1' (default 'auto')
        conf_thres: Confidence threshold minima (default 0.01)
        num_classes: Numero classi del modello (default 13)
    
    Returns:
        metrics: Dict con:
          - 'per_image': [{'image': str, 'average_iou': float}, ...]
          - 'per_class': {class_id: {'mean_iou': float|None, 'count': int}, ...}
          - 'mean_iou': float (media su tutte le immagini)
          - 'images_count': int
          - 'images_dir', 'labels_dir', 'model', 'imgsz', 'device', 'conf_thres'
    
    Esempio:
        metrics = compute_iou_metrics(
            model_path='weights/fine_tuned_yolo_final.pt',
            images_dir='dataset/test/images',
            labels_dir='dataset/test/labels',
            device='0'
        )
        print(f"Mean IOU: {metrics['mean_iou']:.3f}")
        print(f"Classe 0 (Ball): {metrics['per_class'][0]['mean_iou']:.3f}")
    """
    # Carica modello YOLO
    model = YOLO(model_path)
    
    # Lista immagini da valutare
    images = sorted([p for p in Path(images_dir).glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    
    per_image = []
    all_iou_per_class = {c: [] for c in range(num_classes)}
    
    # Per ogni immagine
    for img_path in images:
        # Carica ground truth
        label_path = Path(labels_dir) / (img_path.stem + '.txt')
        gt_boxes = load_yolo_labels(label_path)
        
        # Inferenza YOLO
        pred = model.predict(source=str(img_path), device=device, verbose=False, imgsz=imgsz, conf=conf_thres)
        
        # Estrai predictions normalizzate
        pred_boxes = []
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2] if img is not None else (1, 1)
        if pred and hasattr(pred[0], 'boxes') and pred[0].boxes is not None:
            for box in pred[0].boxes:
                cls = int(box.cls.item())
                xywh = box.xywh[0].cpu().numpy()
                x, y, bw, bh = xywh
                # Normalizza coordinate in [0,1]
                x /= w; y /= h; bw /= w; bh /= h
                pred_boxes.append((cls, x, y, bw, bh))
        
        # ⭐ SKIP frame senza GT (non c'è niente da valutare)
        if not gt_boxes:
            continue
        
        # Calcola IOU per classe
        iou_pc = per_class_iou(gt_boxes, pred_boxes, num_classes=num_classes)
        for c, ious in iou_pc.items():
            all_iou_per_class[c].extend(ious)
        
        # Media IOU per questa immagine
        avg_iou_img = float(np.mean([v for lst in iou_pc.values() for v in lst])) if any(iou_pc.values()) else 0.0
        per_image.append({"image": img_path.name, "average_iou": avg_iou_img})

    # Aggrega statistiche per classe
    per_class = {}
    for c, ious in all_iou_per_class.items():
        per_class[c] = {"mean_iou": float(np.mean(ious)) if ious else None, "count": int(len(ious))}
    
    # Media globale (per box)
    all_ious = [iou for ious in all_iou_per_class.values() for iou in ious]
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    
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
    """
    Salva metriche di valutazione in JSON.
    
    Args:
        metrics: Dict con metriche da salvare
        out_json: Path file JSON output
    
    Esempio:
        metrics = compute_iou_metrics(...)
        save_eval_json(metrics, Path('runs/eval/eval_cam13_raw.json'))
        # File salvato: runs/eval/eval_cam13_raw.json
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open('w', encoding='utf-8') as f:
        json.dump(metrics, f)
    print(f"[EVAL] Salvate metriche in {out_json}")


# -----------------------
# Label-vs-label evaluation (no model) for video label directories
# -----------------------

def load_labels_dir_map(labels_dir: Path, camera_id: int | None = None) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """
    Carica labels YOLO da directory txt organizzata per frame.
    Supporta due pattern di naming:
    - Standard: frame_XXXXX.txt
    - Roboflow: out{cam}_frame_{num}_png.rf.{hash}.txt
    
    Args:
        labels_dir: Path directory con file .txt (es. Path('dataset/val/labels'))
        camera_id: Se specificato, filtra solo frame per quella camera (pattern Roboflow)
    
    Returns:
        frames: Dict {frame_idx: [(class_id, x, y, w, h), ...]}
        
    Esempio Standard:
        # labels_dir contiene: frame_00001.txt, frame_00002.txt, ...
        frames = load_labels_dir_map(Path('dataset/gt_video/labels'))
        print(frames[1])  # [(0, 0.5, 0.5, 0.1, 0.1), (5, 0.3, 0.4, 0.08, 0.09)]
    
    Esempio Roboflow:
        # labels_dir contiene: out13_frame_0001_png.rf.{hash}.txt, out4_frame_0012_png.rf.{hash}.txt, ...
        frames_cam13 = load_labels_dir_map(Path('dataset/val/labels'), camera_id=13)
        frames_cam4 = load_labels_dir_map(Path('dataset/val/labels'), camera_id=4)
        print(len(frames_cam13))  # 110
    """
    frames: dict[int, list[tuple[int, float, float, float, float]]] = {}
    if not labels_dir.exists():
        return frames
    
    # Pattern regex per Roboflow: out{cam}_frame_{num}_png.rf.{hash} oppure out{cam}_frame_{num}
    import re
    
    for p in sorted(labels_dir.glob('*.txt')):
        stem = p.stem
        fi = None  # frame index
        cam = None  # camera id
        
        # Pattern Roboflow: out{cam}_frame_{num}_png.rf.{hash}
        m = re.match(r"out(\d+)_frame_(\d+)", stem)
        if m:
            cam = int(m.group(1))
            fi = int(m.group(2))
        # Pattern standard: frame_00012
        elif stem.startswith('frame_') and '_' not in stem[6:]:
            if stem[6:].isdigit():
                fi = int(stem[6:])
        else:
            # Fallback: prendi numeri finali
            m2 = re.search(r"(\d+)$", stem)
            if m2:
                fi = int(m2.group(1))
        
        if fi is None:
            continue
        
        # Filtra per camera se specificato
        if camera_id is not None and cam is not None and cam != camera_id:
            continue
        
        # Carica bounding boxes da file
        boxes = load_yolo_labels(str(p))  # [(cls,x,y,w,h)]
        frames[fi] = [(int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])) for b in boxes]
    
    return frames


def frames_from_perframe(perframe: dict[int, dict[int, tuple[float, float, float, float, float]]]) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """
    Converte struttura PerFrame {frame: {cls: (x,y,w,h,conf)}} in formato evaluation {frame: [(cls,x,y,w,h),...]}.
    
    Args:
        perframe: Dict {frame_idx: {class_id: (x, y, w, h, conf)}}
                  Struttura interna pipeline con confidence
    
    Returns:
        frames: Dict {frame_idx: [(class_id, x, y, w, h), ...]}
                Formato per evaluation (senza confidence)
    
    Esempio:
        # Input da pipeline dopo interpolazione
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
    
    # Per ogni frame
    for fi, dets in perframe.items():
        lst = []
        # Per ogni detection (class_id: (x,y,w,h,conf))
        for cls_id, (x, y, w, h, _conf) in dets.items():
            # Scarta confidence, mantieni solo bbox
            lst.append((int(cls_id), float(x), float(y), float(w), float(h)))
        out[int(fi)] = lst
    
    return out


def compute_iou_metrics_from_predictions(pred_frames: dict[int, list[tuple[int, float, float, float, float]]],
                                         gt_frames: dict[int, list[tuple[int, float, float, float, float]]],
                                         num_classes: int | None = None) -> dict:
    """
    Calcola metriche IOU confrontando predizioni già pronte (senza modello) con GT per frame/classe.
    
    Args:
        pred_frames: Dict {frame_idx: [(class_id, x, y, w, h), ...]} predizioni normalizzate
        gt_frames: Dict {frame_idx: [(class_id, x, y, w, h), ...]} ground truth normalizzate
        num_classes: Numero classi (auto-detect se None)
    
    Returns:
        metrics: Dict con:
          - 'mode': 'labels-vs-labels'
          - 'frames_evaluated': int numero frame valutati
          - 'per_frame': [{'frame': int, 'average_iou': float}, ...]
          - 'per_class': {class_id: {'mean_iou': float|None, 'count': int}, ...}
          - 'mean_iou': float (media su tutti i frame)
          - 'num_classes': int
    
    Note:
        Usato da pipeline.py per valutazioni RAW e INTERP (senza re-inference).
    
    Esempio:
        # Predizioni dopo tracking
        pred_frames = {1: [(0, 0.5, 0.5, 0.1, 0.1), (5, 0.3, 0.4, 0.08, 0.09)],
                      2: [(0, 0.51, 0.52, 0.1, 0.1)]}
        
        # Ground truth da dataset/val/labels
        gt_frames = {1: [(0, 0.49, 0.51, 0.11, 0.09), (5, 0.31, 0.39, 0.09, 0.1)],
                    2: [(0, 0.50, 0.53, 0.10, 0.11)]}
        
        metrics = compute_iou_metrics_from_predictions(pred_frames, gt_frames, num_classes=13)
        print(f"Mean IOU: {metrics['mean_iou']:.3f}")  # 0.823
        print(f"Frames: {metrics['frames_evaluated']}")  # 2
    """
    # Trova frame comuni tra pred e GT (matching)
    frames_matching = sorted(set(pred_frames.keys()) & set(gt_frames.keys()))

    # Trova tutti i frame GT (per valutazione su tutti i GT)
    frames_gt = sorted(gt_frames.keys())

    # Auto-detect numero classi se non specificato
    if num_classes is None:
        max_cls = -1
        for d in (pred_frames, gt_frames):
            for lst in d.values():
                for it in lst:
                    max_cls = max(max_cls, int(it[0]))
        num_classes = max(0, max_cls + 1)

    # --- METRICA STANDARD: solo frame matching ---
    all_iou_per_class = {c: [] for c in range(num_classes)}
    per_frame = []
    for fi in frames_matching:
        pred = pred_frames.get(fi, [])
        gt = gt_frames.get(fi, [])
        iou_pc = per_class_iou(gt, pred, num_classes=num_classes)
        for c, ious in iou_pc.items():
            all_iou_per_class[c].extend(ious)
        avg_iou = float(np.mean([v for lst in iou_pc.values() for v in lst])) if any(iou_pc.values()) else 0.0
        per_frame.append({"frame": int(fi), "average_iou": avg_iou})
    per_class = {c: {"mean_iou": float(np.mean(v)) if v else None, "count": int(len(v))} for c, v in all_iou_per_class.items()}
    all_ious = [iou for ious in all_iou_per_class.values() for iou in ious]
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    # --- NUOVA METRICA: su tutti i frame GT (assegna IoU=0 se manca la predizione) ---
    all_iou_per_class_gt = {c: [] for c in range(num_classes)}
    per_frame_gt = []
    for fi in frames_gt:
        pred = pred_frames.get(fi, [])  # se manca, lista vuota
        gt = gt_frames.get(fi, [])
        iou_pc = per_class_iou(gt, pred, num_classes=num_classes)
        for c, ious in iou_pc.items():
            all_iou_per_class_gt[c].extend(ious)
        avg_iou = float(np.mean([v for lst in iou_pc.values() for v in lst])) if any(iou_pc.values()) else 0.0
        per_frame_gt.append({"frame": int(fi), "average_iou": avg_iou})
    per_class_gt = {c: {"mean_iou": float(np.mean(v)) if v else None, "count": int(len(v))} for c, v in all_iou_per_class_gt.items()}
    all_ious_gt = [iou for ious in all_iou_per_class_gt.values() for iou in ious]
    mean_iou_gt = float(np.mean(all_ious_gt)) if all_ious_gt else 0.0

    return {
        "mode": "labels-vs-labels",
        "frames_evaluated": len(frames_matching),
        "per_frame": per_frame,
        "per_class": per_class,
        "mean_iou": mean_iou,
        "frames_evaluated_gt": len(frames_gt),
        "per_frame_gt": per_frame_gt,
        "per_class_gt": per_class_gt,
        "mean_iou_gt": mean_iou_gt,
        "num_classes": num_classes,
    }

def main():
    """
    Script standalone per valutazione YOLO su dataset di immagini.
    Esegue inferenza e calcola IOU contro ground truth per ogni frame e classe.
    
    Usage:
        python evaluation.py --model weights/fine_tuned_yolo_final.pt \\
                            --images dataset/test/images \\
                            --labels dataset/test/labels \\
                            --device 0 \\
                            --conf-thres 0.25
    
    Output:
        - Print IOU per immagine e per classe su console
        - Media IOU finale su tutti i frame
    
    Esempio Output:
        frame_00001.jpg PREDICTIONS:
          class=0 x=0.5123 y=0.4567 w=0.0987 h=0.1234
        frame_00001.jpg GROUND TRUTH:
          class=0 x=0.5100 y=0.4550 w=0.1000 h=0.1200
        frame_00001.jpg: average IOU = 0.892
        
        Average IOU per classe (solo frame con GT):
        Classe 0: 0.876 su 45 box
        Classe 5: 0.823 su 32 box
        
        Mean IOU su 110 frame: 0.854
    """
    # Parsing argomenti CLI (usa configurazione centralizzata)
    args = get_args()
    config = PipelineConfig.from_args(args)
    
    # Carica modello YOLO
    model = YOLO(config.model)
    
    # Lista immagini da valutare
    images = sorted([p for p in Path(args.images).glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    
    # Inizializza accumulatori
    results = []
    all_iou_per_class = {c: [] for c in range(13)}
    
    # Per ogni immagine
    for img_path in images:
        # Carica ground truth
        label_path = Path(args.labels) / (img_path.stem + '.txt')
        gt_boxes = load_yolo_labels(label_path)
        
        # Inferenza YOLO
        pred = model.predict(source=str(img_path), device=config.device, verbose=False, imgsz=config.imgsz, conf=config.conf_thres)
        
        # Estrai predictions normalizzate
        pred_boxes = []
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2] if img is not None else (1, 1)
        if pred and hasattr(pred[0], 'boxes') and pred[0].boxes is not None:
            for box in pred[0].boxes:
                cls = int(box.cls.item())
                xywh = box.xywh[0].cpu().numpy()  # in pixel
                # Normalizza rispetto a dimensione immagine
                x, y, bw, bh = xywh
                x /= w; y /= h; bw /= w; bh /= h
                pred_boxes.append((cls, x, y, bw, bh))
        
        # Print predizioni per debug
        print(f"{img_path.name} PREDICTIONS:")
        for pb in pred_boxes:
            print(f"  class={pb[0]} x={pb[1]:.4f} y={pb[2]:.4f} w={pb[3]:.4f} h={pb[4]:.4f}")
        
        # Print ground truth per debug
        print(f"{img_path.name} GROUND TRUTH:")
        for gb in gt_boxes:
            print(f"  class={gb[0]} x={gb[1]:.4f} y={gb[2]:.4f} w={gb[3]:.4f} h={gb[4]:.4f}")
        
        # ⭐ SKIP frame senza GT (non c'è niente da valutare)
        if not gt_boxes:
            print(f"{img_path.name}: SKIPPED (nessuna GT)")
            continue
        
        # Calcola IOU per classe con greedy matching
        iou_per_class = per_class_iou(gt_boxes, pred_boxes, num_classes=13)
        
        # Accumula per statistiche globali
        for c, ious in iou_per_class.items():
            all_iou_per_class[c].extend(ious)
        
        # Media IOU per questa immagine
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
    
    # Statistica finale globale
    if results:
        mean_iou = np.mean([r[1] for r in results])
        print(f"\nMean IOU su {len(results)} frame: {mean_iou:.3f}")
    else:
        print("Nessuna immagine trovata.")


if __name__ == "__main__":
    main()
