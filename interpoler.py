"""
Interpolazione lineare per label YOLO mancanti.
Legge le label da dataset/infer_video/labels_outX/ e riempie i frame dove
una classe è assente usando interpolazione lineare dai frame precedenti/successivi.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
import json


def load_labels_for_camera(labels_dir_or_json):
    """
    Carica tutte le label per una camera da una directory di .txt o da un file JSON.
    
    Args:
        labels_dir_or_json: Path a directory labels_outX/ o file labels_outX.json
    
    Returns:
        frames: dict {frame_idx: {class_id: (x, y, w, h, conf)}}
        max_frame: int, ultimo frame index
    """
    path = Path(labels_dir_or_json)
    frames = {}
    max_frame = -1
    
    # Prova a caricare come JSON
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
    
    # Altrimenti carica da directory .txt
    labels_path = Path(labels_dir_or_json)
    if not labels_path.exists():
        raise FileNotFoundError(f"Directory/file non trovata: {labels_dir_or_json}")
    
    # Leggi tutti i file frame_XXXXX.txt
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
    Interpola le detection mancanti per ogni classe.
    
    Args:
        frames: dict {frame_idx: {class_id: [x, y, w, h, conf]}}
        max_frame: ultimo frame index
        max_gap: numero massimo di frame consecutivi mancanti da interpolare
    
    Returns:
        interpolated_frames: dict completo con detection interpolate
    """
    interpolated = defaultdict(dict)
    
    # Copia i frame esistenti
    for frame_idx, detections in frames.items():
        interpolated[frame_idx] = detections.copy()
    
    # Trova tutte le classi presenti
    all_classes = set()
    for detections in frames.values():
        all_classes.update(detections.keys())
    
    print(f"[INFO] Classi trovate: {sorted(all_classes)}")
    
    # Per ogni classe, interpola i frame mancanti
    total_interpolated = 0
    for cls_id in sorted(all_classes):
        # Trova tutti i frame dove questa classe è presente
        frames_with_class = sorted([f for f in frames.keys() if cls_id in frames[f]])
        if len(frames_with_class) < 2:
            continue
        interpolated_count = 0
        # Interpola solo tra frame consecutivi con gap <= max_gap
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
                interpolated[frame_idx][cls_id] = [
                    det_interp[0], det_interp[1], det_interp[2], det_interp[3], conf_interp
                ]
                interpolated_count += 1
        total_interpolated += interpolated_count
    
    print(f"[INFO] Interpolated {total_interpolated} detections for this camera.")
    return interpolated


def save_interpolated_labels(frames, output_dir, max_frame):
    """
    Salva le label interpolate in formato YOLO.
    
    Args:
        frames: dict {frame_idx: {class_id: [x, y, w, h, conf]}}
        output_dir: directory di output
        max_frame: ultimo frame index
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    
    # Salva tutti i frame (anche quelli vuoti)
    for frame_idx in range(max_frame + 1):
        output_file = output_path / f"frame_{frame_idx:05d}.txt"
        
        with open(output_file, 'w') as f:
            if frame_idx in frames:
                # Ordina per class_id per consistenza
                for cls_id in sorted(frames[frame_idx].keys()):
                    x, y, w, h, conf = frames[frame_idx][cls_id]
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")
                saved_count += 1
    
    print(f"[INFO] Salvati {saved_count} frame con detection in {output_dir}")

def save_interpolated_json(frames, output_file: Path, camera_id: int | None = None, coord_mode: str = "pixel"):
    """
    Salva un unico file JSON con tutte le detection per camera.

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
    print(f"[INFO] JSON salvato: {output_file}")


def process_camera(cam_number, input_base_dir="dataset/infer_video", 
                   output_base_dir="dataset/infer_video_interpolated", 
                   max_gap=10,
                   save_json: bool = True,
                   json_base_dir: str = "dataset/infer_video_interpolated_json"):
    """
    Processa una singola camera: carica, interpola, salva.
    
    Args:
        cam_number: numero camera (es. 2, 4, 13)
        input_base_dir: directory base con labels_outX/
        output_base_dir: directory base output
        max_gap: massimo numero di frame consecutivi mancanti da interpolare
    """
    input_dir = Path(input_base_dir) / f"labels_out{cam_number}"
    output_dir = Path(output_base_dir) / f"labels_out{cam_number}"
    output_json = Path(json_base_dir) / f"labels_out{cam_number}.json"
    
    print(f"\n{'='*60}")
    print(f"[CAMERA {cam_number}] Inizio interpolazione")
    print(f"{'='*60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    
    # Carica label originali
    frames, max_frame = load_labels_for_camera(input_dir)
    print(f"[INFO] Caricati {len(frames)} frame (0-{max_frame})")
    
    # Calcola statistiche pre-interpolazione
    total_detections_before = sum(len(dets) for dets in frames.values())
    
    # Interpola
    interpolated_frames = interpolate_missing_detections(frames, max_frame, max_gap=max_gap)
    
    # Calcola statistiche post-interpolazione
    total_detections_after = sum(len(dets) for dets in interpolated_frames.values())
    added_detections = total_detections_after - total_detections_before
    
    print(f"\n[STATS] Detection originali: {total_detections_before}")
    print(f"[STATS] Detection interpolate: {added_detections}")
    print(f"[STATS] Detection totali: {total_detections_after}")
    
    # Salva risultati
    save_interpolated_labels(interpolated_frames, output_dir, max_frame)
    if save_json:
        save_interpolated_json(interpolated_frames, output_json, camera_id=cam_number, coord_mode="pixel")
    
    return {
        'camera': cam_number,
        'frames': len(frames),
        'max_frame': max_frame,
        'before': total_detections_before,
        'added': added_detections,
        'after': total_detections_after
    }


def main():
    parser = argparse.ArgumentParser(description="Interpola label YOLO mancanti")
    parser.add_argument('--cameras', type=int, nargs='+', default=[2, 4, 13],
                        help='Numeri delle camere da processare (default: 2 4 13)')
    parser.add_argument('--input-dir', type=str, default='dataset/infer_video',
                        help='Directory base input con labels_outX/')
    parser.add_argument('--output-dir', type=str, default='dataset/infer_video_interpolated',
                        help='Directory base output')
    parser.add_argument('--max-gap', type=int, default=10,
                        help='Massimo gap di frame consecutivi da interpolare (default: 10)')
    parser.add_argument('--no-json', action='store_true', help='Non salvare il JSON aggregato per camera')
    parser.add_argument('--json-dir', type=str, default='dataset/infer_video_interpolated_json',
                        help='Directory base per i JSON aggregati')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print("# INTERPOLAZIONE LABEL YOLO")
    print(f"{'#'*60}")
    print(f"Camere: {args.cameras}")
    print(f"Max gap: {args.max_gap} frame")
    
    # Processa ogni camera
    results = []
    for cam_num in args.cameras:
        try:
            result = process_camera(
                cam_num, 
                input_base_dir=args.input_dir,
                output_base_dir=args.output_dir,
                max_gap=args.max_gap,
                save_json=(not args.no_json),
                json_base_dir=args.json_dir
            )
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Camera {cam_num}: {e}")
            continue
    
    # Stampa summary finale
    print(f"\n{'='*60}")
    print("SUMMARY FINALE")
    print(f"{'='*60}")
    for r in results:
        print(f"Camera {r['camera']:2d}: {r['before']:5d} -> {r['after']:5d} detection "
              f"(+{r['added']:4d} interpolate, {r['frames']} frame)")
    
    total_added = sum(r['added'] for r in results)
    print(f"\nTotale detection interpolate: {total_added}")


if __name__ == '__main__':
    main()

# ----------------------------------------
# How to run (PowerShell)
# ----------------------------------------
# Interpola e salva TXT + JSON aggregato per tutte le camere:
#   python interpoler.py --cameras 2 4 13 --max-gap 10 --json-dir dataset/infer_video_interpolated_json
# Solo una camera (es. 13) con gap più piccolo:
#   python interpoler.py --cameras 13 --max-gap 5 --json-dir dataset/infer_video_interpolated_json
# Disabilitare salvataggio JSON aggregato:
#   python interpoler.py --no-json
