"""
Pipeline unificato per ricostruzione 3D multi-camera.

Flusso completo:
  1. (Opzionale) Inferenza YOLO su video per generare detections 2D
  2. Caricamento labels per camera (JSON o txt)
  3. (Opzionale) Tracking 2D per assegnare track_id tra frame consecutivi
  4. (Opzionale) Valutazione RAW contro ground truth
  5. (Opzionale) Interpolazione per riempire gap temporali
  6. (Opzionale) Valutazione INTERP contro ground truth
  7. (Opzionale) Rettificazione distorsioni ottiche
  8. Triangolazione 3D multi-camera (SVD lineare)
  9. Salvataggio risultati (CSV + JSON)
  10. (Opzionale) Visualizzazione interattiva 3D

Minimizza I/O su disco operando in memoria dove possibile.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple
import json
import numpy as np
import cv2

# Local modules
import triangulation_3d as tri
from interpoler import load_labels_for_camera, interpolate_missing_detections, save_interpolated_json
from tracking_2d import track_labels_for_camera, save_tracks_json, run_video_inference, run_images_inference
from config import get_args, PipelineConfig

# Type aliases per strutture dati comuni
Det = Tuple[float, float, float, float, float]  # Detection: (x, y, width, height, confidence)
PerFrame = Dict[int, Dict[int, Det]]  # Struttura frame: {frame_idx: {class_id: Detection}}


def build_frames_by_class_from_cameras(per_cam_frames: Dict[int, PerFrame]) -> List[List[List[Tuple[int, float, float, float, float, float]]]]:
	"""
	Riorganizza le detections da struttura per-camera a struttura per-frame-per-classe per triangolazione.
	
	Args:
		per_cam_frames: {camera_id: {frame_idx: {class_id: (x,y,w,h,conf)}}}
	
	Returns:
		frames_by_class[frame_idx][class_id] = [(cam, x, y, w, h, conf), ...]
		Lista di frame, ogni frame contiene lista per classe, ogni classe ha lista di detections multi-camera.
		Coordinate in pixel.
	
	Note:
		- Determina dinamicamente il numero massimo di frame e classi
		- Espande automaticamente la lista classi se scopre ID più alti
		- Utile per preparare l'input alla triangolazione multi-camera
	"""
	# Trova il numero massimo di frame e di classi presenti in tutte le camere
	max_frame = -1
	max_cls = -1
	for _cam, frames in per_cam_frames.items():
		if frames:
			max_frame = max(max_frame, max(frames.keys()))
			for dets in frames.values():
				if dets:
					max_cls = max(max_cls, max(dets.keys()))
	
	# Se non ci sono frame, ritorna lista vuota
	if max_frame < 0:
		return []

	# Inizializza struttura output: per ogni frame, una lista di liste (una per classe)
	n_classes = max_cls + 1 if max_cls >= 0 else 0
	frames_by_class: List[List[List[Tuple[int, float, float, float, float, float]]]] = []
	
	# Per ogni frame da 0 a max_frame
	for fi in range(max_frame + 1):
		# Crea lista vuota per ogni classe
		by_class: List[List[Tuple[int, float, float, float, float, float]]] = [[] for _ in range(n_classes)]
		
		# Per ogni camera, aggiungi le sue detections per questo frame
		for cam, frames in per_cam_frames.items():
			if fi not in frames:
				continue
			
			# Per ogni classe rilevata in questo frame da questa camera
			for cls_id, (x,y,w,h,conf) in frames[fi].items():
				# Espandi dinamicamente la lista se scopriamo classi con ID più alto
				if cls_id >= len(by_class):
					by_class.extend([] for _ in range(cls_id - len(by_class) + 1))
				
				# Aggiungi detection: (camera_id, x, y, w, h, confidence)
				by_class[cls_id].append((int(cam), float(x), float(y), float(w), float(h), float(conf)))
		
		frames_by_class.append(by_class)
	
	return frames_by_class


def save_points_json(points_per_frame: List[Dict[int, np.ndarray]], frame_indices: List[int], out_json: Path,
					frames_by_class: List[List[List[Tuple[int, float, float, float, float, float]]]] | None = None,
					tracks_by_cam: Dict[int, Dict[int, Dict[int, Dict]]] | None = None):
	"""
	Salva i punti 3D triangolati in formato JSON strutturato.
	
	Args:
		points_per_frame: Lista di dizionari {class_id: punto_3D} per ogni frame
		frame_indices: Indici dei frame corrispondenti
		out_json: Path del file JSON di output
		frames_by_class: (Opzionale) Struttura originale detections per aggiungere mapping track_id
		tracks_by_cam: (Opzionale) {cam: {frame: {class_id: {..., track_id}}}} per collegare 2D↔3D
	
	Output JSON:
		{
		  "units": "meters",
		  "frames": {
		    "0": [
		      {
		        "class_id": 1,
		        "x": 1.234,
		        "y": 2.345,
		        "z": 0.001,
		        "tracks": {"13": 5, "4": 3}  # opzionale: track_id per camera
		      }
		    ]
		  }
		}
	
	Note:
		- Le coordinate 3D sono in metri (scala applicata durante triangolazione)
		- Il campo "tracks" collega le detection 2D usate per questo punto 3D
		- Utile per visualizzare traiettorie 2D↔3D consistenti
	"""
	payload = {
		"units": "meters",
		"frames": {}
	}
	
	# Per ogni frame processato
	for fi, res in zip(frame_indices, points_per_frame):
		items = []
		
		# Per ogni classe triangolata in questo frame
		for cls_id, X in res.items():
			item = {
				"class_id": int(cls_id),
				"x": float(X[0]),
				"y": float(X[1]),
				"z": float(X[2]),
			}
			
			# Aggiungi mapping cam->track_id se disponibile (collegamento 2D↔3D)
			if frames_by_class is not None and tracks_by_cam is not None:
				cam_tracks = {}
				
				# frames_by_class[fi][cls_id] = lista di (cam, x, y, w, h, conf)
				if 0 <= fi < len(frames_by_class) and cls_id < len(frames_by_class[fi]):
					# Per ogni camera che ha contribuito a questo punto 3D
					for cam_entry in frames_by_class[fi][cls_id]:
						cam_id = int(cam_entry[0])
						tr_for_cam = tracks_by_cam.get(cam_id)
						if tr_for_cam is None:
							continue
						
						# Recupera track_id per questo frame e classe da questa camera
						fr_map = tr_for_cam.get(fi)
						if fr_map is None:
							continue
						tr = fr_map.get(int(cls_id))
						if tr and "track_id" in tr:
							cam_tracks[str(cam_id)] = int(tr["track_id"])
				
				# Aggiungi campo tracks solo se non vuoto
				if cam_tracks:
					item["tracks"] = cam_tracks
			
			items.append(item)
		
		# Salva lista punti per questo frame (chiave stringa per JSON)
		payload["frames"][str(fi)] = items
	
	# Scrivi file JSON
	out_json.parent.mkdir(parents=True, exist_ok=True)
	with out_json.open("w", encoding="utf-8") as f:
		json.dump(payload, f)
	print(f"[PIPELINE] Salvato JSON triangolato: {out_json}")


def _rectify_frames_in_memory(frames: PerFrame, K: np.ndarray, dist: np.ndarray) -> PerFrame:
	"""
	Applica correzione distorsioni ottiche ai centri delle bounding box.
	
	Args:
		frames: {frame_idx: {class_id: (x,y,w,h,conf)}} con coordinate in pixel distorte
		K: Matrice intrinseca camera 3x3
		dist: Coefficienti distorsione [k1, k2, p1, p2, k3, ...]
	
	Returns:
		frames_rettificati: Stessa struttura ma con (x,y) corretti, w,h invariati
	
	Note:
		- Usa cv2.undistortPoints con P=K per mantenere coordinate in pixel
		- Corregge solo il centro bbox (x,y), non ridimensiona width/height
		- Migliora accuratezza triangolazione rimuovendo distorsioni radiali/tangenziali
		- Consigliato prima della triangolazione per camere con forte distorsione
	"""
	rect: PerFrame = {}
	
	# Per ogni frame
	for fi, dets in frames.items():
		rect_dets: Dict[int, Det] = {}
		
		# Per ogni detection in questo frame
		for cls_id, (x, y, w, h, conf) in dets.items():
			# Prepara punto in formato OpenCV: array shape (1,1,2)
			pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
			
			# Applica undistort: P=K mantiene output in pixel coordinates
			pt_rect = cv2.undistortPoints(pt, K, dist, P=K)
			
			# Estrai coordinate rettificate
			rx, ry = float(pt_rect[0, 0, 0]), float(pt_rect[0, 0, 1])
			
			# Salva detection con centro rettificato, dimensioni invariate
			rect_dets[cls_id] = (rx, ry, float(w), float(h), float(conf))
		
		rect[fi] = rect_dets
	
	return rect


def filter_3d_movements(results_per_frame: List[Dict[int, np.ndarray]], idxs: List[int]) -> List[Dict[int, np.ndarray]]:
    """
    Filtra i punti 3D che si muovono troppo velocemente (rumore).
    Assumendo 25 fps (dt = 0.04s):
      - Umani: max 11 m/s (sprint elite) -> ~0.45 m/frame
      - Palla: max 30 m/s (passaggio forte) -> ~1.2 m/frame
    """
    # Riorganizza per classe per analizzare le traiettorie
    # class_id -> list of (frame_idx, point, original_list_index)
    trajectories = {}
    
    for i, (fi, res) in enumerate(zip(idxs, results_per_frame)):
        for cls_id, point in res.items():
            if cls_id not in trajectories:
                trajectories[cls_id] = []
            trajectories[cls_id].append((fi, point, i))
            
    points_removed = 0
    
    for cls_id, points in trajectories.items():
        # Ordina per frame
        points.sort(key=lambda x: x[0])
        
        # Soglie più strette
        threshold = 1.2 if cls_id == 0 else 0.5
        
        if not points:
            continue
            
        # Strategia: Manteniamo una finestra di punti validi
        # Se il punto corrente è troppo lontano dall'ultimo valido, lo scartiamo.
        # MA: Se scartiamo troppi punti consecutivi, forse era l'ultimo valido ad essere sbagliato?
        # Per semplicità, qui implementiamo il filtro forward semplice.
        
        last_valid_frame = points[0][0]
        last_valid_point = points[0][1]
        
        for k in range(1, len(points)):
            curr_frame, curr_point, list_idx = points[k]
            
            dt = curr_frame - last_valid_frame
            if dt <= 0: continue
            
            dist = np.linalg.norm(curr_point - last_valid_point)
            max_dist = threshold * dt
            
            # Se il gap temporale è enorme (es. > 2 secondi = 50 frame), 
            # accettiamo la nuova posizione come "nuovo inizio" per evitare di perdere traccia
            # se il giocatore ha attraversato il campo mentre non era visibile.
            if dt > 50:
                max_dist = float('inf')

            if dist <= max_dist:
                # Movimento valido
                last_valid_frame = curr_frame
                last_valid_point = curr_point
            else:
                # Movimento impossibile -> Rimuovi
                if cls_id in results_per_frame[list_idx]:
                    del results_per_frame[list_idx][cls_id]
                    points_removed += 1
                    
    print(f"[PIPELINE] Filtrati {points_removed} punti 3D per movimenti impossibili (soglie: Umani=0.5m/f, Palla=1.2m/f).")
    return results_per_frame


def run_pipeline(cameras: List[int], labels_dir: str, do_interpolate: bool, max_gap: int, anchor: str, min_cams: int,
				 do_track2d: bool = False, track_iou: float = 0.3, track_max_age: int = 15, tracks_out_dir: str = 'runs/tracks', save_tracks: bool = False,
				 rectify: bool = False, infer_videos: bool = False, video_dir: str = 'dataset/video', infer_images_dir: str | None = None, model_path: str = 'weights/fine_tuned_yolo_final.pt', device: str = 'auto',
				 conf_thres: float = 0.25, imgsz: int = 1920, evaluate_labels: bool = False, eval_gt_dir: str = 'dataset/val/labels', eval_out_dir: str = 'runs/eval'):
	"""
	Esegue la pipeline completa di ricostruzione 3D multi-camera.
	
	Args:
		cameras: Lista ID camere da processare (es. [2, 4, 13])
		labels_dir: Directory base con labels per camera (JSON o txt)
		do_interpolate: Se True, riempie gap temporali nelle detections
		max_gap: Numero massimo frame da interpolare (default 10)
		anchor: Punto bbox per triangolazione ('center' o 'bottom_center')
		min_cams: Minimo camere richieste per triangolare un punto (default 2)
		
		do_track2d: Se True, assegna track_id per seguire oggetti tra frame
		track_iou: Soglia IoU per matching tracce (default 0.3)
		track_max_age: Frame massimi senza match prima di chiudere traccia (default 15)
		tracks_out_dir: Directory output JSON tracce 2D (default 'runs/tracks')
		save_tracks: Se True, salva JSON tracce su disco (default False)
		
		rectify: Se True, corregge distorsioni ottiche prima di triangolare
		
		infer_videos: Se True, esegue inferenza YOLO su video prima di tutto
		video_dir: Directory con video input (default 'dataset/video')
		model_path: Path modello YOLO (default 'weights/fine_tuned_yolo_final.pt')
		device: Device per inferenza ('auto', 'cpu', '0', ecc.)
		
		evaluate_labels: Se True, valuta IOU RAW e INTERP contro GT
		eval_gt_dir: Directory GT per valutazione (default 'dataset/val/labels')
		eval_out_dir: Directory output metriche (default 'runs/eval')
	
	Flusso esecuzione:
		0. [Opzionale] Inferenza YOLO su video → labels JSON per camera
		1. Caricamento labels (preferenza JSON, fallback txt)
		   Per ogni camera:
		     a. [Opzionale] Tracking 2D → assegna track_id
		     b. [Opzionale] Salva tracce JSON su disco
		     c. [Opzionale] Valutazione RAW contro GT
		     d. [Opzionale] Interpolazione temporale
		     e. [Opzionale] Valutazione INTERP contro GT
		2. [Opzionale] Rettificazione distorsioni ottiche
		3. Riorganizzazione dati per triangolazione
		4. Caricamento calibrazioni camere
		5. Triangolazione 3D multi-camera (SVD)
		6. Salvataggio risultati (CSV + JSON)
	
	Output:
		- runs/triangulation/points.csv: Punti 3D in metri
		- runs/triangulation/points.json: Punti 3D con metadata
		- [Opzionale] runs/tracks/labels_out{cam}.json: Tracce 2D
		- [Opzionale] runs/eval/eval_cam{cam}_raw.json: Metriche pre-interpolazione
		- [Opzionale] runs/eval/eval_cam{cam}_interp.json: Metriche post-interpolazione
	"""
	
	# Smart default for eval_gt_dir if inferring on images
	if infer_images_dir and evaluate_labels and eval_gt_dir == 'dataset/val/labels':
		# Check if sibling 'labels' dir exists
		candidate_gt = Path(infer_images_dir).parent / 'labels'
		if candidate_gt.exists():
			print(f"[PIPELINE] Auto-detected GT dir: {candidate_gt} (overriding default dataset/val/labels)")
			eval_gt_dir = str(candidate_gt)

	# ========================================
	# 0) STEP OPZIONALE: INFERENZA VIDEO
	# ========================================
	# Detect video resolution for normalization (default 4K)
	vid_w, vid_h = 3840, 2160
	video_base = Path(video_dir)
	for cam in cameras:
		vp = video_base / f"out{cam}.mp4"
		if vp.exists():
			cap = cv2.VideoCapture(str(vp))
			if cap.isOpened():
				vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
				vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
				cap.release()
				print(f"[PIPELINE] Detected video resolution: {vid_w}x{vid_h}")
				break

	if infer_videos:
		# Esegue YOLO su video per generare labels JSON per camera
		labels_base = Path(labels_dir)
		detected_cams = []
		
		# Processa ogni video (es. out2.mp4, out4.mp4, out13.mp4)
		for cam in cameras:
			video_path = video_base / f"out{cam}.mp4"
			if not video_path.exists():
				print(f"[PIPELINE] WARN: Video non trovato: {video_path}. Skip.")
				continue
			
			# Inferenza YOLO frame-by-frame → salva labels_out{cam}.json
			cam_num = run_video_inference(video_path, model_path, labels_base, device=device, conf_thres=conf_thres, imgsz=imgsz)
			if cam_num is not None:
				detected_cams.append(cam_num)
		
		if detected_cams:
			print(f"[PIPELINE] Inferenza completata per camere: {detected_cams}")
		else:
			print("[PIPELINE] WARN: Nessun video processato.")
	elif infer_images_dir:
		# Esegue YOLO su directory di immagini
		print(f"[PIPELINE] Running inference on images in: {infer_images_dir}")
		labels_base = Path(labels_dir)
		detected_cams = run_images_inference(Path(infer_images_dir), model_path, labels_base, device=device, conf_thres=conf_thres)
		if detected_cams:
			print(f"[PIPELINE] Inferenza immagini completata per camere: {detected_cams}")
			
			# Try to detect resolution from first image if video resolution was not detected
			if vid_w == 3840 and vid_h == 2160: # Default values
				try:
					# Find a sample image
					img_dir = Path(infer_images_dir)
					sample_img = next(img_dir.glob("*.jpg"), None) or next(img_dir.glob("*.png"), None)
					if sample_img:
						img = cv2.imread(str(sample_img))
						if img is not None:
							vid_h, vid_w = img.shape[:2]
							print(f"[PIPELINE] Detected image resolution: {vid_w}x{vid_h}")
				except Exception as e:
					print(f"[PIPELINE] WARN: Could not detect image resolution: {e}")
		else:
			print("[PIPELINE] WARN: Nessuna immagine processata.")
	
	# ========================================
	# 1) CARICAMENTO LABELS E PROCESSING PER-CAMERA
	# ========================================
	per_cam_frames: Dict[int, PerFrame] = {}  # {cam: {frame: {cls: (x,y,w,h,conf)}}}
	per_cam_tracks: Dict[int, Dict[int, Dict[int, Dict]]] = {}  # {cam: {frame: {cls: {..., track_id}}}}
	labels_base = Path(labels_dir)
	
	for cam in cameras:
		# --- 1a) Caricamento labels da JSON o directory txt ---
		# Preferenza: labels_out{cam}.json (singolo file per camera)
		# Fallback: labels_out{cam}/ (directory con frame_XXXXX.txt)
		json_path = labels_base / f"labels_out{cam}.json"
		dir_path = labels_base / f"labels_out{cam}"
		
		if json_path.exists():
			try:
				frames, _ = load_labels_for_camera(json_path)
			except Exception as e:
				print(f"[PIPELINE] WARN: Errore caricamento JSON {json_path}: {e}. Skip camera {cam}.")
				continue
		elif dir_path.exists():
			try:
				frames, _ = load_labels_for_camera(dir_path)
			except Exception as e:
				print(f"[PIPELINE] WARN: Errore caricamento dir {dir_path}: {e}. Skip camera {cam}.")
				continue
		else:
			print(f"[PIPELINE] WARN: Labels non trovate per camera {cam} (cercato {json_path} e {dir_path}). Skip.")
			continue
		
		# --- 1b) OPZIONALE: Tracking 2D ---
		# Assegna track_id per seguire oggetti tra frame consecutivi
		if do_track2d:
			# Matching per IoU greedy, per classe
			tracks = track_labels_for_camera(frames, iou_thresh=track_iou, max_age=track_max_age)
			per_cam_tracks[cam] = tracks
			
			# Salva JSON tracce su disco solo se richiesto e non esiste già
			if save_tracks:
				out_tracks = Path(tracks_out_dir) / f"labels_out{cam}.json"
				# Sovrascrivi sempre per debugging
				save_tracks_json(tracks, out_tracks, camera_id=cam)
				
				# Genera video tracking
				try:
					from tracking_2d import save_tracked_video_from_images, save_tracked_video_from_video
					out_vid = Path(tracks_out_dir) / f"tracking_out{cam}.mp4"
					
					# Caso 1: Abbiamo una directory di immagini (es. dataset/val/images)
					if infer_images_dir:
						save_tracked_video_from_images(tracks, Path(infer_images_dir), out_vid, camera_id=cam)
					
					# Caso 2: Abbiamo un video sorgente (es. dataset/video/out13.mp4)
					elif video_dir:
						vid_path = Path(video_dir) / f"out{cam}.mp4"
						if vid_path.exists():
							save_tracked_video_from_video(tracks, vid_path, out_vid)
						else:
							print(f"[PIPELINE] Video sorgente non trovato per cam {cam}: {vid_path}")
							
				except ImportError:
					print("[PIPELINE] WARN: Funzioni video tracking non trovate in tracking_2d")
				except Exception as e:
					print(f"[PIPELINE] WARN: Errore generazione video tracking: {e}")
		
		# --- 1c) OPZIONALE: Valutazione RAW contro GT ---
		# Calcola IOU medio tra detections inferite e ground truth
		gt_map = None
		if evaluate_labels:
			try:
				from evaluation import load_labels_dir_map, frames_from_perframe, compute_iou_metrics_from_predictions, save_eval_json
			except Exception as e:
				print(f"[PIPELINE] WARN: impossibile importare funzioni di valutazione: {e}")
			else:
				gt_dir = Path(eval_gt_dir)
				if gt_dir.exists():
					# Carica GT filtrando per questa camera
					gt_map = load_labels_dir_map(gt_dir, camera_id=cam)
					
					# Se stiamo facendo inferenza su video completi, dobbiamo rimappare le chiavi GT
					# GT frame 1 -> Video frame 3 (User observed GT was 1 frame ahead)
					# GT frame k -> Video frame 3 + (k-1)*5
					if infer_videos and gt_map:
						print(f"[PIPELINE] Remapping GT frames for video alignment (offset 3, stride 5)")
						new_gt_map = {}
						for k, v in gt_map.items():
							new_k = 3 + (k - 1) * 5
							new_gt_map[new_k] = v
						gt_map = new_gt_map

					# Converte struttura interna in formato valutazione
					raw_map = frames_from_perframe(frames)
					
					# Normalizza coordinate se necessario (pixel -> [0,1])
					# Assumiamo che se x > 1 siano pixel
					needs_norm = False
					for fi in raw_map:
						if raw_map[fi]:
							if raw_map[fi][0][1] > 1.0: # check x coordinate
								needs_norm = True
							break
					
					if needs_norm and vid_w > 0 and vid_h > 0:
						for fi in raw_map:
							norm_list = []
							for item in raw_map[fi]:
								cls, x, y, bw, bh = item
								norm_list.append((cls, x/vid_w, y/vid_h, bw/vid_w, bh/vid_h))
							raw_map[fi] = norm_list

					# Calcola metriche IOU
					m_raw = compute_iou_metrics_from_predictions(raw_map, gt_map)
					
					# --- TEMPORARY DEBUG VISUALIZATION ---
					if infer_videos and video_dir:
						print(f"[DEBUG] Generating alignment visualization for Cam {cam}...")
						debug_dir = Path("runs/debug_alignment") / f"cam{cam}"
						debug_dir.mkdir(parents=True, exist_ok=True)
						
						vid_path = Path(video_dir) / f"out{cam}.mp4"
						cap = cv2.VideoCapture(str(vid_path))
						
						# Get common frames
						common_frames = sorted(list(set(gt_map.keys()) & set(raw_map.keys())))
						
						# Save first 10 common frames
						for fi in common_frames[:10]: 
							# fi is 1-based index from tracking_2d.py
							# cv2 uses 0-based index
							cap.set(cv2.CAP_PROP_POS_FRAMES, fi - 1)
							ret, frame = cap.read()
							if not ret:
								continue
								
							# Draw GT (Green)
							if fi in gt_map:
								for item in gt_map[fi]:
									cls, x, y, w, h = item
									h_img, w_img = frame.shape[:2]
									x1 = int((x - w/2) * w_img)
									y1 = int((y - h/2) * h_img)
									x2 = int((x + w/2) * w_img)
									y2 = int((y + h/2) * h_img)
									cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
									cv2.putText(frame, f"GT {cls}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

							# Draw Pred (Red)
							if fi in raw_map:
								for item in raw_map[fi]:
									cls, x, y, w, h = item
									h_img, w_img = frame.shape[:2]
									x1 = int((x - w/2) * w_img)
									y1 = int((y - h/2) * h_img)
									x2 = int((x + w/2) * w_img)
									y2 = int((y + h/2) * h_img)
									cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
									cv2.putText(frame, f"Pred {cls}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
							
							cv2.imwrite(str(debug_dir / f"frame_{fi}.jpg"), frame)
						cap.release()
						print(f"[DEBUG] Saved alignment images to {debug_dir}")
					# -------------------------------------

					# Salva JSON metriche
					out_dir_eval = Path(eval_out_dir); out_dir_eval.mkdir(parents=True, exist_ok=True)
					save_eval_json(m_raw, out_dir_eval / f"eval_cam{cam}_raw.json")
					print(f"[PIPELINE][EVAL] Cam {cam} RAW:")
					print(f"    Mean IOU (Matching): {m_raw['mean_iou']:.3f} ({m_raw['frames_evaluated']} frames)")
					print(f"    Mean IOU (GT):       {m_raw['mean_iou_gt']:.3f} ({m_raw['frames_evaluated_gt']} GT frames)")
					print(f"    Per-class IOU:")
					print(f"        Class | Matching (IoU, N) | GT (IoU, N)")
					for cls_id in sorted(m_raw['per_class'].keys()):
						stats_m = m_raw['per_class'][cls_id]
						stats_gt = m_raw['per_class_gt'][cls_id]
						iou_m = f"{stats_m['mean_iou']:.3f}" if stats_m['mean_iou'] is not None else "N/A"
						n_m = stats_m['count']
						iou_gt = f"{stats_gt['mean_iou']:.3f}" if stats_gt['mean_iou'] is not None else "N/A"
						n_gt = stats_gt['count']
						print(f"        {cls_id:5d} | {iou_m:>8} ({n_m:3d}) | {iou_gt:>8} ({n_gt:3d})")
				else:
					print(f"[PIPELINE] WARN: GT non trovata in {gt_dir}, skip valutazione RAW")
		
		# --- 1d) OPZIONALE: Interpolazione temporale ---
		# Riempie gap nelle detections copiando ultimo valore valido
		if do_interpolate:
			frames = interpolate_missing_detections(frames, max_frame=max(frames.keys()) if frames else -1, max_gap=max_gap)
			
			# --- 1e) OPZIONALE: Valutazione INTERP contro GT ---
			# Ricalcola IOU dopo interpolazione per misurare impatto
			if evaluate_labels and gt_map:
				try:
					from evaluation import frames_from_perframe, compute_iou_metrics_from_predictions, save_eval_json
					
					inter_map = frames_from_perframe(frames)
					
					# Normalizza coordinate se necessario
					needs_norm = False
					for fi in inter_map:
						if inter_map[fi]:
							if inter_map[fi][0][1] > 1.0:
								needs_norm = True
							break
					
					if needs_norm and vid_w > 0 and vid_h > 0:
						for fi in inter_map:
							norm_list = []
							for item in inter_map[fi]:
								cls, x, y, bw, bh = item
								norm_list.append((cls, x/vid_w, y/vid_h, bw/vid_w, bh/vid_h))
							inter_map[fi] = norm_list

					m_int = compute_iou_metrics_from_predictions(inter_map, gt_map)
					out_dir_eval = Path(eval_out_dir); out_dir_eval.mkdir(parents=True, exist_ok=True)
					save_eval_json(m_int, out_dir_eval / f"eval_cam{cam}_interp.json")
					print(f"[PIPELINE][EVAL] Cam {cam} INTERP:")
					print(f"    Mean IOU (Matching): {m_int['mean_iou']:.3f} ({m_int['frames_evaluated']} frames)")
					print(f"    Mean IOU (GT):       {m_int['mean_iou_gt']:.3f} ({m_int['frames_evaluated_gt']} GT frames)")
					print(f"    Per-class IOU:")
					print(f"        Class | Matching (IoU, N) | GT (IoU, N)")
					for cls_id in sorted(m_int['per_class'].keys()):
						stats_m = m_int['per_class'][cls_id]
						stats_gt = m_int['per_class_gt'][cls_id]
						iou_m = f"{stats_m['mean_iou']:.3f}" if stats_m['mean_iou'] is not None else "N/A"
						n_m = stats_m['count']
						iou_gt = f"{stats_gt['mean_iou']:.3f}" if stats_gt['mean_iou'] is not None else "N/A"
						n_gt = stats_gt['count']
						print(f"        {cls_id:5d} | {iou_m:>8} ({n_m:3d}) | {iou_gt:>8} ({n_gt:3d})")
				except Exception as e:
					print(f"[PIPELINE] WARN: Errore valutazione INTERP: {e}")
		
		# Salva frames processati per questa camera
		per_cam_frames[cam] = frames

	# ========================================
	# 2) STEP OPZIONALE: RETTIFICAZIONE DISTORSIONI
	# ========================================
	# Corregge distorsioni ottiche delle camere prima di triangolare
	if rectify and per_cam_frames:
		# Carica calibrazioni per le camere processate
		calibs_for_rect = tri.load_calibrations_for_cams(list(per_cam_frames.keys()))
		
		for cam in list(per_cam_frames.keys()):
			params = calibs_for_rect.get(cam, {})
			K = params.get("K")  # Matrice intrinseca
			dist = params.get("dist")  # Coefficienti distorsione
			
			if K is None or dist is None:
				print(f"[PIPELINE] WARN: Calibrazione mancante per cam {cam}, skip rettifica")
				continue
			
			# Applica undistort ai centri bbox
			per_cam_frames[cam] = _rectify_frames_in_memory(per_cam_frames[cam], K, dist)

	# ========================================
	# 3) PREPARAZIONE DATI PER TRIANGOLAZIONE
	# ========================================
	# Riorganizza da struttura per-camera a struttura per-frame-per-classe
	frames_by_class = build_frames_by_class_from_cameras(per_cam_frames)
	if not frames_by_class:
		print("[PIPELINE] Nessuna label caricata. Esco.")
		return
	
	# Genera lista indici frame da processare
	idxs = list(range(len(frames_by_class)))

	# ========================================
	# 4) CARICAMENTO CALIBRAZIONI CAMERE
	# ========================================
	# Legge parametri intrinseci ed estrinseci da JSON
	calibs = tri.load_calibrations_for_cams(cameras)

	# ========================================
	# 5) TRIANGOLAZIONE 3D MULTI-CAMERA
	# ========================================
	# Ricostruzione 3D lineare (SVD) da proiezioni 2D multi-camera
	# Applica scala 0.001 per conversione millimetri→metri
	results_per_frame = tri.triangulate_frames_list(frames_by_class, idxs, calibs, anchor=anchor, scale=0.001, min_cams=min_cams)

	# ========================================
	# 5b) FILTRAGGIO MOVIMENTI IMPOSSIBILI
	# ========================================
	# Rimuove punti che si spostano troppo velocemente (rumore)
	results_per_frame = filter_3d_movements(results_per_frame, idxs)

	# ========================================
	# 6) SALVATAGGIO RISULTATI
	# ========================================
	out_dir = Path("runs") / "triangulation"
	out_dir.mkdir(parents=True, exist_ok=True)
	
	# --- 6a) Salva CSV compatto ---
	import csv
	out_csv = out_dir / "points.csv"
	total_points = 0
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["frame", "class_id", "x_m", "y_m", "z_m"])
		for fi, res in zip(idxs, results_per_frame):
			for cls_id, X in res.items():
				writer.writerow([fi, cls_id, float(X[0]), float(X[1]), float(X[2])])
				total_points += 1
	print(f"[PIPELINE] Salvati {total_points} punti in {out_csv}")
	
	# --- 6b) Salva JSON strutturato con metadata ---
	save_points_json(
		results_per_frame,
		idxs,
		out_dir / "points.json",
		frames_by_class=frames_by_class,
		tracks_by_cam=per_cam_tracks if do_track2d else None,
	)

	# Nota: valutazione già eseguita per-camera durante il loop (step 1c e 1e)


def main():
	"""
	Entry point principale: parse argomenti, esegue pipeline, visualizza risultati.
	
	Flusso:
		1. Parse argomenti da command-line (usa config.get_args())
		2. Esegue run_pipeline con parametri configurati
		3. Se richiesto (--visualize), avvia visualizzatore 3D interattivo
	"""
	args = get_args()
	
	# Esegue pipeline completa
	run_pipeline(
		args.cameras, args.labels_dir, args.interpolate, args.max_gap, args.anchor, args.min_cams,
		do_track2d=args.track2d, track_iou=args.track_iou, track_max_age=args.track_max_age, 
		tracks_out_dir=args.tracks_out_dir, save_tracks=args.save_tracks,
		rectify=args.rectify, infer_videos=args.infer_videos, video_dir=args.video_dir, infer_images_dir=args.infer_images_dir,
		model_path=args.model, device=args.device, conf_thres=args.conf_thres, imgsz=args.imgsz,
		evaluate_labels=args.evaluate_labels, eval_gt_dir=args.eval_gt_dir, eval_out_dir=args.eval_out_dir
	)
	
	# Visualizzazione 3D interattiva (lazy import per evitare dipendenze matplotlib in run headless)
	if args.visualize:
		from visualizer_3d import visualize_triangulated_points
		visualize_triangulated_points()


if __name__ == '__main__':
	main()

# ----------------------------------------
# How to run (PowerShell)
# ----------------------------------------
# FLUSSO COMPLETO: Inferenza video -> Interpolazione -> Rettificazione -> Triangolazione -> Valutazione -> Visualizzazione:
#   python pipeline.py --infer-videos --interpolate --rectify --evaluate-labels --visualize --device 0
#
# Con modello custom e confidence threshold per catturare più Ball detection (sensibilità maggiore):
#   python pipeline.py --infer-videos --model weights/fine_tuned_yolo2k.pt --conf-thres 0.15 --interpolate --rectify --evaluate-labels --device 0
#
# Con confidence threshold alta per ridurre falsi positivi:
#   python pipeline.py --infer-videos --conf-thres 0.4 --interpolate --rectify --evaluate-labels --device 0
#
# Con tracking 2D (opzionale, disattivato di default):
#   python pipeline.py --infer-videos --track2d --save-tracks --interpolate --rectify --evaluate-labels --visualize --device 0
#
# Se hai già le labels e vuoi solo triangolare e valutare:
#   python pipeline.py --labels-dir dataset/infer_video --interpolate --rectify --evaluate-labels --visualize
#
# Solo triangolazione (labels già pronte, no tracking/interpolazione/valutazione):
#   python pipeline.py --labels-dir dataset/infer_video --visualize
#
# Valutazione personalizzata con GT in altra directory:
#   python pipeline.py --labels-dir dataset/infer_video --interpolate --evaluate-labels --eval-gt-dir path/to/gt/labels
