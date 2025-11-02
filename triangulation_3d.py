from pathlib import Path
from typing import Dict, List, Tuple
import os
import json
import random
import csv
from datetime import datetime

import numpy as np
import cv2

def load_yolo_pairs_by_frame(
	cam_numbers: List[int],
	base_dir: Path = Path("rectified"),
	image_width: int = 3840,
	image_height: int = 2160,
) -> List[List[List[Tuple[int, float, float, float, float, float]]]]:
	"""Legge le label YOLOv8 da rectified/labels_out{numero_cam} e costruisce
	una struttura indicizzata per frame e per classe.

	Output: frames[frame_idx][cls_id] -> list of (cam, x, y, w, h, conf)
	- Supporta sia coordinate in pixel (come nell'esempio) sia normalizzate [0,1].
	- Se una cartella/camera ha meno frame, i frame mancanti vengono ignorati.
	- Il numero di classi è determinato dinamicamente dal massimo cls_id trovato.
	"""
	# Raccogli tutti i frame disponibili per ogni camera
	per_cam_files: Dict[int, List[Path]] = {}
	max_frame = -1
	for cam in cam_numbers:
		label_dir = base_dir / f"labels_out{cam}"
		if not label_dir.exists():
			continue
		files = sorted(
			label_dir.glob("frame_*.txt"),
			key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem else -1,
		)
		per_cam_files[cam] = files
		if files:
			max_frame = max(max_frame, max(int(p.stem.split("_")[1]) for p in files))

	if max_frame < 0:
		return []

	# Mappa temporanea frame -> lista di (cls_id, (cam, x, y, w, h, conf))
	frames_map: Dict[int, List[Tuple[int, Tuple[int, float, float, float, float, float]]]] = {}
	max_cls = -1

	for cam, files in per_cam_files.items():
		for label_path in files:
			try:
				frame_idx = int(label_path.stem.split("_")[1])
			except Exception:
				continue
			with label_path.open("r", encoding="utf-8") as handle:
				for line in handle:
					parts = line.strip().split()
					if len(parts) < 5:
						continue
					try:
						cls_id = int(float(parts[0]))
						x, y, w, h = map(float, parts[1:5])
						conf = float(parts[5]) if len(parts) > 5 else 1.0
					except ValueError:
						continue
					# Se i valori sembrano normalizzati, converti in pixel
					if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
						x *= image_width
						y *= image_height
						w *= image_width
						h *= image_height
					frames_map.setdefault(frame_idx, []).append((cls_id, (cam, x, y, w, h, conf)))
					if cls_id > max_cls:
						max_cls = cls_id

	# Costruisci lista compatta: per ogni frame una lista per classe (fuori dal loop delle camere)
	n_classes = max_cls + 1 if max_cls >= 0 else 0
	frames_list: List[List[List[Tuple[int, float, float, float, float, float]]]] = []
	for idx in range(max_frame + 1):
		by_class: List[List[Tuple[int, float, float, float, float, float]]] = [
			[] for _ in range(n_classes)
		]
		for cls_id, det in frames_map.get(idx, []):
			cam, x, y, w, h, conf = det
			# Estendi dinamicamente nel caso compaiano classi più alte in questo frame
			if cls_id >= len(by_class):
				by_class.extend([] for _ in range(cls_id - len(by_class) + 1))
			by_class[cls_id].append((cam, x, y, w, h, conf))
		frames_list.append(by_class)
	return frames_list

def load_camera_calibration_cv(
	cam_number: int,
	base_dir: Path = Path("camparams"),
) -> Dict[str, np.ndarray]:
	"""Carica la calibrazione di una camera usando lo stesso formato dei file in camparams.

	Atteso un file JSON: out{cam}_camera_calib.json con chiavi: mtx, dist, rvecs, tvecs

	Ritorna un dizionario con:
	  - K: (3x3) intrinseca
	  - dist: (N,) coeff. distorsione (appiattiti)
	  - rvec: (3, 1)
	  - tvec: (3, 1)
	  - P: (3x4) matrice di proiezione K [R|t]
	"""
	calib_path = base_dir / f"out{cam_number}_camera_calib.json"
	if not calib_path.exists():
		raise FileNotFoundError(f"Calibration JSON not found: {calib_path}")

	with calib_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	required = ("mtx", "dist", "rvecs", "tvecs")
	for key in required:
		if key not in data:
			raise KeyError(f"Key '{key}' missing in calibration file {calib_path}")

	K = np.asarray(data["mtx"], dtype=np.float32)
	dist = np.asarray(data["dist"], dtype=np.float32).reshape(-1)
	rvec = np.asarray(data["rvecs"], dtype=np.float32).reshape(3, 1)
	tvec = np.asarray(data["tvecs"], dtype=np.float32).reshape(3, 1)

	R, _ = cv2.Rodrigues(rvec)
	Rt = np.hstack((R.astype(np.float32), tvec.astype(np.float32)))
	P = (K @ Rt).astype(np.float32)

	return {"K": K, "dist": dist, "rvec": rvec, "tvec": tvec, "P": P}

def load_calibrations_for_cams(
	cam_numbers: List[int],
	base_dir: Path = Path("camparams"),
) -> Dict[int, Dict[str, np.ndarray]]:
	"""Carica le calibrazioni per una lista di camere e ritorna un dizionario {cam: params}.

	Ogni voce contiene K, dist, rvec, tvec e P.
	"""
	result: Dict[int, Dict[str, np.ndarray]] = {}
	for cam in cam_numbers:
		result[cam] = load_camera_calibration_cv(cam, base_dir)
	return result

def triangulate_point_svd(
	observations: List[Tuple[int, float, float]],
	calibrations: Dict[int, Dict[str, np.ndarray]],
	scale: float = 1.0,
) -> np.ndarray:
	"""Triangola un punto 3D a partire da osservazioni 2D multi-camera usando SVD.

	Parametri:
	  - observations: lista di tuple (cam_id, u, v) con u,v in pixel
	  - calibrations: dict {cam_id: {"P": (3x4), ...}}

	Ritorna: np.ndarray shape (3,) con il punto 3D; np.nan se triangolazione fallisce.
	"""
	if len(observations) < 2:
		return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

	rows = []
	for cam_id, u, v in observations:
		if cam_id not in calibrations or "P" not in calibrations[cam_id]:
			return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
		P = calibrations[cam_id]["P"].astype(np.float64)  # 3x4
		rows.append(u * P[2, :] - P[0, :])
		rows.append(v * P[2, :] - P[1, :])

	A = np.stack(rows, axis=0)
	_, _, vh = np.linalg.svd(A)
	Xh = vh[-1]
	if np.isclose(Xh[3], 0.0):
		return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
	X = Xh[:3] / Xh[3]
	if scale != 1.0:
		X = X * float(scale)
	return X.astype(np.float64)

def triangulate_centroid_from_boxes(
	boxes: List[Tuple[int, float, float, float, float, float]],
	calibrations: Dict[int, Dict[str, np.ndarray]],
	anchor: str = "center",
	scale: float = 0.001,
) -> np.ndarray:
	"""Triangola il baricentro 3D di un oggetto dato l'array di bbox della stessa istanza.

	- boxes: lista di (cam, x, y, w, h, conf) estratti da frames[frame][class]
			 x,y sono il centro YOLO in pixel; w,h sono larghezza/altezza in pixel
	- calibrations: dict {cam: params} con almeno la matrice di proiezione P (3x4)
	- anchor:
		* "center" (default): usa il centro della bbox (x, y)
		* "bottom_center": usa (x, y + h/2) utile per oggetti a terra

	Ritorna: punto 3D (x,y,z) come np.ndarray shape (3,), np.nan se non triangolabile.
	"""
	if not boxes:
		return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

	observations: List[Tuple[int, float, float]] = []
	for cam, x, y, w, h, _conf in boxes:
		if anchor == "center":
			u, v = float(x), float(y)
		elif anchor == "bottom_center":
			u, v = float(x), float(y + 0.5 * h)
		else:
			# default fallback
			u, v = float(x), float(y)
		observations.append((int(cam), u, v))

	# Rimuovi eventuali duplicati di camera tenendo la prima osservazione
	seen = set()
	uniq_obs: List[Tuple[int, float, float]] = []
	for cam_id, u, v in observations:
		if cam_id in seen:
			continue
		seen.add(cam_id)
		uniq_obs.append((cam_id, u, v))

	return triangulate_point_svd(uniq_obs, calibrations, scale=scale)

def triangulate_all_classes_in_frame(
	frames_by_class: List[List[List[Tuple[int, float, float, float, float, float]]]],
	frame_idx: int,
	calibrations: Dict[int, Dict[str, np.ndarray]],
	anchor: str = "center",
	scale: float = 0.001,
	min_cams: int = 2,
) -> Dict[int, np.ndarray]:
	"""Triangola tutte le classi presenti in un frame.

	- Usa tutte le camere disponibili per quella classe filtrando a una bbox per camera (conf massima)
	- Se almeno 'min_cams' camere sono disponibili (e calibrate), triangola e restituisce il 3D
	- Se < min_cams, la classe non viene inclusa nell'output

	Ritorna: dict {cls_id: np.ndarray shape (3,)} in metri
	"""
	results: Dict[int, np.ndarray] = {}
	if frame_idx < 0 or frame_idx >= len(frames_by_class):
		return results

	by_class = frames_by_class[frame_idx]
	for cls_id, dets in enumerate(by_class):
		if not dets:
			continue
		# Filtra a camere con calibrazione
		dets_cal = [d for d in dets if int(d[0]) in calibrations]
		if len(dets_cal) < min_cams:
			continue
		# Seleziona la prima per ciascuna camera (si assume unicità per cam)
		seen = set()
		selected: List[Tuple[int, float, float, float, float, float]] = []
		for d in dets_cal:
			cam_id = int(d[0])
			if cam_id in seen:
				continue
			seen.add(cam_id)
			selected.append(d)
		# Verifica quante camere rimangono
		cams_used = {int(d[0]) for d in selected}
		if len(cams_used) < min_cams:
			continue
		X = triangulate_centroid_from_boxes(selected, calibrations, anchor=anchor, scale=scale)
		if not np.any(np.isnan(X)):
			results[cls_id] = X
	return results

def triangulate_frames_list(
	frames_by_class: List[List[List[Tuple[int, float, float, float, float, float]]]],
	frame_indices: List[int],
	calibrations: Dict[int, Dict[str, np.ndarray]],
	anchor: str = "center",
	scale: float = 0.001,
	min_cams: int = 2,
) -> List[Dict[int, np.ndarray]]:
	"""Triangola un insieme di frame e ritorna un vettore di risultati.

	L'output è una lista di dict, nello stesso ordine di 'frame_indices',
	dove ogni elemento è {cls_id: punto3D} per quel frame.
	"""
	results: List[Dict[int, np.ndarray]] = []
	for fi in frame_indices:
		res = triangulate_all_classes_in_frame(
			frames_by_class,
			fi,
			calibrations,
			anchor=anchor,
			scale=scale,
			min_cams=min_cams,
		)
		results.append(res)
	return results

if __name__ == "__main__":
	# Esegue triangolazione per tutto il video, tutte le classi, usando tutte le camere disponibili
	cams = [13, 2, 4]
	frames = load_yolo_pairs_by_frame(cams)
	n_frames = len(frames)
	print(f"Frames letti: {n_frames}")

	if n_frames == 0:
		print("Nessun frame trovato, esco.")
		exit(0)

	# Carica le calibrazioni (si assume che i file esistano per queste cam)
	calibs = load_calibrations_for_cams(cams)

	# Triangola tutti i frame e tutte le classi con almeno 2 camere
	idxs = list(range(n_frames))
	results_per_frame = triangulate_frames_list(
		frames_by_class=frames,
		frame_indices=idxs,
		calibrations=calibs,
		anchor="center",
		scale=0.001,
		min_cams=2,
	)

	# Prepara output directory e file
	out_dir = Path("runs") / "triangulation"
	out_dir.mkdir(parents=True, exist_ok=True)
	out_csv = out_dir / "points.csv"
	out_json = out_dir / "points.json"

	# Salva CSV con colonne: frame,class_id,x,y,z
	total_points = 0
	json_frames: Dict[int, List[Dict[str, float]]] = {}
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["frame", "class_id", "x_m", "y_m", "z_m"])
		for fi, res in zip(idxs, results_per_frame):
			for cls_id, X in res.items():
				writer.writerow([fi, cls_id, float(X[0]), float(X[1]), float(X[2])])
				total_points += 1
				json_frames.setdefault(fi, []).append({
					"class_id": int(cls_id),
					"x": float(X[0]),
					"y": float(X[1]),
					"z": float(X[2])
				})

	# Save JSON aggregate
	payload = {
		"generated_at": datetime.utcnow().isoformat() + "Z",
		"units": "meters",
		"frames": {str(k): v for k, v in json_frames.items()}
	}
	with out_json.open("w", encoding="utf-8") as jf:
		json.dump(payload, jf)

	print(f"Salvati {total_points} punti triangolati in {out_csv} e {out_json}")

# ----------------------------------------
# How to run (PowerShell)
# ----------------------------------------
# Triangola tutti i frame usando le label YOLO per le camere [13,2,4] sotto 'rectified/labels_outX':
#   python triangulation_3d.py
# Note:
# - Richiede i file di calibrazione: camparams/out{cam}_camera_calib.json
# - Preferisci usare la pipeline unificata se lavori con dataset/infer_video:
#   python pipeline.py --interpolate --labels-dir dataset/infer_video --visualize

