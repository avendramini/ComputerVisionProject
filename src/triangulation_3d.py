from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import cv2

def load_camera_calibration_cv(
	cam_number: int,
	base_dir: Path = Path("camparams"),
) -> Dict[str, np.ndarray]:
	"""Loads camera calibration using the same format as files in camparams.

	Expects a JSON file: out{cam}_camera_calib.json with keys: mtx, dist, rvecs, tvecs

	Returns a dictionary with:
	  - K: (3x3) intrinsic matrix
	  - dist: (N,) distortion coefficients (flattened)
	  - rvec: (3, 1) rotation vector
	  - tvec: (3, 1) translation vector
	  - P: (3x4) projection matrix K [R|t]
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
	"""Loads calibrations for a list of cameras and returns a dictionary {cam: params}.

	Each entry contains K, dist, rvec, tvec, and P.
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
	"""Triangulates a 3D point from multi-camera 2D observations using SVD.

	Parameters:
	  - observations: list of tuples (cam_id, u, v) with u,v in pixels
	  - calibrations: dict {cam_id: {"P": (3x4), ...}}

	Returns: np.ndarray shape (3,) with the 3D point; np.nan if triangulation fails.
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
	"""Triangulates the 3D centroid of an object given the array of bboxes of the same instance.

	- boxes: list of (cam, x, y, w, h, conf) extracted from frames[frame][class]
			 x,y are the YOLO center in pixels; w,h are width/height in pixels
	- calibrations: dict {cam: params} with at least the projection matrix P (3x4)
	- anchor:
		* "center" (default): uses the bbox center (x, y)
		* "bottom_center": uses (x, y + h/2) useful for objects on the ground

	Returns: 3D point (x,y,z) as np.ndarray shape (3,), np.nan if not triangulatable.
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

	# Remove any camera duplicates keeping the first observation
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
	"""Triangulates all classes present in a frame.

	- Uses all available cameras for that class filtering to one bbox per camera (max conf)
	- If at least 'min_cams' cameras are available (and calibrated), triangulates and returns the 3D point
	- If < min_cams, the class is not included in the output

	Returns: dict {cls_id: np.ndarray shape (3,)} in meters
	"""
	results: Dict[int, np.ndarray] = {}
	if frame_idx < 0 or frame_idx >= len(frames_by_class):
		return results

	by_class = frames_by_class[frame_idx]
	for cls_id, dets in enumerate(by_class):
		if not dets:
			continue
		# Filter to cameras with calibration
		dets_cal = [d for d in dets if int(d[0]) in calibrations]
		if len(dets_cal) < min_cams:
			continue
		# Select the first one for each camera (assumes uniqueness per cam)
		seen = set()
		selected: List[Tuple[int, float, float, float, float, float]] = []
		for d in dets_cal:
			cam_id = int(d[0])
			if cam_id in seen:
				continue
			seen.add(cam_id)
			selected.append(d)
		# Check how many cameras remain
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
	"""Triangulates a set of frames and returns a vector of results.

	The output is a list of dicts, in the same order as 'frame_indices',
	where each element is {cls_id: 3Dpoint} for that frame.
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

