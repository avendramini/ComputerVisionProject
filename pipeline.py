"""
Unified pipeline for multi-camera 3D reconstruction.

Full workflow:
	1. (Optional) YOLO inference on video to generate 2D detections
	2. Load labels for each camera (JSON or txt)
	3. (Optional) 2D tracking to assign track_id between consecutive frames
	4. (Optional) RAW evaluation against ground truth
	5. (Optional) Interpolation to fill temporal gaps
	6. (Optional) INTERP evaluation against ground truth
	7. (Optional) Optical distortion rectification
	8. Multi-camera 3D triangulation (linear SVD)
	9. Save results (CSV + JSON)
	10. (Optional) Interactive 3D visualization

Minimizes disk I/O by operating in memory where possible.
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
from interpoler import load_labels_for_camera, interpolate_missing_detections
from tracking_2d import save_tracks_json, run_video_inference, run_images_inference
from config import get_args


# Type aliases for common data structures
Det = Tuple[float, float, float, float, float]  # Detection: (x, y, width, height, confidence)
PerFrame = Dict[int, Dict[int, Det]]  # Frame structure: {frame_idx: {class_id: Detection}}


def build_frames_by_class_from_cameras(per_cam_frames: Dict[int, PerFrame]) -> List[List[List[Tuple[int, float, float, float, float, float]]]]:
	"""
	Reorganizes detections from per-camera structure to per-frame-per-class structure for triangulation.

	Args:
		per_cam_frames: {camera_id: {frame_idx: {class_id: (x,y,w,h,conf)}}}

	Returns:
		frames_by_class[frame_idx][class_id] = [(cam, x, y, w, h, conf), ...]
		List of frames, each frame contains a list per class, each class has a list of multi-camera detections.
		Coordinates are in pixels.

	Note:
		- Dynamically determines the maximum number of frames and classes
		- Automatically expands the class list if higher IDs are found
		- Useful for preparing input for multi-camera triangulation
	"""
	# Find the maximum number of frames and classes present across all cameras
	max_frame = -1
	max_cls = -1
	for _cam, frames in per_cam_frames.items():
		if frames:
			max_frame = max(max_frame, max(frames.keys()))
			for dets in frames.values():
				if dets:
					max_cls = max(max_cls, max(dets.keys()))
	
	# If there are no frames, return an empty list
	if max_frame < 0:
		return []

	# Initialize output structure: for each frame, a list of lists (one per class)
	n_classes = max_cls + 1 if max_cls >= 0 else 0
	frames_by_class: List[List[List[Tuple[int, float, float, float, float, float]]]] = []
	
	# For each frame from 0 to max_frame
	for fi in range(max_frame + 1):
		# Create an empty list for each class
		by_class: List[List[Tuple[int, float, float, float, float, float]]] = [[] for _ in range(n_classes)]
		
		# For each camera, add its detections for this frame
		for cam, frames in per_cam_frames.items():
			if fi not in frames:
				continue
			
			# For each class detected in this frame by this camera
			for cls_id, (x,y,w,h,conf) in frames[fi].items():
				# Dynamically expand the list if we discover higher class IDs
				if cls_id >= len(by_class):
					by_class.extend([] for _ in range(cls_id - len(by_class) + 1))
				
				# Add detection: (camera_id, x, y, w, h, confidence)
				by_class[cls_id].append((int(cam), float(x), float(y), float(w), float(h), float(conf)))
		
		frames_by_class.append(by_class)
	
	return frames_by_class


def save_points_json(points_per_frame: List[Dict[int, np.ndarray]], frame_indices: List[int], out_json: Path,
					frames_by_class: List[List[List[Tuple[int, float, float, float, float, float]]]] | None = None,
					tracks_by_cam: Dict[int, Dict[int, Dict[int, Dict]]] | None = None):
	"""
	Saves triangulated 3D points in structured JSON format.
	
	Args:
		points_per_frame: List of dictionaries {class_id: 3D_point} for each frame
		frame_indices: Corresponding frame indices
		out_json: Output JSON file path
		frames_by_class: (Optional) Original detections structure to add track_id mapping
		tracks_by_cam: (Optional) {cam: {frame: {class_id: {..., track_id}}}} to link 2D↔3D
	
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
		        "tracks": {"13": 5, "4": 3}  # optional: track_id per camera
		      }
		    ]
		  }
		}
	
	Notes:
		- 3D coordinates are in meters (scale applied during triangulation)
		- The "tracks" field links the 2D detections used for this 3D point
		- Useful for visualizing consistent 2D↔3D trajectories
	"""
	payload = {
		"units": "meters",
		"frames": {}
	}
	
	# For each processed frame
	for fi, res in zip(frame_indices, points_per_frame):
		items = []
		
		# For each triangulated class in this frame
		for cls_id, X in res.items():
			item = {
				"class_id": int(cls_id),
				"x": float(X[0]),
				"y": float(X[1]),
				"z": float(X[2]),
			}
			
			# Add mapping cam->track_id if available (2D↔3D linkage)
			if frames_by_class is not None and tracks_by_cam is not None:
				cam_tracks = {}
				
				# frames_by_class[fi][cls_id] = list of (cam, x, y, w, h, conf)
				if 0 <= fi < len(frames_by_class) and cls_id < len(frames_by_class[fi]):
					# For each camera that contributed to this 3D point
					for cam_entry in frames_by_class[fi][cls_id]:
						cam_id = int(cam_entry[0])
						tr_for_cam = tracks_by_cam.get(cam_id)
						if tr_for_cam is None:
							continue
						
						# Retrieve track_id for this frame and class from this camera
						fr_map = tr_for_cam.get(fi)
						if fr_map is None:
							continue
						tr = fr_map.get(int(cls_id))
						if tr and "track_id" in tr:
							cam_tracks[str(cam_id)] = int(tr["track_id"])
				
				# Add 'tracks' field only if not empty
				if cam_tracks:
					item["tracks"] = cam_tracks
			
			items.append(item)
		
		# Save list of points for this frame (string key for JSON)
		payload["frames"][str(fi)] = items
	
	# Write JSON file
	out_json.parent.mkdir(parents=True, exist_ok=True)
	with out_json.open("w", encoding="utf-8") as f:
		json.dump(payload, f)
	print(f"[PIPELINE] Saved triangulated JSON: {out_json}")


def _rectify_frames_in_memory(frames: PerFrame, K: np.ndarray, dist: np.ndarray) -> PerFrame:
	"""
	Applies optical distortion correction to bounding box centers.
	
	Args:
		frames: {frame_idx: {class_id: (x,y,w,h,conf)}} with distorted pixel coordinates
		K: Intrinsic camera matrix 3x3
		dist: Distortion coefficients [k1, k2, p1, p2, k3, ...]
	
	Returns:
		rectified_frames: Same structure but with corrected (x,y), w,h unchanged
	
	Notes:
		- Uses cv2.undistortPoints with P=K to keep output in pixel coordinates
		- Corrects only the bbox center (x,y), does not resize width/height
		- Improves triangulation accuracy by removing radial/tangential distortions
		- Recommended before triangulation for cameras with strong distortion
	"""
	rect: PerFrame = {}
	
	# For each frame
	for fi, dets in frames.items():
		rect_dets: Dict[int, Det] = {}
		
		# For each detection in this frame
		for cls_id, (x, y, w, h, conf) in dets.items():
			# Prepare point in OpenCV format: array shape (1,1,2)
			pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
			
			# Apply undistort: P=K keeps output in pixel coordinates
			pt_rect = cv2.undistortPoints(pt, K, dist, P=K)
			
			# Extract rectified coordinates
			rx, ry = float(pt_rect[0, 0, 0]), float(pt_rect[0, 0, 1])
			
			# Save detection with rectified center, unchanged dimensions
			rect_dets[cls_id] = (rx, ry, float(w), float(h), float(conf))
		
		rect[fi] = rect_dets
	
	return rect


def filter_3d_movements(results_per_frame: List[Dict[int, np.ndarray]], idxs: List[int]) -> List[Dict[int, np.ndarray]]:
    """
    Filters 3D points that move too fast (noise).
    Assuming 25 fps (dt = 0.04s):
      - Humans: max 11 m/s (elite sprint) -> ~0.45 m/frame
      - Ball: max 30 m/s (strong pass) -> ~1.2 m/frame
    """
    # Reorganize by class to analyze trajectories
    # class_id -> list of (frame_idx, point, original_list_index)
    trajectories = {}
    
    for i, (fi, res) in enumerate(zip(idxs, results_per_frame)):
        for cls_id, point in res.items():
            if cls_id not in trajectories:
                trajectories[cls_id] = []
            trajectories[cls_id].append((fi, point, i))
            
    points_removed = 0
    
    for cls_id, points in trajectories.items():
        # Sort by frame
        points.sort(key=lambda x: x[0])
        
        # Tighter thresholds
        threshold = 1.2 if cls_id == 0 else 0.5
        
        if not points:
            continue
            
        # Strategy: Keep a window of valid points
        # If current point is too far from last valid, discard it.
        # BUT: If we discard too many consecutive points, maybe the last valid one was wrong?
        # For simplicity, here we implement simple forward filtering.
        
        last_valid_frame = points[0][0]
        last_valid_point = points[0][1]
        
        for k in range(1, len(points)):
            curr_frame, curr_point, list_idx = points[k]
            
            dt = curr_frame - last_valid_frame
            if dt <= 0: continue
            
            dist = np.linalg.norm(curr_point - last_valid_point)
            max_dist = threshold * dt
            
            # If temporal gap is huge (e.g. > 2 seconds = 50 frames), 
            # accept new position as "new start" to avoid losing track
            # if player crossed the field while not visible.
            if dt > 50:
                max_dist = float('inf')

            if dist <= max_dist:
                # Valid movement
                last_valid_frame = curr_frame
                last_valid_point = curr_point
            else:
                # Impossible movement -> Remove
                if cls_id in results_per_frame[list_idx]:
                    del results_per_frame[list_idx][cls_id]
                    points_removed += 1
                    
    print(f"[PIPELINE] Filtered {points_removed} 3D points for impossible movements (thresholds: Humans=0.5m/f, Ball=1.2m/f).")
    return results_per_frame


def run_pipeline(cameras: List[int], labels_dir: str, do_interpolate: bool, max_gap: int, anchor: str, min_cams: int,
				 tracks_out_dir: str = 'runs/tracks', save_tracks: bool = False,
				 rectify: bool = False, infer_videos: bool = False, video_dir: str = 'dataset/video', infer_images_dir: str | None = None, model_path: str = 'weights/fine_tuned_yolo_final.pt', device: str = 'auto',
				 conf_thres: float = 0.25, imgsz: int = 1920, evaluate_labels: bool = False, eval_gt_dir: str = 'dataset/val/labels', eval_out_dir: str = 'runs/eval'):
	"""
	Executes the complete multi-camera 3D reconstruction pipeline.
	
	Args:
		cameras: List of camera IDs to process (e.g. [2, 4, 13])
		labels_dir: Base directory with labels per camera (JSON or txt)
		do_interpolate: If True, fills temporal gaps in detections
		max_gap: Maximum number of frames to interpolate (default 10)
		anchor: Bbox point for triangulation ('center' or 'bottom_center')
		min_cams: Minimum cameras required to triangulate a point (default 2)
		
		tracks_out_dir: Output directory for JSON labels and videos (default 'runs/tracks')
		save_tracks: If True, saves JSON labels and video to disk (default False)
		
		rectify: If True, corrects optical distortions before triangulation
		
		infer_videos: If True, runs YOLO inference on videos first
		video_dir: Directory with input videos (default 'dataset/video')
		model_path: YOLO model path (default 'weights/fine_tuned_yolo_final.pt')
		device: Inference device ('auto', 'cpu', '0', etc.)
		
		evaluate_labels: If True, evaluates RAW and INTERP IOU against GT
		eval_gt_dir: GT directory for evaluation (default 'dataset/val/labels')
		eval_out_dir: Metrics output directory (default 'runs/eval')
	
	
	Output:
		- runs/triangulation/points.csv: 3D points in meters
		- runs/triangulation/points.json: 3D points with metadata
		- [Optional] runs/tracks/labels_out{cam}.json: 2D tracks
		- [Optional] runs/eval/eval_cam{cam}_raw.json: Pre-interpolation metrics
		- [Optional] runs/eval/eval_cam{cam}_interp.json: Post-interpolation metrics
	"""
	
	# Smart default for eval_gt_dir if inferring on images
	if infer_images_dir and evaluate_labels and eval_gt_dir == 'dataset/val/labels':
		# Check if sibling 'labels' dir exists
		candidate_gt = Path(infer_images_dir).parent / 'labels'
		if candidate_gt.exists():
			print(f"[PIPELINE] Auto-detected GT dir: {candidate_gt} (overriding default dataset/val/labels)")
			eval_gt_dir = str(candidate_gt)

	# ========================================
	# 0) Video inference
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
		# Run YOLO on video to generate JSON labels per camera
		labels_base = Path(labels_dir)
		detected_cams = []
		
		# Process each video (e.g. out2.mp4, out4.mp4, out13.mp4)
		for cam in cameras:
			video_path = video_base / f"out{cam}.mp4"
			if not video_path.exists():
				print(f"[PIPELINE] WARN: Video not found: {video_path}. Skip.")
				continue
			
			# YOLO inference frame-by-frame → save labels_out{cam}.json
			cam_num = run_video_inference(video_path, model_path, labels_base, device=device, conf_thres=conf_thres, imgsz=imgsz)
			if cam_num is not None:
				detected_cams.append(cam_num)
		
		if detected_cams:
			print(f"[PIPELINE] Inference completed for cameras: {detected_cams}")
		else:
			print("[PIPELINE] WARN: No video processed.")
	elif infer_images_dir:
		# Run YOLO on image directory
		print(f"[PIPELINE] Running inference on images in: {infer_images_dir}")
		labels_base = Path(labels_dir)
		detected_cams = run_images_inference(Path(infer_images_dir), model_path, labels_base, device=device, conf_thres=conf_thres)
		if detected_cams:
			print(f"[PIPELINE] Image inference completed for cameras: {detected_cams}")
			
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
			print("[PIPELINE] WARN: No images processed.")
	
	# ========================================
	# 1) Load labels
	# ========================================
	per_cam_frames: Dict[int, PerFrame] = {}  # {cam: {frame: {cls: (x,y,w,h,conf)}}}
	labels_base = Path(labels_dir)
	
	for cam in cameras:
		# --- 1a) Load labels from JSON or txt directory ---
		# Preference: labels_out{cam}.json (single file per camera)
		# Fallback: labels_out{cam}/ (directory with frame_XXXXX.txt)
		json_path = labels_base / f"labels_out{cam}.json"
		dir_path = labels_base / f"labels_out{cam}"
		
		if json_path.exists():
			try:
				frames, _ = load_labels_for_camera(json_path)
			except Exception as e:
				print(f"[PIPELINE] WARN: Error loading JSON {json_path}: {e}. Skip camera {cam}.")
				continue
		elif dir_path.exists():
			try:
				frames, _ = load_labels_for_camera(dir_path)
			except Exception as e:
				print(f"[PIPELINE] WARN: Error loading dir {dir_path}: {e}. Skip camera {cam}.")
				continue
		else:
			print(f"[PIPELINE] WARN: Labels not found for camera {cam} (checked {json_path} and {dir_path}). Skip.")
			continue
		
		# --- 1b) SAVE LABELS  ---
		# Save camera labels to JSON and optionally generate visualization video
		if save_tracks:
			out_tracks = Path(tracks_out_dir) / f"labels_out{cam}.json"
			# Convert frames to save format (no track_id, only class_id, x, y, w, h, conf)
			frames_for_save = {}
			for fi, dets in frames.items():
				frames_for_save[fi] = {}
				for cls_id, (x, y, w, h, conf) in dets.items():
					frames_for_save[fi][cls_id] = {
						"x": float(x), "y": float(y), "w": float(w), "h": float(h),
						"conf": float(conf)
					}
			save_tracks_json(frames_for_save, out_tracks, camera_id=cam)
			
			# Generate visualization video (without track_id)
			try:
				from tracking_2d import save_tracked_video_from_images, save_tracked_video_from_video
				out_vid = Path(tracks_out_dir) / f"tracking_out{cam}.mp4"
				
				# Case 1: We have an image directory (e.g. dataset/val/images)
				if infer_images_dir:
					save_tracked_video_from_images(frames_for_save, Path(infer_images_dir), out_vid, camera_id=cam)
				
				# Case 2: We have a source video (e.g. dataset/video/out13.mp4)
				elif video_dir:
					vid_path = Path(video_dir) / f"out{cam}.mp4"
					if vid_path.exists():
						save_tracked_video_from_video(frames_for_save, vid_path, out_vid)
					else:
						print(f"[PIPELINE] Source video not found for cam {cam}: {vid_path}")
						
			except ImportError:
				print("[PIPELINE] WARN: Video tracking functions not found in tracking_2d")
			except Exception as e:
				print(f"[PIPELINE] WARN: Error generating video: {e}")
		
		# --- 1c) OPTIONAL: RAW evaluation against GT ---
		# Calculate mean IOU between inferred detections and ground truth
		gt_map = None
		if evaluate_labels:
			try:
				from evaluation import load_labels_dir_map, frames_from_perframe, compute_iou_metrics_from_predictions, save_eval_json
			except Exception as e:
				print(f"[PIPELINE] WARN: cannot import evaluation functions: {e}")
			else:
				gt_dir = Path(eval_gt_dir)
				if gt_dir.exists():
					# Load GT filtering for this camera
					gt_map = load_labels_dir_map(gt_dir, camera_id=cam)
					
					# If inferring on full videos, we must remap GT keys
					# GT frame 1 -> Video frame 3 (User observed GT was 1 frame ahead)
					# GT frame k -> Video frame 3 + (k-1)*5
					if infer_videos and gt_map:
						print(f"[PIPELINE] Remapping GT frames for video alignment (offset 3, stride 5)")
						new_gt_map = {}
						for k, v in gt_map.items():
							new_k = 3 + (k - 1) * 5
							new_gt_map[new_k] = v
						gt_map = new_gt_map
					
					# Convert internal structure to evaluation format
					raw_map = frames_from_perframe(frames)
					
					# Normalize coordinates if necessary (pixel -> [0,1])
					# Assume if x > 1 they are pixels
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

					# Calculate IOU metrics
					m_raw = compute_iou_metrics_from_predictions(raw_map, gt_map)
					
					# Save metrics JSON
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
					print(f"[PIPELINE] WARN: GT not found in {gt_dir}, skip RAW evaluation")
		
		# --- 1d) OPTIONAL: Temporal interpolation ---
		# Fills gaps in detections by copying last valid value
		if do_interpolate:
			frames = interpolate_missing_detections(frames, max_frame=max(frames.keys()) if frames else -1, max_gap=max_gap)
			
			# --- 1e) OPTIONAL: INTERP evaluation against GT ---
			# Recalculate IOU after interpolation to measure impact
			if evaluate_labels and gt_map:
				try:
					from evaluation import frames_from_perframe, compute_iou_metrics_from_predictions, save_eval_json
					
					inter_map = frames_from_perframe(frames)
					
					# Normalize coordinates if necessary
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
					print(f"[PIPELINE] WARN: Error evaluating INTERP: {e}")
		
		# Save processed frames for this camera
		per_cam_frames[cam] = frames

	# ========================================
	# 2) OPTIONAL STEP: DISTORTION RECTIFICATION
	# ========================================
	# Corrects optical distortions of cameras before triangulation
	if rectify and per_cam_frames:
		# Load calibrations for processed cameras
		calibs_for_rect = tri.load_calibrations_for_cams(list(per_cam_frames.keys()))
		
		for cam in list(per_cam_frames.keys()):
			params = calibs_for_rect.get(cam, {})
			K = params.get("K")  # Intrinsic matrix
			dist = params.get("dist")  # Distortion coefficients
			
			if K is None or dist is None:
				print(f"[PIPELINE] WARN: Missing calibration for cam {cam}, skip rectification")
				continue
			
			# Apply undistort to bbox centers
			per_cam_frames[cam] = _rectify_frames_in_memory(per_cam_frames[cam], K, dist)

	# ========================================
	# 3) DATA PREPARATION FOR TRIANGULATION
	# ========================================
	# Reorganize from per-camera to per-frame-per-class structure
	frames_by_class = build_frames_by_class_from_cameras(per_cam_frames)
	if not frames_by_class:
		print("[PIPELINE] No labels loaded. Exiting.")
		return
	
	# Generate list of frame indices to process
	idxs = list(range(len(frames_by_class)))

	# ========================================
	# 4) LOAD CAMERA CALIBRATIONS
	# ========================================
	# Reads intrinsic and extrinsic parameters from JSON
	calibs = tri.load_calibrations_for_cams(cameras)

	# ========================================
	# 5) MULTI-CAMERA 3D TRIANGULATION
	# ========================================
	# Linear 3D reconstruction (SVD) from multi-camera 2D projections
	# Applies scale 0.001 for millimeters→meters conversion
	results_per_frame = tri.triangulate_frames_list(frames_by_class, idxs, calibs, anchor=anchor, scale=0.001, min_cams=min_cams)

	# ========================================
	# 5b) IMPOSSIBLE MOVEMENT FILTERING
	# ========================================
	# Removes points that move too fast (noise)
	results_per_frame = filter_3d_movements(results_per_frame, idxs)

	# ========================================
	# 6) SAVE RESULTS
	# ========================================
	out_dir = Path("runs") / "triangulation"
	out_dir.mkdir(parents=True, exist_ok=True)
	
	# --- 6a) Save compact CSV ---
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
	print(f"[PIPELINE] Saved {total_points} points in {out_csv}")
	
	# --- 6b) Save structured JSON with metadata ---
	save_points_json(
		results_per_frame,
		idxs,
		out_dir / "points.json",
		frames_by_class=frames_by_class,
		tracks_by_cam=None,
	)

	# Note: per-camera evaluation already performed during loop (step 1c and 1e)


def main():
	"""
	Main entry point: parse arguments, run pipeline, visualize results.
	
	Flow:
		1. Parse command-line arguments (uses config.get_args())
		2. Run run_pipeline with configured parameters
		3. If requested (--visualize), launch interactive 3D visualizer
	"""
	args = get_args()
	
	# Run full pipeline
	run_pipeline(
		args.cameras, args.labels_dir, args.interpolate, args.max_gap, args.anchor, args.min_cams,
		tracks_out_dir=args.tracks_out_dir, save_tracks=args.save_tracks,
		rectify=args.rectify, infer_videos=args.infer_videos, video_dir=args.video_dir, infer_images_dir=args.infer_images_dir,
		model_path=args.model, device=args.device, conf_thres=args.conf_thres, imgsz=args.imgsz,
		evaluate_labels=args.evaluate_labels, eval_gt_dir=args.eval_gt_dir, eval_out_dir=args.eval_out_dir
	)
	
	# Interactive 3D visualization (lazy import to avoid matplotlib dependencies in headless run)
	if args.visualize:
		from visualizer_3d import visualize_triangulated_points
		visualize_triangulated_points()


if __name__ == '__main__':
	main()

# ----------------------------------------
# How to run (PowerShell)
# ----------------------------------------
# FULL WORKFLOW: Video Inference -> Interpolation -> Rectification -> Triangulation -> Evaluation -> Visualization:
#   python pipeline.py --infer-videos --interpolate --rectify --evaluate-labels --visualize --device 0
#
# With custom model and confidence threshold to capture more Ball detections (higher sensitivity):
#   python pipeline.py --infer-videos --model weights/fine_tuned_yolo2k.pt --conf-thres 0.15 --interpolate --rectify --evaluate-labels --device 0
#
# With high confidence threshold to reduce false positives:
#   python pipeline.py --infer-videos --conf-thres 0.4 --interpolate --rectify --evaluate-labels --device 0
#
# With 2D tracking (optional, disabled by default):
#   python pipeline.py --infer-videos --track2d --save-tracks --interpolate --rectify --evaluate-labels --visualize --device 0
#
# If you already have labels and just want to triangulate and evaluate:
#   python pipeline.py --labels-dir dataset/infer_video --interpolate --rectify --evaluate-labels --visualize
#
# Triangulation only (labels ready, no tracking/interpolation/evaluation):
#   python pipeline.py --labels-dir dataset/infer_video --visualize
#
# Custom evaluation with GT in another directory:
#   python pipeline.py --labels-dir dataset/infer_video --interpolate --evaluate-labels --eval-gt-dir path/to/gt/labels
