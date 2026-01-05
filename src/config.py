"""
Centralized configuration for multi-camera 3D reconstruction pipeline.

This module manages all configurable parameters of the project in one place.
All other modules import from here to access the configuration.

Usage:
    from config import get_args, PipelineConfig
    
    # In main script
    args = get_args()
    config = PipelineConfig.from_args(args)
    
    # In other modules
    config = PipelineConfig.from_args(args)  # pass received args
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class PipelineConfig:
	"""
	Immutable configuration for multi-camera 3D reconstruction pipeline.
	
	Attributes:
		--- Base ---
		cameras: List of camera IDs to process
		labels_dir: Base directory with labels per camera
		interpolate: If True, interpolate missing detections
		max_gap: Maximum number of frames to interpolate
		anchor: Bounding box anchor point for triangulation ('center' or 'bottom_center')
		min_cams: Minimum cameras required for triangulation
		visualize: If True, launch 3D viewer on completion
		
		--- Labels & Video ---
		tracks_out_dir: Output directory for JSON labels and videos
		save_tracks: If True, save JSON labels and generate visualization video
		
		--- Rectification ---
		rectify: If True, apply optical distortion correction
		
		--- Video Inference ---
		infer_videos: If True, run YOLO inference on videos
		video_dir: Directory containing input videos
		infer_images_dir: Directory containing input images (optional)
		model: Path to YOLO model file
		device: Device for inference (cpu, auto, cuda, etc.)
		conf_thres: Confidence threshold for YOLO predictions
		imgsz: Input image size for YOLO inference
		
		--- Evaluation ---
		evaluate_labels: If True, evaluate IOU against ground truth
		eval_gt_dir: Ground truth directory for evaluation
		eval_out_dir: Output directory for metrics
	"""
	# --- Base ---
	cameras: List[int]
	labels_dir: str
	interpolate: bool
	max_gap: int
	anchor: str
	min_cams: int
	visualize: bool
	
	# --- Labels & Video ---
	tracks_out_dir: str
	save_tracks: bool
	
	# --- Rectification ---
	rectify: bool
	
	# --- Video Inference ---
	infer_videos: bool
	video_dir: str
	infer_images_dir: str | None
	model: str
	device: str
	conf_thres: float
	imgsz: int
	
	# --- Evaluation ---
	evaluate_labels: bool
	eval_gt_dir: str
	eval_out_dir: str
	
	@classmethod
	def from_args(cls, args: argparse.Namespace) -> PipelineConfig:
		"""
		Create configuration from parsed command-line arguments.
		
		Args:
			args: argparse Namespace containing all parameters
		
		Returns:
			PipelineConfig: Immutable configuration instance
		
		Raises:
			ValueError: If incompatible argument combinations are detected
		"""
		if args.evaluate_labels and not args.infer_videos:
			raise ValueError("Configuration error: Cannot run --evaluate-labels unless --infer-videos is also run.")

		return cls(
			# Base
			cameras=args.cameras,
			labels_dir=args.labels_dir,
			interpolate=args.interpolate,
			max_gap=args.max_gap,
			anchor=args.anchor,
			min_cams=args.min_cams,
			visualize=args.visualize,
			
			# Labels & Video
			tracks_out_dir=args.tracks_out_dir,
			save_tracks=args.save_tracks,
			
			# Rectification
			rectify=args.rectify,
			
			# Video inference
			infer_videos=args.infer_videos,
			video_dir=args.video_dir,
			infer_images_dir=args.infer_images_dir,
			model=args.model,
			device=args.device,
			conf_thres=args.conf_thres,
			imgsz=getattr(args, 'imgsz', 1920),  # fallback for compatibility
						# Evaluation
			evaluate_labels=args.evaluate_labels,
			eval_gt_dir=args.eval_gt_dir,
			eval_out_dir=args.eval_out_dir,
		)
	
	@classmethod
	def default(cls) -> PipelineConfig:
		"""
		Create default configuration
		
		Returns:
			PipelineConfig with configured parameters
		"""
		return cls(
			# Base
			cameras=[2, 4, 13],
			labels_dir='dataset/infer_video',
			interpolate=False,
			max_gap=10,
			anchor='center',
			min_cams=2,
			visualize=False,
			
			# Labels & Video
			tracks_out_dir='runs/tracks',
			save_tracks=False,
			
			# Rectification
			rectify=False,
			
			# Video inference
			infer_videos=False,
			video_dir='dataset/video',
			infer_images_dir=None,
			model='weights/fine_tuned_yolo_final.pt',
			device='auto',
			conf_thres=0.25,
			imgsz=1920,
			
			# Evaluation
			evaluate_labels=False,
			eval_gt_dir='dataset/val/labels',
			eval_out_dir='runs/eval',
		)


def get_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Unified multi-camera 3D reconstruction pipeline")
	
	# --- Base options ---
	p.add_argument('--cameras', type=int, nargs='+', default=[2,4,13], 
				   help='List of camera IDs to process (default: 2 4 13)')
	p.add_argument('--labels-dir', type=str, default='dataset/infer_video', 
				   help='Base directory with labels per camera (JSON or txt)')
	p.add_argument('--no-interpolate', dest='interpolate', action='store_false', 
				   help='Disable detection interpolation (enabled by default)')
	# Default is True due to store_false
	p.add_argument('--max-gap', type=int, default=10, 
				   help='Maximum number of frames to interpolate (default: 10)')
	p.add_argument('--anchor', type=str, default='center', choices=['center','bottom_center'], 
				   help='Bounding box anchor point for triangulation: center or bottom_center')
	p.add_argument('--min-cams', type=int, default=2, 
				   help='Minimum number of cameras required for triangulation (default: 2)')
	p.add_argument('--visualize', action='store_true', 
				   help='Launch interactive 3D visualizer upon completion')
	
	# --- Label and video output ---
	p.add_argument('--tracks-out-dir', type=str, default='runs/tracks', 
				   help='Output directory for JSON labels and video (default: runs/tracks)')
	p.add_argument('--save-tracks', action='store_true', 
				   help='Save JSON labels and generate visualization video')
	
	# --- Rectification option ---
	p.add_argument('--no-rectify', dest='rectify', action='store_false', 
				   help='Disable optical distortion correction (enabled by default)')
	
	# --- Video inference options ---
	p.add_argument('--infer-videos', action='store_true', 
				   help='Run YOLO inference on videos first')
	p.add_argument('--video-dir', type=str, default='dataset/video', 
				   help='Directory containing input videos out{cam}.mp4 (default: dataset/video)')
	p.add_argument('--infer-images-dir', type=str, default=None, 
				   help='Directory containing input images out{cam}_frame_{num}*.jpg (default: None)')
	p.add_argument('--model', type=str, default='weights/fine_tuned_yolo_final.pt', 
				   help='Path to fine-tuned YOLO model (default: weights/fine_tuned_yolo_final.pt)')
	p.add_argument('--device', type=str, default='auto', 
				   help='Inference device: auto, cpu, 0, 0,1 (default: auto)')
	p.add_argument('--conf-thres', type=float, default=0.25, 
				   help='Confidence threshold for YOLO predictions (default: 0.25)')
	p.add_argument('--imgsz', type=int, default=1920, 
				   help='Image size for YOLO inference (default: 1920)')
	
	# --- Evaluation options ---
	p.add_argument('--evaluate-labels', action='store_true', 
				   help='Evaluate RAW and INTERP IoU metrics against ground truth')
	p.add_argument('--eval-gt-dir', type=str, default='dataset/val/labels', 
				   help='Ground truth directory with out{cam}_frame_{num}*.txt files (default: dataset/val/labels)')
	p.add_argument('--eval-out-dir', type=str, default='runs/eval', 
				   help='Output directory for evaluation metrics JSON (default: runs/eval)')
	
	return p.parse_args()


# ----------------------------------------
# Usage examples
# ----------------------------------------
if __name__ == '__main__':
	# Test default configuration
	config = PipelineConfig.default()
	print("Default config:")
	print(f"  Model: {config.model}")
	print(f"  Conf threshold: {config.conf_thres}")
	print(f"  Cameras: {config.cameras}")
	print(f"  Device: {config.device}")
	
	# Test command-line argument parsing
	print("\nParsing command-line arguments...")
	args = get_args()
	config = PipelineConfig.from_args(args)
	print(f"  Model from args: {config.model}")
	print(f"  Conf threshold from args: {config.conf_thres}")
