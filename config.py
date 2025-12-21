"""
Configurazione centralizzata per la pipeline di ricostruzione 3D multi-camera.

Questo modulo gestisce tutti i parametri configurabili del progetto in un unico posto.
Tutti gli altri moduli importano da qui per accedere alla configurazione.

Usage:
    from config import get_args, PipelineConfig
    
    # In script principale
    args = get_args()
    config = PipelineConfig.from_args(args)
    
    # In altri moduli
    config = PipelineConfig.from_args(args)  # passa args ricevuto
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class PipelineConfig:
	"""
	Configurazione immutabile per la pipeline di ricostruzione 3D.
	
	Attributes:
		--- Base ---
		cameras: Lista ID camere da processare
		labels_dir: Directory base con labels per camera
		interpolate: Se True, interpola detections mancanti
		max_gap: Numero massimo frame da interpolare
		anchor: Punto bbox per triangolazione ('center' o 'bottom_center')
		min_cams: Minimo camere richieste per triangolare
		visualize: Se True, avvia visualizzatore 3D al termine
		
		--- Tracking 2D ---
		track2d: Se True, esegue tracking 2D IoU-based
		track_iou: Soglia IoU per matching tracce
		track_max_age: Frame massimi senza match prima di chiudere traccia
		tracks_out_dir: Directory output JSON tracce 2D
		save_tracks: Se True, salva JSON tracce su disco
		
		--- Rettificazione ---
		rectify: Se True, corregge distorsioni ottiche
		
		--- Inferenza video ---
		infer_videos: Se True, esegue inferenza YOLO su video
		video_dir: Directory con video input
		model: Path modello YOLO
		device: Device per inferenza
		conf_thres: Confidence threshold per YOLO predict
		imgsz: Dimensione immagine per inferenza YOLO
		
		--- Valutazione ---
		evaluate_labels: Se True, valuta IOU contro GT
		eval_gt_dir: Directory GT per valutazione
		eval_out_dir: Directory output metriche
	"""
	# --- Base ---
	cameras: List[int]
	labels_dir: str
	interpolate: bool
	max_gap: int
	anchor: str
	min_cams: int
	visualize: bool
	
	# --- Tracking 2D ---
	track2d: bool
	track_iou: float
	track_max_age: int
	tracks_out_dir: str
	save_tracks: bool
	
	# --- Rettificazione ---
	rectify: bool
	
	# --- Inferenza video ---
	infer_videos: bool
	video_dir: str
	infer_images_dir: str | None
	model: str
	device: str
	conf_thres: float
	imgsz: int
	
	# --- Valutazione ---
	evaluate_labels: bool
	eval_gt_dir: str
	eval_out_dir: str
	
	@classmethod
	def from_args(cls, args: argparse.Namespace) -> PipelineConfig:
		"""
		Crea configurazione da argomenti parsed.
		
		Args:
			args: Namespace da argparse con tutti i parametri
		
		Returns:
			PipelineConfig istanza immutabile
		"""
		if args.evaluate_labels and not args.infer_videos:
			raise ValueError("Errore configurazione: Non è possibile eseguire --evaluate-labels se non viene eseguito anche --infer-videos.")

		return cls(
			# Base
			cameras=args.cameras,
			labels_dir=args.labels_dir,
			interpolate=args.interpolate,
			max_gap=args.max_gap,
			anchor=args.anchor,
			min_cams=args.min_cams,
			visualize=args.visualize,
			
			# Tracking 2D
			track2d=args.track2d,
			track_iou=args.track_iou,
			track_max_age=args.track_max_age,
			tracks_out_dir=args.tracks_out_dir,
			save_tracks=args.save_tracks,
			
			# Rettificazione
			rectify=args.rectify,
			
			# Inferenza video
			infer_videos=args.infer_videos,
			video_dir=args.video_dir,
			infer_images_dir=args.infer_images_dir,
			model=args.model,
			device=args.device,
			conf_thres=args.conf_thres,
			imgsz=getattr(args, 'imgsz', 1920),  # fallback per compatibilità
			
			# Valutazione
			evaluate_labels=args.evaluate_labels,
			eval_gt_dir=args.eval_gt_dir,
			eval_out_dir=args.eval_out_dir,
		)
	
	@classmethod
	def default(cls) -> PipelineConfig:
		"""
		Crea configurazione con valori di default.
		
		Returns:
			PipelineConfig con parametri predefiniti
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
			
			# Tracking 2D
			track2d=False,
			track_iou=0.3,
			track_max_age=15,
			tracks_out_dir='runs/tracks',
			save_tracks=False,
			
			# Rettificazione
			rectify=False,
			
			# Inferenza video
			infer_videos=False,
			video_dir='dataset/video',
			infer_images_dir=None,
			model='weights/fine_tuned_yolo_final.pt',
			device='auto',
			conf_thres=0.25,
			imgsz=1920,
			
			# Valutazione
			evaluate_labels=False,
			eval_gt_dir='dataset/val/labels',
			eval_out_dir='runs/eval',
		)


def get_args() -> argparse.Namespace:
	"""
	Parser argomenti command-line centralizzato per tutta la pipeline.
	
	Returns:
		args: Namespace con tutti i parametri configurati
	
	Gruppi opzioni:
		- Base: cameras, labels-dir, interpolate, anchor, min-cams, visualize
		- Tracking 2D: track2d, track-iou, track-max-age, tracks-out-dir, save-tracks
		- Rettificazione: rectify
		- Inferenza video: infer-videos, video-dir, model, device, conf-thres, imgsz
		- Valutazione: evaluate-labels, eval-gt-dir, eval-out-dir
	
	Examples:
		>>> args = get_args()
		>>> config = PipelineConfig.from_args(args)
		>>> print(config.model, config.conf_thres)
	"""
	p = argparse.ArgumentParser(description="Pipeline unificata per ricostruzione 3D multi-camera")
	
	# --- Opzioni base ---
	p.add_argument('--cameras', type=int, nargs='+', default=[2,4,13], 
				   help='Lista ID camere da processare (default: 2 4 13)')
	p.add_argument('--labels-dir', type=str, default='dataset/infer_video', 
				   help='Directory base con labels per camera (JSON o txt)')
	p.add_argument('--no-interpolate', dest='interpolate', action='store_false', 
				   help='Disabilita interpolazione detections (attiva di default)')
	# Default is True due to store_false
	p.add_argument('--max-gap', type=int, default=10, 
				   help='Numero massimo frame da interpolare (default: 10)')
	p.add_argument('--anchor', type=str, default='center', choices=['center','bottom_center'], 
				   help='Punto bbox per triangolazione: center o bottom_center')
	p.add_argument('--min-cams', type=int, default=2, 
				   help='Minimo camere richieste per triangolare (default: 2)')
	p.add_argument('--visualize', action='store_true', 
				   help='Avvia visualizzatore 3D interattivo al termine')
	
	# --- Opzioni tracking 2D ---
	p.add_argument('--track2d', action='store_true', 
				   help='Esegui tracking 2D IoU-based (DISATTIVATO di default)')
	p.add_argument('--track-iou', type=float, default=0.3, 
				   help='Soglia IoU per matching tracce (default: 0.3)')
	p.add_argument('--track-max-age', type=int, default=15, 
				   help='Frame massimi senza match prima di chiudere traccia (default: 15)')
	p.add_argument('--tracks-out-dir', type=str, default='runs/tracks', 
				   help='Directory output JSON tracce 2D (default: runs/tracks)')
	p.add_argument('--save-tracks', action='store_true', 
				   help='Salva JSON tracce su disco (una volta, skip se esiste)')
	
	# --- Opzione rettificazione ---
	p.add_argument('--no-rectify', dest='rectify', action='store_false', 
				   help='Disabilita correzione distorsioni ottiche (attiva di default)')
	
	# --- Opzioni inferenza video ---
	p.add_argument('--infer-videos', action='store_true', 
				   help='Esegui inferenza YOLO su video prima di tutto')
	p.add_argument('--video-dir', type=str, default='dataset/video', 
				   help='Directory con video input out{cam}.mp4 (default: dataset/video)')
	p.add_argument('--infer-images-dir', type=str, default=None, 
				   help='Directory con immagini input out{cam}_frame_{num}*.jpg (default: None)')
	p.add_argument('--model', type=str, default='weights/fine_tuned_yolo_final.pt', 
				   help='Path modello YOLO fine-tuned (default: weights/fine_tuned_yolo_final.pt)')
	p.add_argument('--device', type=str, default='auto', 
				   help='Device inferenza: auto, cpu, 0, 0,1 (default: auto)')
	p.add_argument('--conf-thres', type=float, default=0.25, 
				   help='Confidence threshold per YOLO predict (default: 0.25)')
	p.add_argument('--imgsz', type=int, default=1920, 
				   help='Dimensione immagine per inferenza YOLO (default: 1920)')
	
	# --- Opzioni valutazione ---
	p.add_argument('--evaluate-labels', action='store_true', 
				   help='Valuta IOU RAW e INTERP contro ground truth')
	p.add_argument('--eval-gt-dir', type=str, default='dataset/val/labels', 
				   help='Directory GT con file out{cam}_frame_{num}*.txt (default: dataset/val/labels)')
	p.add_argument('--eval-out-dir', type=str, default='runs/eval', 
				   help='Directory output metriche JSON (default: runs/eval)')
	
	return p.parse_args()


# ----------------------------------------
# Esempi di utilizzo
# ----------------------------------------
if __name__ == '__main__':
	# Test configurazione default
	config = PipelineConfig.default()
	print("Config default:")
	print(f"  Model: {config.model}")
	print(f"  Conf threshold: {config.conf_thres}")
	print(f"  Cameras: {config.cameras}")
	print(f"  Device: {config.device}")
	
	# Test parsing argomenti
	print("\nParsing command-line args...")
	args = get_args()
	config = PipelineConfig.from_args(args)
	print(f"  Model from args: {config.model}")
	print(f"  Conf threshold from args: {config.conf_thres}")
