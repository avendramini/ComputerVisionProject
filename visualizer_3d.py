from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import csv
import math

import numpy as np

# Matplotlib for interactive 3D visualization
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# Reuse calibration loader
from triangulation_3d import load_calibrations_for_cams
import json


def load_points_csv(csv_path: Path) -> Tuple[Dict[int, List[Tuple[int, float, float, float]]], List[int]]:
	"""Load points.csv into a frame-indexed structure.

	Returns:
	  - frames_points: {frame: [(class_id, x, y, z), ...]}
	  - sorted unique class ids present
	"""
	frames_points: Dict[int, List[Tuple[int, float, float, float]]] = {}
	classes = set()
	with csv_path.open("r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			fi = int(row["frame"]) if "frame" in row else int(row["Frame"])  # tolerant
			ci = int(row["class_id"]) if "class_id" in row else int(row.get("class", 0))
			x = float(row.get("x_m", row.get("x", 0.0)))
			y = float(row.get("y_m", row.get("y", 0.0)))
			z = float(row.get("z_m", row.get("z", 0.0)))
			frames_points.setdefault(fi, []).append((ci, x, y, z))
			classes.add(ci)
	return frames_points, sorted(classes)

def load_points_json(json_path: Path) -> Tuple[Dict[int, List[Tuple[int, float, float, float]]], List[int]]:
    """Load points.json with schema {frames: {"0":[{class_id,x,y,z},...]}} into same structure as CSV loader."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frames_points: Dict[int, List[Tuple[int, float, float, float]]] = {}
    classes = set()
    frames = data.get("frames", {})
    for k, items in frames.items():
        fi = int(k)
        if not items:
            continue
        lst: List[Tuple[int, float, float, float]] = []
        for it in items:
            cid = int(it.get("class_id", 0))
            x = float(it.get("x", 0.0))
            y = float(it.get("y", 0.0))
            z = float(it.get("z", 0.0))
            lst.append((cid, x, y, z))
            classes.add(cid)
        frames_points[fi] = lst
    return frames_points, sorted(classes)


def find_available_cams(camparams_dir: Path = Path("camparams")) -> List[int]:
	cams: List[int] = []
	if not camparams_dir.exists():
		return cams
	for p in camparams_dir.glob("out*_camera_calib.json"):
		name = p.stem  # e.g. out13_camera_calib
		try:
			num = int(name.replace("out", "").split("_")[0])
			cams.append(num)
		except Exception:
			continue
	return sorted(set(cams))


def compute_camera_centers(calibs: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Compute camera centers C and view directions dir (world) from calibrations.

    Returns {cam_id: (C (3,), dir (3,))}
    """
    import cv2  # local import
    
    result: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for cam, params in calibs.items():
        K = params.get("K")
        rvec = params.get("rvec")
        tvec = params.get("tvec")
        if K is None or rvec is None or tvec is None:
            continue
        R, _ = cv2.Rodrigues(rvec)
        R = R.astype(np.float64)
        # tvec è in millimetri, convertiamo in metri
        t = (tvec.reshape(3, 1) / 1000.0).astype(np.float64)
        C = (-R.T @ t).reshape(3)  # camera center in world (metri)
        z_axis_world = (R.T @ np.array([0.0, 0.0, 1.0])).reshape(3)
        result[cam] = (C, z_axis_world / (np.linalg.norm(z_axis_world) + 1e-9))
    return result
def draw_court(ax, width_m: float = 28.0, length_m: float = 15.0, height_axis: str = "z"):
	"""Draw a simple basketball court rectangle on ground plane.

	height_axis: 'z' assumes ground plane z=0; if 'y', ground plane y=0
	"""
	hw = width_m / 2.0
	hl = length_m / 2.0
	rect = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl), (-hw, -hl)]
	if height_axis == "z":
		xs = [x for x, y in rect]
		ys = [y for x, y in rect]
		zs = [0.0] * len(xs)
		ax.plot(xs, ys, zs, color="gray", linewidth=2)
	else:
		xs = [x for x, y in rect]
		zs = [y for x, y in rect]
		ys = [0.0] * len(xs)
		ax.plot(xs, ys, zs, color="gray", linewidth=2)


def set_equal_aspect_3d(ax, X, Y, Z):
	max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
	if max_range <= 0:
		max_range = 1.0
	mid_x = (X.max()+X.min()) * 0.5
	mid_y = (Y.max()+Y.min()) * 0.5
	mid_z = (Z.max()+Z.min()) * 0.5
	ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
	ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
	ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)


class FrameViewer3D:
    def __init__(self, frames_points: Dict[int, List[Tuple[int, float, float, float]]], classes: List[int], calibs: Dict[int, Dict[str, np.ndarray]] | None = None):
        self.frames_points = frames_points
        self.sorted_frames = sorted(frames_points.keys())
        self.classes = classes
        self.calibs = calibs or {}
        self.cam_geom = compute_camera_centers(self.calibs) if self.calibs else {}

        # Compute global fixed bounds to include court + all points + cameras
        court_width = 28.0
        court_length = 15.0
        
        all_pts = [(x,y,z) for pts in frames_points.values() for (_c,x,y,z) in pts]
        
        # Start with court bounds
        x_min, x_max = -court_width/2 - 2, court_width/2 + 2
        y_min, y_max = -court_length/2 - 2, court_length/2 + 2
        z_min, z_max = -1, 5  # ground to reasonable height
        
        # Expand to include all data points
        if all_pts:
            arr = np.array(all_pts)
            x_min = min(x_min, arr[:,0].min() - 1)
            x_max = max(x_max, arr[:,0].max() + 1)
            y_min = min(y_min, arr[:,1].min() - 1)
            y_max = max(y_max, arr[:,1].max() + 1)
            z_min = min(z_min, arr[:,2].min() - 1)
            z_max = max(z_max, arr[:,2].max() + 1)
        
        # Expand to include cameras
        for cam, (C, _dir) in self.cam_geom.items():
            x_min = min(x_min, C[0] - 2)
            x_max = max(x_max, C[0] + 2)
            y_min = min(y_min, C[1] - 2)
            y_max = max(y_max, C[1] + 2)
            z_min = min(z_min, C[2] - 2)
            z_max = max(z_max, C[2] + 2)
        
        self.fixed_bounds = (x_min, x_max, y_min, y_max, z_min, z_max)

        # Simple color map for classes
        cmap = plt.cm.get_cmap('tab20', max(len(classes), 1))
        self.class_colors = {cid: cmap(i % 20) for i, cid in enumerate(classes)}
        
        # Assume height axis is z (as used in triangulation outputs)
        self.height_axis = 'z'

        # Matplotlib setup
        self.fig = plt.figure(figsize=(10, 8))
        self.ax: Axes3D = self.fig.add_subplot(111, projection='3d')
        self.scatter = None
        self.title = None
        self.current_index = 0

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.draw_frame(self.sorted_frames[self.current_index])

    def on_key(self, event):
        if event.key in ['right', 'd']:
            self.current_index = min(self.current_index + 1, len(self.sorted_frames) - 1)
            self.draw_frame(self.sorted_frames[self.current_index])
        elif event.key in ['left', 'a']:
            self.current_index = max(self.current_index - 1, 0)
            self.draw_frame(self.sorted_frames[self.current_index])
        elif event.key == 'home':
            self.current_index = 0
            self.draw_frame(self.sorted_frames[self.current_index])
        elif event.key == 'end':
            self.current_index = len(self.sorted_frames) - 1
            self.draw_frame(self.sorted_frames[self.current_index])
        elif event.key == 's':
            self.save_top_down_trajectories()

    def save_top_down_trajectories(self):
        """Save top-down view (X-Y) of trajectories for each class to files."""
        output_dir = Path('runs') / 'trajectories'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving trajectory plots to {output_dir}...")

        # Collect all points by class
        class_points = {cid: [] for cid in self.classes}
        for frame_idx in self.sorted_frames:
            pts = self.frames_points[frame_idx]
            for (cid, x, y, z) in pts:
                if cid in class_points:
                    class_points[cid].append((frame_idx, x, y))
        
        # Generate one plot per class
        for cid, points in class_points.items():
            if not points:
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Draw court (28x15)
            # Court is centered at 0,0. Width=28 (X), Length=15 (Y) based on draw_court logic?
            # Wait, draw_court uses: width_m=28, length_m=15.
            # rect = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl), (-hw, -hl)]
            # hw = width/2 = 14. hl = length/2 = 7.5.
            # So X goes from -14 to 14. Y goes from -7.5 to 7.5.
            
            hw = 28.0 / 2.0
            hl = 15.0 / 2.0
            rect_x = [-hw, hw, hw, -hw, -hw]
            rect_y = [-hl, -hl, hl, hl, -hl]
            ax.plot(rect_x, rect_y, 'k-', linewidth=2, label='Court Boundary')
            
            # Sort points by frame
            points.sort(key=lambda p: p[0])
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            
            color = self.class_colors.get(cid, 'blue')
            
            # Plot trajectory
            ax.plot(xs, ys, c=color, linewidth=1, alpha=0.6, label='Trajectory')
            ax.scatter(xs, ys, c=color, s=10, alpha=0.8)
            
            # Mark start and end
            if xs:
                ax.plot(xs[0], ys[0], 'go', markersize=8, label='Start')
                ax.plot(xs[-1], ys[-1], 'rx', markersize=8, label='End')

            ax.set_title(f'Trajectory - Class {cid}')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set limits with some padding
            ax.set_xlim(-16, 16)
            ax.set_ylim(-9, 9)
            
            out_path = output_dir / f'trajectory_class_{cid}.png'
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")
        
        print("Done saving trajectories.")

    def save_side_view_trajectory_ball(self):
        """Save side view (X-Z) of trajectory for Ball (class 0)."""
        output_dir = Path('runs') / 'trajectories'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cid = 0  # Ball class
        if cid not in self.classes:
            return

        # Collect points
        points = []
        for frame_idx in self.sorted_frames:
            pts = self.frames_points[frame_idx]
            for (c, x, y, z) in pts:
                if c == cid:
                    points.append((frame_idx, x, z))
        
        if not points:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Draw ground line (-14 to 14)
        ax.plot([-14, 14], [0, 0], 'k-', linewidth=2, label='Ground')
        
        points.sort(key=lambda p: p[0])
        xs = [p[1] for p in points]
        zs = [p[2] for p in points]
        
        color = self.class_colors.get(cid, 'orange')
        
        ax.plot(xs, zs, c=color, linewidth=1, alpha=0.6, label='Trajectory')
        ax.scatter(xs, zs, c=color, s=10, alpha=0.8)
        
        if xs:
            ax.plot(xs[0], zs[0], 'go', markersize=8, label='Start')
            ax.plot(xs[-1], zs[-1], 'rx', markersize=8, label='End')

        ax.set_title(f'Side View Trajectory (Length vs Height) - Class {cid} (Ball)')
        ax.set_xlabel('Length X (m)')
        ax.set_ylabel('Height Z (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Limits
        ax.set_xlim(-15, 15)
        ax.set_ylim(0, 6)  # Reasonable height for basketball
        
        out_path = output_dir / f'trajectory_side_class_{cid}.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    def draw_cameras(self):
        if not self.cam_geom:
            return
        for cam, (C, dir_vec) in self.cam_geom.items():
            self.ax.scatter([C[0]], [C[1]], [C[2]], marker='^', color='black', s=50)
            # small ray to show view direction
            tip = C + dir_vec * 1.0
            self.ax.plot([C[0], tip[0]], [C[1], tip[1]], [C[2], tip[2]], color='black', linewidth=1)
            self.ax.text(C[0], C[1], C[2], f"cam {cam}", color='black')

    def draw_frame(self, frame_idx: int):
        self.ax.clear()

        # Draw court
        draw_court(self.ax, width_m=28.0, length_m=15.0, height_axis=self.height_axis)

        # Draw cameras if available
        self.draw_cameras()

        # Points for this frame
        pts = self.frames_points.get(frame_idx, [])
        if pts:
            X = np.array([[x, y, z] for (_c, x, y, z) in pts])
            colors = [self.class_colors[cid] for (cid, _x, _y, _z) in pts]
            self.ax.scatter(X[:,0], X[:,1], X[:,2], c=colors, s=40, depthshade=True)

        # Add legend for classes present in this frame
        classes_in_frame = sorted(set(cid for (cid, _x, _y, _z) in pts))
        if classes_in_frame:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=self.class_colors[cid], label=f'Class {cid}') 
                             for cid in classes_in_frame]
            self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # Axes labels and limits
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m) [height]')

        # Use fixed global bounds
        xmin, xmax, ymin, ymax, zmin, zmax = self.fixed_bounds
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_zlim(zmin, zmax)

        # Set aspect ratio to match the data ranges so the court doesn't look square
        self.ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))

        self.ax.set_title(f"Frame {frame_idx} | {len(pts)} punti")
        plt.draw()

def visualize_triangulated_points(points_json: Path | None = None, points_csv: Path | None = None, camparams_dir: Path = Path('camparams')):
    """
    Visualizza i punti triangolati in modo interattivo.
    
    Args:
        points_json: Percorso al file points.json (prioritario se fornito)
        points_csv: Percorso al file points.csv (fallback)
        camparams_dir: Directory con le calibrazioni delle camere
    
    Se nessun file è specificato, cerca automaticamente in runs/triangulation/
    """
    # Load triangulated points (prefer JSON, fallback to CSV)
    if points_json is None and points_csv is None:
        base_dir = Path('runs') / 'triangulation'
        points_json = base_dir / 'points.json'
        points_csv = base_dir / 'points.csv'
    
    frames_points = None
    classes = []
    
    if points_json and points_json.exists():
        frames_points, classes = load_points_json(points_json)
    elif points_csv and points_csv.exists():
        frames_points, classes = load_points_csv(points_csv)
    else:
        print(f"Punti non trovati in {points_json} o {points_csv}. Esegui triangulation_3d.py.")
        return

    # Try to load camera params (optional)
    cams = find_available_cams(camparams_dir)
    calibs = load_calibrations_for_cams(cams) if cams else {}

    viewer = FrameViewer3D(frames_points, classes, calibs if calibs else None)
    
    # Automatically save trajectories on startup
    viewer.save_top_down_trajectories()
    viewer.save_side_view_trajectory_ball()

    print("Navigazione: frecce sinistra/destra (o A/D), Home, End")
    print("Funzioni: 's' per salvare le traiettorie (vista dall'alto) come immagini")
    plt.show()


def main():
    """Entry point per esecuzione standalone da CLI"""
    visualize_triangulated_points()


if __name__ == '__main__':
	main()

# ----------------------------------------
# How to run (PowerShell)
# ----------------------------------------
# Visualizza i punti triangolati (preferisce runs/triangulation/points.json, fallback a CSV):
#   python visualizer_3d.py
# Navigazione: frecce sinistra/destra (o A/D) per frame precedente/successivo, Home/End per primo/ultimo.
