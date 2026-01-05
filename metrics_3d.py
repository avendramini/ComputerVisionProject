import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def calculate_metrics(json_path: str, fps: float = 25.0):
    """
    Calculates 3D trajectory metrics from a JSON file.
    
    Metrics per class:
    - Max Speed (m/s)
    - Avg Speed (m/s)
    - Total Distance (m)
    - Trajectory Smoothness (Mean Acceleration magnitude in m/s^2)
    """
    path = Path(json_path)
    if not path.exists():
        print(f"Error: File {path} not found.")
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Organize data by class_id -> list of (frame_idx, x, y, z)
    trajectories = defaultdict(list)
    
    frames = data.get("frames", {})
    # Sort frame indices to ensure temporal order
    sorted_frame_indices = sorted([int(k) for k in frames.keys()])
    
    for fi in sorted_frame_indices:
        frame_data = frames[str(fi)]
        for item in frame_data:
            cls_id = item["class_id"]
            pt = np.array([item["x"], item["y"], item["z"]])
            trajectories[cls_id].append((fi, pt))

    print(f"{'Class ID':<10} | {'Max Speed (m/s)':<15} | {'Avg Speed (m/s)':<15} | {'Total Dist (m)':<15} | {'Smoothness (m/s^2)':<20}")
    print("-" * 85)

    dt_frame = 1.0 / fps

    for cls_id, points in sorted(trajectories.items()):
        if len(points) < 3: # Need at least 3 points for Acceleration
            print(f"{cls_id:<10} | {'N/A':<15} | {'N/A':<15} | {'0.00':<15} | {'N/A':<20}")
            continue

        # Calculate velocities
        velocities = []
        speeds = []
        distances = []
        velocity_times = [] # Center time of the velocity interval
        
        for i in range(len(points) - 1):
            f1, p1 = points[i]
            f2, p2 = points[i+1]
            
            # Time difference in seconds
            dt = (f2 - f1) * dt_frame
            
            if dt <= 0:
                continue

            dist = np.linalg.norm(p2 - p1)
            velocity = (p2 - p1) / dt
            speed = dist / dt
            
            distances.append(dist)
            velocities.append(velocity)
            speeds.append(speed)
            velocity_times.append((f1 + f2) / 2.0)

        if not speeds:
            print(f"{cls_id:<10} | {'N/A':<15} | {'N/A':<15} | {'0.00':<15} | {'N/A':<20}")
            continue

        # Calculate accelerations
        accelerations = []
        for i in range(len(velocities) - 1):
            # Time difference between velocity samples
            t1 = velocity_times[i]
            t2 = velocity_times[i+1]
            dt_acc = (t2 - t1) * dt_frame
            
            if dt_acc <= 0:
                continue

            acc = (velocities[i+1] - velocities[i]) / dt_acc
            accelerations.append(np.linalg.norm(acc))

        total_dist = sum(distances)
        max_speed = max(speeds)
        avg_speed = np.mean(speeds)
        
        # Smoothness: Mean magnitude of Acceleration (lower is smoother)
        smoothness = np.mean(accelerations) if accelerations else 0.0

        print(f"{cls_id:<10} | {max_speed:<15.2f} | {avg_speed:<15.2f} | {total_dist:<15.2f} | {smoothness:<20.2f}")

