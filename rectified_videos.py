def rectify_labels(label_dir, calib_path, img_w, img_h, output_dir):
    """
    Rettifica le label YOLO (x_center, y_center) usando la stessa mappa di undistortion del video.
    Salva i file rettificati in output_dir.
    """
    mtx, dist = load_calibration(calib_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    for fname in label_files:
        in_path = os.path.join(label_dir, fname)
        out_path = os.path.join(output_dir, fname)
        new_lines = []
        with open(in_path, 'r') as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = parts[0]
                x, y, w, h = map(float, parts[1:5])
                conf = parts[5] if len(parts) > 5 else None
                # Denormalizza centro bbox
                px = x * img_w
                py = y * img_h
                # Rettifica centro bbox
                pt = np.array([[[px, py]]], dtype=np.float32)  # shape (1,1,2)
                pt_rect = cv2.undistortPoints(pt, mtx, dist, P=mtx)
                rx, ry = pt_rect[0,0,0], pt_rect[0,0,1]
                # Rinormalizza
                x_new = rx / img_w
                y_new = ry / img_h
                # Mantieni w,h invariati (approssimazione)
                if conf is not None:
                    new_line = f"{cls} {x_new:.6f} {y_new:.6f} {w:.6f} {h:.6f} {conf}\n"
                else:
                    new_line = f"{cls} {x_new:.6f} {y_new:.6f} {w:.6f} {h:.6f}\n"
                new_lines.append(new_line)
        with open(out_path, 'w') as fout:
            fout.writelines(new_lines)
import cv2
import numpy as np
import json
import os
import glob
import re

def load_calibration(calib_path):
    # Load the camera calibration parameters from a JSON file.
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

def process_video(video_path, calib_path, output_path):
    mtx, dist = load_calibration(calib_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    pts = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)
    

    undistorted_pts = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    undistorted_map = undistorted_pts.reshape(height, width, 2)
    map_x = undistorted_map[:, :, 0]
    map_y = undistorted_map[:, :, 1]
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Apply the undistortion map to the frame
        rectified_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        out.write(rectified_frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames for {video_path}")
    
    cap.release()
    out.release()
    print(f"Finished processing video: {video_path}")

def main():
    video_files = glob.glob("./dataset/video/out*.mp4") # path to the video files
    output_dir = "./rectified/" # folder path where to save the rectified videos
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Rettifica anche le label YOLO
    for video_path in video_files:
        basename = os.path.basename(video_path)
        match = re.search(r'out(\d+)\.mp4', basename)
        if match:
            cam_index = match.group(1)
            calib_path = os.path.join("Camera_config2", f"cam_{cam_index}", "calib", "camera_calib.json")
            label_dir = os.path.join("dataset", "infer_video", f"labels_out{cam_index}")
            output_label_dir = os.path.join(output_dir, f"labels_out{cam_index}")
            # Ricava dimensione immagine dal video
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if not ret:
                print(f"Impossibile leggere {video_path} per dimensione immagine")
                continue
            img_h, img_w = frame.shape[:2]
            cap.release()
            rectify_labels(label_dir, calib_path, img_w, img_h, output_label_dir)

    for video_path in video_files:
        basename = os.path.basename(video_path)
        match = re.search(r'out(\d+)\.mp4', basename)
        if match:
            cam_index = match.group(1)
            calib_path = os.path.join("Camera_config2", f"cam_{cam_index}", "calib", "camera_calib.json")
        else:
            print("Could not extract camera index from filename:", video_path)
            continue
        output_path = os.path.join(output_dir, '', basename)
        if not os.path.exists(os.path.join(output_dir, '')):
            os.makedirs(os.path.join(output_dir, ''))
        print(f"Processing {video_path} using calibration file {calib_path}...")
        process_video(video_path, calib_path, output_path)

if __name__ == "__main__":
    main()

# ----------------------------------------
# How to run (PowerShell)
# ----------------------------------------
# Rettifica tutti i video in dataset/video/out*.mp4 secondo le calibrazioni e salva in ./rectified/:
#   python rectified_videos.py
# Nota: Il percorso delle calibrazioni è impostato su Camera_config2/cam_{id}/calib/camera_calib.json;
# adatta questi percorsi se le tue calibrazioni sono in cartelle diverse (es. camparams/...).
