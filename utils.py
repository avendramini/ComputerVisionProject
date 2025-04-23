import os
from typing import List, Tuple
import cv2
import random
import re

def split_and_sort_by_camera(images_dir: str, labels_dir: str, exts: List[str] = [".jpg", ".png"]):
    """
    Divide il dataset in base alla telecamera e ordina i frame per numero di frame.
    Ritorna un dizionario: {camera_number: [(img, labels, frame_number), ...]}
    """
    pattern = re.compile(r"out(\d+)_frame_(\d+)")
    camera_dict = {}

    for fname in os.listdir(images_dir):
        if any(fname.lower().endswith(ext) for ext in exts):
            match = pattern.search(fname)
            if not match:
                continue
            cam_num = int(match.group(1))
            frame_num = int(match.group(2))
            img_path = os.path.join(images_dir, fname)
            label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")
            img = cv2.imread(img_path)
            labels = []
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            labels.append([float(x) for x in parts])
            if cam_num not in camera_dict:
                camera_dict[cam_num] = []
            camera_dict[cam_num].append((img, labels, frame_num))

    # Ordina per frame_number
    for cam_num in camera_dict:
        camera_dict[cam_num].sort(key=lambda x: x[2])

    return camera_dict

def show_image_with_labels(image, labels, class_names: List[str], class_colors: List[Tuple[int, int, int]], max_size: int = 800):
    """
    Mostra un'immagine con i rettangoli delle labels disegnati sopra, ridimensionata se troppo grande.
    Args:
        image (any): Immagine su cui disegnare (BGR).
        labels (List[List[float]]): Lista di labels in formato YOLO [class, x_center, y_center, width, height].
        class_names (List[str]): Lista dei nomi delle classi.
        class_colors (List[Tuple[int, int, int]]): Lista dei colori per ogni classe.
        max_size (int): Dimensione massima per il lato più lungo dell'immagine.
    """
    h, w, _ = image.shape
    scale = min(max_size / h, max_size / w, 1.0)
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))

    for label in labels:
        class_id, x_center, y_center, width, height = label
        x1 = int((x_center - width / 2) * resized_image.shape[1])
        y1 = int((y_center - height / 2) * resized_image.shape[0])
        x2 = int((x_center + width / 2) * resized_image.shape[1])
        y2 = int((y_center + height / 2) * resized_image.shape[0])
        color = class_colors[int(class_id)] 
        cv2.rectangle(resized_image, (x1, y1), (x2, y2), color, 2)
        if int(class_id) < len(class_names):
            label_text = class_names[int(class_id)]
            cv2.putText(resized_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Image with Labels", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_frames_from_video(video_path):
    """
    Legge un video .mp4 e restituisce i frame come una lista di immagini.
    
    Args:
        video_path (str): Il percorso del file video.
    
    Returns:
        list: Una lista di frame (immagini) estratti dal video.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Errore nell'aprire il video: {video_path}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def show_frame(frame, window_name="Frame", max_size=800):
    """
    Mostra un singolo frame ridimensionato se troppo grande.
    
    Args:
        frame (any): Il frame da mostrare (immagine in formato BGR).
        window_name (str): Nome della finestra di visualizzazione.
        max_size (int): Dimensione massima per il lato più lungo del frame.
    """
    h, w, _ = frame.shape
    scale = min(max_size / h, max_size / w, 1.0)
    resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cv2.imshow(window_name, resized_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_side_by_side(rgb_frame, binary_mask, window_name="Side by Side", max_size=800):
    """
    Mostra due immagini affiancate: un frame RGB e una maschera binaria.
    
    Args:
        rgb_frame (any): Il frame RGB (immagine in formato BGR).
        binary_mask (any): La maschera binaria (immagine in scala di grigi).
        window_name (str): Nome della finestra di visualizzazione.
        max_size (int): Dimensione massima per il lato più lungo delle immagini.
    """
    # Convert binary mask to 3 channels for visualization
    binary_mask_colored = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Resize both images to the same scale
    h1, w1, _ = rgb_frame.shape
    h2, w2, _ = binary_mask_colored.shape
    scale = min(max_size / max(h1, w1), max_size / max(h2, w2), 1.0)
    resized_rgb = cv2.resize(rgb_frame, (int(w1 * scale), int(h1 * scale)))
    resized_mask = cv2.resize(binary_mask_colored, (int(w2 * scale), int(h2 * scale)))

    # Concatenate the images side by side
    side_by_side = cv2.hconcat([resized_rgb, resized_mask])

    # Show the concatenated image
    cv2.imshow(window_name, side_by_side)
    cv2.waitKey(0)
    cv2.destroyAllWindows()