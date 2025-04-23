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
