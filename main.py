from utils import *
from tracking.tracking import *
from ultralytics import YOLO
import cv2
import random

# Funzione per assegnare un colore unico a ogni ID
def get_color(idx):
    random.seed(idx)
    return tuple(random.randint(0, 255) for _ in range(3))

images_path = "dataset/train/images"
labels_path = "dataset/train/labels"
camera_datasets = split_and_sort_by_camera(images_path, labels_path)
tiler = YOLO("yolo12x.pt")
# Salva un'immagine di esempio con i risultati della predizione
# Esegui il tracking e salva i risultati frame per frame
results = tiler.track(source='dataset/video/out13.mp4',conf=0.1, show=False, imgsz=(1280, 720), save=False, stream=True,persist=True)

saved_frames = []
for result in results:
    frame = result.orig_img.copy()
    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    classes = result.boxes.cls.cpu().numpy().astype(int)
    # Prendi gli ID di tracking se disponibili
    ids = result.boxes.id.cpu().numpy().astype(int) if hasattr(result.boxes, "id") and result.boxes.id is not None else [None]*len(boxes)
    for box, cls, track_id in zip(boxes, classes, ids):
        x1, y1, x2, y2 = map(int, box)
        color = get_color(track_id) if track_id is not None else (0, 255, 0)
        label = f"ID {track_id}" if track_id is not None else str(cls)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    frame_resized = cv2.resize(frame, (1280, 720))
    saved_frames.append(frame_resized)
    if len(saved_frames) >= 100:
        break

# Mostra i frame salvati in una finestra 1280x720
for frame in saved_frames:
    cv2.imshow("Risultati YOLO", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
