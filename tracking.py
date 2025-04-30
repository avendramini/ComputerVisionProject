import numpy as np
from utils import *
# Extract the image and labels from YOLO results

def label_distance(label1, label2):
    """
    Calcola la distanza tra due etichette in formato YOLO [class, x_center, y_center, width, height].
    """
    return np.sqrt((label1[1] - label2[1]) ** 2 + (label1[2] - label2[2]) ** 2)

def convert_model_labels(results):
    image_width, image_height = results[0].orig_shape[1], results[0].orig_shape[0]
    labels=[]
    for box in results[0].boxes.data.cpu().numpy():
        x_min, y_min, x_max, y_max, confidence, class_id = box
        x_center = ((x_min + x_max) / 2 )/image_width
        y_center = ((y_min + y_max) / 2 )/image_height
        width = (x_max - x_min)/image_width
        height = (y_max - y_min)/image_height
        labels.append([int(class_id), x_center, y_center, width, height])
    return labels

def class_to_yolo_class_id(tracking_class_id): 
    """
    Converte l'ID della classe di tracking in ID YOLO.
    """
    if tracking_class_id != 0:
        return 37 #palla in yolo
    return 0 #persona in yolo
    
def find_labels(new_frame, old_frame, prev_labels,model):
    """
    Trova le etichette nel nuovo frame confrontando con il frame precedente.
    Ritorna le etichette aggiornate.
    """
    results = model(new_frame, conf=0.2)
    obj_labels = convert_model_labels(results) # Contiene ogni oggetto trovato da yolo
    show_image_with_labels(new_frame,obj_labels,max_size=800)
    new_labels=[]
    rimaste=prev_labels.copy()
    for obj in obj_labels:
        if len(rimaste)==0:
            break
        rimaste.sort(key=lambda x: label_distance(obj, x))
        scelta=rimaste.pop(0) # Soglia di distanza
        scelta=scelta[0],obj[1],obj[2],obj[3],obj[4] # Cambia solo la posizione e le dimensioni
        new_labels.append(scelta)
    return new_labels

def bbox_iou(boxA, boxB):
    """
    Calcola la IoU tra due box in formato YOLO [class, x_center, y_center, width, height].
    """
    # Converti da YOLO a [x1, y1, x2, y2]
    def yolo_to_xy(box):
        _, xc, yc, w, h = box
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        return [x1, y1, x2, y2]
    a = yolo_to_xy(boxA)
    b = yolo_to_xy(boxB)
    # Intersezione
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # Aree
    boxAArea = (a[2] - a[0]) * (a[3] - a[1])
    boxBArea = (b[2] - b[0]) * (b[3] - b[1])
    # IoU
    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou

def compare_labels(old_labels, new_labels):
    """
    Confronta le etichette di due frame usando la mean IoU tra box della stessa classe.
    Ritorna 1 - mIoU come loss (più basso è meglio).
    """
    if len(old_labels) == 0 or len(new_labels) == 0:
        return 1.0  # Loss massima se non ci sono box

    ious = []
    for old in old_labels:
        same_class = [new for new in new_labels if int(new[0]) == int(old[0])]
        if not same_class:
            ious.append(0.0)
            continue
        ious.append(max(bbox_iou(old, new) for new in same_class))
    mIoU = np.mean(ious)
    return 1 - mIoU