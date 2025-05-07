from utils import (
    split_and_sort_by_camera, 
    select_polygon_scaled, 
    select_roi_scaled, 
    is_center_in_polygon, 
    is_center_in_rectangle
)
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tracking.assignment_tracker import AssignmentTracker

images_path="dataset/train/images"
labels_path="dataset/train/labels"
camera_datasets=split_and_sort_by_camera(images_path, labels_path)

detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path='yolo12x.pt',
    confidence_threshold=0.1,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

# Convert the BGR image to RGB
#ok = cv2.cvtColor(camera_datasets[13][3][0], cv2.COLOR_BGR2RGB)
# Apply deblurring using a GaussianBlur filter
#deblurred_image = cv2.GaussianBlur(ok, (5, 5), 0)
#result = get_sliced_prediction(
#    ok, 
#    detection_model=detection_model, 
#    slice_height=800,
#    slice_width=800,
#    overlap_height_ratio=0.3,
#    overlap_width_ratio=0.3
#)
#result.export_visuals(export_dir="demo_data/", hide_conf=False,hide_labels=False)

class_names= ['Ball', 'Red_0', 'Red_11', 'Red_12', 'Red_16', 'Red_2', 'Refree_F', 'Refree_M', 'White_13', 'White_16', 'White_25', 'White_27', 'White_34']
class_colors = [
    (255, 0, 0),       # blue
    (238, 130, 238),   # violet
    (235, 206, 135),   # skyblue
    (255, 0, 255),     # fuchsia
    (211, 0, 148),     # darkviolet
    (0, 128, 0),       # green
    (42, 42, 165),     # brown
    (203, 192, 255),   # pink
    (0, 255, 255),     # yellow
    (144, 238, 144),   # lightgreen
    (0, 0, 255),       # red
    (0, 165, 255),     # orange
    (255, 255, 0)      # cyan
]

path_videos={4:'dataset/video/out4.mp4', 13:'dataset/video/out13.mp4'}
frame_videos={
    4:[],#extract_frames_from_video(path_videos[0]),
    13:[]#extract_frames_from_video(path_videos[1])
}

selected_classes=[0]
offset = 2
cameras=[13]
prev_labels={4:[],13:[]}

# --- Selezione poligono e rettangolo all'inizio ---
for cam in cameras:
    # Prima selezione del ROI
    cap = cv2.VideoCapture(path_videos[cam])
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Impossibile leggere il primo frame per la selezione della zona")
    polygon = select_polygon_scaled(first_frame, display_width=1200)
    rectangle = select_roi_scaled(first_frame, display_width=1200)
    cap.release()
    
    # Riapriamo il video per il processing
    cap = cv2.VideoCapture(path_videos[cam])
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video per la camera {cam}")
    
    # Reset del tracker per ogni camera
    tracker = AssignmentTracker()
    
    frame_idx=0
    while True:
        print(frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        x, y, w, h = cv2.boundingRect(polygon)
        crop = frame[y:y+h, x:x+w]
        # Detection solo sul crop
        result = get_sliced_prediction(
            crop,
            detection_model=detection_model,
            slice_height=800,
            slice_width=800,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3
        )
        object_prediction_list = result.object_prediction_list
        # Filtra solo le classi di interesse
        filtered_predictions = [
            pred for pred in object_prediction_list if pred.category.id in selected_classes
        ]
        # Prepara le detections per il tracking
        detections = []
        frame_height, frame_width = frame.shape[:2]
        for pred in filtered_predictions:
            bbox = pred.bbox
            # Trasla le box alle coordinate globali
            x_min = bbox.minx + x
            y_min = bbox.miny + y
            x_max = bbox.maxx + x
            y_max = bbox.maxy + y
            # Normalizza le coordinate
            x_min = max(0, min(x_min, frame_width))
            y_min = max(0, min(y_min, frame_height))
            x_max = max(0, min(x_max, frame_width))
            y_max = max(0, min(y_max, frame_height))
            
            # Calcola l'intersezione con il rettangolo
            rect_x, rect_y, rect_w, rect_h = rectangle
            rect_x2 = rect_x + rect_w
            rect_y2 = rect_y + rect_h
            
            # Calcola l'area di intersezione
            inter_x1 = max(x_min, rect_x)
            inter_y1 = max(y_min, rect_y)
            inter_x2 = min(x_max, rect_x2)
            inter_y2 = min(y_max, rect_y2)
            
            # Se non c'è intersezione, salta
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue
                
            # Calcola l'area di intersezione
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            box_area = (x_max - x_min) * (y_max - y_min)
            
            # Se l'intersezione è troppo piccola (meno del 10% dell'area della box), salta
            if inter_area < 0.1 * box_area:
                continue
            
            # Filtro: centro della box deve essere nel poligono
            if not is_center_in_polygon([x_min, y_min, x_max, y_max], polygon):
                continue
                
            # Controllo di validità
            if x_max <= x_min or y_max <= y_min:
                continue
            # Controllo dimensione
            box_width = x_max - x_min
            box_height = y_max - y_min
            if box_width > frame_width * 0.5 or box_height > frame_height * 0.5:
                continue
            conf = pred.score.value if hasattr(pred, "score") else 1.0
            class_id = pred.category.id
            detections.append([float(x_min), float(y_min), float(x_max), float(y_max), float(conf), int(class_id)])
        
        # Tracking basato su IOU
        # Converti le detections in formato lista
        tracks = []
        if detections:
            # detections è già lista di [x1,y1,x2,y2,conf,class_id]
            tracks = tracker.update(detections=detections)

        print(f"Tracks attivi: {len(tracks)}")
        
        # Visualizza i risultati
        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #plt.figure(figsize=(12, 8))
        #plt.imshow(frame_rgb)
        #ax = plt.gca()
        
        # Disegna il poligono
        #polygon_points = np.array(polygon)
        #ax.add_patch(plt.Polygon(polygon_points, fill=False, edgecolor='red', linewidth=2))
        
        # Disegna il rettangolo
        #rect_x, rect_y, rect_w, rect_h = rectangle
        #ax.add_patch(plt.Rectangle((rect_x, rect_y), rect_w, rect_h, 
        #                         fill=False, edgecolor='blue', linewidth=2))
        
        #for track in tracks:
        #    x_min, y_min, x_max, y_max, conf, class_id, track_id = track
            
            # Controllo dimensione
            #box_width = x_max - x_min
            #box_height = y_max - y_min
            #if box_width > 300 or box_height > 300:
            #    continue
                
            #color = "lime" if class_id == 0 else "orange"
            
            # Disegna la box
            #rect = plt.Rectangle(
            #    (x_min, y_min),
            #    box_width,
            #    box_height,
            #    edgecolor=color,
            #    facecolor="none",
            #    linewidth=2,
            #)
            #ax.add_patch(rect)
            
            # Aggiungi il testo
            #text = plt.text(
            #    x_min,
            #    y_min - 5,
            #    f"ID {track_id} (cls {class_id})",
            #    color=color,
            #    fontsize=10,
            #    bbox=dict(facecolor="black", alpha=0.5),
            #)
        
        #plt.axis("off")
        #plt.title(f"Frame {frame_idx}")
        #plt.show()
        if frame_idx == 200:
            break
        # Per demo: ferma dopo 5 frame
        frame_idx+=1
        frame_videos[cam].append(frame)
        prev_labels[cam].append(tracks)
    
    cap.release()

    # Salva il video con le track
    print(f"\nSalvataggio del video con le track per la camera {cam}...")
    display_width = 1280
    
    # Prendi le dimensioni del primo frame per inizializzare il video writer
    h, w = frame_videos[cam][0].shape[:2]
    scale = display_width / w
    display_height = int(h * scale)
    
    # Inizializza il video writer
    output_path = f"video_tracking_{cam}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (display_width, display_height))
    
    for frame_idx, frame in enumerate(frame_videos[cam]):
        # Ridimensiona il frame per la visualizzazione
        frame_display = cv2.resize(frame, (display_width, display_height))
        
        
        
        # Disegna le track per questo frame
        if frame_idx < len(prev_labels[cam]):
            for track in prev_labels[cam][frame_idx]:
                x_min, y_min, x_max, y_max, conf, class_id, track_id = track
                
                # Ridimensiona le coordinate della box
                x_min, y_min = int(x_min * scale), int(y_min * scale)
                x_max, y_max = int(x_max * scale), int(y_max * scale)
                
                # Controllo dimensione
                box_width = x_max - x_min
                box_height = y_max - y_min
                if box_width > 300 * scale or box_height > 300 * scale:
                    continue
                    
                color = (0, 255, 0) if class_id == 0 else (0, 165, 255)  # Verde per palla, arancione per giocatori
                
                # Disegna la box
                cv2.rectangle(frame_display, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Aggiungi il testo
                text = f"ID {track_id} (cls {class_id})"
                cv2.putText(frame_display, text, (x_min, y_min - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Aggiungi il numero del frame
        cv2.putText(frame_display, f"Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Scrivi il frame nel video
        out.write(frame_display)
    
    # Rilascia il video writer
    out.release()
    print(f"Video salvato in: {output_path}")

#loss={4:0,13:0}
#for cam in cameras:
#    for i in range(len(camera_datasets[cam])):
#        loss[cam]+=compare_labels(camera_datasets[cam][i][1],prev_labels[cam][offset+i*5])
#    
#    print(f"Loss for camera {cam}: {loss[cam]}")