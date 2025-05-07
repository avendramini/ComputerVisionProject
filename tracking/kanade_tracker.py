import numpy as np
import cv2

class KanadeTracker:
    def __init__(self, max_age=5, win_size=(15, 15), max_level=2):
        self.next_id = 0
        self.tracks = {}  # id -> {'bbox': [x1,y1,x2,y2], 'points': np.array, 'age': 0, 'conf': 1.0, 'class_id': 0}
        self.max_age = max_age
        self.win_size = win_size
        self.max_level = max_level
        self.prev_gray = None
        self.min_iou = 0.3
        self.max_distance = 100  # Distanza massima tra centri delle box
        self.max_tracks = 13  # Numero massimo di tracce attive

    def extract_points(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        points = cv2.goodFeaturesToTrack(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), maxCorners=10, qualityLevel=0.01, minDistance=3)
        if points is not None:
            points = points.reshape(-1, 2) + np.array([x1, y1])
        return points

    def compute_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def compute_center_distance(self, bbox1, bbox2):
        center1 = np.array([(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2])
        center2 = np.array([(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2])
        return np.linalg.norm(center1 - center2)

    def update(self, frame, detections):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        updated_tracks = {}
        assigned = set()
        
        # Aggiorna tracce esistenti
        for tid, track in self.tracks.items():
            if track['points'] is not None and self.prev_gray is not None:
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, track['points'].astype(np.float32), None, winSize=self.win_size, maxLevel=self.max_level)
                if new_points is not None and status.sum() > 0:
                    mean_shift = new_points[status.flatten() == 1].mean(axis=0) - track['points'][status.flatten() == 1].mean(axis=0)
                    new_bbox = np.array(track['bbox']) + np.tile(mean_shift, 2)
                    updated_tracks[tid] = {
                        'bbox': new_bbox,
                        'points': new_points,
                        'age': 0,
                        'conf': track['conf'],
                        'class_id': track['class_id']
                    }
                    assigned.add(tid)
                else:
                    # Se non ci sono punti validi, invecchia la traccia
                    if track['age'] + 1 < self.max_age:
                        updated_tracks[tid] = {
                            'bbox': track['bbox'],
                            'points': None,
                            'age': track['age'] + 1,
                            'conf': track['conf'],
                            'class_id': track['class_id']
                        }

        # Associa nuove detections a tracce non assegnate
        unassigned_dets = []
        for det in detections:
            best_score = 0
            best_tid = None
            
            # Cerca il track esistente con il miglior matching
            for tid, track in updated_tracks.items():
                if tid not in assigned:
                    iou = self.compute_iou(det[:4], track['bbox'])
                    distance = self.compute_center_distance(det[:4], track['bbox'])
                    
                    # Calcola uno score combinato
                    score = iou * (1 - min(distance/self.max_distance, 1))
                    
                    if score > best_score and iou > self.min_iou and distance < self.max_distance:
                        best_score = score
                        best_tid = tid
            
            if best_tid is not None:
                # Aggiorna il track esistente
                points = self.extract_points(frame, det[:4])
                updated_tracks[best_tid] = {
                    'bbox': det[:4],
                    'points': points,
                    'age': 0,
                    'conf': det[4],
                    'class_id': det[5]
                }
                assigned.add(best_tid)
            else:
                # Crea una nuova traccia solo se non è troppo vicina a tracks esistenti
                too_close = False
                for tid, track in updated_tracks.items():
                    if self.compute_center_distance(det[:4], track['bbox']) < self.max_distance:
                        too_close = True
                        break
                
                if not too_close:
                    unassigned_dets.append(det)

        # Aggiungi nuove tracce per detections non assegnate
        for det in unassigned_dets:
            points = self.extract_points(frame, det[:4])
            updated_tracks[self.next_id] = {
                'bbox': det[:4],
                'points': points,
                'age': 0,
                'conf': det[4],
                'class_id': det[5]
            }
            self.next_id += 1

        # Limita il numero di tracce attive a max_tracks
        if len(updated_tracks) > self.max_tracks:
            # Ordina le tracce per confidenza e mantieni solo le prime max_tracks
            sorted_tracks = sorted(updated_tracks.items(), 
                                 key=lambda x: (x[1]['conf'], -x[1]['age']), 
                                 reverse=True)
            updated_tracks = dict(sorted_tracks[:self.max_tracks])

        self.tracks = updated_tracks
        self.prev_gray = frame_gray.copy()
        
        # Ritorna le tracce attive con ID
        return [list(self.tracks[tid]['bbox']) + [self.tracks[tid]['conf'], self.tracks[tid]['class_id'], tid] for tid in self.tracks] 