import numpy as np
from scipy.optimize import linear_sum_assignment

class AssignmentTracker:
    def __init__(self, max_age=5, dist_thresh=100):
        self.next_id = 0
        self.tracks = {}  # id -> {'bbox': [x1,y1,x2,y2], 'age': 0, 'conf': 1.0, 'class_id': 0}
        self.max_age = max_age
        self.dist_thresh = dist_thresh

    def bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def compute_cost_matrix(self, detections):
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                c1 = self.bbox_center(self.tracks[tid]['bbox'])
                c2 = self.bbox_center(det[:4])
                cost_matrix[i, j] = np.linalg.norm(c1 - c2)  # distanza euclidea
        return cost_matrix, track_ids

    def update(self, detections):
        # detections: lista di [x1, y1, x2, y2, conf, class_id]
        if len(self.tracks) == 0:
            # Inizializza tutte le tracce
            for det in detections:
                self.tracks[self.next_id] = {
                    'bbox': det[:4],
                    'age': 0,
                    'conf': det[4],
                    'class_id': det[5]
                }
                self.next_id += 1
            return [list(self.tracks[tid]['bbox']) + [self.tracks[tid]['conf'], self.tracks[tid]['class_id'], tid] for tid in self.tracks.keys()]

        cost_matrix, track_ids = self.compute_cost_matrix(detections)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_dets = set()
        updated_tracks = {}

        # Aggiorna le tracce assegnate
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.dist_thresh:
                tid = track_ids[r]
                updated_tracks[tid] = {
                    'bbox': detections[c][:4],
                    'age': 0,
                    'conf': detections[c][4],
                    'class_id': detections[c][5]
                }
                assigned_tracks.add(tid)
                assigned_dets.add(c)

        # Aggiorna le tracce non assegnate (invecchiano)
        for tid in self.tracks:
            if tid not in assigned_tracks:
                age = self.tracks[tid]['age'] + 1
                if age < self.max_age:
                    updated_tracks[tid] = {
                        'bbox': self.tracks[tid]['bbox'],
                        'age': age,
                        'conf': self.tracks[tid]['conf'],
                        'class_id': self.tracks[tid]['class_id']
                    }

        # Aggiungi nuove tracce per detections non assegnate
        for i, det in enumerate(detections):
            if i not in assigned_dets:
                updated_tracks[self.next_id] = {
                    'bbox': det[:4],
                    'age': 0,
                    'conf': det[4],
                    'class_id': det[5]
                }
                self.next_id += 1

        self.tracks = updated_tracks

        # Ritorna le tracce attive con ID
        return [list(self.tracks[tid]['bbox']) + [self.tracks[tid]['conf'], self.tracks[tid]['class_id'], tid] for tid in self.tracks] 