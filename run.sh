#!/bin/bash
# Esegue la pipeline con i parametri di default definiti in config.py
# Default:
# - Cameras: 2, 4, 13
# - Labels dir: dataset/infer_video
# - Interpolate: True
# - Rectify: True
# - Visualize: False
# - Infer videos: False (usa labels esistenti)
# - Evaluate labels: True

python pipeline.py --infer-videos --evaluate-labels --device 0 --imgsz 1920 --visualize